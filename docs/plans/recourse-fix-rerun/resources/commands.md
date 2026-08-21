# Commands

## Local: regenerate a config from scratch

The laptop's checked-in `data/scm_t/` is dated 2026-07-07/08 and carries the **pre-retune**
nonlinear hyperparameters (`gain=0.8`, `decay_range=[0.3, 0.8]`, `spectral_cap=0.9`,
`init_gain=0.7`). Always regenerate before validating metric/reporting outputs against the
current configuration.

```bash
cd /Users/ofurman/pwr/causaltemp-xai
SD=/tmp/ctxai_repro
.venv/bin/python experiments/01_generate_benchmarks.py --config smoke_nl --out-dir $SD/data
.venv/bin/python experiments/02_train_classifiers.py   --config smoke_nl --out-dir $SD/data
.venv/bin/python experiments/03_run_cf_methods.py      --config smoke_nl --n-cf 20 \
    --methods CARLA PearlCARLA --out-dir $SD/data
```

Regeneration plus training takes about 2 minutes and must reproduce the Helios classifier
accuracies exactly (`smoke_nl`: train 0.96 / val 0.98 / test 0.96). If yours differ, stop.

**Two traps:**
- Phases 01 and 02 write `axis_a_benchmark.json` and `train_report.json` into the repo's
  **tracked** `results/` regardless of `--out-dir` (`experiments/_common.py:64` — `RESULTS_DIR`
  is not env-overridable). Phase 03 likewise overwrites tracked files under
  `results/<config>/lstm/`. Restore with `git checkout` afterwards.
- Use `.venv/bin/python` directly. A bare `uv run` will resync the venv.

## Local: the full smoke tier (stage 5)

```bash
METHODS="CARLA PearlCARLA CftsWachter CftsCOMTE CftsConfeti CftsCels TSCausal"
for C in smoke smoke_nl smoke_spring; do
  .venv/bin/python experiments/01_generate_benchmarks.py --config "$C"
  .venv/bin/python experiments/02_train_classifiers.py   --config "$C"
  .venv/bin/python experiments/03_run_cf_methods.py      --config "$C" --n-cf 20 --methods $METHODS
  .venv/bin/python experiments/04_evaluate_axes.py       --config "$C"
  .venv/bin/python experiments/05_run_oracle_control.py  --config "$C" --n-cf 20
  .venv/bin/python experiments/07_auxiliary_methods.py   --config "$C" --method pns --n-cf 20
done
.venv/bin/python experiments/make_final_table.py --configs smoke smoke_nl smoke_spring
```

`CftsCounts` is excluded from `$METHODS` deliberately — it trains a per-instance conditional VAE
and costs 2-6 hours on its own.

## Helios: mandatory job preamble

```bash
#!/bin/bash -l
PROJECT=/net/home/plgrid/plgofurman/projects/causaltemp-xai
HEAVY=/net/storage/pr3/plgrid/plggcfsgenwro/users/plgofurman/causaltemp-xai
export XDG_CACHE_HOME="$HEAVY/cache/xdg"
export TORCH_HOME="$HEAVY/cache/torch"
export MPLCONFIGDIR="$HEAVY/cache/matplotlib"
export MPLBACKEND=Agg
module purge
module load ML-bundle/25.10
source "$HEAVY/envs/gh200/bin/activate"
cd "$PROJECT"
```

## Helios: the threading fix (mandatory)

```bash
unset OMP_NUM_THREADS MKL_NUM_THREADS
CORES=${SLURM_CPUS_PER_TASK:-$(nproc)}
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
```

**Never write `export OMP_NUM_THREADS=$(nproc)`.** `module load ML-bundle/25.10` sets
`OMP_NUM_THREADS=1`, and GNU `nproc` reports the CPUs available to the current process, honouring
that variable. So that line reads back 1 and re-exports 1, pinning the job single-threaded.
Verified on Helios: plain `nproc` = 64, `OMP_NUM_THREADS=1 nproc` = 1, `OMP_NUM_THREADS=4 nproc` = 4,
`nproc --all` = 64.

Gate it in preflight — `torch.get_num_threads() < 2 -> sys.exit(1)`. Otherwise the failure is
silent at minute zero and appears as a TIMEOUT at hour sixteen.

## Helios: the phase sequence (from run 1)

```bash
METHODS="CARLA PearlCARLA CftsWachter CftsCOMTE CftsConfeti CftsCels TSCausal"

# Job A: full
python experiments/01_generate_benchmarks.py --config full
python experiments/02_train_classifiers.py   --config full
python experiments/03_run_cf_methods.py      --config full    --n-cf 100 --methods $METHODS
python experiments/04_evaluate_axes.py       --config full
python experiments/05_run_oracle_control.py  --config full    --n-cf 100
python experiments/07_auxiliary_methods.py   --config full    --method pns --n-cf 100

# Job B: full_nl   (identical, --config full_nl)

# Job C: reports and table
python experiments/08_aggregate_and_report.py figures
python experiments/08_aggregate_and_report.py pns --config full    --seeds
python experiments/08_aggregate_and_report.py pns --config full_nl --seeds
python experiments/make_final_table.py --configs full full_nl
```

Note the bare `--seeds` on the `pns` subcommand. It is `nargs="*"`, so bare gives `[]`, which
targets the base config directory. Passing nothing at all makes it default to `[0, 1, 2]` and
look for nonexistent `results/<config>_seed{0,1,2}/`.

Phase 03 also runs Shift-VR for Axis B — a second full generation pass on noise-shifted data, so
phase 03 costs roughly twice its base. `--skip-aux` turns it off and halves the cost, at the
price of Axis B.

## Helios: submission

```bash
HEAVY=/net/storage/pr3/plgrid/plggcfsgenwro/users/plgofurman/causaltemp-xai
mkdir -p "$HEAVY/run2"
A=$(sbatch --parsable slurm/helios_full_run.sbatch full)
B=$(sbatch --parsable --dependency=afterany:$A slurm/helios_full_run.sbatch full_nl)
C=$(sbatch --parsable --dependency=afterany:$B slurm/helios_reports.sbatch)
printf '%s\n' "$A" "$B" "$C" > "$HEAVY/run2/run2.jobids"
squeue -j "$A,$B,$C" -o '%.18i %.12P %.24j %.2t %.10M %R'
```

Never write the ID file inside the Git checkout: dependent jobs reject untracked source-tree
files during preflight. The executor records the returned IDs in the plan index after submission.

Sequential execution is **mandatory**: `experiments/_common.py::append_table` does an unlocked
read-modify-write on the shared `results/tables/table_axis_c_cf_faith.csv` at
`experiments/04_evaluate_axes.py:135`, before `summary.json` is written at `:157`. Two concurrent
configs corrupt it.

## Helios: monitoring and verification

```bash
squeue -j "$job_id" -o '%.18i %.12P %.24j %.10u %.2t %.10M %.6D %R'
sacct  -j "$job_id" --format=JobID,JobName,Partition,Account,State,Elapsed,AllocTRES,ReqMem,ExitCode
```

Check **every** job in a dependency chain, not only the last. Accounting lags briefly after a job
leaves the queue — recheck before declaring a job missing.

## Cost constants (measured, run 1)

| quantity | value |
|---|---|
| Phase 03, config `full` | `17s + 381.3s per n_cf unit`, **including** the Shift-VR second pass |
| Phase 03, config `full_nl` | 1.3-1.5x the `full` figure |
| Phase 02 | 342s (test accuracy 0.9905) |
| Phases 01 / 04 / 05 / 07 | minutes |
| Peak MaxRSS | 3.07 GB — 24G is ample, do not over-request |
| Run-1 elapsed | `full` 12:07:25, `full_nl` 12:05:35, reports 01:53:11; ~26 GPU-h total |
| Per-method cost | CftsConfeti 84 s/instance, PearlCARLA 52, CARLA 32, TSCausal 17; Wachter/COMTE/Cels near-free |

The stage-1 prediction-only fallback can at most double optimizer work on failed cases. Compare
the stage-5 smoke runtime with these constants and re-size walltimes before submission if needed.
