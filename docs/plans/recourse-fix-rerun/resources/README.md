# Shared Resources

- **[commands.md](commands.md)** — local regeneration recipes, the mandatory Helios job preamble,
  the threading fix, the phase sequence, submission and monitoring, and the measured cost
  constants.
- **[checklist.md](checklist.md)** — the 25-item defect checklist that gates whether a generated
  table is usable. Run it in stage 7 (smoke tier) and again in stage 9 (full run).

## Standing constraints

- **`.env` holds `PLG_PASSWORD`.** Never read, print, echo or transfer it. Cluster access is
  key-based only. Every `rsync` carries `--exclude '.env'`.
- **Never `uv sync` on Helios.** The committed `uv.lock` resolves torch 2.12.0 against CUDA 13;
  Helios needs torch 2.9.0+cu129 from the site wheelhouse. A wrong build is a silent CPU
  fallback, not an error.
- **Never run two configs as concurrent Slurm jobs.** `experiments/_common.py::append_table`
  does an unlocked read-modify-write on a shared CSV. Chain with `--dependency=afterany`.
- **Keep submission metadata outside the checkout.** Store job IDs at
  `$HEAVY/run2/run2.jobids`; otherwise dependent jobs fail the source-clean preflight.
- **Publication scope is explicit.** Run 2's final table is built with
  `--configs full full_nl`; smoke tables are validation diagnostics only.
- **Never `export OMP_NUM_THREADS=$(nproc)`.** See [commands.md](commands.md#helios-the-threading-fix-mandatory).
