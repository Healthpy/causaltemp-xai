"""Static and executable regressions for the Helios run-2 harness."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm"
HELIOS_JOBS = sorted(SLURM.glob("helios_*.sbatch"))


def test_all_shell_files_parse():
    for path in sorted(SLURM.glob("*.sbatch")) + sorted(SLURM.glob("*.sh")):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_every_helios_job_has_resource_threading_and_cleanliness_gates():
    assert HELIOS_JOBS
    for path in HELIOS_JOBS:
        body = path.read_text()
        assert "#SBATCH --account=plgcountercontex-gpu-gh200" in body
        assert "#SBATCH --partition=plgrid-gpu-gh200" in body
        assert "SLURM_CPUS_PER_TASK" in body
        assert "unset OMP_NUM_THREADS MKL_NUM_THREADS" in body
        assert "export OMP_NUM_THREADS=$CORES MKL_NUM_THREADS=$CORES" in body
        assert "preflight_source_clean" in body or "preflight_runtime" in body
        assert "finish" in body


def test_no_legacy_cluster_or_environment_runner_survives():
    bodies = "\n".join(path.read_text() for path in SLURM.iterdir() if path.is_file())
    assert "tesr123566" not in bodies
    assert "gpu_a100" not in bodies
    assert "uv sync" not in bodies
    assert "uv run" not in bodies
    assert "OMP_NUM_THREADS=$(nproc)" not in bodies


def test_full_job_contains_the_six_phase_sequence_and_hard_checks():
    body = (SLURM / "helios_full_run.sbatch").read_text()
    for phase in (
        "01_generate_benchmarks.py",
        "02_train_classifiers.py",
        "03_run_cf_methods.py",
        "04_evaluate_axes.py",
        "05_run_oracle_control.py",
        "07_auxiliary_methods.py",
    ):
        assert phase in body
    assert '[[ "$1" != "full" && "$1" != "full_nl" ]]' in body
    assert "do_complexity_mean_pearl_scorable" in body
    assert "NoiselessSCMRecourse" in body and "PearlSCMRecourse" in body
    assert 'mv "results/$CONFIG" "$archive_root/results/$CONFIG"' in body
    assert 'if [[ "$CONFIG" == "full" && -d results/tables ]]' in body


def test_reports_job_contains_exact_publication_scope_and_checks():
    body = (SLURM / "helios_reports.sbatch").read_text()
    for command in (
        "08_aggregate_and_report.py figures",
        "08_aggregate_and_report.py pns --config full --seeds",
        "08_aggregate_and_report.py pns --config full_nl --seeds",
        "make_final_table.py --configs full full_nl",
    ):
        assert command in body
    assert "final_table_long.csv" in body
    assert "final_table_provenance.json" in body
    assert "if (( $# != 0 ))" in body


def test_smoke_archives_old_config_before_running():
    body = (SLURM / "helios_smoke.sbatch").read_text()
    archive = 'mv "results/$CONFIG" "$archive_root/results/$CONFIG"'
    first_phase = "experiments/01_generate_benchmarks.py"
    assert archive in body
    assert body.index(archive) < body.index(first_phase)


def test_submission_is_sequential_and_jobids_live_outside_checkout():
    body = (SLURM / "submit_helios_run2.sh").read_text()
    assert "afterany:$A" in body
    assert "afterany:$B" in body
    assert '"$HEAVY/run2/run2.jobids"' in body
    assert "slurm/run2.jobids" not in body


def test_sync_excludes_secrets_results_and_notebooks():
    body = (SLURM / "sync_helios_code.sh").read_text()
    for exclusion in ("'.env'", "'results'", "'notebooks'"):
        assert f"--exclude {exclusion}" in body
    assert "--delete-excluded" not in body
    assert "test ! -e .env" in body
    assert body.count("rsync -az") == body.count("--exclude '.env'") == 2
    assert "--files-from" in body


def test_step_failure_does_not_skip_later_work_but_finish_fails(tmp_path):
    helper = SLURM / "helios_harness.sh"
    marker = tmp_path / "later-step-ran"
    command = f'source "{helper}"; step false; step touch "{marker}"; finish'
    proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert proc.returncode == 1
    assert marker.exists()
    assert "FAILED(rc=1): false" in proc.stdout
    assert "JOB FAILED: 1 step(s) failed" in proc.stderr
