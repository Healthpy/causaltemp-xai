---
name: plgrid-run
description: Set up and use PLGrid environments for GPU experiments, especially Helios GH200 and Athena A100. Use when Codex needs to connect to PLGrid over SSH, verify grants and storage, organize code and heavy files around the home quota, create project-scoped environments and cache links, prepare Slurm jobs, submit or monitor experiments, diagnose failures, or retrieve results.
---

# Run workloads on PLGrid

Use this skill as procedural guidance. It provides no `scripts/` tools or
controller. Treat executable-looking files under `assets/` as examples to
inspect and adapt, not as commands to invoke blindly. Execute standard local,
SSH, filesystem, and Slurm commands only as required by the user's request.

## Route the task

- Read [references/setup-environment.md](references/setup-environment.md) in
  full when connecting, checking access, preparing storage, creating Python
  environments, or arranging project directories and symlinks.
- Read [references/running-experiments.md](references/running-experiments.md)
  in full when preparing, submitting, monitoring, troubleshooting, or
  retrieving a Slurm experiment.
- Read both references when taking a project from first login through a smoke
  experiment.

## Apply these rules

1. Prefer an SSH key or SSH agent. Never print, commit, or copy `.env` or
   passwords to PLGrid.
2. Inspect live `hpc-grants`, `hpc-fs`, partition, and module information
   before relying on a bundled value. Treat the documented cluster profiles as
   verified examples, not permanent service discovery.
3. Keep code and small configuration in the cluster home directory. Keep
   environments, caches, datasets, models, and durable outputs in allocated
   group storage.
4. Create project-scoped links such as `project/.venv` and `project/.cache`.
   Do not replace global `~/.local` or `~/.cache`.
5. Refuse to overwrite an existing path or unexpected symbolic link during
   storage setup.
6. Run `sbatch --test-only` before a new job shape. Submit work that consumes
   allocation only when the user asks to run or submit it.
7. Treat `$SCRATCH` as temporary. Copy durable results back to group storage.
8. Verify jobs with `sacct`, logs, and expected result files; a submitted job
   ID alone is not evidence of success.

## Use the examples

Use files under `assets/examples/` as copyable starting points when the user
asks for an example project or smoke test:

- `assets/examples/helios-gh200-smoke/` is the verified ARM64 GH200 example.
- `assets/examples/athena-a100-smoke/` is the verified x86_64 A100 example.

Adapt account, storage group, project name, requested resources, module
versions, and paths to live output. Do not copy example credentials; the
assets contain none.
