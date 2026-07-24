from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path.home() / "projects" / "timestamp_gen"

BLEND_SCRIPT = PROJECT_DIR / "create_blend.slurm"
RENDER_SCRIPT = PROJECT_DIR / "render_visionsim.slurm"
TIMESTAMP_SCRIPT = PROJECT_DIR / "run_timestamp_gen.slurm"


def submit_job(script: Path, dependency: str | None = None) -> str:
    if not script.exists():
        raise FileNotFoundError(f"SLURM file not found: {script}")

    command = ["sbatch", "--parsable"]

    if dependency is not None:
        command.append(f"--dependency=afterok:{dependency}")

    command.append(str(script))

    result = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        check=True,
        text=True,
        capture_output=True,
    )

    # Some clusters return "12345", while others may return "12345;cluster".
    job_id = result.stdout.strip().split(";")[0]

    if not job_id:
        raise RuntimeError(
            f"Could not determine job ID from sbatch output: {result.stdout!r}"
        )

    return job_id


def main() -> None:
    try:
        blend_job = submit_job(BLEND_SCRIPT)
        print(f"Blend job submitted:     {blend_job}")

        render_job = submit_job(
            RENDER_SCRIPT,
            dependency=blend_job,
        )
        print(f"VisionSIM job submitted: {render_job}")

        timestamp_job = submit_job(
            TIMESTAMP_SCRIPT,
            dependency=render_job,
        )
        print(f"Timestamp job submitted: {timestamp_job}")

        print()
        print("Pipeline submitted successfully.")
        print(f"Track jobs with: squeue -j {blend_job},{render_job},{timestamp_job}")

    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"Pipeline submission failed: {exc}", file=sys.stderr)

        if isinstance(exc, subprocess.CalledProcessError):
            if exc.stdout:
                print(exc.stdout, file=sys.stderr)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr)

        raise SystemExit(1)


if __name__ == "__main__":
    main()