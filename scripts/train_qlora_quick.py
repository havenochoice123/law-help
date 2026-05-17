"""Run a QLoRA quick training job with LLaMA-Factory."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLAMA_FACTORY_DIR = PROJECT_ROOT / "LLaMA-Factory"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "qlora_quick.yaml"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "finetune" / "train.jsonl"
DEFAULT_VAL_DATASET = PROJECT_ROOT / "data" / "finetune" / "val.jsonl"
DEFAULT_TEST_DATASET = PROJECT_ROOT / "data" / "eval" / "test.jsonl"
DEFAULT_LOG_FILE = PROJECT_ROOT / "outputs" / "logs" / "train_qlora_quick.log"
LLAMA_FACTORY_DATASETS = {
    "train": LLAMA_FACTORY_DIR / "data" / "disc_law_sft_train_split.jsonl",
    "val": LLAMA_FACTORY_DIR / "data" / "disc_law_sft_val.jsonl",
    "test": LLAMA_FACTORY_DIR / "data" / "disc_law_sft_test.jsonl",
}


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def find_cli() -> list[str]:
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "Python 3.14 is not supported by the current training stack. "
            "Use a Python 3.10 or 3.11 virtual environment."
        )

    cli = shutil.which("llamafactory-cli")
    if cli:
        return [cli]

    train_py = LLAMA_FACTORY_DIR / "src" / "train.py"
    if train_py.exists():
        return [sys.executable, str(train_py)]

    raise FileNotFoundError(
        "Cannot find llamafactory-cli or LLaMA-Factory/src/train.py. "
        "Install LLaMA-Factory first."
    )


def ensure_datasets(dataset_path: Path, val_path: Path, test_path: Path, link_dataset: bool) -> None:
    required = {
        "train": dataset_path,
        "val": val_path,
        "test": test_path,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset split files not found:\n"
            + "\n".join(missing)
            + "\nRun scripts/data_prep.py and scripts/split_dataset.py first."
        )

    if not LLAMA_FACTORY_DIR.exists():
        raise FileNotFoundError(f"LLaMA-Factory directory not found: {LLAMA_FACTORY_DIR}")

    if link_dataset:
        for name, source_path in required.items():
            target_path = LLAMA_FACTORY_DATASETS[name]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)


def build_command(config: Path) -> list[str]:
    command = find_cli()
    if Path(command[-1]).stem == "llamafactory-cli":
        return command + ["train", str(config)]
    return command + [str(config)]


def stream_process(command: list[str], cwd: Path, env: dict, log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QLoRA quick training.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to LLaMA-Factory YAML config.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to train JSONL dataset.")
    parser.add_argument("--val-dataset", default=str(DEFAULT_VAL_DATASET), help="Path to validation JSONL dataset.")
    parser.add_argument("--test-dataset", default=str(DEFAULT_TEST_DATASET), help="Path to test JSONL dataset.")
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE), help="Path to write the training log.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the resolved command.")
    parser.add_argument(
        "--no-link-dataset",
        action="store_true",
        help="Do not copy split datasets into LLaMA-Factory/data/.",
    )
    args = parser.parse_args()

    config = resolve_path(args.config)
    dataset = resolve_path(args.dataset)
    val_dataset = resolve_path(args.val_dataset)
    test_dataset = resolve_path(args.test_dataset)
    log_file = resolve_path(args.log_file)

    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")

    ensure_datasets(dataset, val_dataset, test_dataset, link_dataset=not args.no_link_dataset)
    command = build_command(config)

    print("Training command:")
    print(" ".join(f'"{item}"' if " " in item else item for item in command))

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(LLAMA_FACTORY_DIR / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return_code = stream_process(command, cwd=LLAMA_FACTORY_DIR, env=env, log_file=log_file)
    print(f"\nTraining process exited with code {return_code}")
    print(f"Log file: {log_file}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
