"""Verify a saved LoRA adapter directory."""
import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "models" / "lora-quick"


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def require_file(directory: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    expected = ", ".join(names)
    raise FileNotFoundError(f"Missing required file in {directory}: {expected}")


def read_adapter_base_model(adapter_config: Path) -> str | None:
    with open(adapter_config, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config.get("base_model_name_or_path")


def verify_files(adapter_dir: Path) -> str | None:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    adapter_config = require_file(adapter_dir, ("adapter_config.json",))
    require_file(adapter_dir, ("adapter_model.safetensors", "adapter_model.bin"))
    return read_adapter_base_model(adapter_config)


def verify_load(adapter_dir: Path, base_model: str) -> None:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Loading check requires transformers and peft. Install requirements first."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
    )
    PeftModel.from_pretrained(model, str(adapter_dir))

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer loaded, but no pad_token_id or eos_token_id is available.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify saved LoRA adapter files and loading.")
    parser.add_argument("--adapter-dir", default=str(DEFAULT_ADAPTER_DIR), help="Adapter output directory.")
    parser.add_argument("--base-model", help="Base model path/name. Defaults to adapter_config value.")
    parser.add_argument("--load", action="store_true", help="Actually load base model and adapter.")
    args = parser.parse_args()

    adapter_dir = resolve_path(args.adapter_dir)
    base_model = verify_files(adapter_dir)
    selected_base_model = args.base_model or base_model

    print(f"Adapter files OK: {adapter_dir}")
    if selected_base_model:
        print(f"Base model: {selected_base_model}")
    else:
        print("Base model: not recorded in adapter_config.json")

    if args.load:
        if not selected_base_model:
            raise ValueError("Use --base-model when adapter_config.json has no base model path.")
        verify_load(adapter_dir, selected_base_model)
        print("Adapter load OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
