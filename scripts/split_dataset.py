"""Split Alpaca-style JSONL data into train/val/test sets."""
import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "finetune" / "all.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "finetune"
DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "eval"


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def check_unique_ids(rows: list[dict]) -> None:
    seen = set()
    duplicates = []
    for index, row in enumerate(rows):
        row_id = row.get("id", f"row-{index}")
        if row_id in seen:
            duplicates.append(row_id)
        seen.add(row_id)
    if duplicates:
        preview = ", ".join(str(item) for item in duplicates[:5])
        raise ValueError(f"Duplicate ids found: {preview}")


def split_rows(
    rows: list[dict],
    val_ratio: float,
    test_ratio: float,
    val_size: int | None,
    test_size: int | None,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    if val_size is not None or test_size is not None:
        val_count = val_size if val_size is not None else round(total * val_ratio)
        test_count = test_size if test_size is not None else round(total * test_ratio)
        if val_count < 0 or test_count < 0:
            raise ValueError("--val-size and --test-size must be non-negative")
    else:
        if not 0 < val_ratio < 1:
            raise ValueError("--val-ratio must be between 0 and 1")
        if not 0 < test_ratio < 1:
            raise ValueError("--test-ratio must be between 0 and 1")
        if val_ratio + test_ratio >= 1:
            raise ValueError("--val-ratio + --test-ratio must be less than 1")
        test_count = round(total * test_ratio)
        val_count = round(total * val_ratio)

    if val_count + test_count >= total:
        raise ValueError("Validation and test sizes must leave at least one training sample")

    test_rows = shuffled[:test_count]
    val_rows = shuffled[test_count:test_count + val_count]
    train_rows = shuffled[test_count + val_count:]
    return train_rows, val_rows, test_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Split SFT data into train/val/test JSONL files.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input Alpaca JSONL path.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for train/val.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR), help="Output directory for test.")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation set ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test set ratio.")
    parser.add_argument("--val-size", type=int, help="Fixed validation set size.")
    parser.add_argument("--test-size", type=int, help="Fixed test set size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--allow-duplicate-ids", action="store_true", help="Skip duplicate id validation.")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    out_dir = resolve_path(args.out_dir)
    eval_dir = resolve_path(args.eval_dir)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    test_path = eval_dir / "test.jsonl"

    if input_path == train_path:
        raise ValueError(
            "Input path must be the full converted dataset, not the train split. "
            "Use data/finetune/all.jsonl or regenerate it with scripts/data_prep.py."
        )

    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")
    if not args.allow_duplicate_ids:
        check_unique_ids(rows)

    train_rows, val_rows, test_rows = split_rows(
        rows,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, test_rows)

    print(f"input: {len(rows)}")
    print(f"train: {len(train_rows)} -> {train_path}")
    print(f"val: {len(val_rows)} -> {val_path}")
    print(f"test: {len(test_rows)} -> {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
