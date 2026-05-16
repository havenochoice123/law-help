# Training Engineering

This module provides the quick-run training path for LoRA and QLoRA with LLaMA-Factory.

## Files

- `configs/lora_quick.yaml`: LoRA SFT quick-run config.
- `configs/qlora_quick.yaml`: QLoRA SFT quick-run config.
- `scripts/train_lora_quick.py`: LoRA launcher with dataset checks.
- `scripts/train_qlora_quick.py`: QLoRA launcher with dataset checks.
- `scripts/verify_adapter.py`: Saved adapter file and optional load check.

## Prerequisites

1. Use Python 3.10 or 3.11. Avoid Python 3.14 because the current `datasets/dill`
   stack is not compatible with its `pickle` API.
2. Install project dependencies.
3. Prepare SFT data:

```powershell
python scripts/data_prep.py --from-disc-law-sft data/raw/disc_law_sft.jsonl --out data/
```

This creates `data/finetune/all.jsonl` in Alpaca format. Split it before training:

```powershell
python scripts/split_dataset.py --input data/finetune/all.jsonl --out-dir data/finetune --eval-dir data/eval
```

For Triplet data with `id`, `input`, `output`, and `reference`:

- `input` and `output` are used for SFT training and evaluation.
- `reference` is converted into `data/knowledge_chunks/` for the RAG knowledge base.
- Do not build the RAG knowledge base from `output`, because that leaks target answers into retrieval.
- Keep `data/eval/test.jsonl` out of training and out of any answer-derived knowledge base.

## Dry Run

Use dry-run first to verify paths and the resolved LLaMA-Factory command:

```powershell
python scripts/train_lora_quick.py --config configs/lora_quick.yaml --dry-run
python scripts/train_qlora_quick.py --config configs/qlora_quick.yaml --dry-run
```

The launchers copy `data/finetune/train.jsonl` to `LLaMA-Factory/data/disc_law_sft_train.jsonl`.
The dataset is registered as `disc_law_sft_train` in `LLaMA-Factory/data/dataset_info.json`.

## Train

```powershell
python scripts/train_lora_quick.py --config configs/lora_quick.yaml
python scripts/train_qlora_quick.py --config configs/qlora_quick.yaml
```

Outputs:

- LoRA: `outputs/models/lora-quick`
- QLoRA: `outputs/models/qlora-quick`

## Verify

Check adapter files only:

```powershell
python scripts/verify_adapter.py --adapter-dir outputs/models/lora-quick
python scripts/verify_adapter.py --adapter-dir outputs/models/qlora-quick
```

Check actual loading:

```powershell
python scripts/verify_adapter.py --adapter-dir outputs/models/lora-quick --load
```

If `adapter_config.json` does not record the base model path, pass it explicitly:

```powershell
python scripts/verify_adapter.py --adapter-dir outputs/models/lora-quick --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --load
```

## Notes

- `ollama://DeepSeek-R1-1.5B` is not used for training because LoRA/QLoRA training needs loadable base weights.
- The default base model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
- QLoRA requires a working `bitsandbytes` installation and compatible GPU support.
