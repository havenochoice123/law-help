# 50-Sample Evaluation Summary

Evaluation date: 2026-05-17

Input file: `outputs/eval/rag_compare_50.jsonl`

Score file: `outputs/eval/rag_compare_50_scores.json`

Command:

```powershell
python .\scripts\compare_rag.py --limit 50 --modes lora,rag,lora_rag --out outputs\eval\rag_compare_50.jsonl --max-new-tokens 180
python .\scripts\score_generations.py --input outputs\eval\rag_compare_50.jsonl --out outputs\eval\rag_compare_50_scores.json
```

## Metrics

| Method | BLEU-4 | ROUGE-L | Non-empty | Cited answers | English-fragment answers | Avg chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 0.1221 | 0.0705 | 50/50 | 0/50 | 23/50 | 270.5 |
| RAG | 0.0000 | 0.0368 | 50/50 | 50/50 | 19/50 | 205.4 |
| LoRA + RAG | 0.0000 | 0.0363 | 50/50 | 50/50 | 27/50 | 237.2 |

## Analysis

In this 50-sample evaluation, all three methods generated non-empty answers. The LoRA-only model achieved the best BLEU-4 and ROUGE-L scores, which suggests that supervised fine-tuning improved the model's answer style and surface similarity to the reference answers.

The RAG and LoRA + RAG methods achieved 100% citation coverage because the answer pipeline appends retrieved evidence identifiers when the model does not cite them by itself. This makes the answers more traceable and demonstrates the retrieval-augmented pipeline required by the project. However, their BLEU-4 and ROUGE-L scores were lower than LoRA-only in this run.

The combined LoRA + RAG method did not outperform LoRA-only on automatic metrics. The main observed limitations are weak instruction following, occasional English fragments, and unstable use of retrieved legal clauses by the 1.5B local model. Therefore, the recommended report conclusion is not that LoRA + RAG is best, but that the project implements and compares the three required settings:

- LoRA improves legal-domain generation quality.
- RAG improves evidence traceability through hybrid retrieval and citation.
- LoRA + RAG is functionally complete, but its quality is limited by the small base model and needs stronger generation ability or more targeted alignment to consistently improve over LoRA-only.

