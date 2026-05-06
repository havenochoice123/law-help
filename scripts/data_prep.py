"""数据准备脚本（最小实现）
功能：
- 从 DISC-Law-SFT jsonl 导出微调样本（SFT 格式）
- 简单清洗与分块（chunking）并输出 chunks 到 data/chunks/

用法示例：
python scripts/data_prep.py --from-disc-law-sft data/raw/disc_law_sft.jsonl --out data/
"""
import argparse
import json
import os
from pathlib import Path


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def clean_text(text: str) -> str:
    # 简单清洗：去除多余空白
    return ' '.join(text.split())


def chunk_text(text: str, chunk_size=1024, overlap=128):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks


def from_disc_law_sft(in_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'finetune'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'chunks'), exist_ok=True)

    finetune_out = os.path.join(out_dir, 'finetune', 'train.jsonl')
    chunks_dir = os.path.join(out_dir, 'chunks')

    count = 0
    with open(finetune_out, 'w', encoding='utf-8') as fout:
        for obj in read_jsonl(in_path):
            # 假定 obj 包含 instruction/input/output
            instruction = obj.get('instruction') or obj.get('prompt') or ''
            inp = obj.get('input') or ''
            output = obj.get('output') or obj.get('response') or ''
            if not output:
                continue
            sample = {
                'id': obj.get('id', f'sft-{count}'),
                'instruction': clean_text(instruction),
                'input': clean_text(inp),
                'output': clean_text(output),
                'meta': obj.get('meta', {})
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
            # 生成 chunks
            full_text = (instruction + '\n' + inp + '\n' + output).strip()
            if full_text:
                chunks = chunk_text(full_text, chunk_size=200, overlap=20)
                for idx, c in enumerate(chunks):
                    chunk_obj = {
                        'id': f"{sample['id']}-chunk-{idx}",
                        'text': c,
                        'title': obj.get('meta', {}).get('title', ''),
                        'source': 'DISC-Law-SFT'
                    }
                    chunk_path = os.path.join(chunks_dir, f"{chunk_obj['id']}.json")
                    with open(chunk_path, 'w', encoding='utf-8') as cf:
                        json.dump(chunk_obj, cf, ensure_ascii=False)
            count += 1
    print(f'Wrote {count} samples to {finetune_out} and chunks to {chunks_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-disc-law-sft', type=str, help='输入 DISC-Law-SFT jsonl 路径')
    parser.add_argument('--out', type=str, default='data')
    args = parser.parse_args()
    if args.from_disc_law_sft:
        from_disc_law_sft(args.from_disc_law_sft, args.out)
    else:
        print('No input specified. Use --from-disc-law-sft <path>')
