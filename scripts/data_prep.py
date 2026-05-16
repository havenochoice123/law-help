"""Prepare SFT data and knowledge chunks from DISC-Law-SFT style JSONL."""
import argparse
import json
import os
import shutil


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid JSON at {path}:{line_number}') from error


def normalize_text_field(value) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            text = normalize_text_field(item)
            if text:
                parts.append(text)
        return '\n'.join(parts)
    if isinstance(value, dict):
        for key in ('text', 'content', 'article', 'reference', 'value'):
            if key in value:
                return normalize_text_field(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def clean_text(text: str) -> str:
    return ' '.join(normalize_text_field(text).split())


def chunk_text(text: str, chunk_size=1024, overlap=128):
    tokens = text.split()
    chunks = []
    index = 0
    step = max(1, chunk_size - overlap)
    while index < len(tokens):
        chunk = tokens[index:index + chunk_size]
        chunks.append(' '.join(chunk))
        index += step
    return chunks


def write_chunk(path, chunk_obj):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(chunk_obj, file, ensure_ascii=False)


def recreate_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def from_disc_law_sft(in_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'finetune'), exist_ok=True)

    finetune_out = os.path.join(out_dir, 'finetune', 'all.jsonl')
    chunks_dir = os.path.join(out_dir, 'chunks')
    knowledge_chunks_dir = os.path.join(out_dir, 'knowledge_chunks')
    recreate_dir(chunks_dir)
    recreate_dir(knowledge_chunks_dir)

    sample_count = 0
    qa_chunk_count = 0
    knowledge_chunk_count = 0

    with open(finetune_out, 'w', encoding='utf-8') as output_file:
        for obj in read_jsonl(in_path):
            instruction = normalize_text_field(obj.get('instruction') or obj.get('prompt') or '')
            input_text = normalize_text_field(obj.get('input') or '')
            output_text = normalize_text_field(obj.get('output') or obj.get('response') or '')
            reference_text = normalize_text_field(obj.get('reference') or obj.get('references') or '')

            if not output_text:
                continue
            if not instruction and input_text:
                instruction = input_text
                input_text = ''

            sample_id = obj.get('id', f'sft-{sample_count}')
            sample = {
                'id': sample_id,
                'instruction': clean_text(instruction),
                'input': clean_text(input_text),
                'output': clean_text(output_text),
                'meta': obj.get('meta', {}),
            }
            output_file.write(json.dumps(sample, ensure_ascii=False) + '\n')

            qa_text = clean_text('\n'.join([instruction, input_text, output_text]))
            if qa_text:
                for chunk_index, chunk in enumerate(chunk_text(qa_text, chunk_size=200, overlap=20)):
                    chunk_obj = {
                        'id': f'{sample_id}-chunk-{chunk_index}',
                        'text': chunk,
                        'title': obj.get('meta', {}).get('title', ''),
                        'source': 'DISC-Law-SFT',
                        'qa_id': sample_id,
                    }
                    write_chunk(os.path.join(chunks_dir, f"{chunk_obj['id']}.json"), chunk_obj)
                    qa_chunk_count += 1

            knowledge_text = clean_text(reference_text)
            if knowledge_text:
                for chunk_index, chunk in enumerate(chunk_text(knowledge_text, chunk_size=200, overlap=20)):
                    chunk_obj = {
                        'id': f'{sample_id}-reference-{chunk_index}',
                        'text': chunk,
                        'title': obj.get('meta', {}).get('title', sample_id),
                        'source': 'DISC-Law-SFT-reference',
                        'qa_id': sample_id,
                    }
                    write_chunk(os.path.join(knowledge_chunks_dir, f"{chunk_obj['id']}.json"), chunk_obj)
                    knowledge_chunk_count += 1

            sample_count += 1

    print(
        f'Wrote {sample_count} samples to {finetune_out}, '
        f'{qa_chunk_count} QA chunks to {chunks_dir}, '
        f'and {knowledge_chunk_count} knowledge chunks to {knowledge_chunks_dir}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-disc-law-sft', type=str, help='Input DISC-Law-SFT jsonl path')
    parser.add_argument('--out', type=str, default='data')
    args = parser.parse_args()
    if args.from_disc_law_sft:
        from_disc_law_sft(args.from_disc_law_sft, args.out)
    else:
        print('No input specified. Use --from-disc-law-sft <path>')
