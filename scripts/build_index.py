import os
import json
import glob
import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def load_paragraphs(text_dir):
    paragraphs = []
    for txt_file in sorted(glob.glob(os.path.join(text_dir, '*.txt'))):
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    paragraphs.append({'text': line, 'source': os.path.basename(txt_file)})
    return paragraphs


def build_index(paragraphs, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    model = SentenceTransformer(model_name)
    texts = [p['text'] for p in paragraphs]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    return np.array(embeddings), paragraphs


def load_index(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data['embeddings'], dtype=np.float32), data['metadata']


def main():
    parser = argparse.ArgumentParser(description='Build vector index for Sinhala search')
    parser.add_argument('--text-dir', default='textfile', help='Folder with .txt files')
    parser.add_argument('--out', default='index.json', help='Output index file')
    parser.add_argument('--append', action='store_true', help='Append to existing index')
    args = parser.parse_args()

    paragraphs = load_paragraphs(args.text_dir)
    emb_new, meta_new = build_index(paragraphs)

    if args.append and Path(args.out).exists():
        emb_old, meta_old = load_index(args.out)
        emb = np.concatenate([emb_old, emb_new], axis=0)
        metadata = meta_old + meta_new
    else:
        emb, metadata = emb_new, meta_new

    data = {'embeddings': emb.tolist(), 'metadata': metadata}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    print(f'Wrote {len(meta_new)} new paragraphs. Total {len(metadata)} saved to {args.out}')


if __name__ == '__main__':
    main()
