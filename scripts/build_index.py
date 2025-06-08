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


def main():
    parser = argparse.ArgumentParser(description='Build vector index for Sinhala search')
    parser.add_argument('--text-dir', default='textfile', help='Folder with .txt files')
    parser.add_argument('--out', default='index.json', help='Output index file')
    args = parser.parse_args()

    paragraphs = load_paragraphs(args.text_dir)
    emb, metadata = build_index(paragraphs)
    data = {'embeddings': emb.tolist(), 'metadata': metadata}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f'Wrote {len(metadata)} paragraphs to {args.out}')


if __name__ == '__main__':
    main()
