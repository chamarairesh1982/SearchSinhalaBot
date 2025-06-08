import os
import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

HF_API = 'https://api-inference.huggingface.co/models/google/mt5-small'
HF_TOKEN = os.getenv('HF_TOKEN')

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'


def load_index(index_file):
    with open(index_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data['embeddings'], dtype=np.float32), data['metadata']


def embed(text, model=None):
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec


def search(query, embeddings, metadata, top_k=5):
    model = SentenceTransformer(MODEL_NAME)
    q = embed(query, model)
    sims = embeddings @ q
    idx = np.argsort(-sims)[:top_k]
    results = [metadata[i] | {'score': float(sims[i])} for i in idx]
    return results


def ask_hf(question, context, max_tokens=256):
    if HF_TOKEN is None:
        raise RuntimeError('HF_TOKEN environment variable not set')
    payload = {
        'inputs': f"{context}\n\nප්‍රශ්නය: {question}\n\nපිළිතුර:",
        'parameters': {'max_new_tokens': max_tokens}
    }
    headers = {'Authorization': f'Bearer {HF_TOKEN}'}
    resp = requests.post(HF_API, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()[0]['generated_text']


def main():
    parser = argparse.ArgumentParser(description='Answer Sinhala question via RAG')
    parser.add_argument('question', help='Question in Sinhala')
    parser.add_argument('--index', default='index.json', help='Index file from build_index')
    parser.add_argument('--top-k', type=int, default=3, help='Number of docs to use')
    args = parser.parse_args()

    embeddings, metadata = load_index(args.index)
    results = search(args.question, embeddings, metadata, top_k=args.top_k)
    context = "\n".join(r['text'] for r in results)
    answer = ask_hf(args.question, context)
    print('--- Context ---')
    for r in results:
        print(f"({r['source']}) {r['text'][:100]}...")
    print('\n--- Answer ---')
    print(answer)


if __name__ == '__main__':
    main()
