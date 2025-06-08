# 🛕 Sinhala Dhamma Search Bot

A fast, intelligent Sinhala-language Dhamma search engine built with HTML, JavaScript, and GitHub Pages.
Now enhanced with AI-powered semantic search using TensorFlow.js.

Searches Buddhist teachings (ධර්ම දේශනා), Nibbana concepts (නිවන්), and more — 100% free and offline-capable.

## 🚀 Live Site
👉 [Visit Dhamma Search Bot](https://chamarairesh1982.github.io/SearchSinhalaBot/)

## 🧠 Features
- Sinhala Intelligent Search (සිංහල බුද්ධිමත් සෙවුම)
- Synonym matching for Dhamma terms
- Fast, no server — pure HTML & JS
- Semantic ranking powered by TensorFlow.js Universal Sentence Encoder
- Voice search support (🎤 Speak your question in Sinhala)
- Optimized for Buddhist teachings, Nibbana, and Dhamma.

## 📂 Project Structure
/index.html -> Main search app
/textfile/ -> Sinhala text files

sinhala-text1.txt

sinhala-text2.txt
file-list.json -> List of all text files
favicon.ico -> Temple favicon for website


## 🤖 Local RAG Scripts
Several helper scripts support RAG workflows with the Hugging Face Hub:

1. `scripts/build_index.py` – read text files and create `index.json` with embeddings. Use `--append` to update an existing index.
2. `scripts/upload_index.py` – push `index.json` to a Hugging Face dataset repository.
3. `scripts/answer.py` – retrieve relevant paragraphs and query `google/mt5-small` via the Inference API.

Set the `HF_TOKEN` environment variable with your Hugging Face access token before running these scripts.

Example:

```bash
python scripts/build_index.py --text-dir textfile --out index.json
HF_TOKEN=your_token_here python scripts/upload_index.py --repo your-user/dataset
HF_TOKEN=your_token_here python scripts/answer.py "නිවන් දකින්නේ කෙසේද?"
```

This provides a lightweight RAG workflow for more intelligent answers and allows storing the index on the Hugging Face Hub.

## 🛠️ Tech Stack
- HTML
- CSS
- JavaScript
- GitHub Pages

## 🙏 Contributions
Feel free to fork and contribute! Sadhu Sadhu Sadhu! 🛕
---
### 🧘 *May all beings be happy, may all beings be free from suffering.*
