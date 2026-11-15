"""Exploration, tokenizer comparison, and regex evidence for EducationQ."""
from pathlib import Path
import json
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "educationq_questions.csv"
OUT = ROOT / "outputs"

def ensure_nltk():
    for package in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass

def regex_tokenize(text):
    return re.findall(r"\b[\w]+(?:['’][\w]+)?\b|[^\w\s]", str(text), flags=re.UNICODE)

def main():
    if not RAW.exists():
        raise FileNotFoundError("Run `python src/download_data.py` first.")
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_nltk()
    df = pd.read_csv(RAW).dropna(subset=["question", "category"])
    texts = df["question"].astype(str)
    token_lists = [regex_tokenize(x) for x in texts]
    tokens = [t for row in token_lists for t in row]
    vocab = sorted(set(t.lower() for t in tokens))
    word_counts = texts.str.split().str.len()
    summary = {
        "documents": int(len(df)),
        "average_document_length_words": round(float(word_counts.mean()), 3),
        "sample_documents_for_token_type_count": int(min(200, len(df))),
        "sample_token_count": int(sum(len(x) for x in token_lists[:200])),
        "sample_type_count": int(len(set(t.lower() for row in token_lists[:200] for t in row))),
        "vocabulary_size_full_corpus_regex_tokens_lowercase": int(len(vocab)),
        "category_distribution": {str(k): int(v) for k, v in df["category"].value_counts().sort_index().items()},
        "messy_examples": texts.head(5).tolist(),
        "note": "All metrics are computed from the downloaded EducationQ split by this script."
    }
    (OUT / "exploration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    for i, text in enumerate(texts.head(5), 1):
        try:
            nltk_tokens = word_tokenize(text)
        except LookupError:
            nltk_tokens = nltk.wordpunct_tokenize(text)
        lines += [
            f"EXAMPLE {i}: {text}",
            f"Regex tokenizer: {regex_tokenize(text)}",
            f"NLTK tokenizer:  {nltk_tokens}",
            ""
        ]

    regex_tests = {
        "URL": (r"https?://\S+|www\.\S+", "Visit https://example.com/help"),
        "EMAIL": (r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "Contact help@example.edu today"),
        "DATE": (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "The test is on 15/11/2026"),
    }
    lines.append("REGEX TESTS")
    for label, (pattern, sample) in regex_tests.items():
        lines.append(f"{label}: pattern={pattern!r}; sample={sample!r}; matches={re.findall(pattern, sample)}")
    (OUT / "tokenizer_regex_examples.txt").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved tokenizer and regex evidence to {OUT / 'tokenizer_regex_examples.txt'}")

if __name__ == "__main__":
    main()
