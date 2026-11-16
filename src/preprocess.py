"""Reusable cleaning pipeline and cleaned CSV generation."""
from pathlib import Path
import json
import re
import unicodedata
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "educationq_questions.csv"
PROCESSED = ROOT / "data" / "processed"
OUTPUT = PROCESSED / "educationq_cleaned.csv"
OUT_SUMMARY = ROOT / "outputs" / "cleaning_summary.json"

def prepare_nltk():
    for package in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        nltk.download(package, quiet=True)

def clean_text(text):
    """Apply NFC -> lowercase -> noise normalization -> stopword filtering -> lemmatization."""
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)             
    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", " EMAILTOKEN ", text)
    text = re.sub(r"\s+", " ", text).strip()

    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = nltk.wordpunct_tokenize(text)

    stop_set = set(stopwords.words("english"))
    stop_set.difference_update({"no", "not", "nor", "never"})
    lemmatizer = WordNetLemmatizer()
    cleaned = []
    for token in tokens:
        if not re.search(r"[a-z0-9]", token):
            continue
        if token in stop_set:
            continue
        if token == "urltoken" or token == "emailtoken":
            cleaned.append(token)
        else:
            cleaned.append(lemmatizer.lemmatize(token))
    return " ".join(cleaned)

def vocab_size(series):
    vocab = set()
    for text in series.astype(str):
        vocab.update(re.findall(r"\b\w+\b", text.lower()))
    return len(vocab)

def main():
    if not RAW.exists():
        raise FileNotFoundError("Run `python src/download_data.py` first.")
    prepare_nltk()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW).dropna(subset=["question", "category"]).copy()
    df["question"] = df["question"].astype(str)
    before_vocab = vocab_size(df["question"])
    df["cleaned_text"] = df["question"].apply(clean_text)
    after_vocab = vocab_size(df["cleaned_text"])
    df[["cleaned_text", "category"]].to_csv(OUTPUT, index=False, encoding="utf-8")
    summary = {
        "documents_processed": int(len(df)),
        "vocabulary_before_cleaning": int(before_vocab),
        "vocabulary_after_cleaning": int(after_vocab),
        "cleaned_output": str(OUTPUT.relative_to(ROOT)),
        "columns": ["cleaned_text", "category"],
        "pipeline_order": ["Unicode NFC", "lowercase", "HTML/URL/email/whitespace normalization",
                           "English stop-word removal (negations retained)", "WordNet lemmatization"],
        "stemming_applied": False
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved cleaned corpus: {OUTPUT}")

if __name__ == "__main__":
    main()
