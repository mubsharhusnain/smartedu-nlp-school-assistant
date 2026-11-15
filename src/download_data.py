"""Download the EducationQ public dataset and save question/category fields."""
from pathlib import Path
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT = RAW_DIR / "educationq_questions.csv"

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("SunriserFuture/EducationQ", "educationq_full", split="train")
    df = ds.to_pandas()
    required = {"question", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset schema changed; missing columns: {sorted(missing)}")
    df = df[["question", "category"]].dropna(subset=["question", "category"]).copy()
    df["question"] = df["question"].astype(str)
    df["category"] = df["category"].astype(str)
    df = df[df["question"].str.strip().ne("")]
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Saved {len(df):,} rows to {OUT}")
    print(f"Columns: {list(df.columns)}")
    print("Dataset card: https://huggingface.co/datasets/SunriserFuture/EducationQ")
    print("License stated on dataset card: CC BY 4.0. Verify source-dataset terms before redistribution.")

if __name__ == "__main__":
    main()
