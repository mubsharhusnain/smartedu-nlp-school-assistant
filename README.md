# SmartEdu — NLP Assignment 1 (Phase 1)

**Student:** Mubshar Husnain  
**Registration No.:** L1F23BSCS1075  
**Project:** SmartEdu — NLP-Based Intelligent School Assistant Chatbot for The Smart Learners Academy

## 1. Assignment scope

This repository implements Phase 1: dataset acquisition, exploration, tokenization, regular expressions, cleaning/normalization, and reproducible output.

For the public NLP experiment, this project uses the **EducationQ** dataset from Hugging Face:
- Dataset page: https://huggingface.co/datasets/SunriserFuture/EducationQ
- Configuration: `educationq_full`
- Split: `train`
- Dataset card reports approximately **1.5K rows**, English text, 13 category values, and **CC BY 4.0**.
- Text field: `question`
- Label / target: `category` (multi-class subject/domain category)
- Planned Phase 1 prediction target: classify a question into its dataset category, as a subject-routing prototype.

**Approval status:** The dataset must be approved by the course instructor before final submission. Do not claim approval until it is received.

The public benchmark is used for reproducible preprocessing and subject/category routing experiments. It is **not** the academy's factual FAQ database. The eventual SmartEdu chatbot must use a separate, verified academy knowledge base for admissions, classes, fees, timings, policies, tests, and homework. No academy facts are invented in this project.

## 2. Repository structure

```text
SmartEdu_NLP_Assignment_1/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── download_data.py
│   ├── explore.py
│   └── preprocess.py
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
└── reports/
    ├── SmartEdu_NLP_Assignment_1_Report.docx
    └── SmartEdu_NLP_Assignment_1_Report.pdf
```

## 3. Setup (Windows PowerShell)

Run commands from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\download_data.py
python src\explore.py
python src\preprocess.py
```

If PowerShell blocks activation for this terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. Setup (macOS / Linux)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/download_data.py
python src/explore.py
python src/preprocess.py
```

## 5. Generated outputs

After the commands run successfully:
- `data/raw/educationq_questions.csv` — downloaded source subset used locally (do not commit if course policy says not to).
- `outputs/exploration_summary.json` — row count, mean words/document, token/type counts, category distribution, and vocabulary.
- `outputs/tokenizer_regex_examples.txt` — side-by-side regex and NLTK tokenizer output plus regex test output.
- `data/processed/educationq_cleaned.csv` — cleaned text and original category label.
- `outputs/cleaning_summary.json` — vocabulary before/after and pipeline configuration.

The scripts create the directories automatically. All report metrics must be copied from these generated files after running the code. No statistics are fabricated in this report package.

## 6. Cleaning decisions

1. Unicode NFC normalization is applied first.
2. English case folding/lowercasing is used because capitalization is not the intended category signal for this experiment.
3. URLs, email addresses, HTML tags, and repeated whitespace are normalized/removed using regex patterns.
4. Stop words are removed using NLTK's English list, **but negations** (`no`, `not`, `nor`, `never`) are retained because they can change the meaning of a question.
5. WordNet lemmatization is used (one normalization method; no stemming is applied).
6. The pipeline is a single reusable `clean_text()` function.

## 7. Git workflow

```bash
git init
git add README.md requirements.txt .gitignore src/
git commit -m "chore: initialize SmartEdu NLP phase 1"
git add reports/
git commit -m "docs: add NLP assignment 1 report"
git add outputs/ data/processed/
git commit -m "feat: add reproducible exploration and cleaning outputs"
git tag assignment-1
```

For a team repository, add all registered team members and the instructor as collaborators. Use branches and pull requests if multiple members are contributing. Replace this section with the actual repository URL and actual team contribution details before submission.

## 8. Important submission checklist

- [ ] Instructor approval recorded for EducationQ dataset.
- [ ] Team name and all team members/roll numbers entered from the registered team.
- [ ] All scripts run successfully in a fresh environment.
- [ ] Actual statistics copied from `outputs/*.json` into the report.
- [ ] Genuine screenshots captured from the user's own machine and inserted into the report.
- [ ] GitHub repository URL and exact `assignment-1` tag URL added.
- [ ] Raw dataset is not committed if prohibited by the assignment.


## 9. Assignment 2 — Word Representations (Phase 2)

Assignment 2 continues in this same repository and uses the exact cleaned corpus from Assignment 1:
`data/processed/educationq_cleaned.csv` with columns `cleaned_text` and `category`.
Do not run a different cleaning process for Phase 2.

### Run Assignment 2

```powershell
pip install -r requirements.txt
python src\assignment2.py
```

The script builds:
- BoW and TF-IDF document-term matrices; reports matrix shapes and sparsity.
- Unigram, bigram, and trigram feature counts and sparsity.
- A tiny 3-document hand calculation for one TF-IDF value and one bigram, checked against scikit-learn.
- Word2Vec CBOW (`sg=0`) and Skip-gram (`sg=1`) with identical hyperparameters.
- fastText with character n-grams, plus rare/misspelled-word probing.
- Similar-word, cosine similarity, and analogy probes (where vocabulary permits).
- Saved vectorizers and embedding models in `models/`.
- A summary JSON at `outputs/assignment2/assignment2_summary.json`.

### Saved artefacts

```text
models/bow_vectorizer.joblib
models/tfidf_vectorizer.joblib
models/ngram_1_3_vectorizer.joblib
models/word2vec_cbow.model
models/word2vec_skipgram.model
models/fasttext.model
outputs/assignment2/assignment2_summary.json
```

### Phase 3 vectorization interface

Import `vectorize_document` from `src/assignment2.py` and choose:
`bow`, `tfidf`, `ngram`, `cbow`, `skipgram`, or `fasttext`.
The function loads saved artefacts and does not retrain models.

```python
from src.assignment2 import vectorize_document
vector = vectorize_document("How can students improve learning?", "tfidf")
print(vector.shape)
```

After successful execution, inspect the JSON output and insert real metrics/screenshots into the Assignment 2 report. The script does not fabricate metrics.
