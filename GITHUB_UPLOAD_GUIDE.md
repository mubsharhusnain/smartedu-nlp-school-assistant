# Assignment 2 — GitHub Upload Guide

Assignment 2 must continue in the **same repository** used for Assignment 1. Do not create a separate repository unless your instructor specifically tells you to.

## 1. Keep these files and folders in the repository

```text
SmartEdu_NLP_Assignment_1/
├── README.md
├── requirements.txt
├── .gitignore
├── CONTRIBUTION_STATEMENT.md
├── GITHUB_UPLOAD_GUIDE.md
├── src/
│   ├── download_data.py
│   ├── explore.py
│   ├── preprocess.py
│   └── assignment2.py
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   └── assignment2/
├── models/
└── reports/
    ├── SmartEdu_NLP_Assignment_1_Report.pdf
    ├── SmartEdu_NLP_Assignment_2_Report.pdf
    └── (DOCX working copies, if desired)
```

`models/` and `outputs/assignment2/` are generated when the Assignment 2 script runs. The script itself is included so these files can be regenerated; the generated artefacts are not present in this starter package yet.

## 2. Run the project before uploading

Open PowerShell in the repository folder:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\download_data.py
python src\explore.py
python src\preprocess.py
python src\assignment2.py
```

Run the Assignment 1 steps first because Assignment 2 reads the cleaned file produced by Assignment 1. Then confirm that `data/processed/educationq_cleaned.csv`, `outputs/assignment2/assignment2_summary.json`, and the saved files under `models/` exist.

## 3. Check the report before final submission

- Replace team-member and contribution placeholders with actual information.
- Copy actual corpus metrics and probe results from `outputs/assignment2/assignment2_summary.json` into the report.
- Add genuine screenshots of your own program execution and the handwritten CAM-Scan copy required by the assignment.
- Ensure the PDF report is 5–8 pages and contains Parts A–D plus reproducibility and contribution statement.
- Do not claim the dataset is instructor-approved unless approval was actually received.

## 4. Commit and tag Assignment 2

Run these commands from the repository root after reviewing `git status`:

```powershell
git status
git add README.md requirements.txt .gitignore src/ reports/ CONTRIBUTION_STATEMENT.md GITHUB_UPLOAD_GUIDE.md outputs/ models/
git commit -m "feat: add NLP assignment 2 word representations"
git tag assignment-2
git push origin main
git push origin assignment-2
```

If your default branch is not named `main`, push your actual branch name instead. If Git reports that the tag already exists, inspect it before deciding whether to replace it; do not force-push a tag blindly.

## 5. What to submit

Submit the GitHub repository URL and the PDF report, as required by the assignment sheet. The repository should contain the Assignment 1 work plus the Assignment 2 code, generated models/results (or reproducible generation script), updated dependencies, README, and contribution statement.
