---
title: Clinical Safety Auditor
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: 3.11
app_file: app.py
pinned: false
---
# Agentic Clinical Safety Auditor

Hybrid ML + LangGraph workflow for pharmacovigilance and patient sentiment. The system converts drug reviews into a 1–10 satisfaction score and, when the score is low, triggers deeper adverse-event analysis.

---

## What This Does
- **Satisfaction scoring:** DistilBERT regression model predicts a 1–10 rating from free‑text reviews.
- **Agentic auditing:** A LangGraph state machine decides when to run deeper safety checks and extraction.
- **Actionable signal:** Flags low‑satisfaction cases for clinical review.

## Architecture
The core logic runs as a stateful graph to keep processing reliable and bounded.

![Graph Architecture](output.png)

## Model Performance (Summary)
Benchmarks compare DistilBERT to classical baselines using TF–IDF and Word2Vec features.

| Feature Set | Input Format | Best Model | $R^2$ | RMSE | MAE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TF–IDF** | Review Only | **Ridge** | **0.499** | **2.33** | **1.84** |
| **Word2Vec** | Review Only | XGBoost | 0.485 | 2.36 | 1.85 |
| **TF–IDF** | Drug + Cond + Rev | **Ridge** | **0.503** | **2.32** | **1.84** |
| **Word2Vec** | Drug + Cond + Rev | XGBoost | 0.459 | 2.42 | 1.91 |

Key takeaway: the transformer model captures nuanced medical sentiment that classical frequency‑based models miss.

## Tech Stack
- **Core AI:** Python, PyTorch, Transformers (BERT), LangGraph
- **MLOps:** Docker, FastAPI, AWS SageMaker, AWS Step Functions
- **UI:** Gradio (Hugging Face Spaces compatible)

## Live Demo
Hugging Face Space: https://huggingface.co/spaces/rukshan1015/drug_review

## Local Setup
```bash
git clone https://github.com/your-username/drug-review-rating.git
cd drug-review-rating

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Local Inference
The fine‑tuned DistilBERT model is hosted on Hugging Face Hub (weights not in this repo).

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "rukshan1015/drug-review-bert-regression-fullmodel"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
```

## Docker
```bash
docker build -t drug-rating-app .
docker run --rm -p 7860:7860 drug-rating-app
```
Open: http://localhost:7860

## Data
- **Dataset:** Drug review dataset with `review`, `rating` (1–10), optional `drugName` and `condition`
- **Source:** https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018
- **Splits:** Train / Validation / Test (e.g., 80/10/10)

## Training Code
Scripts live in `ml/`:
- `train_tfidf_models.py`
- `train_word2vec_models.py`
- `train_bert_rating.py`

## Future Work
Aspect‑based satisfaction modeling (Effectiveness, Side Effects, Ease of Use, Cost) using smaller classifiers and aggregated analytics.
