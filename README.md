# Drug Review Satisfaction Scoring with DistilBERT (1–10 Regression)

This project fine-tunes a **DistilBERT** model to predict a **1–10 satisfaction rating** from free-text drug reviews[cite: 1, 2]. It is **Stage 1** of a larger plan to turn patient reviews into an overall satisfaction score (regression), with future work focused on aspect-based satisfaction[cite: 3, 4, 5].

---

## 🏥 Business Problem (Medical Executive Perspective)

Healthcare authorities and drug manufacturers face several challenges when it comes efficacy and effectiveness of a drug from pateint's point of view. They inlcude,
* Receiving **thousands of patient reviews** across many medications, but no resouces have been allocated to read them all.
* Simple numeric ratings don’t explain **why** patients are unhappy.
* By the time low satisfaction shows up in formal surveys, it may already be hurting adherence, increasing call-center volume, and affecting institutional satisfaction scores

**The Solution:**
You need a way to **convert unstructured patient reviews into a numeric satisfaction signal** that can be tracked over time and compared across drugs. This project addresses the first step:

> **Given a free-text drug review, estimate how satisfied the patient is on a 1–10 scale.** 

---

## 🔍 Problem Definition

### Inputs
We explored two input variants:
1.  **Review only**:  
    `"It has no side effect, I take it in combination of Bystolic 5 mg..."` 
2.  **Drug + Condition + Review**:  
    "Drug: Valsartan. Condition: Hypertension. Review: It has no side effect..."` 

### Output
A continuous satisfaction score from **1 to 10**, where:
* **1** = Very Dissatisfied
* **10** = Very Satisfied 

This allows us to score reviews even when patients don't provide a numeric rating and aggregate predictions across conditions or time periods.

---

## 🧹 Data & Preprocessing

* **Dataset:** Drug review dataset containing `review` (text), `rating` (1–10), and optional `drugName`/`condition`.
    * *Source:* Kaggle (https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)
* **Splits:** Train / Validation / Test (e.g., 80/10/10).
* **Text Cleaning:** Minimal cleaning for transformers. We strip extra whitespace but keep punctuation, numbers, and units (e.g., "5 mg", "3/10") without aggressive symbol removal.

---

## 📊 Model Comparison & Results

This project compares classical baselines against a transformer-based model[cite: 58].

### 1. Classical Models
We evaluated Linear Regression, Lasso, Ridge, ElasticNet, and XGBoost using two feature sets: **TF–IDF** and **Word2Vec**.

| Feature Set | Input Format | Best Model | $R^2$ | RMSE | MAE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TF–IDF** | Review Only | **Ridge** | **0.499** | **2.33** | **1.84** |
| **Word2Vec** | Review Only | XGBoost | 0.485 | 2.36 | 1.85 |
| **TF–IDF** | Drug + Cond + Rev | **Ridge** | **0.503** | **2.32** | **1.84** |
| **Word2Vec** | Drug + Cond + Rev | XGBoost | 0.459 | 2.42 | 1.91 |


>**Observation:** Averaged Word2Vec embeddings underperformed TF–IDF. Averaging compresses long reviews into a single vector, losing specific high-signal phrases like "no side effects" that TF–IDF retains.

### 2. Transformer Model (DistilBERT)
* **Base Model:** `distilbert-base-uncased`
* **Objective:** Regression on the 1–10 rating (MSE loss)

#### Test Set Metrics
| Metric | Value |
| :--- | :--- |
| **$R^2$** | **0.805** |
| **RMSE** | **1.450** |
| **MAE** | **0.865** |

### 🏆 Key Takeaway
Compared to the best classical baseline (Ridge with $R^2 \approx 0.503$), **DistilBERT achieves $R^2 \approx 0.805$**. RMSE drops significantly from ~2.32 to **1.45**, demonstrating the value of modern transformers for capturing nuanced medical sentiment.

---

## 🛠 Installation

```bash
git clone [https://github.com/your-username/drug-review-rating-bert.git](https://github.com/your-username/drug-review-rating.git)
cd drug-review-rating-bert

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
---

## ▶️ Local Inference

The fine-tuned DistilBERT model is hosted on Hugging Face Hub (weights are not stored in this repo).

You can use the provided script `src/infer_rating.py` which launches a simple Gradio UI, or run Python code directly:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "rukshan1015/drug-review-bert-regression-fullmodel"  
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
```
---

## 📓 Training Code

All training scripts are located in the `/ml` folder:

* `train_tfidf_models.py`: TF–IDF + linear/XGBoost regressors 
* `train_word2vec_models.py`: Word2Vec + regressors 
* `train_bert_rating.py`: DistilBERT rating regression fine-tuning 

These scripts handle data loading, feature extraction (TF–IDF, Word2Vec, BERT tokenization), model training, and saving classical models to `/models`.

---

## 🔮 Future Work: Stage 2 (Aspect-Based Satisfaction)

This project is the foundation for a richer aspect-based satisfaction system. Planned next steps include:

1.  **Annotation:** Use a large LLM (e.g., OpenAI) to annotate reviews with aspect-wise sentiment: **Effectiveness, Side Effects, Ease of Use, Cost**.
2.  **Training:** Train smaller DistilBERT classifiers on these specific aspects.
3.  **Analytics:** Combine the overall rating (this model) with aspect-level sentiment to identify drugs that are effective but poorly tolerated or have high cost complaints. This will provide fine-grained granularity in analysing drug's effectiveness. 