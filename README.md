# 🎯 Product Review Intelligence

> AI-powered product review analysis using a fine-tuned DistilBERT model and aspect-based sentiment analysis.

An end-to-end NLP project that analyzes product reviews, predicts overall sentiment, detects product-related aspects, and determines sentiment for each aspect through an interactive Streamlit dashboard.

---

## ✨ Features

- 🤖 **DistilBERT Sentiment Classification**
- 📊 **Positive / Neutral / Negative prediction**
- 🎯 **Confidence & probability scores**
- 🔍 **Aspect detection**
- 🧠 **Aspect-level sentiment analysis**
- 📝 **Aspect-specific context extraction**
- 🖥️ **Interactive Streamlit dashboard**

---

## 🧠 How It Works

```text
Product Review
      ↓
Text Preprocessing
      ↓
Fine-tuned DistilBERT
      ↓
 ┌───────────────┬────────────────────┐
 │               │                    │
 ▼               ▼                    ▼
Overall       Aspect Detection    Context Extraction
Sentiment           ↓                    ↓
 │              Aspect Sentiment ← DistilBERT
 │                    │
 └────────────┬───────┘
              ↓
      Streamlit Dashboard


📦 Dataset

The final dataset contains 49,989 reviews:

Split	Samples
Train	39,991
Validation	4,999
Test	4,999

Classes:

Negative · Neutral · Positive

🚀 Model Performance

The fine-tuned DistilBERT model achieved:

| Metric            |      Score |
| ----------------- | ---------: |
| **Test Accuracy** | **81.54%** |
| **Weighted F1**   | **81.55%** |


| Sentiment | Precision | Recall |     F1 |
| --------- | --------: | -----: | -----: |
| Negative  |    83.31% | 85.14% | 84.22% |
| Neutral   |    58.59% | 58.30% | 58.45% |
| Positive  |    91.33% | 89.55% | 90.43% |

🖥️ Dashboard

The dashboard provides:

- Overall sentiment
- Confidence score
- Sentiment probabilities
- Detected aspects
- Aspect-level sentiment
- Aspect confidence
- Relevant review context
- Analysis summary

#------------------------------------------
# Dashboard Images
#------------------------------------------

<p align="center">
  <img src="" alt="Project Banner" width="600">
</p>


⚙️ Run Locally

1. Clone the repository

git clone <YOUR_REPOSITORY_URL>
cd product-review-intelligence

2. Install dependencies

pip install -r requirements.txt

3. Launch the dashboard

streamlit run app/streamlit_app.py

🛠️ Tech Stack

Python · PyTorch · Hugging Face Transformers · DistilBERT · Streamlit · Pandas · NumPy · Scikit-learn · Matplotlib


## 🤗 Model

The fine-tuned DistilBERT sentiment model is hosted on Hugging Face:

[View the model on Hugging Face](https://huggingface.co/nisarg0902/product-review-distilbert-sentiment)

👤 Author

Nisarg Chaudhari