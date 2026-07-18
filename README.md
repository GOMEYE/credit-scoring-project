# Credit Risk Scoring System

A machine learning system for predicting credit risk (Good/Bad) using the German Credit dataset, with model explainability (SHAP), fairness auditing, and an interactive Streamlit web application.

## 📋 Project Overview

This project was developed as a final year Computer Science project. It covers the full machine learning lifecycle:

- Exploratory data analysis and preprocessing
- Training and comparison of multiple classification models (Logistic Regression, Random Forest, Extra Trees, SVM, XGBoost)
- Handling of class imbalance via class weighting
- Model explainability using SHAP (global and local)
- Fairness/bias auditing across the `Sex` attribute
- Deployment as an interactive Streamlit web application

## 🗂️ Project Structure

```
credit-scoring/
├── data/
│   └── german_credit_data.csv       # Raw dataset
├── models/
│   ├── xgboost_balanced_credit_model.pkl   # Final trained model
│   ├── shap_background.pkl                 # Background sample for SHAP explainer
│   └── *_encoder.pkl                       # Label encoders for categorical features
├── notebooks/
│   └── analysis_model.ipynb         # Full analysis: EDA, model training, evaluation, fairness audit
├── app/
│   └── app.py                       # Streamlit web application
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

## ⚙️ Setup

**1. Clone or download this repository, then navigate into it:**
```bash
cd credit-scoring
```

**2. Create and activate a virtual environment:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 Running the App

```bash
cd app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 📓 Running the Analysis

Open `notebooks/analysis_model.ipynb` in VS Code or Jupyter and run the cells in order. The notebook covers:

1. Data loading and preprocessing
2. Model training and comparison across 5 algorithms
3. Evaluation using accuracy, precision, recall, F1 (weighted and macro), ROC-AUC, and cross-validation
4. Class imbalance mitigation via `class_weight='balanced'` / `scale_pos_weight`
5. SHAP-based global and local explainability
6. Fairness auditing (Demographic Parity, Equal Opportunity, Disparate Impact Ratio) across the `Sex` attribute, including a proxy-discrimination experiment (retraining without `Sex`)

## 🤖 Model

The final deployed model is an **XGBoost classifier** trained with `scale_pos_weight` to address class imbalance in the dataset (approximately 70% "Good" risk vs. 30% "Bad" risk).

**Performance (test set):**

| Metric | Score |
|---|---|
| F1 (Macro) | 0.652 |
| ROC-AUC | 0.714 |
| Recall (Bad risk) | 0.609 |

XGBoost was selected over Logistic Regression, Random Forest, Extra Trees, and SVM after evaluation showed it had the best balance of overall performance and — critically for this use case — the best ability to correctly identify high-risk ("Bad") applicants, which naive accuracy or F1 scores alone tend to obscure on this imbalanced dataset.

## 🔍 Explainability

The app provides two levels of model explanation using [SHAP](https://github.com/shap/shap):

- **Global importance**: which features the model relies on most across all predictions
- **Local explainability**: a waterfall chart showing exactly why a specific applicant received their prediction, with contributions from each input feature

## ⚖️ Fairness Considerations

A fairness audit was conducted across the `Sex` attribute using standard fairness metrics. Key findings:

- **Disparate Impact Ratio**: 0.910 (passes the commonly cited "80% rule")
- **Equal Opportunity Difference**: 0.118 — qualified male applicants are correctly approved at a meaningfully higher rate than equally qualified female applicants
- Retraining without `Sex` as an input feature only marginally reduced this gap (0.118 → 0.113), while reducing overall model performance — indicating other features likely act as proxies for `Sex`, a known limitation of "fairness through unawareness" as a mitigation strategy

**This is a known limitation of the current model.** See the notebook's fairness analysis section for full methodology, metrics, and discussion. Before any real-world deployment, further mitigation (e.g., fairness-constrained training, reweighing, or post-processing threshold adjustment) would be required.

## ⚠️ Disclaimer

This project is for academic purposes only. The model is trained on a small, dated, and non-representative dataset (the German Credit dataset), and its predictions should not be used for real lending decisions. See the Fairness Considerations section above for known limitations.

## 🛠️ Tech Stack

- **Python 3.11**
- **scikit-learn**, **XGBoost** — model training
- **SHAP** — explainability
- **Streamlit** — web application
- **pandas**, **numpy** — data processing
- **matplotlib**, **seaborn** — visualization

## 📄 License

Academic project — for educational use.
