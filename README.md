# Customer Churn Prediction Using Machine Learning

Project ini bertujuan untuk memprediksi pelanggan yang berpotensi **churn** menggunakan pendekatan Machine Learning. Project ini cocok untuk menunjukkan kemampuan Data Scientist dalam mengolah data pelanggan, membangun model klasifikasi, mengevaluasi performa model, dan memberikan rekomendasi bisnis berbasis data.

## Business Problem

Perusahaan perlu mengetahui pelanggan mana yang berisiko berhenti menggunakan layanan. Dengan prediksi churn, tim bisnis dapat melakukan strategi retensi lebih awal, seperti campaign personal, promo loyalitas, follow-up layanan, atau perbaikan pengalaman pelanggan.

## Dataset

Dataset pada folder `data/` adalah **synthetic sample data** yang dibuat untuk kebutuhan portofolio. Struktur datanya menyerupai data customer analytics dan dapat diganti dengan data real perusahaan jika tersedia.

Fitur utama:
- `tenure_month`
- `contract_type`
- `internet_service`
- `monthly_charges`
- `total_charges`
- `usage_gb`
- `support_tickets`
- `late_payments`
- `churn`

## Methodology

1. Data understanding
2. Data cleaning
3. Exploratory Data Analysis
4. Feature engineering
5. Train-test split
6. Model training
7. Model evaluation
8. Business recommendation

## Model

Model utama yang digunakan:
- Random Forest Classifier

Model pembanding yang dapat ditambahkan:
- Logistic Regression
- Decision Tree
- Gradient Boosting
- XGBoost

## Evaluation Result

Baseline result pada synthetic data:

| Metric | Score |
|---|---:|
| Accuracy | 0.66 |
| Precision | 0.62 |
| Recall | 0.59 |
| F1-Score | 0.60 |

## Business Insight

Pelanggan dengan kontrak bulanan, masa berlangganan pendek, komplain tinggi, dan keterlambatan pembayaran memiliki risiko churn lebih besar. Perusahaan dapat melakukan strategi retensi pada segmen ini melalui program loyalitas, diskon personal, dan peningkatan kualitas layanan.

## Project Structure

```text
customer-churn-prediction-machine-learning/
├── data/
│   └── customer_churn.csv
├── notebook/
│   └── churn_prediction_analysis.ipynb
├── src/
│   └── train_model.py
├── model/
│   └── churn_model.joblib
├── images/
│   ├── churn_distribution.png
│   └── feature_importance.png
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## How to Run

```bash
pip install -r requirements.txt
python src/train_model.py
```

Atau buka notebook:

```bash
jupyter notebook notebook/churn_prediction_analysis.ipynb
```

## Portfolio Summary for CV

Built a machine learning classification model to predict customer churn using customer behavior and transaction data. The project includes data cleaning, EDA, feature engineering, model training, evaluation using accuracy, precision, recall, F1-score, and business recommendations for customer retention strategy.
