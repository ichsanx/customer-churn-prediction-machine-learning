# Customer Churn Prediction Using Machine Learning

This project aims to predict customers who are likely to **churn** using a Machine Learning approach. This project is suitable for demonstrating Data Scientist skills in processing customer data, building a classification model, evaluating model performance, and providing data-driven business recommendations.

## Business Problem

The company needs to identify which customers are at risk of stopping the use of its services. By predicting churn, the business team can take early retention actions, such as personalized campaigns, loyalty promotions, service follow-ups, or customer experience improvements.

## Dataset

The dataset in the `data/` folder is **synthetic sample data** created for portfolio purposes. The data structure resembles customer analytics data and can be replaced with real company data if available.

Main features:

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

Main model used:

- Random Forest Classifier

Comparison models that can be added:

- Logistic Regression
- Decision Tree
- Gradient Boosting
- XGBoost

## Evaluation Result

Baseline result on synthetic data:

| Metric | Score |
|---|---:|
| Accuracy | 0.66 |
| Precision | 0.62 |
| Recall | 0.59 |
| F1-Score | 0.60 |

## Business Insight

Customers with monthly contracts, short tenure, a high number of complaints, and late payments have a higher risk of churn. The company can apply retention strategies for this segment through loyalty programs, personalized discounts, and service quality improvements.

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

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python src/train_model.py
```

Or open the notebook:

```bash
jupyter notebook notebook/churn_prediction_analysis.ipynb
```

## Portfolio Summary for CV

Built a machine learning classification model to predict customer churn using customer behavior and transaction data. The project includes data cleaning, exploratory data analysis, feature engineering, model training, evaluation using accuracy, precision, recall, F1-score, and business recommendations for customer retention strategy.
