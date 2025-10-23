# 🧠 Customer Churn Forecasting System

## 📋 Project Overview

This project develops a **Customer Churn Forecasting system** to predict which customers are most likely to discontinue using a company’s services.
By analyzing customer behavior, demographics, and service usage, the model enables businesses to take **proactive retention measures**, such as targeted marketing campaigns and personalized offers.

---

## 🎯 Objectives

* Predict customer churn using historical and behavioral data.
* Identify **key drivers** of churn through feature importance and exploratory analysis.
* Build and compare multiple machine learning models for the best performance.
* Provide **actionable business insights** for customer retention strategies.

---

## 📊 Dataset

The dataset (used locally for this project) contains anonymized customer-level information such as:

* Customer demographics
* Account information (tenure, contract type, charges, etc.)
* Usage or activity patterns
* Target variable: **`Churn`** — 1 if the customer left, 0 otherwise

> **Note:** The dataset was preprocessed and engineered in `Feature_Engineering.ipynb`.
> Due to confidentiality, only a sample or synthetic version may be shared.

---

## 🧹 Data Preprocessing & Cleaning

Steps implemented in the pipeline:

1. **Missing values handling** – imputed using mean/median strategies.
2. **Feature scaling** – applied `StandardScaler` to normalize numeric columns.
3. **Encoding** – categorical variables encoded using one-hot or label encoding.
4. **Outlier treatment** – used interquartile range (IQR) clipping for continuous features.
5. **Balancing** – handled class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**.
6. **Train-test split** – 80% training and 20% testing, maintaining class distribution.

---

## 🧠 Modeling Approach

Multiple supervised machine learning models were developed and evaluated to predict churn probability:

| Model                     | Description                                  | Notebook                          |
| ------------------------- | -------------------------------------------- | --------------------------------- |
| Logistic Regression       | Baseline interpretable model                 | `CCF - Logistic Regression.ipynb` |
| Decision Tree             | Nonlinear model with rule-based segmentation | `CCF - Decision Tree.ipynb`       |
| K-Nearest Neighbors (KNN) | Distance-based classification                | `CCF - Modeling KNN.ipynb`        |
| Random Forest + SMOTE     | Ensemble model with resampling for imbalance | `CCF prediction rf SMOTE.ipynb`   |
| Model Comparison          | Evaluated all models using common metrics    | `Model_Comparison.ipynb`          |

The trained and tuned models were saved as:

* `fraud_detection_model.pkl` – trained Random Forest model
* `scaler.pkl` – fitted feature scaler used for data normalization

---

## ⚙️ Evaluation Metrics

To measure model performance, the following metrics were used:

| Metric        | Meaning                                                               |
| ------------- | --------------------------------------------------------------------- |
| **Accuracy**  | Overall correctness of predictions                                    |
| **Precision** | % of predicted churns that were actual churns                         |
| **Recall**    | % of actual churns correctly predicted                                |
| **F1-Score**  | Balance between Precision and Recall                                  |
| **ROC-AUC**   | Ability of the model to distinguish between churners and non-churners |

Example (Random Forest SMOTE Results):

```
Accuracy: 0.96
Precision: 0.92
Recall: 0.90
F1-Score: 0.91
ROC-AUC: 0.97
```

---

## 📈 Exploratory Data Analysis (EDA)

Performed in `Feature_Engineering.ipynb` and visualized using **Matplotlib** and **Seaborn**:

* Churn distribution and imbalance analysis
* Correlation heatmap of numerical features
* Churn rate by customer tenure, plan type, and region
* Top predictors identified through feature importance plots

---

## 🔍 Key Insights

* **Tenure**, **account balance**, and **transaction frequency** were the strongest predictors of churn.
* Customers with **low engagement and short tenure** showed the highest churn probability.
* High **service complaints** and **low transaction volume** correlated with increased churn.
* Applying **SMOTE** significantly improved recall, reducing false negatives.

---

## 💡 Business Recommendations

1. **Customer Retention Programs**
   Offer loyalty rewards or discounts for customers within their first 3 months (high churn risk).
2. **Personalized Outreach**
   Use churn probability scores to trigger retention campaigns for at-risk users.
3. **Customer Support Optimization**
   Reduce unresolved complaint rates — strongly linked to churn.
4. **Product Feedback Loops**
   Target users with declining engagement for personalized feedback collection.

---

## 🧾 Repository Structure

```
Customer-Churn-Forecasting/
│
├── notebooks/
│   ├── Feature_Engineering.ipynb
│   ├── CCF - Logistic Regression.ipynb
│   ├── CCF - Decision Tree.ipynb
│   ├── CCF - Modeling KNN.ipynb
│   ├── CCF prediction rf SMOTE.ipynb
│   ├── Model_Comparison.ipynb
│   └── Model_Deployment.ipynb
│
├── models/
│   ├── fraud_detection_model.pkl
│   └── scaler.pkl
│
├── outputs/
│   └── figures, metrics.json
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/Customer-Churn-Forecasting.git
   cd Customer-Churn-Forecasting
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and execute the notebooks in order:

   1. `Feature_Engineering.ipynb`
   2. Model notebooks (Logistic, Decision Tree, KNN, RF SMOTE)
   3. `Model_Comparison.ipynb`
   4. `Model_Deployment.ipynb`

4. (Optional) Use the saved model to predict:

   ```python
   import joblib
   model = joblib.load('models/fraud_detection_model.pkl')
   scaler = joblib.load('models/scaler.pkl')
   preds = model.predict(scaler.transform(new_data))
   ```

---

## 🧩 Tech Stack

* **Languages:** Python (3.10+)
* **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn, Seaborn, Matplotlib, Joblib
* **Environment:** Jupyter Notebooks
* **Modeling:** Logistic Regression, Decision Tree, Random Forest, KNN
* **Deployment:** Model serialization using Pickle

---

## 📘 Results Summary

| Model                 | Accuracy | Precision | Recall   | F1-Score | ROC-AUC  |
| --------------------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression   | 0.90     | 0.87      | 0.84     | 0.85     | 0.91     |
| Decision Tree         | 0.93     | 0.89      | 0.88     | 0.88     | 0.94     |
| KNN                   | 0.92     | 0.88      | 0.87     | 0.87     | 0.93     |
| Random Forest + SMOTE | **0.96** | **0.92**  | **0.90** | **0.91** | **0.97** |

---

## 🏁 Conclusion

The **Random Forest model with SMOTE balancing** provided the most robust and accurate churn predictions.
The results enable businesses to **prioritize retention strategies** by identifying customers at high churn risk and improving service engagement.

---

## 👨‍💻 Author

**P Prakash**
📧 prakashsivam725@gmail.com
🔗 GitHub: https://github.com/Prakash2524

---
