# Machine Learning Capstone Project

**Author:** Mozammel Hoque Sagor 

**Course:** Machine Learning, Chapter 14 Capstone Project

This repository contains two end-to-end machine learning projects. Each one follows the same pipeline: data acquisition and EDA, preprocessing, a baseline model plus a complex model, hyperparameter tuning, and rigorous evaluation.

| # | Project | Type | Baseline | Complex model | Main metrics |
|---|---------|------|----------|---------------|--------------|
| 1 | Uber/Lyft Ride Fare Estimator | Regression | Linear Regression | Random Forest (RandomizedSearchCV) | RMSE, MAE, R² |
| 2 | Credit Card Fraud Detection | Binary classification (imbalanced) | Logistic Regression (class weights, SMOTE) | XGBoost (RandomizedSearchCV) | Recall, Precision, F1, PR-AUC, Confusion Matrix |


## Repository layout

```
.
├── project1_uber_fare_estimator.ipynb
├── Uber_Fare_Estimator_Report.pdf
├── Uber_Fare_Estimator_Presentation.pptx
├── project2_credit_card_fraud.ipynb
├── Credit Card Fraud Detection I Re...pdf    (Project 2 report)
├── Credit Card Fraud Detection.pptx          (Project 2 presentation)
├── requirements.txt
└── README.md
```

## Datasets (not included, download them yourself)

| Project | Dataset | Expected file name | Required columns |
|---------|---------|--------------------|------------------|
| 1 | NYC taxi / Uber fare data from Kaggle (for example "Uber Fares Dataset" or the NYC Taxi Fare `train.csv`) | `data/uber.csv` | `fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count` |
| 2 | "Credit Card Fraud Detection" from Kaggle | `data/creditcard.csv` | `Time, V1 ... V28, Amount, Class` |

Search for the dataset names on Kaggle and check the column names after downloading. Each notebook stops with a clear error message if a required column is missing. If your file has a different name, change `DATA_PATH` in the Setup cell or set the `DATA_PATH` environment variable.

## How to run

### Locally
```bash
pip install -r requirements.txt
jupyter notebook
```
Open a notebook and choose **Run All**.

### Google Colab
1. Upload the notebook and the CSV file.
2. In the first code cell add `!pip install -q xgboost imbalanced-learn` if the imports fail.
3. Set `DATA_PATH` to where you uploaded the CSV, then **Run all**.

## Notes on the method

- **No data leakage:** the train/test split is done first. Scalers are fitted inside pipelines, so they only see training data. In Project 2, SMOTE is applied only to training data (via an `imblearn` pipeline), duplicates are removed before splitting, and the decision threshold is chosen on out-of-fold training predictions, never on the test set.
- **Project 1:** trips are cleaned (invalid fares, passenger counts, coordinates outside New York City), then features are engineered: Haversine distance, time features in New York local time, and distance to the three major airports.
- **Project 2:** because fraud is extremely rare, accuracy is not used to judge models. A "dummy" model that always predicts "not fraud" is included to show why.
- Settings such as `READ_ROWS`, `N_ITER` and `TUNE_ROWS` at the top of each notebook control how long the run takes. Increase them for a more thorough search.


## Results

| Project | Best model | Key metrics |
|---------|-----------|-------------|
| 1 | Random Forest (tuned) | Test RMSE 2.998, MAE 1.562, R² 0.897 |
| 2 | XGBoost (tuned + threshold) | Precision 0.986, Recall 0.758, F1 0.857, PR-AUC 0.817 |
