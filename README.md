# Secom Failure Classification (End-to-End ML + MLOps)
## Overview
This project focuses on building a robust machine learning pipeline to detect manufacturing failures using the SECOM dataset (high-dimensional, highly imbalanced data).

The workflow covers:
- Data preprocessing & feature reduction
- Model experimentation & selection
- MLflow-based experiment tracking
- Threshold tuning for imbalanced classification
- API deployment using FastAPI
- Containerization with Docker
- Cloud deployment on AWS EC2

## Project Structure
```
├── DockerFile           # for Dockerization
├── Secom_Notebook.ipynb       # Notebook file 
├── config.json          # saved features, threshold
├── exported_model.zip   # saved model for predictions
├── main.py              # fastapi backend
├── sample_test.py       # sample testcases 
├── requirements.txt     # requirements for application
└── README.md
```
## Results
**Top K- experimentation aganist PR-AUC Scores**

![K-values](k_values.png)

*K=150 has the highest PR-AUC score of 0.27*

**Hyperparameter Tuning of the best model (RandomForest) with selected Top-150 features**

![pr_values](pr_values.png)

Best Parameters found: {n_estimators: 200, max_depth: 10, min_split: 2, min_leaf: 1} *resulted in pr_auc score: 0.297 and roc-auc score: 0.797*

**7.5% improvement from baseline in pr_auc acheieved by hyperparamter tuning**



Predicted Probabilities on test data and evaluated **Precision**, **Recall**, **F1-Score** at multiple thresholds and optimal threshold is selected based on max **F1-Score** for balanced performance.

**Untuned Performance**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Pass)** | 0.93 | 1.00 | 0.97 | 440 |
| **1 (Fail)** | 0.00 | 0.00 | 0.00 | 31 |
| **Accuracy** | | | **0.93** | 471 |

**Tuned Performance** *@0.189*

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Pass)** | 0.96 | 0.97 | 0.96 | 440 |
| **1 (Fail)** | 0.46 | 0.42 | 0.44 | 31 |
| **Accuracy** | | | **0.93** | 471 |

**Improved Minority Class F1 from 0 → 0.44**\
This model is saved in local directory using `mlflow.sklearn.save_model(model,"exported_model")`

## Dataset Details
* Rows: 1567
* Columns: 592
* Problem Type: Binary Classification (Highly Imbalanced) (0:1463, 1:104) with target mapping ({-1:0,1:1})

## Data Preprocessing Pipeline
**1. Missng Value Analysis :** Computed null percentage per columns and selected a cutoff of 60 % through histogram, dropping columns having >60% of null values.\
**2. Feature Reduction :** Applied Variance Threshold reducing 566 → 301 columns.\
**3. Imputation :** Used Median to impute Missing Values.\
**4. Stratified Train Test Split :** to preserve class distribution in both train and test datasets.

## Feature Selection Strategy
* Used **SelectKBest(f_classif)**
* Evaluated multiple *k* values: [50,75,100,125,150,175,200]
* Tracked experiments using **MLFlow**
* Metric used: **PR-AUC** for comparision across k values with RandomForest Classifier
**Selected Top 150** features based on the best **PR-AUC**

## Model Training & Comparision
Trained multiple models with selected top 150 features evaluated using PR-AUC
* RandomForest
* XGBoost
* LightGBM
* Logistic Regression

Handling Imbalance
* `class_weight="balanced"` (where applicable)
* LightGBM → `scale_pos_weight`\
**Best Model: RandomForest, PR-AUC: 0.2718** (outperformed all other models)

## Hyperparameter Tuning
* Tuned:
  * `n_estimators`
  * `max_depth`
  * `min_samples_split`
  * `min_samples_leaf`
* Logged every run in MLFlow
* Selected best configuration based on PR-AUC

## Configuration Management 
Saved in (config.json):
* Selected features (Top 150)
* Optimal threshold

## API Deployment
Built inference API using **FastAPI**
Features:
* Loads trained model
* Arranges features in the training order of the model using `config.json`
* Used Tuned threshold
* Returns classification output and probability

## Dockerization
* Entire project containerized using **Docker**
* Pushed image to [Docker Repository](https://hub.docker.com/repository/docker/abhinay1289/secom_api) for reproducible environment

## Cloud Deployment
* Deployed on **AWS(Amazon Web Services) EC2** (Ubuntu instance)
* Pulled Docker image in the instance and served API on cloud

## Try it out
run `docker pull abhinay1289/secom_api:latest`\
run `docker run -p 8000:8000 abhinay1289/secom_api:latest`\

access API on `http://localhost:8000` and interactive docs on `http://localhost:8000/docs`\

*Example response*
```
{
  "prediction": pass,
  "probability": 0.63
}
```
