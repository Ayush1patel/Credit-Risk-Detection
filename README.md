# Credit Risk Modelling

## Introduction
Credit risk modelling refers to estimating the risk associated with lending credit to a borrower. If a lender fails to detect credit risk in advance, it exposes them to the risk of default and loss of funds. Therefore, lenders rely on the validation provided by credit risk analysis models to make key lending decisions, such as whether or not to extend credit to a borrower and what interest rate should be charged.

The probability of default (POD) is the likelihood that a borrower will default on their loan obligations. This project aims to accurately predict this probability using various machine learning algorithms.

## Project Overview
In this project, different learning algorithms including **K-Nearest Neighbors (KNN), Logistic Regression, Decision Tree, and Random Forest** have been evaluated to find the best algorithm for predicting credit risk. 

Key techniques applied in this project include:
* **KNN Imputer** for imputing missing numerical values.
* **SMOTE (Synthetic Minority Over-sampling Technique)** for handling imbalanced dataset issues.
* **Exploratory Data Analysis (EDA)** to discover hidden patterns and handle outliers.

## Dataset and Features
The dataset for this credit risk modelling project is sourced from Kaggle. It classifies the credit risk of borrowers with respect to various demographic and financial features.

**Key Features include:**
* `person_age`: Age of the borrower.
* `person_income`: Annual income of the borrower.
* `person_home_ownership`: Type of home ownership (e.g., RENT, MORTGAGE, OWN).
* `person_emp_length`: Employment length of the borrower in years.
* `loan_intent`: Purpose of the loan (e.g., PERSONAL, EDUCATION, MEDICAL).
* `loan_grade`: Assigned loan grade.
* `loan_amnt`: Total loan amount requested.
* `loan_int_rate`: Interest rate of the loan.
* `loan_percent_income`: Percentage of income going towards the loan.
* `cb_person_default_on_file`: Historical default record (Y/N).
* `cb_person_cred_hist_length`: Length of credit history.

**Target Variable:**
* `loan_status`: Binary variable where `1` indicates default and `0` indicates non-default.

## Tech Stack & Libraries Used
* **Python 3**
* **Pandas & NumPy**: For data manipulation and numerical computations.
* **Matplotlib & Seaborn**: For data visualization and EDA.
* **Scikit-Learn (sklearn)**: For model building, preprocessing, and evaluation metrics.
* **Imbalanced-Learn**: For SMOTE (handling imbalanced data).

## Workflow and Methodology

### 1. Data Preprocessing
* **Duplicate Removal:** Identified and removed 165 duplicate data points to ensure data integrity.
* **Handling Missing Values:**
  * `person_emp_length` (895 missing values): Imputed using the statistical **mode** (most frequent value).
  * `loan_int_rate` (3116 missing values): Handled using a **KNN Imputer**.
* **Categorical Encoding:** Categorical text features (like `person_home_ownership`, `loan_intent`, `loan_grade`, etc.) were mapped and encoded into numerical formats (e.g., RENT -> 1, MORTGAGE -> 2, OWN -> 3) so they could be fed into the machine learning models.

### 2. Exploratory Data Analysis (EDA) & Outlier Handling
* Visualized feature distributions and relationships (e.g., analyzing `loan_amnt` grouped by loan purpose using boxplots).
* Identified extreme, unrealistic outliers in continuous variables such as `person_age` (e.g., age > 100), `person_income`, and `person_emp_length` (e.g., 123 years of employment). These outliers were systematically removed to prevent model distortion.

### 3. Data Splitting & Balancing
* **Train-Test Split:** The cleaned data was split into training and testing sets (80% training, 20% testing).
* **Handling Imbalanced Data:** The target class (`loan_status`) was highly imbalanced (far more non-defaults than defaults). **SMOTE** was applied to the training set to synthetically generate minority class samples, resulting in a perfectly balanced training dataset.

### 4. Model Training & Evaluation
Four different algorithms were trained and tested:
1. **K-Nearest Neighbors (KNN):** Hyperparameter tuning was performed iteratively to find the optimal `K`. `K=11` yielded the best predictions.
2. **Logistic Regression**
3. **Decision Tree Classifier**
4. **Random Forest Classifier**

Each model was evaluated using:
* **Accuracy Score**
* **Precision, Recall, and F1-Score** (via Classification Report)
* **Confusion Matrix** (Visualized using Seaborn heatmaps to assess False Positives and False Negatives).

## Conclusion and Results
After fitting and testing the different models on the processed data, the **Random Forest** algorithm performed better than all other algorithms evaluated. It was able to successfully predict the credit risk on the unseen test set with an impressive **accuracy of 93%**. I observed a significant improvement in the model's predictive power and recall metrics after properly handling imperfect data (outliers, missing values) and addressing the class imbalance with SMOTE.

## How to Run the Notebook
1. Clone the repository or download the `credit-risk-notebook.ipynb` file.
2. Upload the notebook to Google Colab or open it via a local Jupyter Notebook environment.
3. Download the `credit_risk_dataset.csv` from Kaggle and place it in your working directory.
4. Run the notebook cells sequentially. If using Colab, you will be prompted to upload the CSV file in the data loading section.
