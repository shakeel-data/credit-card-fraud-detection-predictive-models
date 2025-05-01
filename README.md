# 💳 Credit Card Fraud Deduction Predictive Models | RadomforestClassifier + XGBOOST
![image](https://github.com/user-attachments/assets/525744be-e352-4bdb-a882-1bc3555641b6)

Credit card fraud detection predictive models are crucial for protecting finances, enhancing security by proactively identifying evolving threats in real-time, and improving customer experience by minimizing disruptions from false positives. They also boost operational efficiency through automation and scalability, and help financial institutions meet regulatory requirements. These models analyze transaction data using machine learning to identify fraud patterns, enabling timely intervention and reducing financial losses for both cardholders and institutions. In short, these models are vital for a secure and efficient financial ecosystem.

## 📘 Project Overview
This project aims to develop and evaluate machine learning models for detecting fraudulent credit card transactions. The analysis uses a dataset containing transactions made by European cardholders over two consecutive days in September 2013. This dataset is notable for its severe class imbalance, with only 492 fraudulent transactions (0.172%) out of a total of 284,807. Due to confidentiality, the original transaction features are not available. Instead, the dataset primarily consists of 28 numerical features (V1 to V28) derived from a Principal Component Analysis (PCA) transformation of the original data. The only features not transformed by PCA are 'Time' (seconds elapsed between each transaction and the first transaction) and 'Amount' (transaction value). The target variable, 'Class', indicates fraud (1) or legitimacy (0).

## 🎯 Objectives
- Develop robust machine learning models capable of accurately identifying fraudulent credit card transactions within a highly imbalanced dataset.
- Perform thorough exploratory data analysis to understand transaction patterns and feature characteristics.
- Compare the predictive performance of RandomForestClassifier and XGBoost using the ROC-AUC metric.
- Establish a baseline for fraud detection on this dataset.

## 📂 Dataset used
- Kaggle
  <a href="">csv</a>
- Python
  <a href="">codes</a>

## 🔄 Project Workflow
### 1. Load pakages and Data Ingestion
Import necessary Python libraries (Pandas, NumPy, Scikit-learn, XGBoost, visualization tools) and load the creditcard.csv dataset.

```python
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

RFC_METRIC = 'gini'  #metric used for RandomForrestClassifier
NUM_ESTIMATORS = 100 #number of estimators used for RandomForrestClassifier
NO_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier


#TRAIN/VALIDATION/TEST SPLIT
#VALIDATION
VALID_SIZE = 0.20 # simple validation using train_test_split
TEST_SIZE = 0.20 # test size using_train_test_split


RANDOM_STATE = 2018

MAX_ROUNDS = 1000 #lgb iterations
EARLY_STOP = 50 #lgb early stop 
OPT_ROUNDS = 1000  #To be adjusted based on best validation rounds
VERBOSE_EVAL = 50 #Print out metric result
```

**Read the data**















































































































