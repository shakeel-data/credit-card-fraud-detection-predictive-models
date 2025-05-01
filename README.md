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

```python
df = pd.read_csv('path')
```

### 2. Data Inspection & Preparation
**Check the data**

```python
print("Credit Card Fraud Detection data -  rows:",df.shape[0],", columns:", df.shape[1])
```
![image](https://github.com/user-attachments/assets/d9d148aa-cba8-49e7-b46b-ba7541d3f37a)

**Glimpse the data**

```python
df.head()
```

Generate descriptive statistics
```python
df.describe()
```
we can confirm that the data contains 284,807 transactions, during 2 consecutive days (or 172792 seconds).

**Check missing data**

```python
print(df.isnull().sum())
```
![image](https://github.com/user-attachments/assets/ead3040f-68f7-48c1-ac3c-23901824b861)
There is no missing data in the entire dataset.

**Check data unbalance**

```python
sns.countplot(x='Class', data=df)
plt.title("0: Legitimate vs 1: Fraud")
plt.show()
print(df['Class'].value_counts(normalize=True))
```
![image](https://github.com/user-attachments/assets/3de49219-cb04-4e3a-bb83-2054650f37f7)

### 3. Exploratory Data Analysis (EDA)
Visualize transaction density over time for both classes using distribution plots (Plotly ff.create_distplot) to observe patterns; fraudulent transactions showed a more even distribution.

**Transactions in time**

```python
class_0 = df.loc[df['Class'] == 0]["Time"]
class_1 = df.loc[df['Class'] == 1]["Time"]

hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))
iplot(fig, filename='dist_only')
```
![image](https://github.com/user-attachments/assets/0a7966d4-cafb-4802-8204-55cc82970c0c)

**Aggregate transaction statistics (min, max, count, sum, mean, median, variance) per hour for both classes.**

```python
df['Hour'] = df['Time'].apply(lambda x: np.floor(x / 3600))

tmp = df.groupby(['Hour', 'Class'])['Amount'].aggregate(['min', 'max', 'count', 'sum', 'mean', 'median', 'var']).reset_index()
df = pd.DataFrame(tmp)
df.columns = ['Hour', 'Class', 'Min', 'Max', 'Transactions', 'Sum', 'Mean', 'Median', 'Var']
df.head()
```
![image](https://github.com/user-attachments/assets/eecfb8c8-0377-456b-b8af-016a1cd0fdee)

**Visualize aggregated transaction sums and means per hour.**

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Sum", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Sum", data=df.loc[df.Class==1], color="red")
plt.suptitle("Total Amount")
plt.show();
```
![image](https://github.com/user-attachments/assets/76119677-7472-4c3f-8f67-0d2ca2789b9b)

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Transactions", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Transactions", data=df.loc[df.Class==1], color="red")
plt.suptitle("Total Number of Transactions")
plt.show();
```
![image](https://github.com/user-attachments/assets/04012331-74b5-4b5f-b2b4-e7cffdc0136f)

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Mean", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Mean", data=df.loc[df.Class==1], color="red")
plt.suptitle("Average Amount of Transactions")
plt.show();
```
![image](https://github.com/user-attachments/assets/e875fdac-8de4-410a-9edc-ac348ed8568a)

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Max", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Max", data=df.loc[df.Class==1], color="red")
plt.suptitle("Maximum Amount of Transactions")
plt.show();
```
![image](https://github.com/user-attachments/assets/e207e27c-94c5-41a6-afe6-dbd6e5617de3)

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Median", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Median", data=df.loc[df.Class==1], color="red")
plt.suptitle("Median Amount of Transactions")
plt.show();
```
![image](https://github.com/user-attachments/assets/683b163d-1683-43fd-b776-70b153092c74)

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Min", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Min", data=df.loc[df.Class==1], color="red")
plt.suptitle("Minimum Amount of Transactions")
plt.show();
```
![image](https://github.com/user-attachments/assets/14af20a8-08ee-4e77-96a5-281500463991)























































































































