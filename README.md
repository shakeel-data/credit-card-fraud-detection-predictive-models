#  💳 Real-time Credit Card Fraud Detection Project | XGBoost + LightGBM 

![image](https://github.com/user-attachments/assets/c3b421eb-de00-4415-8669-b02b517f02dd)

Credit card fraud detection predictive models are crucial for **protecting finances, enhancing security by proactively identifying evolving threats** in real-time, and improving customer experience by minimizing disruptions from false positives. They also boost operational efficiency through automation and scalability, and help financial institutions meet regulatory requirements. These models analyze transaction data using machine learning to identify fraud patterns, enabling timely intervention and reducing financial losses for both cardholders and institutions. In short, these models are **vital for a secure and efficient financial ecosystem.**

## 📘 Project Overview
- This project aims to develop and evaluate machine learning models for detecting fraudulent credit card transactions. The analysis uses a dataset containing transactions made by **European cardholders** over two consecutive days in **September 2013**. This dataset is notable for its severe class imbalance, with only **492 fraudulent transactions (0.172%)** out of a total of 284,807. Due to confidentiality, the original transaction features are not available. 
- Instead, the dataset primarily consists of **28** numerical features **(V1 to V28)** derived from a **Principal Component Analysis (PCA)** transformation of the original data. The only features not transformed by PCA are 'Time' (seconds elapsed between each transaction and the first transaction) and 'Amount' (transaction value). The target variable, 'Class', indicates **fraud (1) or legitimacy (0).**

## 🎯 Key Objectives
- Develop robust machine learning models capable of **accurately identifying fraudulent credit card transactions** within a highly imbalanced dataset.
- Perform thorough **exploratory data analysis** to understand transaction patterns and **feature characteristics.**
- Compare the predictive performance of **XGBoost** and **LightGBM** using the ROC-AUC metric.
- Establish a baseline for **fraud detection on this dataset**.

### 📁 Data Sources
- Kaggle
  - https://drive.google.com/file/d/1HRVxmmhNT3-OPhYFt_YeKwK5cfHVW6K4/view?usp=sharing (Hosted externally due to GitHub's file size limits)
- Python
  <a href="https://github.com/shakeel-data/credit-card-fraud-deduction-predictive-models/blob/c19aa61004a6017a6b824b2ce8304a78884b9ffb/credit_card_fraud_deduction_predictive_models.ipynb">codes</a>

## 🔧 Project Workflow
### 1. 📥 Load Packages and Data Ingestion

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
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier


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
df = pd.read_csv('file-path')
```

### 2. 🛠️ Data Inspection and Preparation
**Check the data**

```python
print("Credit Card Fraud Detection data -  rows:",df.shape[0],", columns:", df.shape[1])
```
![image](https://github.com/user-attachments/assets/d9d148aa-cba8-49e7-b46b-ba7541d3f37a)

**Glimpse the data**

```python
df.head()
```

**Generate descriptive statistics**
```python
df.describe()
```
we can confirm that the data contains **284,807 transactions, during 2 consecutive days (or 172792 seconds).**

**Check missing data**

```python
print(df.isnull().sum())
```
![image](https://github.com/user-attachments/assets/ead3040f-68f7-48c1-ac3c-23901824b861)
There is **no missing data** in the entire dataset.

**Check data unbalance**

```python
sns.countplot(x='Class', data=df)
plt.title("0: Legitimate vs 1: Fraud")
plt.show()
print(df['Class'].value_counts(normalize=True))
```
![image](https://github.com/user-attachments/assets/3de49219-cb04-4e3a-bb83-2054650f37f7)

### 3. 📊 Exploratory Data Analysis (EDA)
**Visualize transaction density over time for both classes using distribution plots (Plotly ff.create_distplot) to observe patterns; fraudulent transactions showed a more even distribution**

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

**Aggregate transaction statistics (min, max, count, sum, mean, median, variance) per hour for both classes**

```python
df['Hour'] = df['Time'].apply(lambda x: np.floor(x / 3600))

tmp = df.groupby(['Hour', 'Class'])['Amount'].aggregate(['min', 'max', 'count', 'sum', 'mean', 'median', 'var']).reset_index()
df = pd.DataFrame(tmp)
df.columns = ['Hour', 'Class', 'Min', 'Max', 'Transactions', 'Sum', 'Mean', 'Median', 'Var']
df.head()
```
![image](https://github.com/user-attachments/assets/eecfb8c8-0377-456b-b8af-016a1cd0fdee)

**Visualize aggregated transaction sums and means per hour**

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

**Compute and visualize feature correlations using a heatmap (Seaborn heatmap)**

```python
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm', linewidths=0.2)
plt.title("Feature Correlation")
plt.show()
```
![image](https://github.com/user-attachments/assets/7de2add5-ef0e-41d2-97e9-092c9b8b5ca8)

## 4. 🤖 Preparing Predictive Models
**Define predictor features (V1-V28, Time, Amount) and the target feature ('Class')**
- Let's define the predictor features and the target features. Categorical features, if any, are also defined. In our case, there are no categorical feature.

```python
target = 'Class'
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\
       'Amount']
```
*Split data in train, test and validation set*
**Split the data into training (64%), validation (16%), and test (20%) sets using train_test_split**

```python
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True )
train_df, valid_df = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )
```

### ✖️ XGBoost
- Prepare data using **xgb.DMatrix.**
- Define XGBoost parameters **(objective='binary:logistic', eta=0.039, max_depth=2, etc.).**
- Train the XGBoost model using the **training set, monitoring performance** on the validation set with early stopping **(based on AUC).**
- **Predict probabilities** on the test set using the best iteration.
- Evaluate the test set predictions using **ROC-AUC score.**

**Prepare the model**

```python
# Prepare the train and valid datasets
dtrain = xgb.DMatrix(train_df[predictors], train_df[target].values)
dvalid = xgb.DMatrix(valid_df[predictors], valid_df[target].values)
dtest = xgb.DMatrix(test_df[predictors], test_df[target].values)

#What to monitor (in this case, **train** and **valid**)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Set xgboost parameters
params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.039
params['silent'] = True
params['max_depth'] = 2
params['subsample'] = 0.8
params['colsample_bytree'] = 0.9
params['eval_metric'] = 'auc'
params['random_state'] = RANDOM_STATE
```

**Train the model**

```python
model = xgb.train(params, 
                dtrain, 
                MAX_ROUNDS, 
                watchlist, 
                early_stopping_rounds=EARLY_STOP, 
                maximize=True, 
                verbose_eval=VERBOSE_EVAL)
```
![image](https://github.com/user-attachments/assets/aea6aa61-14a9-4264-8d40-a50ed42a75a0)
The best validation score **(ROC-AUC)** was **0.979**

**Plot Variable Importance**
```python
fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))
xgb.plot_importance(model, height=0.8, title="Features importance (XGBoost)", ax=ax, color="green") 
plt.show()
```
![image](https://github.com/user-attachments/assets/a9afdfc1-f4eb-4bf3-a48d-8a68823dbae5)

**Predict test set**

```python
preds = model.predict(dtest)
```

**Area Under Curve**

```python
roc_auc_score(test_df[target].values, preds)
```
![image](https://github.com/user-attachments/assets/8a6713ea-3fb2-4b9d-8907-975df37ede4e)
The **AUC score** for the prediction of **fresh data** (test set) is **0.976**.

### ⚡ LightGBM
- Define **model parameters**
 - Set the parameters for the model. We will use these parameters only for the first lgb model.

```python
params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric':'auc',
          'learning_rate': 0.05,
          'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
          'max_depth': 4,  # -1 means no limit
          'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
          'max_bin': 100,  # Number of bucketed bin for feature values
          'subsample': 0.9,  # Subsample ratio of the training instance.
          'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
          'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
          'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
          'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
          'nthread': 8,
          'verbose': 0,
          'scale_pos_weight':150, # because training data is extremely unbalanced 
         }
```

**Prepare the model**

```python
dtrain = lgb.Dataset(train_df[predictors].values, 
                     label=train_df[target].values,
                     feature_name=predictors)

dvalid = lgb.Dataset(valid_df[predictors].values,
                     label=valid_df[target].values,
                     feature_name=predictors)
```

**Run the model**

```python
evals_results = {}

model = lgb.train(params, 
                  dtrain, 
                  valid_sets=[dtrain, dvalid], 
                  valid_names=['train','valid'], 
                  num_boost_round=MAX_ROUNDS,
                  callbacks=[lgb.early_stopping(2 * EARLY_STOP), lgb.log_evaluation(VERBOSE_EVAL)],
                  feval=None)

# Store evaluation results
evals_results = model.evals_result
```
![image](https://github.com/user-attachments/assets/afb85053-1e32-4789-8668-cfedae12ad29)
Best validation score was obtained for **AUC** ~= **0.957.**

**Plot variable Importance.**

```python
fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))
lgb.plot_importance(model, height=0.8, title="Features importance (LightGBM)", ax=ax,color="purple") 
plt.show()
```
![image](https://github.com/user-attachments/assets/81dd76c3-1c8d-4d3b-803a-e8f57dabecfb)

**Predict test data**

```python
preds = model.predict(test_df[predictors])
```
**Area Under Curve**
Calculate the **ROC-AUC score** for the prediction.

```python
roc_auc_score(test_df[target].values, preds)
```
![image](https://github.com/user-attachments/assets/f39385e1-231a-48dc-b74e-404190ac254c)
The **ROC-AUC** score obtained for the test set is **0.947.**


## 🌟 Highlights and Key Insights
- **Data Imbalance:** The dataset presents a significant challenge due to the rarity of **fraud cases (0.172%).**
- **PCA Transformation:** Anonymization via **PCA** means feature interpretability is limited, requiring **models robust to abstract features.**
- **Temporal Patterns:** Visualization revealed differing **temporal distributions between fraudulent and legitimate transactions,** suggesting 'Time' might hold predictive value despite PCA.
- **Model Performance:** **XGBoost** demonstrated significantly better performance **(Test ROC-AUC: 0.976)** compared to the baseline **LightGBM** **(Test ROC-AUC: 0.947)** on this task. **XGBoost's** validation AUC reached **0.979** and  **LightGBM's** validation AUC reached **0.957** during training.
- **Data Quality:** The dataset was complete with no missing values.

## ☁️ Technologies and Tools
- **Kaggle** – Dataset source
- **Google Colab** – Interactive environment for coding and presenting analysis
- **Python** – Data analysis, Manipulation and Visualization 
  - Libraries: ```numpy```, ```pandas```, ```matplotlib```, ```seaborn```, ```plotly```
- **Machine Learning** – Model development and evaluation
  - Scikit-learn:```train_test_split```, ```roc_auc_score```, ```LGBMClassifier```
  - XGBoost: ```xgb.DMatrix```, ```xgb.train```
  - LightGBM: ```lgb.train```

## 🔚🔁 Conclusion & Next Steps
The analysis successfully demonstrated the application of machine learning for credit card fraud detection on an imbalanced, PCA-transformed dataset. **XGBoost** emerged as the superior model, achieving a high ROC-AUC score of **0.976** on the unseen test data, indicating its effectiveness in distinguishing fraudulent from legitimate transactions under these conditions. The **LightGBM** provided a baseline but was significantly outperformed.

### Next Steps:
- Conduct comprehensive hyperparameter tuning for XGBoost (e.g., using **GridSearchCV or RandomizedSearchCV**) to potentially further enhance performance.
- Explore **feature engineering** possibilities, particularly with the **'Time'** and **'Amount'** features, perhaps by creating cyclical time features or scaling 'Amount'.
- Evaluate other advanced classification models suitable for imbalanced data, such as **AdaBoost, CatBoost**, or potentially deep learning approaches (e.g., **Autoencoders, LSTMs** if sequential patterns are relevant).
