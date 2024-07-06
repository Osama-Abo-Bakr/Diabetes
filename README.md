# Diabetes Prediction

## Project Overview

This project focuses on predicting diabetes in patients using machine learning classification algorithms. The dataset includes medical records with various features. The workflow involves data preprocessing, model building, hyperparameter tuning, and evaluation.

## Table of Contents

1. [Libraries and Tools](#libraries-and-tools)
2. [Project Steps](#project-steps)
3. [Models Used](#models-used)
4. [Results](#results)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Conclusion](#conclusion)
8. [Contact](#contact)

## Libraries and Tools

- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Data preprocessing, model building, and hyperparameter tuning
- **XGBoost**: Gradient boosting

## Project Steps

1. **Reading and Exploring Data**:
    - Loaded the dataset and inspected its structure.
    - Verified the data types and checked for null values.

2. **Data Cleaning**:
    - Identified and removed outliers in specific features like Insulin, Skin Thickness, and Glucose.

3. **Data Visualization**:
    - Created histograms and boxplots to visualize feature distributions.
    - Used heatmaps to analyze feature correlations.

4. **Data Splitting**:
    - Divided the data into training and testing sets with a 70:30 ratio.

5. **Model Building and Evaluation**:
    - Implemented and evaluated various classification models.
    - Applied GridSearchCV for hyperparameter optimization.

## Models Used

1. **Logistic Regression**:
    - Configured with a maximum of 5000 iterations.
    - Evaluated performance on training and testing data.

2. **Random Forest**:
    - Set with 30 estimators and specific parameters for depth, samples, and features.
    - Hyperparameters were fine-tuned using GridSearchCV.

3. **XGBoost**:
    - Applied for gradient boosting classification.

4. **AdaBoost**:
    - Implemented with Random Forest and Decision Tree as base estimators.
    - Evaluated performance on both training and testing data.

## Results

- **Logistic Regression**:
    - Training Accuracy: 0.7883
    - Testing Accuracy: 0.7657

- **Random Forest**:
    - Training Accuracy: 0.8194
    - Testing Accuracy: 0.7883

- **AdaBoost (Random Forest)**:
    - Training Accuracy: 0.8233
    - Testing Accuracy: 0.7883

- **AdaBoost (Decision Tree)**:
    - Training Accuracy: 0.8097
    - Testing Accuracy: 0.7928

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/diabetes-prediction.git
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or script to see the results.

## Usage

1. **Data Preparation**:
    - Ensure the dataset is located at the specified path in the code.

2. **Model Training**:
    - Execute the script or Jupyter notebook to train the models and view performance metrics.

3. **Evaluation**:
    - Review the output accuracy scores to assess model efficacy.

## Conclusion

This project demonstrates the use of machine learning for predicting diabetes. The results underscore the importance of data preprocessing and model selection in achieving high accuracy.

## Contact

For questions or collaborations, feel free to get in touch.

- **Email**: [Gmail](mailto:osamaoabobakr12@gmail.com)
- **LinkedIn**: [LinkedIn](https://linkedin.com/in/osama-abo-bakr-293614259/)

---

### Sample Code (for reference)

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Read data from CSV
data = pd.read_csv('path_to_dataset.csv')

# Explore data
print(data.shape)
data.info()

# Data visualization
data.hist(figsize=(20, 10))
data.boxplot(figsize=(20, 10))

# Handle outliers
data = data[data["Insulin"] <= 350]
data = data[data["SkinThickness"] <= 80]
data = data[data["Glucose"] >= 5]

# Visualize correlations
plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, square=True, cmap="Blues", fmt="0.1f")

# Split data
X = data.drop(columns="Outcome", axis=1)
Y = data["Outcome"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=5)

# Model 1: Logistic Regression
model_LR = LogisticRegression(max_iter=5000)
model_LR.fit(x_train, y_train)
print(f"Logistic Regression - Train Accuracy: {model_LR.score(x_train, y_train)}, Test Accuracy: {model_LR.score(x_test, y_test)}")

# Model 2: Random Forest
model_RF = RandomForestClassifier(n_estimators=30, max_depth=5, min_samples_leaf=3, min_samples_split=4, max_samples=0.2, max_features=4, n_jobs=-1)
model_RF.fit(x_train, y_train)
print(f"Random Forest - Train Accuracy: {model_RF.score(x_train, y_train)}, Test Accuracy: {model_RF.score(x_test, y_test)}")

# Hyperparameter tuning for Random Forest
param2 = {"n_estimators": np.arange(65, 70, 1), "max_depth": np.arange(5, 8, 1), "min_samples_split": np.arange(1, 3, 1), "min_samples_leaf": np.arange(2, 4, 1), "max_samples": [0.18, 0.2, 0.21, 0.24]}
new_model_RF = GridSearchCV(estimator=model_RF, param_grid=param2, verbose=6, cv=5, n_jobs=-1)
new_model_RF.fit(x_train, y_train)
model_RF = new_model_RF.best_estimator_
print(f"Random Forest (Tuned) - Train Accuracy: {model_RF.score(x_train, y_train)}, Test Accuracy: {model_RF.score(x_test, y_test)}")

# Model 3: XGBoost
model_XGB = xgb.XGBClassifier()
model_XGB.fit(x_train, y_train)
print(f"XGBoost - Train Accuracy: {model_XGB.score(x_train, y_train)}, Test Accuracy: {model_XGB.score(x_test, y_test)}")

# Model 4: AdaBoost
model_AD = AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=5, max_samples=0.4, min_samples_leaf=5, n_estimators=18, n_jobs=-1), n_estimators=5, learning_rate=0.1)
model_AD.fit(x_train, y_train)
print(f"AdaBoost (Random Forest) - Train Accuracy: {model_AD.score(x_train, y_train)}, Test Accuracy: {model_AD.score(x_test, y_test)}")

# AdaBoost with Decision Tree
model_AD1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=25, min_samples_leaf=4, max_features=2, max_leaf_nodes=3), n_estimators=5, learning_rate=1)
model_AD1.fit(x_train, y_train)
print(f"AdaBoost (Decision Tree) - Train Accuracy: {model_AD1.score(x_train, y_train)}, Test Accuracy: {model_AD1.score(x_test, y_test)}")

# User input prediction
feature_input = np.asarray(list(map(float, input().split(",")))).reshape(1, -1)
prediction = model_AD1.predict(feature_input)
if prediction == [0]:
    print("The Person is Not Diabetic")
else:
    print("The Person is Diabetic")
```
