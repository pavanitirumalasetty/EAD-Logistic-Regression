# EAD-Logistic-Regression
# Titanic Survival Prediction using Logistic Regression

This project demonstrates the implementation of Logistic Regression to predict survival on the Titanic using the [Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic). This dataset is a common starting point for students in machine learning, focusing on binary classification: **Survived** or **Deceased**.

---

## Project Overview
The objective of this project is to predict whether a passenger survived the Titanic disaster using various features such as age, gender, passenger class, and fare. The model is built using Logistic Regression, a popular classification algorithm.

---

## Dataset
The dataset used is a semi-cleaned version of the Titanic data set. It contains the following relevant features:

- **Pclass**: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Fare paid by the passenger
- **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis was performed using **Seaborn** and **Matplotlib** to understand the data distribution and relationships between features. Key visualizations include:

- **Heatmap** for missing data visualization
- **Count plots** for survival rate by gender and passenger class
- **Histograms** for age and fare distributions

---

## Data Cleaning and Preprocessing
Data cleaning involved:
- Imputing missing values in the **Age** column based on passenger class averages.
- Dropping the **Cabin** column due to excessive missing values.
- Removing rows with missing values in the **Embarked** column.

Categorical features were converted to numerical dummy variables using **pandas.get_dummies()**, including:
- **Sex**: Converted to binary (0 = Female, 1 = Male)
- **Embarked**: Dummy variables for ports, dropping the first category to avoid multicollinearity.

---

## Model Building

### Logistic Regression
- Implemented using **scikit-learn's** `LogisticRegression()` model.
- Split the dataset into training and testing sets using `train_test_split()` with a 70-30 ratio.

---

## Model Evaluation
Model performance was evaluated using:
- **Confusion Matrix** to visualize true positives, false positives, true negatives, and false negatives.
- **Accuracy Score** for overall model accuracy.
- **Classification Report** to evaluate precision, recall, and F1-score.

### Results
The model showed reasonable accuracy, demonstrating the effectiveness of logistic regression for binary classification. Future improvements can be achieved through advanced feature engineering and hyperparameter tuning.

---

## Dependencies
- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- cufflinks (for interactive visualizations)

Install dependencies using:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn cufflinks
