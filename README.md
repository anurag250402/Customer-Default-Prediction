# 🏦 Customer Default Prediction

This project focuses on predicting whether a customer will default on a loan using machine learning. It covers data cleaning, preprocessing, model training, and evaluation.

---

## 🎯 Project Objective

* Predict the likelihood of loan default based on customer financial and loan data.
* Clean and prepare raw data to be model-ready.
* Use classification algorithms like AdaBoost to improve accuracy.

---

## 🛠️ Libraries Used

### Core Libraries

* **Pandas:** For loading and exploring the dataset.
* **NumPy:** For working with numerical arrays and math operations.

### Visualization

* **Matplotlib & Seaborn:** Used to plot data distributions, trends, and model performance.

### Preprocessing & Modeling (scikit-learn)

* **train\_test\_split:** Splits the data into training and validation sets.
* **LabelEncoder:** Converts categories like home ownership or loan intent into numeric format.
* **StandardScaler:** Scales features for better model performance.
* **CountVectorizer:** Converts text data (e.g., genres) into numeric vectors.
* **metrics:** Offers tools for evaluating model accuracy.

### Advanced Modeling

* **XGBoost / AdaBoost:** High-performance gradient boosting algorithms used for better predictions.

### Utility

* **warnings.filterwarnings('ignore'):** Hides unnecessary warning messages for cleaner output.

---

## 📂 Dataset Overview

The dataset includes:

* `customer_income`
* `loan_amnt`
* `home_ownership`
* `loan_intent`
* `loan_grade`
* `historical_default`
* `Current_loan_status`

---

## 🧹 Data Preprocessing Steps

### 1. Handle Missing Values

```python
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
```

**Explanation:**

* Numeric columns are filled with the column mean.
* Categorical columns use the most frequent value.

### 2. Clean Numeric Columns Stored as Strings

```python
columns_to_clean = ['customer_income', 'loan_amnt']

for col in columns_to_clean:
    df[col] = df[col].replace({',': ''}, regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')
```

**Explanation:**

* Commas are removed and values are converted to numeric.

### 3. Recheck for NaNs After Conversion

```python
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
```

**Explanation:**

* Ensures no missing numeric values remain.

### 4. Encode Categorical Variables

```python
label_encoder = LabelEncoder()
df['home_ownership'] = label_encoder.fit_transform(df['home_ownership'])
df['loan_intent'] = label_encoder.fit_transform(df['loan_intent'])
df['loan_grade'] = label_encoder.fit_transform(df['loan_grade'])
df['historical_default'] = label_encoder.fit_transform(df['historical_default'])
df['Current_loan_status'] = label_encoder.fit_transform(df['Current_loan_status'])
```

**Explanation:**

* Transforms string-based categories into numeric codes.

---

## 🧠 Model Training

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
```

### Training AdaBoost Classifier

```python
model = AdaBoostClassifier()
model.fit(X_train, y_train)
```

### Evaluation

```python
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
```

---

## ✅ Results

* **Accuracy Score:** *92%*

---

## 📁 Files Included

* `Customer_Default_Prediction.ipynb`: Full code and walkthrough
* `README.md`: Project overview
* `LoanDataset---LoansDatasest`: Loan default dataset

---

## 🙌 Credits

* Data cleaning techniques adapted from industry-standard practices
* Model implementation and documentation by \[Anurag Tripathi]

---

## 📬 Contact

For collaboration or questions:
**Email:** \[[2005051530009anurag@gmail.com](mailto:2005051530009anurag@gmail.com)]
