# 🌸 Iris Flower Classification

A machine learning project that classifies Iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — using classic ML algorithms with the well-known Iris dataset.

---

## 📁 Project Structure

```
IRIS-FLOWER-CLASSIFICATION/
├── iris_flowers_classification.ipynb  # Main Jupyter notebook
├── Iris.csv                           # Dataset
├── iris_rf_model.pkl                  # Saved Random Forest model
├── app.py                             # Streamlit web app
├── web.py                             # Web utilities
├── prediction.py                      # Prediction script
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## 📊 Dataset

- **Source:** Iris dataset (`Iris.csv`)
- **Samples:** 150 (50 per class)
- **Features:**
  - `SepalLengthCm`
  - `SepalWidthCm`
  - `PetalLengthCm`
  - `PetalWidthCm`
- **Target:** `Species` — `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`

---

## 🔍 Workflow

### 1. Data Loading & Cleaning
- Load dataset using `pandas`
- Drop the redundant `Id` column
- Check for missing/null values

### 2. Exploratory Data Analysis (EDA)
- Summary statistics with `df.describe()`
- Histogram plots for all features
- **Pairplot** (using Seaborn) colored by species to visualize feature separability
- **Correlation heatmap** of features

### 3. Preprocessing
- Encode the `Species` target column using `LabelEncoder` (0, 1, 2)
- Split data into training and testing sets (70/30 split)

### 4. Model Training & Evaluation

| Model | Type |
|---|---|
| Logistic Regression | Linear classifier |
| Decision Tree | Tree-based classifier |
| Random Forest | Ensemble (100 estimators) |

- **Cross-validation:** 5-fold KFold cross-validation on Random Forest
- **Metrics:** Accuracy score on train and test sets

### 5. Model Saving
- Final Random Forest model saved using `joblib` as `iris_rf_model.pkl`

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook iris_flowers_classification.ipynb
```

### Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas**, **numpy** — data manipulation
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — ML models, preprocessing, evaluation
- **joblib** — model serialization
- **Streamlit** — web app interface

---

## 📈 Results

The **Random Forest Classifier** (with 5-fold cross-validation) achieved the best and most stable performance on this dataset, with near-perfect accuracy on both training and test sets.

---

## 📌 Notes

- The `Id` column is dropped during preprocessing as it carries no predictive value.
- Label encoding maps species alphabetically: `Iris-setosa → 0`, `Iris-versicolor → 1`, `Iris-virginica → 2`.
- The saved model (`iris_rf_model.pkl`) can be loaded directly for inference without retraining.

```python
import joblib
model = joblib.load("iris_rf_model.pkl")
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
```
