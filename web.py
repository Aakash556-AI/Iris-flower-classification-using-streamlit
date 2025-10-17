# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from prediction import predict

# ---------------------------
# Load dataset and model
# ---------------------------
iris = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
model = joblib.load("iris_rf_model.pkl")

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="🌸 Iris Classifier", layout="wide")
st.title("🌼Iris Flower Classification & Analytics")
st.markdown("Predict **Setosa, Versicolor, Virginica** and explore feature analytics.")

# ---------------------------
# Input sliders
# ---------------------------
st.header("Enter Flower Features")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Sepal")
    sepal_l = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_w = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
with col2:
    st.subheader("Petal")
    petal_l = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_w = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict"):
    flower, probabilities = predict(features)
    
    st.subheader("🌼 Predicted Flower")
    st.success(flower["name"])
    st.image(flower["image"], width=400)

    # Probabilities table
    df_prob = pd.DataFrame({
        "Species": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        "Probability (%)": np.round(probabilities*100, 2)
    })
    st.subheader("Prediction Probabilities")
    st.table(df_prob)

    # Probability Distribution Bar Chart
    st.subheader("Probability Distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x="Species", y="Probability (%)", data=df_prob, palette="Set2", ax=ax)
    ax.set_ylim(0, 100)
    st.pyplot(fig)

# ---------------------------
# Analytics Section
# ---------------------------
st.header("📊 Feature Analytics")

# 1️⃣ Feature Importance
st.subheader("Random Forest Feature Importance")
importances = model.feature_importances_
features_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df_importance = pd.DataFrame({
    'Feature': features_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=df_importance, palette="viridis", ax=ax)
st.pyplot(fig)

# 2️⃣ Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
corr = iris.iloc[:, :4].corr()
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)


st.markdown("---")
st.markdown("Made by Aakash Kumar Sinha 🌟")
