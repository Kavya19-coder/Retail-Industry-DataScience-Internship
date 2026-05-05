import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Retail Customer Sales Prediction")
st.write("Capstone Project by Kavya19-coder")

@st.cache_data
def load_data():
    return pd.read_csv('customer_shopping_data.csv')

df = load_data()
st.write("Available columns:", df.columns.tolist()) # This will show columns on website

# === CHANGE THIS LINE WITH YOUR REAL FEATURE COLUMNS ===
X = df[['Age', 'Gender', 'Category']] # <-- CHANGE THIS
# === CHANGE THIS LINE WITH YOUR REAL TARGET COLUMN ===
y = df['Price'] # <-- CHANGE THIS

X = pd.get_dummies(X)
model = LinearRegression()
model.fit(X, y)

st.sidebar.header("Predict Sales")
age = st.sidebar.slider("Customer Age", 18, 70, 30)

if st.sidebar.button("Predict"):
    st.success("Model trained! Now update the input fields above to match your columns")
