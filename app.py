import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Retail Customer Sales Prediction")
st.write("Capstone Project by Kavya19-coder")

@st.cache_data
def load_data():
    return pd.read_csv('customer_shopping_data.csv')

df = load_data()

# === CHANGE THESE 2 LINES TO YOUR ACTUAL COLUMNS ===
X = df[['age', 'quantity', 'price']] # Your input features
y = df['total_amount'] # Your target column
# ===================================================

X = pd.get_dummies(X)
model = LinearRegression()
model.fit(X, y)

st.sidebar.header("Predict Sales")
age = st.sidebar.slider("Customer Age", 18, 70, 30)
quantity = st.sidebar.number_input("Quantity", 1, 20, 2)
price = st.sidebar.number_input("Price ₹", 100, 5000, 500)

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([[age, quantity, price]], 
                           columns=['age', 'quantity', 'price'])
    input_df = pd.get_dummies(input_df)
    
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]
    
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Sales: ₹{pred:,.2f}")
