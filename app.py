import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Retail Sales Prediction")
st.title("Retail Customer Sales Prediction")
st.write("Capstone Project by Kavya19-coder")

@st.cache_data
def load_data():
    df = pd.read_csv('customer_shopping_data.csv')
    df['total_sales'] = df['price'] * df['quantity']
    return df

df = load_data()

# Using lowercase column names from your CSV
X = df[['age', 'gender', 'category', 'quantity', 'payment_method', 'shopping_mall']]
y = df['total_sales']

X = pd.get_dummies(X, drop_first=True)
model = LinearRegression()
model.fit(X, y)

st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Customer Age", 18, 70, 30)
gender = st.sidebar.selectbox("Gender", df['gender'].unique())
category = st.sidebar.selectbox("Product Category", df['category'].unique())
quantity = st.sidebar.number_input("Quantity", 1, 20, 2)
payment_method = st.sidebar.selectbox("Payment Method", df['payment_method'].unique())
shopping_mall = st.sidebar.selectbox("Shopping Mall", df['shopping_mall'].unique())

if st.sidebar.button("Predict Total Sales"):
    input_data = [[age, gender, category, quantity, payment_method, shopping_mall]]
    input_df = pd.DataFrame(input_data, columns=['age', 'gender', 'category', 'quantity', 'payment_method', 'shopping_mall'])
    input_df = pd.get_dummies(input_df)
    
    # Match training columns
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]
    
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Total Sales: ₹{prediction:,.2f}")
    st.balloons()
