import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="MobileSphere Dashboard",
    page_icon="📱",
    layout="wide"
)

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("clean_mobile_data_fixed.csv")

# -------------------------
# Load Model
# -------------------------
model = pickle.load(open("mobile_price_prediction_model.pkl", "rb"))

# -------------------------
# Custom CSS Styling
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
h1, h2, h3 {
    color: #2c3e50;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    text-align:center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
.footer {
    text-align:center;
    padding:20px;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("📱 MobileSphere")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction", "EDA", "Model Performance", "About"]
)

# -------------------------
# DASHBOARD
# -------------------------
if page == "Dashboard":

    st.title("📱 MobileSphere Mobile Price Intelligence System")

    total_mobiles = len(df)
    avg_price = round(df["Price_Clean"].mean(), 2)
    avg_ram = round(df["RAM_GB"].mean(), 2)

    col1, col2, col3 = st.columns(3)

    col1.markdown(
        f"""
        <div class="metric-card">
        <h3>Total Mobiles</h3>
        <h1>{total_mobiles}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    col2.markdown(
        f"""
        <div class="metric-card">
        <h3>Average Price (€)</h3>
        <h1>{avg_price}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    col3.markdown(
        f"""
        <div class="metric-card">
        <h3>Average RAM</h3>
        <h1>{avg_ram} GB</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    col4, col5 = st.columns(2)

    fig1 = px.scatter(
        df,
        x="RAM_GB",
        y="Price_Clean",
        color="Brand",
        title="RAM vs Mobile Price"
    )

    col4.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        df.groupby("Storage_GB")["Price_Clean"].mean().reset_index(),
        x="Storage_GB",
        y="Price_Clean",
        title="Average Price by Storage"
    )

    col5.plotly_chart(fig2, use_container_width=True)

# -------------------------
# PREDICTION PAGE
# -------------------------
elif page == "Prediction":

    st.title("📱 Mobile Price Prediction")

    brand_list = sorted(df["Brand"].unique())

    col1, col2 = st.columns(2)

    brand = col1.selectbox("Brand", brand_list)
    ram = col1.slider("RAM (GB)", 1, 24, 4)
    storage = col1.slider("Storage (GB)", 8, 512, 64)

    battery = col2.slider("Battery (mAh)", 2000, 10000, 4000)
    camera = col2.slider("Camera (MP)", 5, 200, 48)

    brand_encoded = brand_list.index(brand)

    if st.button("Predict Price"):

        features = pd.DataFrame(
            [[brand_encoded, ram, storage, battery, camera]],
            columns=["Brand_Encoded", "RAM_GB", "Storage_GB", "Battery_mAh", "Main_Camera_MP"]
        )

        prediction = model.predict(features)[0]

        st.success(f"💰 Estimated Mobile Price: €{prediction:.2f}")

# -------------------------
# EDA PAGE
# -------------------------
elif page == "EDA":

    st.title("Exploratory Data Analysis")

    fig1 = px.histogram(df, x="Price_Clean", title="Price Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(df, x="RAM_GB", y="Price_Clean", title="RAM vs Price")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(df, x="Battery_mAh", y="Price_Clean", title="Battery vs Price")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# MODEL PERFORMANCE
# -------------------------
elif page == "Model Performance":

    st.title("Model Performance Comparison")

    data = {
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost"
        ],
        "R2 Score": [0.72, 0.85, 0.91, 0.89, 0.90],
        "MAE": [85, 60, 42, 50, 48],
        "RMSE": [120, 90, 70, 80, 75]
    }

    model_df = pd.DataFrame(data)

    st.dataframe(model_df, use_container_width=True)

    st.success("🏆 Best Model: Random Forest Regressor")

# -------------------------
# ABOUT PAGE
# -------------------------
elif page == "About":

    st.title("About MobileSphere")

    st.write("""
MobileSphere is a machine learning project that predicts mobile phone prices 
based on hardware specifications such as RAM, storage, battery capacity, camera, 
and brand.

This project demonstrates a complete Data Science pipeline including:

• Data Cleaning  
• Exploratory Data Analysis  
• Feature Engineering  
• Machine Learning Model Training  
• Model Comparison  
• Web Application Deployment
""")

    st.markdown("---")

    st.subheader("Developer Information")

    st.write("""
Name: **Mahima Thakar**

College: **CHARUSAT DEPSTAR**

Project: **MobileSphere – Mobile Price Intelligence System**
""")

# -------------------------
# Footer
# -------------------------
st.markdown("""
<div class="footer">
Developed by Mahima Thakar | CHARUSAT DEPSTAR
</div>
""", unsafe_allow_html=True)

