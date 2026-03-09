import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="MobileSphere Dashboard",
    page_icon="📱",
    layout="wide"
)

# -----------------------
# CUSTOM CSS
# -----------------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.header {
    background: linear-gradient(90deg,#667eea,#764ba2);
    padding: 25px;
    border-radius: 10px;
    color: white;
    text-align:center;
}

.metric-card {
    background:white;
    padding:20px;
    border-radius:10px;
    text-align:center;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
}

.footer {
    text-align:center;
    padding:20px;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("clean_mobile_data_fixed.csv")

# -----------------------
# LOAD MODEL
# -----------------------
model = pickle.load(open("mobile_price_prediction_model.pkl","rb"))

# -----------------------
# HEADER
# -----------------------
st.markdown("""
<div class="header">
<h1>📱 MobileSphere</h1>
<h3>Mobile Price Intelligence System</h3>
</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    ["Dashboard","Prediction","EDA","Model Performance","About"]
)

# -----------------------
# DASHBOARD
# -----------------------
if page == "Dashboard":

    st.subheader("Market Overview")

    total = len(df)
    avg_price = round(df["Price_Clean"].mean(),2)
    avg_ram = round(df["RAM_GB"].mean(),2)

    c1,c2,c3 = st.columns(3)

    c1.markdown(f"""
    <div class="metric-card">
    <h3>Total Mobiles</h3>
    <h1>{total}</h1>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-card">
    <h3>Average Price</h3>
    <h1>€ {avg_price}</h1>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="metric-card">
    <h3>Average RAM</h3>
    <h1>{avg_ram} GB</h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1,col2 = st.columns(2)

    fig1 = px.scatter(
        df,
        x="RAM_GB",
        y="Price_Clean",
        color="Brand",
        title="RAM vs Price"
    )

    col1.plotly_chart(fig1,use_container_width=True)

    fig2 = px.bar(
        df.groupby("Storage_GB")["Price_Clean"].mean().reset_index(),
        x="Storage_GB",
        y="Price_Clean",
        title="Average Price by Storage"
    )

    col2.plotly_chart(fig2,use_container_width=True)

# -----------------------
# PREDICTION
# -----------------------
elif page == "Prediction":

    st.subheader("Mobile Price Prediction")

    brands = sorted(df["Brand"].unique())

    col1,col2 = st.columns(2)

    brand = col1.selectbox("Brand",brands)
    ram = col1.slider("RAM (GB)",1,24,6)
    storage = col1.slider("Storage (GB)",8,512,128)

    battery = col2.slider("Battery (mAh)",2000,10000,4500)
    camera = col2.slider("Camera (MP)",5,200,50)

    brand_encoded = brands.index(brand)

    if st.button("Predict Price"):

        features = pd.DataFrame(
            [[brand_encoded,ram,storage,battery,camera]],
            columns=["Brand_Encoded","RAM_GB","Storage_GB","Battery_mAh","Main_Camera_MP"]
        )

        prediction = model.predict(features)[0]

        st.success(f"Estimated Mobile Price: € {round(prediction,2)}")

# -----------------------
# EDA
# -----------------------
elif page == "EDA":

    st.subheader("Exploratory Data Analysis")

    fig1 = px.histogram(df,x="Price_Clean",title="Price Distribution")
    st.plotly_chart(fig1,use_container_width=True)

    fig2 = px.box(df,x="RAM_GB",y="Price_Clean",title="RAM vs Price")
    st.plotly_chart(fig2,use_container_width=True)

    fig3 = px.scatter(df,x="Battery_mAh",y="Price_Clean",title="Battery vs Price")
    st.plotly_chart(fig3,use_container_width=True)

# -----------------------
# MODEL PERFORMANCE
# -----------------------
elif page == "Model Performance":

    st.subheader("Model Comparison")

    models = pd.DataFrame({
        "Model":[
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost"
        ],
        "R2 Score":[0.72,0.85,0.91,0.89,0.90],
        "MAE":[85,60,42,50,48],
        "RMSE":[120,90,70,80,75]
    })

    st.dataframe(models,use_container_width=True)

    st.success("Best Model: Random Forest")

# -----------------------
# ABOUT
# -----------------------
elif page == "About":

    st.subheader("About MobileSphere")

    st.write("""
MobileSphere is a Machine Learning project that predicts mobile phone prices 
based on specifications such as RAM, storage, battery capacity, camera, and brand.

The project demonstrates a full Data Science workflow including:

• Data Cleaning  
• Exploratory Data Analysis  
• Feature Engineering  
• Machine Learning Model Training  
• Model Comparison  
• Web Deployment with Streamlit
""")

    st.markdown("---")

    st.write("### Developer")

    st.write("""
Mahima Thakar  
CHARUSAT DEPSTAR  
MobileSphere Project
""")

# -----------------------
# FOOTER
# -----------------------
st.markdown("""
<div class="footer">
Developed by Mahima Thakar | CHARUSAT DEPSTAR
</div>
""", unsafe_allow_html=True)


