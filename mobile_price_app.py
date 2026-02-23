import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(
    page_title="MobileSphere Dashboard",
    page_icon="📱",
    layout="wide"
)

# =============================
# CUSTOM CSS
# =============================

st.markdown("""
<style>

.stApp {
    background-color: #f4f6fb;
}

section[data-testid="stSidebar"] {
    background-color: #1e293b;
}

section[data-testid="stSidebar"] * {
    color: white;
}

.kpi-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    text-align: center;
}

.footer {
    text-align: center;
    padding: 15px;
    color: gray;
}

</style>
""", unsafe_allow_html=True)

# =============================
# LOAD DATA AND MODEL
# =============================

df = pd.read_csv("clean_mobile_data_encoded.csv")

model = pickle.load(open("mobile_price_prediction_model.pkl", "rb"))

# =============================
# BRAND MAPPING
# =============================

brand_mapping = {
    "Apple":0,
    "Samsung":1,
    "Xiaomi":2,
    "Realme":3,
    "Oppo":4,
    "Vivo":5,
    "OnePlus":6,
    "Motorola":7,
    "Nokia":8,
    "Huawei":9
}

# =============================
# MODEL RESULTS
# =============================

model_results = pd.DataFrame({
    "Model":["Linear Regression","Decision Tree","Random Forest","Gradient Boosting","XGBoost"],
    "Accuracy":[0.78,0.85,0.92,0.89,0.91]
})

# =============================
# SIDEBAR
# =============================

st.sidebar.title("📱 MobileSphere Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard","Prediction","EDA","Model Performance","About"]
)

# =============================
# HEADER
# =============================

st.title("📱 MobileSphere Mobile Price Intelligence System")

# =============================
# DASHBOARD
# =============================

if page == "Dashboard":

    st.subheader("Mobile Market Insights")

    col1,col2,col3 = st.columns(3)

    col1.markdown(f"""
    <div class="kpi-card">
    <h3>Total Mobiles</h3>
    <h2>{len(df)}</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="kpi-card">
    <h3>Average Price (€)</h3>
    <h2>{round(df['Price_Clean'].mean(),2)}</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="kpi-card">
    <h3>Average RAM</h3>
    <h2>{round(df['RAM_GB'].mean(),2)} GB</h2>
    </div>
    """, unsafe_allow_html=True)

    col1,col2 = st.columns(2)

    fig1 = px.scatter(df,x="RAM_GB",y="Price_Clean",color="Price_Clean",title="RAM vs Price")
    col1.plotly_chart(fig1,use_container_width=True)

    fig2 = px.box(df,x="Storage_GB",y="Price_Clean",title="Storage vs Price")
    col2.plotly_chart(fig2,use_container_width=True)

# =============================
# PREDICTION
# =============================

elif page == "Prediction":

    st.subheader("Predict Mobile Price")

    brand_name = st.selectbox("Select Brand", list(brand_mapping.keys()))

    RAM = st.slider("RAM",1,32,8)
    Storage = st.slider("Storage",8,512,128)
    Battery = st.slider("Battery",1000,10000,5000)
    Camera = st.slider("Camera",5,200,48)

    if st.button("Predict Price"):

        brand_encoded = brand_mapping[brand_name]

        features = pd.DataFrame({
            "RAM_GB":[RAM],
            "Storage_GB":[Storage],
            "Battery_mAh":[Battery],
            "Main_Camera_MP":[Camera],
            "Brand_Encoded":[brand_encoded]
        })

        prediction = model.predict(features)[0]

        price_inr = prediction * 90

        st.success(f"Price: €{prediction:.2f}")
        st.success(f"Price: ₹{price_inr:.0f}")

# =============================
# EDA
# =============================

elif page == "EDA":

    st.subheader("Exploratory Data Analysis")

    fig = px.histogram(df,x="Price_Clean")
    st.plotly_chart(fig)

    fig2 = px.histogram(df,x="RAM_GB")
    st.plotly_chart(fig2)

# =============================
# MODEL PERFORMANCE
# =============================

elif page == "Model Performance":

    st.subheader("Model Accuracy")

    st.dataframe(model_results)

    fig = px.bar(model_results,x="Model",y="Accuracy")
    st.plotly_chart(fig)

# =============================
# ABOUT
# =============================

elif page == "About":

    st.subheader("About Project")

    st.write("""
MobileSphere predicts mobile prices using Machine Learning.

Models used:
• Linear Regression  
• Decision Tree  
• Random Forest  
• Gradient Boosting  
• XGBoost  

Best Model: Random Forest
""")

# =============================
# FOOTER
# =============================

st.markdown("""
<div class="footer">
Developed by MAHIMA THAKAR<br>
CHARUSAT DEPSTAR
</div>
""", unsafe_allow_html=True)
