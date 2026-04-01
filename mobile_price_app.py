import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="MobileSphere",
    page_icon="📱",
    layout="wide"
)

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("mobile_dataset_cleaned.csv")
model = pickle.load(open("mobile_price_prediction_model.pkl", "rb"))

EUR_TO_INR = 90

# ---------------- BRAND VALUE MAP ---------------- #
brand_value = {
    "Apple": 5.0,
    "Samsung": 3.5,
    "OnePlus": 3.0,
    "Google": 3.0,
    "Nothing": 2.8,

    "Xiaomi": 1.2,
    "Realme": 1.1,
    "Redmi": 1.0,
    "Poco": 1.0,
    "Vivo": 1.3,
    "Oppo": 1.3,

    "Motorola": 1.6,
    "Nokia": 1.5,
    "Sony": 2.5,
    "Asus": 1.8,
    "Acer": 1.2,

    "Infinix": 0.9,
    "Tecno": 0.9,
    "Lava": 0.8,
    "Micromax": 0.8
}

budget_brands = ["Redmi", "Poco", "Realme", "Infinix", "Tecno", "Lava", "Micromax"]
mid_brands = ["Xiaomi", "Vivo", "Oppo", "Motorola", "Nokia", "Acer", "Asus"]
premium_brands = ["Samsung", "OnePlus", "Apple", "Google", "Nothing", "Sony"]

# ---------------- HELPERS ---------------- #
def prepare_input_data(brand, ram, storage, battery, camera):
    input_data = pd.DataFrame({
        "RAM_GB": [ram],
        "Storage_GB": [storage],
        "Battery_mAh": [battery],
        "Main_Camera_MP": [camera],
        "Brand_Value": [brand_value.get(brand, 1.2)]
    })

    # Same feature engineering as notebook
    input_data["RAM_Storage"] = input_data["RAM_GB"] * input_data["Storage_GB"]

    return input_data


def calibrate_prediction_inr(prediction_inr, brand):
    # Post-prediction calibration for more realistic INR output
    if brand in budget_brands:
        prediction_inr *= 0.40
    elif brand in mid_brands:
        prediction_inr *= 0.60
    elif brand in premium_brands:
        prediction_inr *= 1.00
    else:
        prediction_inr *= 0.65

    return prediction_inr


def predict_price_euro(brand, ram, storage, battery, camera):
    input_data = prepare_input_data(brand, ram, storage, battery, camera)

    prediction_log = model.predict(input_data)[0]

    prediction_euro = round(np.exp(prediction_log), 2)

    return prediction_euro


def price_label(price_inr):
    if price_inr < 15000:
        return "Budget Range"
    elif price_inr < 30000:
        return "Mid-Range"
    elif price_inr < 60000:
        return "Upper Mid-Range"
    else:
        return "Premium Range"


# ---------------- UI STYLING ---------------- #
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(120deg, #eef2ff, #f8fafc);
        font-family: 'Segoe UI', sans-serif;
    }

    .main-title-box {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.3);
    }

    .section-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin-bottom: 18px;
    }

    .upload-box {
        border: 2px dashed #c7d2fe;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        background: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)
# ---------------- BRAND LIST ---------------- #
brands = sorted(df["Brand"].dropna().unique().tolist())

# ---------------- SIDEBAR ---------------- #
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction", "EDA", "Bulk Scanner", "About"]
)

# ---------------- DASHBOARD ---------------- #
if page == "Dashboard":
    st.markdown("""
    <div class="main-title-box">
        <h1 style="margin-bottom:8px;">📱 MobileSphere</h1>
        <h3 style="margin-top:0; font-weight:500;">AI-Based Mobile Price Prediction System</h3>
    </div>
    """, unsafe_allow_html=True)

    total = len(df)
    avg_price_eur = df["Price_Clean"].mean()
    avg_price_inr = avg_price_eur * EUR_TO_INR
    avg_ram = df["RAM_GB"].mean()
    avg_storage = df["Storage_GB"].mean()


    
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📱 Total Mobiles</div>
            <div class="metric-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💶 Avg Price</div>
            <div class="metric-value">€{round(avg_price_eur,2)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚡ Avg RAM</div>
            <div class="metric-value">{round(avg_ram,1)} GB</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💾 Avg Storage</div>
            <div class="metric-value">{round(avg_storage,1)} GB</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- PREDICTION ---------------- #
elif page == "Prediction":
    st.markdown("""
<div class="main-title-box">
    <h1>💰 Smart Price Prediction</h1>
    <p>AI-powered estimation based on specifications</p>
</div>
""", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        brand = st.selectbox("🏬 Brand", brands)

        ram = st.selectbox(
            "⚡ RAM (GB)",
            [2, 3, 4, 6, 8, 12, 16],
            index=2
        )

        storage = st.selectbox(
            "💾 Storage (GB)",
            [32, 64, 128, 256, 512],
            index=1
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        battery = st.slider("🔋 Battery (mAh)", 2000, 7000, 5000, step=100)
        camera = st.slider("📸 Camera (MP)", 5, 108, 50, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Predict Price"):
        predicted_price = predict_price_euro(brand, ram, storage, battery, camera)
        category = price_label(predicted_price * EUR_TO_INR)

        st.markdown(f"""
        <div class="result-box">
            <div class="result-price">Estimated Price: €{predicted_price:.2f}</div>
            <div class="result-sub">{category}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="small-note">This value is based on model prediction combined with brand-based market calibration for realistic pricing.</div>',
            unsafe_allow_html=True
        )

# ---------------- EDA ---------------- #
elif page == "EDA":
    st.markdown("""
    <div class="main-title-box">
        <h1 style="margin-bottom:8px;">📈 Exploratory Data Analysis</h1>
        <h3 style="margin-top:0; font-weight:500;">Visual insights from the mobile dataset</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📉 Price Distribution")
        fig1 = px.histogram(df, x="Price_Clean", nbins=30)
        fig1.update_layout(
            xaxis_title="Price (€)",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🔋 Battery vs Price")
        fig2 = px.scatter(
            df,
            x="Battery_mAh",
            y="Price_Clean",
            color="RAM_GB",
            hover_data=["Brand", "Storage_GB", "Main_Camera_MP"]
        )
        fig2.update_layout(
            xaxis_title="Battery (mAh)",
            yaxis_title="Price (€)",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📸 Camera vs Price")
        fig3 = px.scatter(
            df,
            x="Main_Camera_MP",
            y="Price_Clean",
            color="Brand"
        )
        fig3.update_layout(
            xaxis_title="Camera (MP)",
            yaxis_title="Price (€)",
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🏷️ Average Price by Brand")
        avg_brand_df = df.groupby("Brand", as_index=False)["Price_Clean"].mean().sort_values("Price_Clean", ascending=False)
        fig4 = px.bar(
            avg_brand_df,
            x="Brand",
            y="Price_Clean"
        )
        fig4.update_layout(
            xaxis_title="Brand",
            yaxis_title="Average Price (€)",
            height=400
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- BULK SCANNER ---------------- #
elif page == "Bulk Scanner":

    st.markdown("""
    <div class="main-title-box">
        <h1>🔎 Bulk Premium Scanner</h1>
    </div>
    """, unsafe_allow_html=True)

    # STEP 1
    st.markdown("## 1. Download Sample Templates")

    sample_df = pd.DataFrame({
        "Brand": ["Xiaomi", "Samsung"],
        "RAM_GB": [4, 8],
        "Storage_GB": [64, 128],
        "Battery_mAh": [5000, 4500],
        "Main_Camera_MP": [50, 64]
    })

    col1, col2, col3 = st.columns(3)

    csv = sample_df.to_csv(index=False).encode("utf-8")
    col1.download_button("📄 Download CSV Sample", csv, "sample.csv")

    excel_file = "sample.xlsx"
    sample_df.to_excel(excel_file, index=False)
    with open(excel_file, "rb") as f:
        col2.download_button("📊 Download Excel Sample", f, "sample.xlsx")

    json_data = sample_df.to_json(orient="records")
    col3.download_button("📦 Download JSON Sample", json_data, "sample.json")

    st.markdown("---")

    # STEP 2
    st.markdown("## 2. Upload File to Scan")

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a file (CSV, Excel, JSON)",
        type=["csv", "xlsx", "json"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            bulk_df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith(".xlsx"):
            bulk_df = pd.read_excel(uploaded_file)

        elif uploaded_file.name.endswith(".json"):
            bulk_df = pd.read_json(uploaded_file)

        else:
            st.error("Unsupported file format")
            st.stop()

        st.subheader("Preview")
        st.dataframe(bulk_df.head(), use_container_width=True)

        required_cols = ["Brand", "RAM_GB", "Storage_GB", "Battery_mAh", "Main_Camera_MP"]
        missing_cols = [col for col in required_cols if col not in bulk_df.columns]

        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            if st.button("🚀 Run Bulk Prediction"):

                result_df = bulk_df.copy()

                def bulk_predict(row):
                    return predict_price_euro(
                        row["Brand"],
                        row["RAM_GB"],
                        row["Storage_GB"],
                        row["Battery_mAh"],
                        row["Main_Camera_MP"]
                    )

                result_df["Predicted_Price_EUR"] = result_df.apply(bulk_predict, axis=1)
                result_df["Price_Category"] = (result_df["Predicted_Price_EUR"] * EUR_TO_INR).apply(price_label)

                st.subheader("Results")
                st.dataframe(result_df, use_container_width=True)

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results", csv, "bulk_predictions.csv")
# ---------------- ABOUT ---------------- #
elif page == "About":

    st.markdown("""
    <div class="main-title-box">
        <h1 style="margin-bottom:8px;">📘 About MobileSphere</h1>
        <h3 style="margin-top:0; font-weight:500;">
        AI-Based Mobile Price Prediction System
        </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <p style="font-size:16px; line-height:1.6;">
        <b>MobileSphere</b> is an AI-based system developed to estimate realistic smartphone prices 
        using machine learning techniques. The system analyzes key specifications such as 
        <b>RAM, Storage, Battery, Camera, and Brand value</b> to predict accurate market prices.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # -------- KEY FEATURES -------- #
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>🎯 Key Features</h3>
            <ul style="line-height:1.8;">
                <li>Realistic price prediction in Indian Rupees</li>
                <li>Brand-aware price adjustment for accuracy</li>
                <li>Interactive dashboard with visual analytics</li>
                <li>Bulk scanner for multiple mobile predictions</li>
                <li>Clean and professional user interface</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # -------- TECHNOLOGIES -------- #
    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>🛠 Technologies Used</h3>
            <ul style="line-height:1.8;">
                <li>Python</li>
                <li>Pandas & NumPy</li>
                <li>XGBoost (Machine Learning Model)</li>
                <li>Streamlit (Web Application)</li>
                <li>Plotly (Data Visualization)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
