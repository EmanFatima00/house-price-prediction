"""
Pakistan House Price Prediction App
Author: Eman Fatima | BS-AI @ PAF-IAST
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

st.set_page_config(page_title="Pakistan House Price Predictor",
                   page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #0f3460, #16213e, #1a1a2e);
        padding: 2rem; border-radius: 12px; text-align: center;
        color: white; margin-bottom: 2rem; }
    .main-header h1 { font-size: 2.1rem; margin: 0; }
    .main-header p  { font-size: 0.95rem; opacity: 0.85; margin: 0.4rem 0 0; }
    .metric-card { background: white; border-radius: 10px; padding: 1.1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
        border-left: 4px solid #6366F1; }
    .metric-card h3 { font-size: 1.7rem; color: #6366F1; margin: 0; }
    .metric-card p  { color: #64748B; margin: 0; font-size: 0.88rem; }
    .result-box { background: linear-gradient(135deg, #EEF2FF, #E0E7FF);
        border: 2px solid #6366F1; border-radius: 12px;
        padding: 2rem; text-align: center; margin: 1rem 0; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all():
    model   = joblib.load(os.path.join(OUTPUT_DIR, "house_model.pkl"))
    scaler  = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
    le_city = joblib.load(os.path.join(OUTPUT_DIR, "le_city.pkl"))
    le_ptype= joblib.load(os.path.join(OUTPUT_DIR, "le_ptype.pkl"))
    le_prov = joblib.load(os.path.join(OUTPUT_DIR, "le_prov.pkl"))
    meta    = json.load(open(os.path.join(OUTPUT_DIR, "meta.json")))
    return model, scaler, le_city, le_ptype, le_prov, meta

model, scaler, le_city, le_ptype, le_prov, meta = load_all()

# city -> province mapping from data
CITY_PROVINCE = {
    'Karachi': 'Sindh', 'Lahore': 'Punjab',
    'Islamabad': 'Islamabad Capital', 'Rawalpindi': 'Punjab',
    'Faisalabad': 'Punjab',
}
CITY_COORDS = {
    'Karachi':    (24.8607, 67.0011),
    'Lahore':     (31.5204, 74.3587),
    'Islamabad':  (33.6844, 73.0479),
    'Rawalpindi': (33.5651, 73.0169),
    'Faisalabad': (31.4504, 73.1350),
}

st.markdown("""
<div class="main-header">
    <h1>🏠 Pakistan House Price Predictor</h1>
    <p>XGBoost regression model trained on 50,000 Zameen.com listings | R²: 0.8785</p>
</div>""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric-card"><h3>{meta["r2"]}</h3><p>R² Score</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><h3>PKR {meta["mae"]/1e6:.1f}M</h3><p>Mean Absolute Error</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><h3>{meta["total_listings"]:,}</h3><p>Listings Trained</p></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card"><h3>4</h3><p>Models Compared</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Predict Price", "Model Insights", "About"])

with tab1:
    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.markdown("### Property Details")
        city         = st.selectbox("City", meta['cities'])
        property_type= st.selectbox("Property Type", meta['property_types'])
        purpose      = st.selectbox("Purpose", ["For Sale", "For Rent"])
        area_unit = st.selectbox("Area Unit", ["Marla", "Kanal"])
        area_input = st.number_input(f"Area Size ({area_unit})", min_value=0.5, max_value=500.0, value=10.0, step=0.5)
        area_size = area_input * 20 if area_unit == "Kanal" else area_input
        st.caption(f"= {area_size:.1f} Marla" if area_unit == "Kanal" else "")
        bedrooms     = st.slider("Bedrooms", 0, 10, 3)
        baths        = st.slider("Bathrooms", 0, 10, 2)
        house_age    = st.slider("Estimated House Age (years)", 0, 50, 5)

        predict_btn = st.button("Predict Price", type="primary", use_container_width=True)

    with col_out:
        st.markdown("### Prediction")
        if predict_btn:
            province = CITY_PROVINCE.get(city, meta['provinces'][0])
            lat, lon = CITY_COORDS.get(city, (30.0, 70.0))

            # Encode
            try: city_enc = le_city.transform([city])[0]
            except: city_enc = 0
            try: ptype_enc = le_ptype.transform([property_type])[0]
            except: ptype_enc = 0
            try: prov_enc = le_prov.transform([province])[0]
            except: prov_enc = 0

            log_area    = np.log1p(area_size)
            total_rooms = bedrooms + baths
            baths_per_bed = baths / (bedrooms + 1)
            is_for_sale = 1 if purpose == "For Sale" else 0

            input_df = pd.DataFrame([[
                area_size, log_area, bedrooms, baths, total_rooms,
                baths_per_bed, house_age, is_for_sale,
                lat, lon, city_enc, ptype_enc, prov_enc
            ]], columns=['Area Size','log_area','bedrooms','baths','total_rooms',
                         'baths_per_bed','house_age','is_for_sale',
                         'latitude','longitude','city_enc','ptype_enc','province_enc'])

            pred_log   = model.predict(input_df)[0]
            pred_price = np.expm1(pred_log)
            low_est    = pred_price * 0.85
            high_est   = pred_price * 1.15

            st.markdown(f"""
            <div class="result-box">
                <p style="color:#4338CA;font-size:1rem;margin-bottom:4px;">Estimated Price</p>
                <h1 style="color:#6366F1;font-size:2.8rem;margin:0;">
                    PKR {pred_price/1e6:.2f}M
                </h1>
                <p style="color:#64748B;margin-top:6px;">
                    Range: PKR {low_est/1e6:.2f}M — PKR {high_est/1e6:.2f}M
                </p>
            </div>""", unsafe_allow_html=True)

            # Price range bar
            fig_r, ax_r = plt.subplots(figsize=(5, 1.5))
            ax_r.barh([''], [high_est/1e6 - low_est/1e6], left=[low_est/1e6],
                      color='#6366F1', height=0.3, alpha=0.7)
            ax_r.axvline(pred_price/1e6, color='#DC2626', linewidth=2, linestyle='--', label='Estimate')
            ax_r.set_xlabel("PKR Millions"); ax_r.set_title("Price Range", fontweight='bold', fontsize=10)
            ax_r.legend(fontsize=8); plt.tight_layout()
            st.pyplot(fig_r); plt.close()

            st.markdown("#### Property Summary")
            st.markdown(f"- **City:** {city} | **Type:** {property_type}")
            display_area = f"{area_input} {area_unit} ({area_size:.1f} Marla)" if area_unit == "Kanal" else f"{area_size} Marla"
            st.markdown(f"- **Area:** {display_area} | **Beds:** {bedrooms} | **Baths:** {baths}")
            st.markdown(f"- **House Age:** {house_age} years | **Purpose:** {purpose}")
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem;color:#64748B;">
                <h3>Fill in property details on the left</h3>
                <p>Enter the details and click Predict Price</p>
            </div>""", unsafe_allow_html=True)

with tab2:
    st.markdown("### Model Performance & Analysis")
    c1, c2 = st.columns(2)
    with c1:
        for img, cap in [("model_comparison.png","Model Comparison"),
                          ("actual_vs_predicted.png","Actual vs Predicted")]:
            p = os.path.join(OUTPUT_DIR, img)
            if os.path.exists(p): st.image(p, caption=cap, use_container_width=True)
    with c2:
        for img, cap in [("feature_importance.png","Feature Importance — XGBoost"),
                          ("price_by_city.png","Price Distribution by City")]:
            p = os.path.join(OUTPUT_DIR, img)
            if os.path.exists(p): st.image(p, caption=cap, use_container_width=True)
    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        p = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
        if os.path.exists(p): st.image(p, caption="Correlation Heatmap", use_container_width=True)
    with c4:
        p = os.path.join(OUTPUT_DIR, "price_by_type.png")
        if os.path.exists(p): st.image(p, caption="Median Price by Property Type", use_container_width=True)

    st.markdown("---")
    st.markdown("### Results Summary")
    perf = {
        "Model":    ["Linear Regression","Ridge Regression","Random Forest","XGBoost"],
        "R² Score": ["-153.17","−136.92","0.8648","0.8785"],
        "MAE (PKR)":["7,028,578","6,937,257","2,661,320","2,496,729"],
        "MAPE (%)": ["3656.29%","3461.80%","48.66%","44.57%"],
    }
    st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)
    st.info("Linear and Ridge regression performed poorly — confirming that house prices follow a non-linear pattern that tree-based models handle better.")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### About This Project

        Regression model trained on 50,000 Zameen.com property listings
        from Pakistan's top 5 cities.

        **Pipeline:**
        - 4 ML models trained and compared
        - XGBoost selected as final model
        - 7 engineered features
        - Log-transformation of target variable (price)
        - Cities: Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad

        **Dataset:** Zameen.com Pakistani Real Estate (168K+ original listings)
        """)
    with c2:
        st.markdown("""
        ### Developer

        **Eman Fatima**
        BS Artificial Intelligence — Semester 6
        PAF Institute of Applied Sciences & Technology, Pakistan

        Tech Stack: Python · XGBoost · Scikit-learn · Streamlit · Pandas

        [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/eman-fatima-99962230b)
        [![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/EmanFatima00)
        """)