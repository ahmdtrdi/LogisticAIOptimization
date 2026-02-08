import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import sys
import json

# Add root path to access src
sys.path.append(os.getcwd())

from src.pipelines import inference_pipeline
from src.pipelines.utils import load_config

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Logistics AI Optimization",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (Combined: Key Insights + Consultant Notes) ---
st.markdown("""
<style>
    /* 1. KEY INSIGHTS GRID (Dark Theme) */
    .insight-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-bottom: 20px;
    }
    .insight-card {
        background-color: #1E293B; /* Dark Blue/Grey */
        color: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #FF4B4B; /* Accent Color */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-title {
        font-size: 0.9em;
        font-weight: bold;
        color: #94A3B8; /* Muted text */
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .insight-value {
        font-size: 1.1em;
        font-weight: 600;
        line-height: 1.4;
    }
    .insight-highlight {
        color: #38BDF8; /* Light Blue highlight */
    }

    /* 2. BUSINESS IMPACT BOX */
    .impact-box {
        background-color: #0F172A;
        color: #E2E8F0;
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        border: 1px solid #334155;
    }
    .impact-list {
        margin-left: 20px;
    }
    .impact-list li {
        margin-bottom: 8px;
    }

    /* 3. CONSULTANT NOTE (Light Blue Box) - DIKEMBALIKAN */
    .consultant-note {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #00a8cc;
        margin-top: 20px;
        margin-bottom: 20px;
        font-size: 0.95em;
        color: #0f172a;
    }

    /* 4. BUTTON STYLING */
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        height: 50px;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHED FUNCTIONS ---
@st.cache_resource
def load_system_config():
    return load_config("config/local.yaml")

@st.cache_resource
def load_model_artifact(model_path):
    return joblib.load(model_path)

def load_model_metrics():
    summary_path = "models/run_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None

# --- MAIN APP ---
def main():
    # --- 1. SIDEBAR ---
    st.sidebar.image("https://img.icons8.com/color/96/delivery--v1.png", width=80)
    st.sidebar.title("Configuration")
    
    with st.sidebar.expander("ℹ️ Definitions", expanded=False):
        st.markdown("""
        **Cost Intervention:** Biaya preventif (Expedite).
        **Late Penalty:** Total kerugian (Denda + Reputasi).
        """)
    
    st.sidebar.subheader("💰 Params")
    cost_intervention = st.sidebar.number_input("Cost of Intervention ($)", value=15.0, step=1.0)
    cost_penalty = st.sidebar.number_input("Late Penalty Cost ($)", value=50.0, step=5.0)
    
    st.sidebar.subheader("🧠 Engine")
    model_dir = "models"
    available_models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')] if os.path.exists(model_dir) else []
    selected_model_file = st.sidebar.selectbox("Select Model", available_models)
    
    st.sidebar.subheader("📂 Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    
    cfg = load_system_config()

    # --- 2. HEADER ---
    st.title("🚚 Logistics Risk Predictor & Delivery Optimize")
    st.markdown("Real-time predictive analytics to minimize late delivery risks and maximize operational ROI.")
    st.divider()

    # --- 3. LOGIC ---
    df_raw = None
    
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file, encoding='latin-1')
        st.success("Custom Data Loaded!")
        
    else:
        demo_path = "data/01-raw/DataCoSupplyChainDataset.csv"
        
        if os.path.exists(demo_path):
            st.info("ℹ️ Using **Demo Data** (DataCo Supply Chain) for showcase purposes. Upload your own CSV to override.")
            try:
                df_raw = pd.read_csv(demo_path, encoding='latin-1')
            except Exception as e:
                st.error(f"Failed to load demo data: {e}")
        else:
            st.warning("👈 Silakan upload file CSV di sidebar untuk memulai.")
            
    if df_raw is not None:
        col_run, col_dummy = st.columns([1, 4])

def process_simulation(df_raw, cfg, model_file, cost_intervention, cost_penalty):
    with st.spinner('Calculating risks and generating insights...'):
        try:
            X_new = inference_pipeline.apply_fe_for_inference(df_raw.copy(), cfg)
            model_path = os.path.join("models", model_file)
            artifact = load_model_artifact(model_path)
            
            y_proba = None
            threshold = 0.5
            
            if 'weights' in artifact: # Ensemble
                threshold = artifact['threshold']
                weights = artifact['weights']
                models = artifact['models']
                weighted_sum, total_weight = 0, 0
                name_map = {'RandomForest': 'w_rf', 'XGBoost': 'w_xgb', 'CatBoost': 'w_cat', 'LogisticReg': 'w_lr'}
                for name, model in models.items():
                    key = name_map.get(name, f"w_{name}")
                    w = weights.get(key, 0)
                    if w > 0:
                        X_aligned = inference_pipeline.align_features(X_new.copy(), model)
                        weighted_sum += w * model.predict_proba(X_aligned)[:, 1]
                        total_weight += w
                y_proba = weighted_sum / (total_weight + 1e-10)
            elif 'model' in artifact: # Single
                threshold = artifact.get('threshold', 0.5)
                model = artifact['model']
                X_aligned = inference_pipeline.align_features(X_new.copy(), model)
                y_proba = model.predict_proba(X_aligned)[:, 1]

            if np.isscalar(y_proba): y_proba = np.full(len(X_new), y_proba)
            
            # --- ROI CALC ---
            loss_no_ai = np.sum(y_proba * cost_penalty)
            expected_loss_per_order = y_proba * cost_penalty
            should_intervene = expected_loss_per_order > cost_intervention
            
            intervention_cost_total = np.sum(should_intervene) * cost_intervention
            residual_risk_total = np.sum(expected_loss_per_order[~should_intervene])
            total_cost_ai = intervention_cost_total + residual_risk_total
            net_savings = loss_no_ai - total_cost_ai
            efficiency_gain = (net_savings / loss_no_ai * 100) if loss_no_ai > 0 else 0

            # --- DF RESULT ---
            df_res = df_raw.copy()
            df_res['risk_score'] = y_proba
            df_res['prediction'] = np.where(y_proba >= threshold, 'LATE', 'ON-TIME')
            conditions = [(y_proba >= 0.8) & should_intervene, (y_proba >= threshold) & should_intervene]
            choices = ['EXPEDITE (Air)', 'PRIORITY (Truck)']
            df_res['recommendation_type'] = np.select(conditions, choices, default='STANDARD (Ground)')
            df_res['action_flag'] = np.where(should_intervene, 'INTERVENE', 'IGNORE')
            df_res['potential_saving'] = np.where(should_intervene, expected_loss_per_order - cost_intervention, 0)
            if 'distance_km' in X_new.columns: df_res['distance_km'] = X_new['distance_km']

            # 1. Top Risk Shipping Mode
            mode_risk = "N/A"
            mode_risk_val = 0
            if 'Shipping Mode' in df_res.columns:
                mode_stats = df_res.groupby('Shipping Mode')['risk_score'].mean()
                mode_risk = mode_stats.idxmax()
                mode_risk_val = mode_stats.max()

            # 2. Top Risk Region
            region_risk = "N/A"
            if 'Order Region' in df_res.columns:
                region_stats = df_res.groupby('Order Region')['risk_score'].mean()
                region_risk = region_stats.idxmax()

            # 3. High Risk Volume
            high_risk_vol = np.sum(df_res['risk_score'] > 0.7)

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    tab1, tab2, tab3 = st.tabs(["Executive Insights", "Deep Dive Analysis", "Action Center"])

    with tab1:
        st.subheader("Key Insights 🔍")
        
        st.markdown(f"""
        <div class="insight-container">
            <div class="insight-card">
                <div class="insight-title">Highest Risk Mode</div>
                <div class="insight-value">{mode_risk}<br><span class="insight-highlight">{mode_risk_val:.1%} Avg Risk</span></div>
                <div style="font-size:0.8em; margin-top:5px;">Consider renegotiating SLA for this mode.</div>
            </div>
            <div class="insight-card">
                <div class="insight-title">Critical Region</div>
                <div class="insight-value">{region_risk}<br><span class="insight-highlight">Requires Attention</span></div>
                <div style="font-size:0.8em; margin-top:5px;">Highest concentration of potential delays.</div>
            </div>
            <div class="insight-card">
                <div class="insight-title">Cost Efficiency</div>
                <div class="insight-value">{efficiency_gain:.1f}%<br><span class="insight-highlight">Reduction</span></div>
                <div style="font-size:0.8em; margin-top:5px;">Driven by targeted intervention.</div>
            </div>
             <div class="insight-card">
                <div class="insight-title">Actionable Volume</div>
                <div class="insight-value">{high_risk_vol:,}<br><span class="insight-highlight">High Risk Orders</span></div>
                <div style="font-size:0.8em; margin-top:5px;">Immediate expedite recommended.</div>
            </div>
        </div>
        
        <div class="impact-box">
            <div style="font-size:1.2em; font-weight:bold; margin-bottom:10px;">Business Impact</div>
            <ul class="impact-list">
                <li><b>Net Savings:</b> Projected savings of <b>${net_savings:,.2f}</b> by preventing late penalties.</li>
                <li><b>Operational Agility:</b> Shifted <b>{np.sum(df_res['recommendation_type'] == 'EXPEDITE (Air)'):,} orders</b> to Air Freight to meet SLA.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        st.markdown("""<div class='consultant-note'>
        <b>💡 Executive Insight:</b><br>
        Grafik di bawah membandingkan skenario <b>'Do Nothing'</b> vs <b>'AI-Driven'</b>. 
        Gap antara batang merah dan hijau adalah <b>Net Savings</b> (uang yang diselamatkan) dengan menerapkan rekomendasi model.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Total Projected Loss (No AI)", f"${loss_no_ai:,.0f}")
            st.metric("Total Optimized Cost (With AI)", f"${total_cost_ai:,.0f}", delta=f"-${net_savings:,.0f}")
        
        with c2:
            cost_data = pd.DataFrame({'Scenario': ['No AI', 'With AI'], 'Cost': [loss_no_ai, total_cost_ai]})
            fig_cost = px.bar(cost_data, x='Cost', y='Scenario', orientation='h', color='Scenario', 
                             color_discrete_map={'No AI': '#EF553B', 'With AI': '#00CC96'}, title="Cost Comparison")
            st.plotly_chart(fig_cost, use_container_width=True)
            
        st.divider()

        st.subheader("Strategic Targets")
        
        st.markdown("""<div class='consultant-note'>
        <b>🔍 Root Cause Analysis:</b><br>
        Grafik ini menjawab <i>'Di mana kebocoran terbesar kita?'</i>. Gunakan data ini untuk negosiasi ulang dengan vendor logistik 
        pada <b>Kategori Produk</b> dan <b>Mode Pengiriman</b> yang paling berisiko.
        </div>""", unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        intervention_df = df_res[df_res['action_flag'] == 'INTERVENE']
        
        with cc1:
            if 'Category Name' in intervention_df.columns:
                cat_counts = intervention_df['Category Name'].value_counts().head(8).reset_index()
                cat_counts.columns = ['Category', 'Count']
                fig_cat = px.bar(cat_counts, x='Count', y='Category', orientation='h', title="Interventions by Category", color='Count', color_continuous_scale='Reds')
                fig_cat.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_cat, use_container_width=True)
                
        with cc2:
            if not intervention_df.empty:
                rec_counts = intervention_df['recommendation_type'].value_counts().reset_index()
                rec_counts.columns = ['Action', 'Count']
                fig_rec = px.pie(rec_counts, values='Count', names='Action', title="Recommended Actions", hole=0.5,
                                color_discrete_map={'EXPEDITE (Air)': '#FF4B4B', 'PRIORITY (Truck)': '#FFA500', 'STANDARD (Ground)': '#00CC96'})
                st.plotly_chart(fig_rec, use_container_width=True)

        st.markdown("""<div class='consultant-note'>
        <b>🛠️ Operational Strategy:</b><br>
        <ul>
        <li><b>EXPEDITE (Air):</b> Risiko > 80%. Wajib kirim kilat. Biaya tinggi tapi mencegah denda besar.</li>
        <li><b>PRIORITY (Truck):</b> Risiko Menengah. Prioritaskan loading, gunakan rute tol/cepat.</li>
        <li><b>STANDARD (Ground):</b> Risiko Rendah. Lanjutkan pengiriman reguler (Cost Saving).</li>
        </ul>
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.subheader("Hypothesis Validation")
        
        st.markdown("""<div class='consultant-note'>
        <b>🔬 Data Scientist Verification:</b><br>
        Bagian ini memvalidasi pola (pattern) yang dipelajari oleh model AI.
        Kita melihat korelasi antara metode pengiriman, jarak, dan geografi terhadap probabilitas keterlambatan.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1. Risk by Shipping Mode**")
            st.caption("Hypothesis Check: Apakah 'Standard Class' konsisten memiliki risiko tertinggi?")
            if 'Shipping Mode' in df_res.columns:
                risk_mode = df_res.groupby('Shipping Mode')['risk_score'].mean().reset_index()
                fig_mode = px.bar(risk_mode, x='Shipping Mode', y='risk_score', color='risk_score', color_continuous_scale='Reds')
                st.plotly_chart(fig_mode, use_container_width=True)
        with c2:
            st.markdown("**2. Distance vs. Risk Correlation**")
            st.caption("Hypothesis Check: Apakah pengiriman jarak jauh selalu lebih berisiko? (Scatter Plot)")
            if 'distance_km' in df_res.columns:
                fig_dist = px.scatter(df_res.head(1000), x='distance_km', y='risk_score', color='prediction', opacity=0.5, color_discrete_map={'LATE':'#FF4B4B', 'ON-TIME':'#00CC96'})
                st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")
        
        c3, c4 = st.columns(2)
        with c3:
             st.markdown("**3. Risk Score Distribution**")
             st.caption("Model Confidence: Seberapa tegas model memisahkan 'Late' (kanan) vs 'On-Time' (kiri)?")
             fig_hist = px.histogram(df_res, x='risk_score', color='prediction', nbins=50, title="Probability Density Function")
             fig_hist.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text="Threshold")
             st.plotly_chart(fig_hist, use_container_width=True)
             
        with c4:
            st.markdown("**4. Top High Risk Market Regions**")
            st.caption("Geographical Bottleneck: Wilayah mana yang memerlukan perhatian khusus tim logistik?")
            if 'Order Region' in df_res.columns:
                top_regions = df_res.groupby('Order Region')['risk_score'].mean().sort_values(ascending=False).head(10)
                fig_reg = px.bar(top_regions, orientation='h', title="Top 10 Riskiest Regions")
                st.plotly_chart(fig_reg, use_container_width=True)
        
        st.markdown("### 🌍 Global Risk Heatmap")
        st.caption("Visualisasi persebaran titik risiko tinggi secara global.")
        
        if 'Latitude' in df_res.columns and 'Longitude' in df_res.columns:
            try:
                map_data = df_res.copy()
                map_data['Latitude'] = pd.to_numeric(map_data['Latitude'], errors='coerce')
                map_data['Longitude'] = pd.to_numeric(map_data['Longitude'], errors='coerce')
                map_data = map_data.dropna(subset=['Latitude', 'Longitude'])
                
                map_data = map_data[(map_data['Latitude'] >= -90) & (map_data['Latitude'] <= 90)]
                map_data = map_data[(map_data['Longitude'] >= -180) & (map_data['Longitude'] <= 180)]

                if not map_data.empty:
                    center_lat = map_data['Latitude'].mean()
                    center_lon = map_data['Longitude'].mean()
                    
                    if len(map_data) > 1000: map_data = map_data.sample(500)
                    
                    fig_map = px.density_mapbox(
                        map_data, lat='Latitude', lon='Longitude', z='risk_score',
                        radius=15,
                        center=dict(lat=center_lat, lon=center_lon), # DYNAMIC CENTER
                        zoom=2,
                        mapbox_style="carto-positron",
                        height=500,
                        color_continuous_scale="RdBu_r"
                    )
                    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.warning("No valid coordinate data found after cleaning.")
            except Exception as e:
                st.error(f"Map Error: {e}")
        else:
            st.info("Latitude/Longitude columns missing.")
            
    with tab3:
        st.subheader("Execution List")
        priority_df = df_res[df_res['action_flag'] == 'INTERVENE'].sort_values(by='risk_score', ascending=False)
        
        valid_cols = [c for c in ['Order Id', 'risk_score', 'recommendation_type', 'potential_saving', 'Shipping Mode'] if c in priority_df.columns]
        
        st.dataframe(priority_df[valid_cols].head(500).style.format({'risk_score': '{:.1%}', 'potential_saving': '${:.2f}'}).background_gradient(subset=['risk_score'], cmap='Reds'), use_container_width=True)
        
        st.download_button("📥 Download Action Plan", priority_df.to_csv(index=False).encode('utf-8'), "action_plan.csv", "text/csv", type="primary")

if __name__ == "__main__":
    main()