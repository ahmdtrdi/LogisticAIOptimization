import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import sys

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

# --- CACHED FUNCTIONS---
@st.cache_resource
def load_system_config():
    return load_config("config/local.yaml")

@st.cache_resource
def load_model_artifact(model_path):
    return joblib.load(model_path)

# --- MAIN APP ---
def main():
    # SIDEBAR: Control Panel
    st.sidebar.title("⚙️ Simulation Params")
    
    # Cost Parameters (Interactive ROI)
    st.sidebar.subheader("💰 Cost Assumptions")
    cost_intervention = st.sidebar.number_input("Cost of Intervention ($)", value=15.0, step=1.0)
    cost_penalty = st.sidebar.number_input("Late Penalty Cost ($)", value=50.0, step=5.0)
    
    # Model Selection
    st.sidebar.subheader("🧠 AI Model")
    model_dir = "models"
    available_models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')] if os.path.exists(model_dir) else []
    selected_model_file = st.sidebar.selectbox("Select Model", available_models, index=0 if available_models else None)
    
    # Data Input
    st.sidebar.subheader("📂 Input Data")
    uploaded_file = st.sidebar.file_uploader("Upload New Orders (CSV)", type=['csv'])
    
    # Load Config & Model
    cfg = load_system_config()
    
    # --- HERO SECTION ---
    st.title("🚚 AI-Driven Logistic Risk Predictor & Supply Chain Optimization")
    st.markdown("""
    **Predict Late Deliveries before they happen.** This system uses Machine Learning to calculate risk scores and recommends optimal intervention strategies to maximize profit.
    """)
    st.divider()

    if uploaded_file is None:
        st.info("Please upload a CSV file in the sidebar to start the simulation. (Using demo data if available)")
        # Demo Data Fallback
        demo_path = "data/01-raw/DataCoSupplyChainDataset.csv"
        if os.path.exists(demo_path):
            if st.button("Load Demo Data"):
                uploaded_file = demo_path
            else:
                st.stop()
        else:
            st.stop()
            
    # --- PROCESSING ENGINE ---
    if uploaded_file:
        with st.spinner('AI is analyzing order risks...'):
            # Load Data
            if isinstance(uploaded_file, str):
                df_raw = pd.read_csv(uploaded_file, encoding='latin-1')
            else:
                df_raw = pd.read_csv(uploaded_file, encoding='latin-1')
            
            # Load Model
            model_path = os.path.join(model_dir, selected_model_file)
            artifact = load_model_artifact(model_path)
            
            # --- RUN PIPELINE LOGIC MANUALLY (To Inject Custom Cost) ---
            # 1. Feature Engineering
            X_new = inference_pipeline.apply_fe_for_inference(df_raw.copy(), cfg)
            
            # 2. Predict Probability
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

            # 3. Business Logic (Dynamic based on Sidebar)
            if np.isscalar(y_proba): y_proba = np.full(len(X_new), y_proba)
            
            expected_loss = y_proba * cost_penalty
            should_intervene = expected_loss > cost_intervention
            
            # Create Result DF
            df_res = df_raw.copy()
            df_res['Risk Score'] = y_proba
            df_res['Prediction'] = np.where(y_proba >= threshold, 'LATE', 'ON-TIME')
            df_res['Action'] = np.where(should_intervene, 'INTERVENE (Expedite)', 'IGNORE (Standard)')
            df_res['Net Savings'] = np.where(should_intervene, expected_loss - cost_intervention, 0)
            
            # Metrics
            total_orders = len(df_res)
            total_intervention = np.sum(should_intervene)
            total_savings = df_res['Net Savings'].sum()
            high_risk_count = np.sum(df_res['Prediction'] == 'LATE')

        # --- DASHBOARD UI ---
        
        # TAB STRUCTURE
        tab1, tab2, tab3 = st.tabs(["📊 Executive Overview", "🧠 Intelligence Engine", "✅ Action Center"])
        
        with tab1:
            st.subheader("Business Impact Simulation")
            
            # Big Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Orders", f"{total_orders:,}")
            col2.metric("High Risk Orders", f"{high_risk_count:,}", delta_color="inverse")
            col3.metric("Rec. Interventions", f"{total_intervention:,}")
            col4.metric("💰 Est. Net Savings", f"${total_savings:,.2f}", delta="Proactive AI")
            
            # Charts Row 1
            c1, c2 = st.columns([2, 1])
            with c1:
                # Risk Map if lat/lon exists
                if 'Latitude' in df_raw.columns and 'Longitude' in df_raw.columns:
                    st.markdown("##### 🌍 Global Risk Distribution")
                    # Filter for heavy plotting
                    map_data = df_res[['Latitude', 'Longitude', 'Risk Score', 'Order City']].dropna()
                    fig_map = px.scatter_geo(
                        map_data, lat='Latitude', lon='Longitude',
                        color='Risk Score', size='Risk Score',
                        hover_name='Order City',
                        color_continuous_scale='Reds',
                        projection='natural earth'
                    )
                    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fig_map, use_container_width=True)
            
            with c2:
                st.markdown("##### Action Recommendation")
                fig_pie = px.pie(df_res, names='Action', hole=0.4, color='Action',
                                 color_discrete_map={'INTERVENE (Expedite)': '#FF4B4B', 'IGNORE (Standard)': '#00CC96'})
                st.plotly_chart(fig_pie, use_container_width=True)

        with tab2:
            st.subheader("Model Performance & Risk Profile")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 📉 Risk Score Distribution")
                fig_hist = px.histogram(df_res, x='Risk Score', nbins=50, title="Probability Density")
                fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Decision Threshold")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with c2:
                st.markdown("##### 🌡️ Risk by Market/Region")
                if 'Order Region' in df_res.columns:
                    risk_by_region = df_res.groupby('Order Region')['Risk Score'].mean().sort_values(ascending=False).head(10)
                    fig_bar = px.bar(risk_by_region, orientation='h', title="Top 10 High-Risk Regions")
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
            st.info(f"ℹ️ Current Model Threshold: **{threshold:.4f}**. Orders above this score are classified as LATE.")

        with tab3:
            st.subheader("🚀 Delivery Optimization (Priority List)")
            st.markdown("Below is the list of orders requiring **Immediate Action** to prevent late delivery penalties.")
            
            # Filter only interventions
            action_df = df_res[df_res['Action'] == 'INTERVENE (Expedite)'].sort_values(by='Risk Score', ascending=False)
            
            # Table
            display_cols = ['Order Id', 'Order City', 'Product Name', 'Risk Score', 'Net Savings']
            # Safety check if cols exist
            valid_cols = [c for c in display_cols if c in action_df.columns]
            
            st.dataframe(
                action_df[valid_cols].style.background_gradient(subset=['Risk Score'], cmap='Reds'),
                use_container_width=True
            )
            
            # Download Button
            csv = action_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Priority List (CSV)",
                data=csv,
                file_name="priority_intervention_list.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()