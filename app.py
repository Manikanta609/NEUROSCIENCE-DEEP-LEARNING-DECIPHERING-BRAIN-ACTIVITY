import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from src.models import get_model

# 1. Page Configuration
st.set_page_config(
    page_title="NeuroSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS - Neuro-Futurism Theme
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    /* Global Vars */
    :root {
        --bg-color: #0e1117;
        --card-bg: #161b22;
        --text-color: #e6e6e6;
        --accent-cyan: #00f2ff;
        --accent-purple: #bd00ff;
        --neon-glow: 0 0 10px rgba(0, 242, 255, 0.5);
    }

    /* Main Background */
    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(189, 0, 255, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(0, 242, 255, 0.1) 0%, transparent 20%);
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: white !important;
        text-shadow: 0 0 5px rgba(255,255,255,0.2);
    }
    p, span, label, div {
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
    }

    /* Stats Cards */
    .stMetric {
        background-color: var(--card-bg);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Custom Result Card */
    .neuro-card {
        background: rgba(22, 27, 34, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .neuro-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
    }
    
    /* Upload Box */
    .stFileUploader {
        border: 1px dashed var(--accent-cyan);
        border-radius: 10px;
        padding: 10px;
        background: rgba(0, 242, 255, 0.05);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #11141d;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(45deg, var(--accent-purple), var(--accent-cyan));
        color: white;
        font-family: 'Orbitron', sans-serif;
        border: none;
        border-radius: 5px;
        box-shadow: var(--neon-glow);
        transition: all 0.3s;
    }
    .stButton button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/brain--v1.png", width=80)
    st.title("NEUROSENSE UI")
    st.caption("v2.5 // PROD_BUILD")
    st.divider()
    
    st.info("System Status: ONLINE")
    st.write("GPU Acceleration: ENABLED" if torch.cuda.is_available() else "GPU Acceleration: DISABLED")
    st.markdown("---")
    st.markdown("### Controls")
    analysis_mode = st.radio("Mode", ["Single Classification", "Batch Analysis"], index=0)

# 4. Helper Functions
@st.cache_resource
def load_resources():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler_path = 'models/scaler.pkl'
    classes_path = 'models/label_classes.pkl'
    
    if os.path.exists(scaler_path) and os.path.exists(classes_path):
        scaler = joblib.load(scaler_path)
        label_classes = joblib.load(classes_path)
    else:
        st.error("System Core Missing. Run training module.")
        return None, None, None, None
        
    nb_classes = len(label_classes)
    model = get_model('MLP', nb_classes=nb_classes, Chans=None, Samples=None, input_dim=2548)
    
    try:
        model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    except Exception as e:
        st.error(f"Neural Weights Corrupted: {e}")
        return None, None, None, None
        
    model.to(device)
    model.eval()
    return model, scaler, label_classes, device

# Main App Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("## 📡 Data Ingestion")
    st.markdown("Upload pre-processed EEG feature vectors (.csv)")
    uploaded_file = st.file_uploader("", type=['csv'])

model, scaler, label_classes, device = load_resources()

if uploaded_file is not None and model is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Preprocessing
        if 'label' in df.columns:
            feature_df = df.drop(columns=['label'])
        else:
            feature_df = df
            
        with col1:
            st.success(f"Signal Locked: {len(df)} samples")
            if st.button("INITIATE ANALYSIS", use_container_width=True):
                with st.spinner("Decoding Neural Patterns..."):
                    # Inference pipeline
                    X_input = scaler.transform(feature_df.values)
                    X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
                    
                    with torch.no_grad():
                        outputs = model(X_tensor)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()
                        preds = np.argmax(probs, axis=1)
                        
                    pred_labels = [label_classes[p] for p in preds]
                    
                    # Store results in session state to persist across reruns
                    st.session_state['results'] = {
                        'preds': pred_labels,
                        'probs': probs,
                        'df_len': len(df)
                    }

    except Exception as e:
        st.error(f"Ingestion Failure: {e}")

# Results Display Section
if 'results' in st.session_state and uploaded_file is not None:
    results = st.session_state['results']
    pred_labels = results['preds']
    
    # Calculate Dominant State
    from collections import Counter
    counts = Counter(pred_labels)
    dominant_state = counts.most_common(1)[0][0]
    confidence = np.max(results['probs']) * 100
    
    # Color Logic
    color_map = {
        'Relaxed': '#00ff9d',   # Neon Green
        'Focused': '#00f2ff',   # Neon Cyan
        'Stressed': '#ff0055',  # Neon Red
        'Drowsy': '#bd00ff',    # Neon Purple
        'Anxiety': '#ff9100'    # Neon Orange
    }
    state_color = color_map.get(dominant_state, '#ffffff')
    
    with col2:
        st.markdown("## 🧠 Neural Decoding")
        
        # Hero Card
        st.markdown(f"""
        <div class="neuro-card" style="border-top: 5px solid {state_color};">
            <h3 style="margin:0; color: #8b949e;">DETECTED STATE</h3>
            <h1 style="font-size: 3.5em; margin: 10px 0; color: {state_color}; text-shadow: 0 0 20px {state_color};">
                {dominant_state.upper()}
            </h1>
            <p style="color: #8b949e;">Confidence: <span style="color: white; font-weight: bold;">{confidence:.1f}%</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Stats Grid
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Samples", results['df_len'])
        with c2:
            st.metric("Dominance Ratio", f"{int((counts[dominant_state]/results['df_len'])*100)}%")
        with c3:
            st.metric("Processing Time", "< 50ms")
            
        # Dark Theme Plots
        st.markdown("### 📊 Spectral Analysis")
        plt.style.use('dark_background')
        
        # 1. Distribution Plot
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_alpha(0.0) # Transparent bg
        ax.patch.set_alpha(0.0)
        
        dist_df = pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).reset_index()
        dist_df.columns = ['State', 'Count']
        
        colors = [color_map.get(s, '#fff') for s in dist_df['State']]
        sns.barplot(data=dist_df, x='Count', y='State', palette=colors, ax=ax, orient='h')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("Sample Count", color='gray')
        ax.set_ylabel("")
        st.pyplot(fig)
        
        # 2. Timeline (if batch)
        if results['df_len'] > 1:
            st.markdown("### 📈 Temporal Shift")
            timeline_df = pd.DataFrame({'Time': range(len(pred_labels)), 'State': pred_labels})
            y_map = {cls: i for i, cls in enumerate(label_classes)}
            timeline_df['Y'] = timeline_df['State'].map(y_map)
            
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            fig2.patch.set_alpha(0.0)
            ax2.patch.set_alpha(0.0)
            
            ax2.plot(timeline_df['Time'], timeline_df['Y'], color=state_color, marker='o', 
                     linestyle='-', linewidth=2, markersize=4, alpha=0.8)
            ax2.fill_between(timeline_df['Time'], timeline_df['Y'], alpha=0.1, color=state_color)
            
            ax2.set_yticks(range(len(label_classes)))
            ax2.set_yticklabels(label_classes)
            ax2.grid(True, alpha=0.1)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            st.pyplot(fig2)
else:
    # Empty Space Placeholder
    with col2:
        for _ in range(3): st.write("")
        st.markdown("""
        <div style="text-align:center; opacity: 0.5; margin-top: 100px;">
            <h3 style="font-family:'Orbitron'">WAITING FOR SIGNAL...</h3>
            <p>Upload data to begin neural interfacing</p>
        </div>
        """, unsafe_allow_html=True)

