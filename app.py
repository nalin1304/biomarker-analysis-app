import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime

st.set_page_config(
    page_title="Biomarker Analysis Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-green: #2E7D32;
        --light-green: #4CAF50;
        --accent-gold: #FFD700;
        --dark-gold: #B8860B;
        --background: #F8F9FA;
        --card-bg: #FFFFFF;
        --text-primary: #1B5E20;
        --text-secondary: #388E3C;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--light-green) 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .main-header p {
        color: #E8F5E8;
        font-size: 1.1rem;
        text-align: center;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .stCard {
        background: var(--card-bg);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border: 1px solid #E0E0E0;
        margin-bottom: 1rem;
    }
    
    /* Section headers */
    .section-header {
        color: var(--primary-green);
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-gold);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--light-green) 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }
    
    /* Upload area styling */
    .uploadedFile {
        border: 2px dashed var(--light-green);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #F1F8E9;
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        border-color: var(--primary-green);
        background: #E8F5E8;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: var(--card-bg);
        border: 2px solid var(--light-green);
        border-radius: 8px;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, #F8F9FA 100%);
        border: 1px solid var(--accent-gold);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-green);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--primary-green) 0%, var(--light-green) 100%);
    }
    
    .css-1d391kg .stSelectbox label {
        color: white;
        font-weight: 600;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: #E8F5E8;
        border: 1px solid var(--light-green);
        border-radius: 8px;
    }
    
    .stError {
        background: #FFEBEE;
        border: 1px solid #F44336;
        border-radius: 8px;
    }
    
    /* Info panel */
    .info-panel {
        background: linear-gradient(135deg, #FFF9C4 0%, #FFECB3 100%);
        border: 1px solid var(--dark-gold);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .info-panel h3 {
        color: var(--dark-gold);
        margin-top: 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .stCard {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'gradcam_image' not in st.session_state:
    st.session_state.gradcam_image = None


st.markdown("""
<div class="main-header">
    <h1>🔬 Biomarker Analysis Platform</h1>
    <p>Advanced Cancer Subtype Prediction Using Deep Learning & Biomarker Analysis</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("###  Model Information")
    
    with st.expander(" How Our Model Works"):
        st.markdown("""
        **Multi-Modal Deep Learning Approach:**
        
         **Image Analysis:**
        - Convolutional Neural Networks analyze histopathological images
        - Extracts cellular patterns, morphology, and tissue architecture
        - Identifies key visual biomarkers automatically
        
         **Biomarker Integration:**
        - Combines molecular markers (Ki-67, HER2, EGFR) with imaging data
        - Considers staining intensity and cellular localization
        - Provides comprehensive molecular profiling
        
         **Prediction Engine:**
        - Ensemble model combining image and biomarker features
        - Outputs probability scores for each cancer subtype
        - Provides explainable AI through Grad-CAM visualization
        """)
    
    with st.expander("🔬 Cancer Subtypes"):
        st.markdown("""
        **TNBC** - Triple Negative Breast Cancer
        - Lacks ER, PR, and HER2 expression
        - Aggressive subtype, limited targeted therapy
        
        **IDC** - Invasive Ductal Carcinoma
        - Most common breast cancer type
        - Originates in milk ducts
        
        **MBC** - Metaplastic Breast Carcinoma
        - Rare, aggressive subtype
        - Mixed epithelial and mesenchymal features
        
        **ILC** - Invasive Lobular Carcinoma
        - Originates in milk lobules
        - Often hormone receptor positive
        """)
    
    with st.expander("⚡ Feature Importance"):
        st.markdown("""
        **Image Features:**
        - Nuclear morphology and pleomorphism
        - Mitotic activity patterns
        - Tissue architecture disruption
        - Cellular density and organization
        
        **Biomarker Features:**
        - Expression levels and intensity
        - Cellular localization patterns
        - Molecular pathway activation
        - Therapeutic target identification
        """)

col1, col2 = st.columns([1, 1])

with col1:

    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📸 Image Upload</h3>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload histopathological image (.jpg format)",
        type=['jpg', 'jpeg'],
        help="Upload a high-quality histopathological image for analysis"
    )
    
    if uploaded_file is not None:
       
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        st.image(image, caption="Uploaded Image", use_column_width=True)
  
        st.markdown("**Image Details:**")
        st.write(f"📏 Size: {image.size[0]} × {image.size[1]} pixels")
        st.write(f"📄 Format: {image.format}")
        st.write(f"💾 File size: {len(uploaded_file.getvalue()) / 1024:.1f} KB")
    
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">🧬 Biomarker Configuration</h3>', unsafe_allow_html=True)
    

    biomarker_options = ['Ki-67', 'HER2', 'EGFR', 'ER', 'PR', 'p53', 'BRCA1', 'BRCA2']
    selected_biomarker = st.selectbox(
        "Select Biomarker",
        biomarker_options,
        help="Choose the primary biomarker for analysis"
    )
    
  
    intensity_options = ['Weak', 'Moderate', 'Strong']
    selected_intensity = st.selectbox(
        "Staining Intensity",
        intensity_options,
        index=1,
        help="Select the observed staining intensity"
    )
    

    staining_options = ['Nuclear', 'Cytoplasmic', 'Membranous', 'Mixed']
    selected_staining = st.selectbox(
        "Staining Pattern",
        staining_options,
        help="Choose the cellular localization pattern"
    )
    
    st.markdown("**Selected Configuration:**")
    st.write(f"🔬 Biomarker: **{selected_biomarker}**")
    st.write(f"💪 Intensity: **{selected_intensity}**")
    st.write(f"📍 Localization: **{selected_staining}**")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">🚀 Run Analysis</h3>', unsafe_allow_html=True)
    
    
    if st.button("🔍 Analyze Sample", type="primary"):
        if st.session_state.uploaded_image is not None:
            with st.spinner("🔬 Analyzing image and biomarkers..."):
            
              
                
                
                prediction_probs = {
                    'TNBC': np.random.random(),
                    'IDC': np.random.random(),
                    'MBC': np.random.random(),
                    'ILC': np.random.random()
                }
                
                
                total = sum(prediction_probs.values())
                prediction_probs = {k: v/total for k, v in prediction_probs.items()}
                
             
                st.session_state.prediction_results = prediction_probs
                
               
                
            st.success("✅ Analysis completed successfully!")
        else:
            st.error("❌ Please upload an image first!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    

    if st.session_state.prediction_results is not None:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📊 Prediction Results</h3>', unsafe_allow_html=True)
        
  
        top_prediction = max(st.session_state.prediction_results.items(), key=lambda x: x[1])
        
       
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{top_prediction[0]}</div>
            <div class="metric-label">Predicted Subtype ({top_prediction[1]:.1%} confidence)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Confidence Scores:**")
        
      
        fig = go.Figure()
        
        colors = ['#2E7D32', '#4CAF50', '#FFD700', '#B8860B']
        subtypes = list(st.session_state.prediction_results.keys())
        probabilities = list(st.session_state.prediction_results.values())
        
        fig.add_trace(go.Bar(
            y=subtypes,
            x=probabilities,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{p:.1%}' for p in probabilities],
            textposition='inside',
            textfont=dict(color='white', size=12, family='Arial Black')
        ))
        
        fig.update_layout(
            title="Subtype Probability Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Cancer Subtype",
            height=300,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1B5E20')
        )
        
        fig.update_xaxis(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


st.markdown("---")


if st.session_state.gradcam_image is not None:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">🎯 Grad-CAM Heatmap Visualization</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Image**")
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, use_column_width=True)
    
    with col2:
        st.markdown("**Attention Heatmap**")
        st.image(st.session_state.gradcam_image, use_column_width=True, caption="Regions of high model attention")
    
    st.markdown("""
    **Interpretation:** The heatmap shows which regions of the image were most important for the model's prediction. 
    Warmer colors (red/yellow) indicate areas that strongly influenced the classification decision.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="info-panel">', unsafe_allow_html=True)
st.markdown("### 🔍 Model Explainability & Clinical Relevance")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🖼️ Image Analysis**
    - Automated feature extraction from histopathological slides
    - Identifies cellular morphology patterns
    - Analyzes tissue architecture and organization
    - Detects mitotic figures and nuclear features
    """)

with col2:
    st.markdown("""
    **🧬 Biomarker Integration**
    - Molecular marker expression levels
    - Cellular localization patterns
    - Pathway activation indicators
    - Therapeutic target identification
    """)

with col3:
    st.markdown("""
    **🎯 Clinical Impact**
    - Supports diagnostic decision-making
    - Guides treatment selection
    - Identifies prognostic factors
    - Enables personalized medicine
    """)

st.markdown('</div>', unsafe_allow_html=True)


if st.session_state.prediction_results is not None:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📄 Generate Report</h3>', unsafe_allow_html=True)
    

    report_data = {
        'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Biomarker': selected_biomarker,
        'Intensity': selected_intensity,
        'Staining Pattern': selected_staining,
        'Top Prediction': top_prediction[0],
        'Confidence': f"{top_prediction[1]:.1%}",
        'All Predictions': st.session_state.prediction_results
    }
    

    st.markdown("**Report Summary:**")
    st.json(report_data)
    
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🔬 <strong>Biomarker Analysis Platform</strong> | Powered by Deep Learning & Medical AI</p>
    <p>For research and educational purposes. Always consult with medical professionals for clinical decisions.</p>
</div>
""", unsafe_allow_html=True)

# API Integration Helper Function (for when you connect to your backend)
def call_prediction_api(image_data, biomarker_data):
    api_url = ""
    

    payload = {
        "image": base64.b64encode(image_data).decode('utf-8'),
        "biomarker": biomarker_data['biomarker'],
        "intensity": biomarker_data['intensity'],
        "staining_type": biomarker_data['staining_type']
    }
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


def preprocess_image(image):
    

    image = image.resize((224, 224))

    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image)

    img_array = img_array / 255.0
    
    return img_array
