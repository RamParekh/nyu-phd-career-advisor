# NYU PhD Career Advisor App
# PERFORMANCE OPTIMIZATION: ML model training disabled for faster deployment
# - Uses instant keyword-based sector prediction instead of slow ML training
# - Eliminates delays on Streamlit Cloud
# - Provides same quality predictions with instant results

# =============================================================================
# 🔧 DATASET CONFIGURATION - UPDATE THESE PATHS FOR YOUR DATA
# =============================================================================
# 📁 RIASEC DATASET - ADD YOUR PATH HERE
RIASEC_DATASET_PATH = "data/RIASEC .xlsx"  # ← CHANGE THIS TO YOUR ACTUAL RIASEC FILE PATH

# 📁 Other datasets (already configured)
NYU_CAREER_DATA_PATH = "data/imputed_data_NYU copy 3.xlsx"  # ← Your main NYU career data
HANDSHAKE_EVENTS_PATH = "data/Events for Graduate Students.csv"  # ← Handshake events data
SENTIMENT_DATA_PATH = "data/sentiment analysis- Final combined .xlsx"  # ← Sentiment analysis data
TRAINING_DATA_PATH = "data/phd_career_sector_training_data_final_1200.xlsx"  # ← ML training data

# =============================================================================
# 🚀 QUICK SETUP GUIDE:
# =============================================================================
# 1. Place your RIASEC dataset in the 'data' folder
# 2. Update RIASEC_DATASET_PATH above with your actual filename
# 3. The app will automatically load and use your dataset
# 4. Check the console output for dataset loading confirmation
# =============================================================================

import streamlit as st
import openai, pandas as pd, requests, warnings, json, traceback
from datetime import datetime
# Try to import ML libraries, with fallback for Python 3.13 compatibility
try:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    import xgboost as xgb
    from sklearn.base import BaseEstimator, ClassifierMixin
    ML_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Some ML libraries not available: {e}")
    st.info("The app will work with basic features, but some advanced ML features may be limited.")
    ML_AVAILABLE = False
import numpy as np
import uuid
from textblob import TextBlob
from collections import defaultdict
import os
import re
import contractions
import time
from functools import lru_cache

# Add new imports for multiple sentiment models
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

warnings.filterwarnings("ignore")

# --- Sentiment Analysis Helper Functions ---
def preprocess_text_for_sentiment(text):
    """Expand contractions, normalize repeated chars, handle sarcasm cues, etc."""
    if not isinstance(text, str):
        return ""
    # Expand contractions
    text = contractions.fix(text)
    # Lowercase
    text = text.lower()
    # Normalize repeated characters (e.g., sooooo -> soo)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Remove excessive punctuation (keep ! and ? for sentiment)
    text = re.sub(r'([!?.])\1+', r'\1', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Sarcasm Cue Detection ---
SARCASM_CUES = [
    'yeah right', 'as if', 'totally', 'sure', 'obviously', 'great...', 'nice...', 'love that',
    'just perfect', 'wonderful', 'amazing', 'sure thing', 'can\'t wait', '/s', 'right...',
    'fantastic', 'brilliant', 'awesome', 'lovely', 'what a surprise', 'what a joy', 'so fun',
    'so helpful', 'thanks a lot', 'thanks for nothing', 'good luck with that', 'i bet',
]

def detect_sarcasm(text):
    text = text.lower()
    for cue in SARCASM_CUES:
        if cue in text:
            return True
    return False

# --- Custom VADER Lexicon for Sarcasm ---
CUSTOM_VADER_LEXICON = {
    'yeah right': -2.0, 'as if': -2.0, 'totally': -1.0, 'sure': -1.0, 'obviously': -1.0,
    'great...': -2.0, 'nice...': -2.0, 'love that': -1.5, 'just perfect': -2.0, 'wonderful': -1.0,
    'amazing': -1.0, 'sure thing': -1.0, 'can\'t wait': -1.5, '/s': -2.0, 'right...': -2.0,
    'fantastic': -1.0, 'brilliant': -1.0, 'awesome': -1.0, 'lovely': -1.0, 'what a surprise': -1.0,
    'what a joy': -1.0, 'so fun': -1.0, 'so helpful': -1.0, 'thanks a lot': -1.5, 'thanks for nothing': -2.0,
    'good luck with that': -2.0, 'i bet': -1.0,
}

# Move set_page_config here, before any other Streamlit commands
st.set_page_config(page_title="PhD Compass", layout="wide", initial_sidebar_state="expanded")

# Initialize session state variables
if 'recommendations_completed' not in st.session_state:
    st.session_state.recommendations_completed = False
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'recommendations_data' not in st.session_state:
    st.session_state.recommendations_data = []

# --- API Keys from Streamlit Secrets ---

# API Keys - Add your actual keys here
# Get free keys from: https://developer.adzuna.com/

# Replace these with your actual Adzuna API keys
ADZUNA_APP_ID = "5a2ee7fb"  # Replace this
ADZUNA_APP_KEY = "cf15da52bcd7e5585e5cd2d7fea154e5"  # Replace this

# OpenAI API Key - Add your actual key here
# Get your key from: https://platform.openai.com/api-keys

# Replace this with your actual OpenAI API key
openai.api_key = ""  # Replace this

# Fallback to environment variables if needed
import os
if openai.api_key == "your-actual-openai-api-key-here":
    openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

# Check if OpenAI API key is configured
if openai.api_key != "your-actual-openai-api-key-here":
    # Add checkbox to control OpenAI usage
    use_openai = st.sidebar.checkbox(
        "🔓 Enable personalization", 
        value=False,
        help="Check this to use AI for personalized career insights (uses API tokens)"
    )
    # No info messages needed - checkbox state is clear
else:
    use_openai = False
    st.sidebar.warning("⚠️ OpenAI API Key Not Set")
    st.sidebar.info("💡 Replace the OpenAI API key in the code with your actual key")

# Show Adzuna API status
if ADZUNA_APP_ID != "your-actual-adzuna-app-id-here" and ADZUNA_APP_KEY != "your-actual-adzuna-app-key-here":
    # Job API keys are configured (no message needed)
    pass
else:
    st.sidebar.warning("⚠️ Job API Keys Not Set")
    st.sidebar.info("💡 Replace the API keys in the code with your actual keys from developer.adzuna.com")





# -----------------------------------------------------------------------------
# Local Data Loading Helper --------------------------------------------
# -----------------------------------------------------------------------------



# -------------------------------------------------------------------
# Reinforcement Learning (basic Q-table using Google Sheets feedback)
# -------------------------------------------------------------------

def load_feedback():
    """Load feedback data from local file (if available)."""
    try:
        # For now, return empty DataFrame since we don't have a local feedback file
        # You can add a local feedback file later if needed
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading feedback data: {e}")
        return pd.DataFrame()

def process_feedback(df_feedback):
    """Returns a dictionary of bad actions to avoid based on feedback."""
    if df_feedback.empty:
        return {}
    avoidance_dict = {}
    grouped = df_feedback.groupby("Division")
    for division, subset in grouped:
        Comments = subset["Comments"].dropna().str.split(", ")
        flat_list = [item for sublist in Comments for item in sublist]
        bad_actions = pd.Series(flat_list).value_counts()
        avoidance_dict[division] = bad_actions.to_dict()
    return avoidance_dict

# Load feedback into app
feedback_df = load_feedback()
q_agent = process_feedback(feedback_df)

# -----------------------------------------------------------------------------
# Streamlit page config & NYU‑style UI  ---------------------------------------
# -----------------------------------------------------------------------------

NYU_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');

:root {
  /* --- NYU brand colours ---- */
  --violet:        #57068c;   /* primary NYU purple */
  --violet-light:  #8850c8;   /* lighter hover / focus shade */
  --violet-dark:   #3d004c;   /* darker shade for contrast */
  --text:          #1c1c1c;   /* default body text */
  --text-light:    #666666;   /* lighter text for secondary info */
  --bg-light:      #f8f9fa;   /* light background */
  --border:        #e1e1e1;   /* border color */
  --success:       #28a745;   /* success color */
  --warning:       #ffc107;   /* warning color */
  --error:         #dc3545;   /* error color */

  --st-header-h: 56px;   /* Streamlit's built-in header */
  --nyu-nav-h:   60px;   /* purple NYU bar height       */
  --hero-h:     240px;   /* hero banner height          */
  --sb-w:       250px;  
}

/* Global Styles */
html, body, .stApp {
  background: var(--bg-light) !important;
  color: var(--text) !important;
  font-family: 'Source Sans Pro', sans-serif !important;
  line-height: 1.6;
}

/* NYU Info Strip */
.nyu-info-strip {
  position: fixed;
  top: 0;
  left: 0; right: 0;
  height: 32px;
  background: #fff;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 18px;
  font-size: 0.85rem;
  border-bottom: 1px solid var(--border);
  z-index: 10002;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.nyu-info-strip a {
  color: var(--violet);
  font-weight: 600;
  text-decoration: none;
  transition: color 0.2s ease;
}

.nyu-info-strip a:hover {
  color: var(--violet-light);
  text-decoration: none;
}

/* Main Navigation Bar */
.nyu-main-bar {
  position: fixed;
  top: var(--st-header-h);
  left: 0; right: 0;
  height: var(--nyu-nav-h);
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 12px;
  background: var(--violet);
  padding: 0 24px;
  z-index: 20000;
  transition: left .25s ease;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.nyu-main-bar img {
  height: 38px;
  max-width: 160px;
  object-fit: contain;
}

.nyu-main-bar span {
  color: #fff;
  font-size: 1.35rem;
  font-weight: 700;
  line-height: 1.25;
  text-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

/* Hero Section */
.nyu-hero {
  position: fixed;
  top: calc(var(--st-header-h) + var(--nyu-nav-h));
  left: 0; right: 0;
  height: var(--hero-h);
  background: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)),
              url('https://teachingsupport.hosting.nyu.edu/wp-content/uploads/page-banner.jpg')
              center center / cover no-repeat;
  border-bottom: 1px solid var(--border);
  z-index: 15000;
}

.nyu-hero-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 2.5rem;
  font-weight: 700;
  color: white;
  text-align: center;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
  max-width: 800px;
  padding: 0 20px;
  z-index: 999;
}

.nyu-hero-text::after {
  content: '';
  position: absolute;
  bottom: -20px;
  left: 50%;
  transform: translateX(-50%);
  width: 100px;
  height: 4px;
  background: var(--violet-light);
  border-radius: 2px;
}

/* Content Container */
.block-container {
  padding-top: calc(var(--st-header-h) + var(--nyu-nav-h) + var(--hero-h) + 24px) !important;
  max-width: 1200px !important;
  margin: 0 auto !important;
}

/* Form Elements */
.stTextInput>div>div>input,
.stTextArea textarea {
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  padding: 8px 12px !important;
  transition: all 0.2s ease !important;
}

.stTextInput>div>div>input:focus,
.stTextArea textarea:focus {
  border-color: var(--violet) !important;
  box-shadow: 0 0 0 2px rgba(87, 6, 140, 0.1) !important;
}

.stFileUploader {
  border-radius: 10px !important;
  border: 2px dashed var(--violet) !important;
  background: #fff !important;
  padding: 20px !important;
  transition: all 0.2s ease !important;
}

.stFileUploader:hover {
  border-color: var(--violet-light) !important;
  background: var(--bg-light) !important;
}

/* Headers */
.header {
  font-size: 2.5rem;
  text-align: center;
  color: var(--violet);
  margin: 2rem 0;
  font-weight: 700;
  text-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.subheader {
  font-size: 1.5rem;
  margin: 1.5rem 0;
  color: var(--violet-dark);
  font-weight: 600;
}

/* Cards and Containers */
.recommendations {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  margin: 20px 0;
}

.opportunity-card {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  padding: 24px;
  margin-bottom: 24px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.opportunity-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.opportunity-card h4 {
  margin: 0 0 12px;
  color: var(--violet);
  font-size: 1.3rem;
  font-weight: 600;
}

.opportunity-card a {
  color: var(--violet);
  text-decoration: none;
  font-weight: 600;
  transition: color 0.2s ease;
}

.opportunity-card a:hover {
  color: var(--violet-light);
  text-decoration: none;
}

/* Tabs */
div[data-testid="stTabs"] {
  background: #fff;
  border-radius: 12px;
  padding: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

div[data-testid="stTabs"] button {
  font-size: 1rem;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  transition: all 0.2s ease;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
  background: var(--violet) !important;
  color: white !important;
}

/* Buttons */
.stButton>button {
  background: var(--violet) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.75rem 1.5rem !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
}

.stButton>button:hover {
  background: var(--violet-light) !important;
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Radio and Select */
.stRadio>div {
  background: #fff;
  border-radius: 8px;
  padding: 12px;
  border: 1px solid var(--border);
}

.stSelectbox>div>div {
  background: #fff;
  border-radius: 8px;
  border: 1px solid var(--border);
}

/* Success/Error Messages */
.stSuccess {
  background: var(--success) !important;
  color: white !important;
  border-radius: 8px !important;
  padding: 12px !important;
}

.stError {
  background: var(--error) !important;
  color: white !important;
  border-radius: 8px !important;
  padding: 12px !important;
}

/* Feedback Section */
.feedback-section {
  background: #fff;
  border-radius: 12px;
  padding: 24px;
  margin-top: 32px;
  border: 1px solid var(--border);
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* Event Cards */
.event-card {
  background: #fff;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
  border: 1px solid var(--border);
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  transition: transform 0.2s ease;
}

.event-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.event-card img {
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Responsive Design */
@media (max-width: 768px) {
  .header {
    font-size: 2rem;
  }
  
  .subheader {
    font-size: 1.25rem;
  }
  
  .block-container {
    padding: 16px !important;
  }
  
  .opportunity-card, .recommendations {
    padding: 16px;
  }
}
</style>
"""

st.markdown("""
<div class='nyu-info-strip'>
  <strong>Information For:</strong>
  <a href='#'>Students</a>
  <a href='#'>Faculty</a>
  <a href='#'>Alumni</a>
  <a href='#'>Employees</a>
  <a href='#'>Community</a>
</div>

<div class='nyu-main-bar'>
  <img src='https://i.pinimg.com/736x/0b/af/fb/0baffb91607677363de658bebdbb1ee1.jpg' alt='NYU Logo'>
  <span>PhD Compass</span>
</div>

<div class='nyu-hero'>
  <div class='nyu-hero-text'>
    PhD Compass
          <div style='font-size: 1.2rem; margin-top: 10px; font-weight: 400;'>
        Research-Based Insights For Navigating Doctoral Careers
      </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(NYU_CSS, unsafe_allow_html=True)

# --- Sidebar: How to Use and Contact Info ---
st.sidebar.markdown("### ℹ️ How to Use This App")

with st.sidebar.expander("📋 Click to expand instructions", expanded=False):
    st.markdown("""
    1. Fill in your academic and career information.
    2. Click **Explore Your Options** (max 3 per session).
    3. Review the personalized advice and explore opportunities.
    4. Use the feedback section to help us improve!
    """)

# Add feedback management section to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 RIASEC Integration")
st.sidebar.info("""
**New: Focused RIASEC Assessment**
- 10 targeted questions across 6 dimensions
- Real-time scoring and visualization
- Career pathway mapping based on personality
- Enhanced AI recommendations using RIASEC data
""")

st.sidebar.markdown("### 📊 Feedback Management")

# Show feedback count if available
if 'feedback_data' in st.session_state and len(st.session_state.feedback_data) > 0:
    st.sidebar.success(f"📝 {len(st.session_state.feedback_data)} feedback entries collected")
    
    # Export all feedback button
    if st.sidebar.button("📥 Export All Feedback"):
        feedback_df = pd.DataFrame(st.session_state.feedback_data)
        csv = feedback_df.to_csv(index=False)
        st.sidebar.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name=f"nyu_career_feedback_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Clear feedback button
    if st.sidebar.button("🗑️ Clear All Feedback"):
        st.session_state.feedback_data = []
        st.sidebar.success("✅ Feedback cleared!")
        st.rerun()
    
    # Recommendations management
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Recommendations")
    
    if 'recommendations_data' in st.session_state and len(st.session_state.recommendations_data) > 0:
        st.sidebar.success(f"📋 {len(st.session_state.recommendations_data)} recommendations generated")
        
        # Download all recommendations as CSV
        recommendations_df = pd.DataFrame(st.session_state.recommendations_data)
        csv = recommendations_df.to_csv(index=False)
        st.sidebar.download_button(
            label="📥 Download All Recommendations (CSV)",
            data=csv,
            file_name="career_recommendations.csv",
            mime="text/csv",
            key="sidebar_download_all"
        )
        
        # Clear recommendations button
        if st.sidebar.button("🗑️ Clear All Recommendations"):
            st.session_state.recommendations_data = []
            st.session_state.recommendations_completed = False
            st.sidebar.success("✅ All recommendations cleared!")
            st.rerun()
    else:
        st.sidebar.info("📋 No recommendations generated yet")
else:
    st.sidebar.info("📝 No feedback collected yet")
    st.sidebar.info("📋 No recommendations generated yet")



# Function to ensure model is loaded (DISABLED for faster performance)
def ensure_model_loaded():
    """Ensure the sector prediction model is loaded."""
    # DISABLED: ML model training causes delays on Streamlit Cloud
    # Use cached model training
    # print("🔄 Loading sector prediction model...")
    # model = train_cached_model()
    # if model is not None:
    #     print("✅ Sector prediction model loaded successfully!")
    #     return model
    # else:
    #     print("❌ Failed to load sector prediction model")
    #     return None
    
    # Return None to use keyword-based predictions instead
    print("🚀 Skipping ML model training for instant performance")
    return None

def predict_sector_from_text(user_goals, desired_industry, work_env):
    if not user_goals:
        st.warning("No user goals provided for sector prediction")
        return "Unknown", "Not available"
    
    # Create comprehensive input text for analysis
    input_text = f"{user_goals} {desired_industry} {work_env}".lower()
    
    # Enhanced keyword-based prediction with better logic
    def analyze_text_for_sector(text):
        # Academic/Research indicators
        academic_keywords = [
            'academic', 'professor', 'university', 'college', 'school',
            'teaching', 'lecture', 'publish', 'paper', 'journal', 'conference',
            'thesis', 'dissertation', 'scholar', 'faculty',
            'postdoc', 'postdoctoral', 'tenure', 'academia', 'higher education'
            # Note: 'research', 'researcher', 'phd', 'doctoral' are handled separately with lower weights
        ]
        
        # Industry/Corporate indicators
        industry_keywords = [
            'industry', 'company', 'business', 'corporate', 'startup', 'tech',
            'product', 'development', 'engineering', 'software', 'data', 'analytics',
            'consulting', 'finance', 'banking', 'investment', 'marketing', 'sales',
            'operations', 'management', 'leadership', 'entrepreneur', 'founder',
            'data scientist', 'research engineer', 'product researcher', 'applied research',
            'industrial research', 'corporate research', 'business research', 'market research',
            'product development', 'r&d', 'research and development', 'machine learning',
            'ai', 'artificial intelligence', 'data analyst', 'business analyst',
            'product manager', 'project manager', 'technical lead', 'architect',
            'developer', 'programmer', 'coder', 'scientist', 'researcher',
            'innovation', 'technology', 'digital', 'online', 'web', 'mobile',
            'cloud', 'cybersecurity', 'blockchain', 'automation', 'optimization'
        ]
        
        # Government/Public sector indicators
        government_keywords = [
            'government', 'public', 'policy', 'federal', 'state', 'local',
            'regulatory', 'compliance', 'law', 'legal', 'justice', 'defense',
            'military', 'intelligence', 'diplomatic', 'foreign service', 'civil service',
            'public health', 'environmental', 'energy', 'transportation', 'education'
        ]
        
        # Non-profit/NGO indicators
        nonprofit_keywords = [
            'non-profit', 'nonprofit', 'ngo', 'charity', 'foundation', 'volunteer',
            'social impact', 'community', 'advocacy', 'humanitarian', 'aid',
            'environmental', 'conservation', 'healthcare', 'education', 'arts',
            'cultural', 'religious', 'faith-based', 'social justice', 'equity'
        ]
        
        # Healthcare indicators
        healthcare_keywords = [
            'healthcare', 'medical', 'hospital', 'clinic', 'patient', 'clinical',
            'pharmaceutical', 'biotech', 'biomedical', 'nursing', 'therapy',
            'public health', 'epidemiology', 'genetics', 'oncology', 'cardiology'
        ]
        
        # Education indicators (K-12, not higher ed)
        education_keywords = [
            'k-12', 'elementary', 'middle school', 'high school', 'primary',
            'secondary', 'curriculum', 'pedagogy', 'student teaching', 'classroom'
        ]
        
        # Enhanced scoring with weighted keywords and better balance
        scores = {
            'Academic': 0,
            'Industry': 0,
            'Government': 0,
            'Non-profit': 0,
            'Healthcare': 0,
            'Education': 0
        }
        
        # Weighted scoring system
        for word in academic_keywords:
            if word in text:
                # Reduce weight for common academic words that might appear in industry contexts
                if word in ['research', 'researcher', 'phd', 'doctoral']:
                    scores['Academic'] += 0.5  # Lower weight for ambiguous terms
                else:
                    scores['Academic'] += 1
        
        for word in industry_keywords:
            if word in text:
                # Give higher weight to clear industry indicators
                if word in ['industry', 'company', 'business', 'corporate', 'startup', 'tech', 'product', 'engineering', 'software', 'data scientist', 'consulting', 'finance']:
                    scores['Industry'] += 2  # Higher weight for strong industry indicators
                else:
                    scores['Industry'] += 1
        
        # Special handling for research + industry combinations
        # If both research and industry keywords are present, prioritize industry for applied research
        if scores['Academic'] > 0 and scores['Industry'] > 0:
            # Strong boost for industry research roles
            if any(word in text for word in ['data scientist', 'research engineer', 'product researcher', 'applied research', 'industrial research', 'corporate research', 'business research', 'market research', 'product development', 'r&d', 'research and development']):
                scores['Industry'] += 3
                print(f"🔍 Boosting Industry score for applied research role in industry")
            # Only boost academic for pure academic roles
            elif any(word in text for word in ['professor', 'university', 'college', 'academia', 'tenure', 'faculty', 'postdoc', 'postdoctoral']):
                scores['Academic'] += 2
                print(f"🔍 Boosting Academic score for pure academic role")
        
        # Additional industry boost for modern tech roles
        tech_industry_indicators = ['data scientist', 'machine learning', 'ai', 'artificial intelligence', 'software engineer', 'product manager', 'data analyst', 'research engineer']
        for indicator in tech_industry_indicators:
            if indicator in text:
                scores['Industry'] += 1.5
                print(f"🔍 Boosting Industry score for tech role: {indicator}")
        
        # Find the sector with the highest score
        max_score = max(scores.values())
        if max_score == 0:
            return "Industry"  # Default to industry if no clear indicators
        
        # Return the sector with the highest score
        best_sector = max(scores, key=scores.get)
        print(f"🔍 Final scores: {scores}")
        
        # Debug: Show what was detected in the input
        detected_academic = [word for word in academic_keywords if word in text]
        detected_industry = [word for word in industry_keywords if word in text]
        print(f"🔍 Detected academic keywords: {detected_academic}")
        print(f"🔍 Detected industry keywords: {detected_industry}")
        
        return best_sector
    
    # Skip ML model training for faster performance - use keyword analysis directly
    # if ML_AVAILABLE:
    #     text_sector_model = ensure_model_loaded()
    #     if text_sector_model is not None:
    #         try:
    #             prediction = text_sector_model.predict([input_text])[0]
    #             return prediction
    #         except Exception as e:
    #             print(f"ML model prediction failed: {e}, falling back to keyword analysis")
    #             # Fall through to keyword analysis
    
    # Use enhanced keyword analysis directly for instant results
    sector = analyze_text_for_sector(input_text)
    print(f"🔍 Sector prediction for '{input_text[:100]}...': {sector}")
    
    # More intelligent logic: prioritize industry for applied research roles
    strong_academic_indicators = ['professor', 'university', 'college', 'academia', 'tenure', 'faculty', 'postdoc', 'postdoctoral']
    strong_industry_indicators = ['industry', 'company', 'business', 'corporate', 'startup', 'tech', 'product', 'engineering', 'software', 'data scientist', 'consulting', 'finance', 'r&d', 'research and development']
    
    has_strong_academic = any(word in input_text for word in strong_academic_indicators)
    has_strong_industry = any(word in input_text for word in strong_industry_indicators)
    
    # Special case: if both research and industry indicators present, favor industry for applied research
    if 'research' in input_text and has_strong_industry:
        print(f"🔍 Research + Industry indicators detected - favoring Industry for applied research")
        return "Industry", "Keyword Analysis (Enhanced - Applied Research)"
    
    # Only override to Academic if there are STRONG academic indicators AND no strong industry indicators
    if has_strong_academic and not has_strong_industry:
        print(f"🔍 Overriding to Academic due to strong academic indicators")
        return "Academic", "Keyword Analysis (Enhanced)"
    elif has_strong_industry:
        print(f"🔍 Overriding to Industry due to strong industry indicators")
        return "Industry", "Keyword Analysis (Enhanced)"
    else:
        # Let the keyword scoring decide
        print(f"🔍 Using keyword scoring result: {sector}")
        return sector, "Keyword Analysis (Enhanced)"



# -----------------------------------------------------------------------------
# Dataset loader (Modified to load NYU dataset directly) ----------------------
# -----------------------------------------------------------------------------

# Data will be loaded fresh each time

def load_local_data():
    """Load all data from local files in the data folder."""
    loaded_data = {}
    
    # For local development, prioritize local files
    print("📁 Loading data from local files...")
    print(f"🔍 DEBUG: Current working directory: {os.getcwd()}")
    print(f"🔍 DEBUG: Data folder exists: {os.path.exists('data')}")
    if os.path.exists('data'):
        print(f"🔍 DEBUG: Data folder contents: {os.listdir('data')}")
    
    # Load NYU career data
    try:
        df = pd.read_excel("data/imputed_data_NYU copy 3.xlsx")
        # Clean the data
        for c in df.select_dtypes(include=[object]):
            df[c] = df[c].astype(str).str[:800]
        loaded_data['nyu_data'] = df
        print(f"✅ Loaded {len(df)} records from NYU career data")
    except Exception as e:
        print(f"❌ Error loading NYU career data: {e}")
    
    # Load Handshake events
    try:
        events_df = pd.read_csv("data/Events for Graduate Students.csv")
        loaded_data['handshake_events'] = events_df
        print(f"✅ Loaded {len(events_df)} Handshake events")
    except Exception as e:
        print(f"❌ Error loading Handshake events: {e}")
    
    # Load sentiment data
    try:
        sentiment_df = pd.read_excel("data/sentiment analysis- Final combined .xlsx")
        loaded_data['sentiment_data'] = sentiment_df
        print(f"✅ Loaded {len(sentiment_df)} sentiment records")
    except Exception as e:
        print(f"❌ Error loading sentiment data: {e}")
    
    # Load training data
    try:
        training_df = pd.read_excel("data/phd_career_sector_training_data_final_1200.xlsx")
        loaded_data['training_data'] = training_df
        print(f"✅ Loaded {len(training_df)} training records")
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
    
    # Try to load from Streamlit secrets as fallback (for production)
    try:
        if 'data_files' in st.secrets:
            print("🔐 Also found Streamlit secrets data (will use for production)...")
            
            # Load NYU career data from base64 if not already loaded
            if 'nyu_data' not in loaded_data and 'nyu_data' in st.secrets['data_files']:
                import base64
                import io
                nyu_data_b64 = st.secrets['data_files']['nyu_data']
                nyu_data_bytes = base64.b64decode(nyu_data_b64)
                df = pd.read_excel(io.BytesIO(nyu_data_bytes))
                # Clean the data
                for c in df.select_dtypes(include=[object]):
                    df[c] = df[c].astype(str).str[:800]
                loaded_data['nyu_data'] = df
                print(f"✅ Loaded {len(df)} records from secrets (NYU career data)")
            
            # Load Handshake events from base64 if not already loaded
            if 'handshake_events' not in loaded_data and 'handshake_events' in st.secrets['data_files']:
                import base64
                import io
                events_data_b64 = st.secrets['data_files']['handshake_events']
                events_data_bytes = base64.b64decode(events_data_b64)
                events_df = pd.read_csv(io.BytesIO(events_data_bytes))
                loaded_data['handshake_events'] = events_df
                print(f"✅ Loaded {len(events_df)} Handshake events from secrets")
            
            # Load training data from base64 if not already loaded
            if 'training_data' not in loaded_data and 'training_data' in st.secrets['data_files']:
                import base64
                import io
                training_data_b64 = st.secrets['data_files']['training_data']
                training_data_bytes = base64.b64decode(training_data_b64)
                training_df = pd.read_excel(io.BytesIO(training_data_bytes))
                loaded_data['training_data'] = training_df
                print(f"✅ Loaded {len(training_df)} training records from secrets")
            
    except Exception as e:
        print(f"⚠️ Could not load from secrets: {e}")
    
    return loaded_data

def load_data():
    """Load NYU career data from local file or secrets."""
    try:
        # Try to load from local file first (for development)
        if os.path.exists("data/imputed_data_NYU copy 3.xlsx"):
            df = pd.read_excel("data/imputed_data_NYU copy 3.xlsx")
            print(f"✅ Loaded {len(df)} NYU records from local file")
        elif 'data_files' in st.secrets and 'nyu_data' in st.secrets['data_files']:
            # Fallback to secrets (for production)
            import base64
            import io
            nyu_data_b64 = st.secrets['data_files']['nyu_data']
            nyu_data_bytes = base64.b64decode(nyu_data_b64)
            df = pd.read_excel(io.BytesIO(nyu_data_bytes))
            print(f"✅ Loaded {len(df)} NYU records from secrets")
        else:
            print("❌ No NYU data available locally or in secrets")
            return None
        
        # Clean the data
        for c in df.select_dtypes(include=[object]):
            df[c] = df[c].astype(str).str[:800]
        return df
    except Exception as e:
        print(f"❌ Error loading NYU data: {e}")
        return None

def load_handshake_events():
    """Load Handshake events from local file or secrets."""
    try:
        # Try to load from local file first (for development)
        if os.path.exists("data/Events for Graduate Students.csv"):
            df = pd.read_csv("data/Events for Graduate Students.csv")
            print(f"✅ Loaded {len(df)} Handshake events from local file")
        elif 'data_files' in st.secrets and 'handshake_events' in st.secrets['data_files']:
            # Fallback to secrets (for production)
            import base64
            import io
            events_data_b64 = st.secrets['data_files']['handshake_events']
            events_data_bytes = base64.b64decode(events_data_b64)
            df = pd.read_csv(io.BytesIO(events_data_bytes))
            print(f"✅ Loaded {len(df)} Handshake events from secrets")
        else:
            print("❌ No Handshake events available locally or in secrets")
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"❌ Error loading Handshake events: {e}")
        return pd.DataFrame()

def load_sentiment_data():
    """Load sentiment analysis data from local file or secrets."""
    try:
        # Try to load from local file first (for development)
        if os.path.exists("data/sentiment analysis- Final combined .xlsx"):
            df = pd.read_excel("data/sentiment analysis- Final combined .xlsx")
            print(f"✅ Loaded {len(df)} sentiment records from local file")
        elif 'data_files' in st.secrets and 'sentiment_data' in st.secrets['data_files']:
            # Fallback to secrets (for production)
            import base64
            import io
            sentiment_data_b64 = st.secrets['data_files']['sentiment_data']
            sentiment_data_bytes = base64.b64decode(sentiment_data_b64)
            df = pd.read_excel(io.BytesIO(sentiment_data_bytes))
            print(f"✅ Loaded {len(df)} sentiment records from secrets")
        else:
            print("❌ No sentiment data available locally or in secrets")
            return None
        return df
    except Exception as e:
        print(f"❌ Error loading sentiment data: {e}")
        return None

def load_training_data():
    """Load training data from local file or secrets."""
    try:
        # Try to load from local file first (for development)
        if os.path.exists("data/phd_career_sector_training_data_final_1200.xlsx"):
            df = pd.read_excel("data/phd_career_sector_training_data_final_1200.xlsx")
            print(f"✅ Loaded {len(df)} training records from local file")
        elif 'data_files' in st.secrets and 'training_data' in st.secrets['data_files']:
            # Fallback to secrets (for production)
            import base64
            import io
            training_data_b64 = st.secrets['data_files']['training_data']
            training_data_bytes = base64.b64decode(training_data_b64)
            df = pd.read_excel(io.BytesIO(training_data_bytes))
            print(f"✅ Loaded {len(df)} training records from secrets")
        else:
            print("❌ No training data available locally or in secrets")
            return None
        
        # Check required columns
        if 'profile' not in df.columns or 'target_sector' not in df.columns:
            print(f"""
            Training dataset must contain 'profile' and 'target_sector' columns!
            Found columns: {list(df.columns)}
            """)
            return None
        
        return df
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None

# Load data efficiently
def load_data_once():
    """Load all data from local files."""
    print("🔄 Loading data from local files...")
    
    loaded_data = load_local_data()
    
    print("✅ Data loading complete!")
    return loaded_data

# Load data once
all_data = load_data_once()

# Extract data
df = all_data.get('nyu_data', None)
handshake_events_df = all_data.get('handshake_events', pd.DataFrame())
sentiment_df = all_data.get('sentiment_data', None)

# -----------------------------------------------------------------------------
# Sentiment Analysis Functions ------------------------------------------------
# -----------------------------------------------------------------------------

def calculate_sentiment_with_multiple_models(text):
    """Calculate sentiment using multiple models and return ensemble result. Hypertuned for sarcasm/informal."""
    results = {}
    pre_text = preprocess_text_for_sentiment(text)
    sarcasm = detect_sarcasm(pre_text)

    # TextBlob
    try:
        blob = TextBlob(pre_text)
        results['TextBlob'] = blob.sentiment.polarity
        results['TextBlob_subjectivity'] = blob.sentiment.subjectivity
    except:
        results['TextBlob'] = 0.0
        results['TextBlob_subjectivity'] = 0.0

    # VADER (with custom lexicon)
    if VADER_AVAILABLE:
        try:
            analyzer = SentimentIntensityAnalyzer()
            # Update lexicon for sarcasm cues
            analyzer.lexicon.update(CUSTOM_VADER_LEXICON)
            vader_scores = analyzer.polarity_scores(pre_text)
            results['VADER'] = vader_scores['compound']
        except:
            results['VADER'] = 0.0
    else:
        results['VADER'] = 0.0

    # Dynamic ensemble weighting
    if sarcasm:
        vader_weight = 0.8
        tb_weight = 0.2
    else:
        vader_weight = 0.6
        tb_weight = 0.4

    # If TextBlob is highly subjective but neutral, nudge negative if sarcasm
    tb_score = results.get('TextBlob', 0.0)
    tb_subj = results.get('TextBlob_subjectivity', 0.0)
    if sarcasm and abs(tb_score) < 0.05 and tb_subj > 0.5:
        tb_score = -0.2
        results['TextBlob'] = tb_score

    # Weighted ensemble - always calculate ensemble score
    results['Ensemble'] = vader_weight * results['VADER'] + tb_weight * tb_score

    # If both models are neutral but sarcasm cues, nudge negative
    if sarcasm and abs(results['Ensemble']) < 0.05:
        results['Ensemble'] = -0.2

    return results



def map_division_to_major_group(division):
    """Map detailed academic divisions to three major groups: Humanities, Natural Science, Social Science."""
    
    # Define mapping dictionaries for each major group
    humanities_keywords = [
        'humanities', 'philosophy', 'history', 'literature', 'english', 'languages', 
        'linguistics', 'classics', 'religious', 'theology', 'art', 'music', 'drama',
        'theater', 'film', 'media', 'cultural', 'anthropology', 'archaeology'
    ]
    
    natural_science_keywords = [
        'biology', 'chemistry', 'physics', 'mathematics', 'computer science', 
        'engineering', 'medical', 'health', 'neuroscience', 'biochemistry',
        'molecular', 'genetics', 'ecology', 'environmental', 'geology',
        'astronomy', 'biomedical', 'pharmaceutical', 'biotechnology'
    ]
    
    social_science_keywords = [
        'psychology', 'sociology', 'economics', 'political', 'education',
        "social work", 'public health', 'business', 'management', 'marketing',
        'finance', 'accounting', 'law', 'criminal justice', 'urban planning',
        'international relations', 'communication', 'journalism', 'public policy'
    ]
    
    division_lower = str(division).lower()
    
    # Check for matches in each category
    for keyword in humanities_keywords:
        if keyword in division_lower:
            return 'Humanities'
    
    for keyword in natural_science_keywords:
        if keyword in division_lower:
            return 'Natural Science'
    
    for keyword in social_science_keywords:
        if keyword in division_lower:
            return 'Social Science'
    
    # Default mapping for common patterns
    if any(word in division_lower for word in ['basic medical', 'medical science']):
        return 'Natural Science'
    elif any(word in division_lower for word in ['computer', 'cs', 'software']):
        return 'Natural Science'
    elif any(word in division_lower for word in ['bio', 'biological']):
        return 'Natural Science'
    
    # If no match found, return the original division
    return division

def calculate_sentiment_scores(sentiment_df):
    """Calculate sentiment scores for each major academic group using ensemble model."""
    if sentiment_df is None or sentiment_df.empty:
        print("❌ Sentiment dataframe is None or empty")
        return None, None
    
    # Create a copy to avoid modifying the original
    df_copy = sentiment_df.copy()
    
    # Debug: Print available columns
    print(f"📋 Available columns in sentiment data: {list(df_copy.columns)}")
    
    # Find the comments column (try different possible names)
    comments_column = None
    possible_comment_columns = ['Comments', 'Comment', 'Text', 'Feedback', 'Response', 'Review']
    
    for col in possible_comment_columns:
        if col in df_copy.columns:
            comments_column = col
            print(f"✅ Found comments column: {col}")
            break
    
    if comments_column is None:
        print(f"❌ No comments column found. Available columns: {list(df_copy.columns)}")
        return None, None
    
    # The Academic_Division column already contains the major groups
    # Just use it directly as Major_Group
    if 'Academic_Division' in df_copy.columns:
        df_copy['Major_Group'] = df_copy['Academic_Division']
        print(f"✅ Using Academic_Division column for Major_Group")
    else:
        print(f"❌ No Academic_Division column found. Available columns: {list(df_copy.columns)}")
        return None, None
    
    # Calculate sentiment scores using ensemble model
    sentiment_scores = []
    print(f"🔄 Processing {len(df_copy)} comments for sentiment analysis...")
    
    for i, comment in enumerate(df_copy[comments_column]):
        try:
            if pd.isna(comment) or str(comment).strip() == '':
                sentiment_scores.append(0.0)
            else:
                # Use ensemble model for better accuracy
                ensemble_result = calculate_sentiment_with_multiple_models(comment)
                sentiment_scores.append(ensemble_result['Ensemble'])
                
                # Show progress every 100 comments
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{len(df_copy)} comments...")
                    
        except Exception as e:
            print(f"⚠️ Error processing comment {i}: {e}")
            sentiment_scores.append(0.0)
    
    # Ensure the length matches
    if len(sentiment_scores) != len(df_copy):
        print(f"❌ Sentiment score length mismatch: {len(sentiment_scores)} vs {len(df_copy)}")
        return None, None
    
    df_copy['Sentiment_Score'] = sentiment_scores
    print(f"✅ Added sentiment scores to dataframe")
    
    # Group by major academic groups and calculate average sentiment
    group_sentiments = {}
    print(f"📊 Grouping by major academic groups...")
    
    for group in ['Humanities', 'Natural Science', 'Social Science']:
        group_data = df_copy[df_copy['Major_Group'] == group]
        if not group_data.empty:
            # Include all sentiment scores, including 0.0 (neutral)
            avg_sentiment = group_data['Sentiment_Score'].mean()
            count = len(group_data)
            group_sentiments[group] = {
                'average_sentiment': avg_sentiment,
                'comment_count': count,
                'sentiment_range': (group_data['Sentiment_Score'].min(), group_data['Sentiment_Score'].max())
            }
            print(f"   {group}: {count} comments, avg sentiment: {avg_sentiment:.3f}")
        else:
            print(f"   {group}: No data found")
    
    print(f"✅ Sentiment analysis complete!")
    return group_sentiments, df_copy

# Load sentiment data and evaluate models with caching
@st.cache_data(ttl=3600)
def get_cached_sentiment_scores(sentiment_data):
    """Cache sentiment analysis results."""
    if sentiment_data is not None:
        return calculate_sentiment_scores(sentiment_data)
    return None, None

sentiment_scores, processed_sentiment_df = get_cached_sentiment_scores(sentiment_df)

# --- Sarcasm Macro/Count in Sidebar ---
def sarcasm_macro_count(df):
    """Count sarcastic comments in the dataset."""
    if df is None or df.empty:
        return 0, 0.0
    
    # Find the comments column
    comments_column = None
    possible_comment_columns = ['Comments', 'Comment', 'Text', 'Feedback', 'Response', 'Review']
    
    for col in possible_comment_columns:
        if col in df.columns:
            comments_column = col
            break
    
    if comments_column is None:
        print(f"❌ No comments column found in sarcasm_macro_count")
        return 0, 0.0
    
    sarcasm_count = 0
    total_comments = 0
    
    for comment in df[comments_column]:
        if pd.notna(comment) and str(comment).strip() != '':
            total_comments += 1
            if detect_sarcasm(str(comment)):
                sarcasm_count += 1
    
    sarcasm_percent = (sarcasm_count / total_comments * 100) if total_comments > 0 else 0.0
    return sarcasm_count, sarcasm_percent

# Evaluate model performance (temporarily disabled to fix sentiment scores)
model_evaluation = None

# ADD THE FUNCTION DEFINITIONS HERE (before they're called)
def extract_theme_sentiments(df, themes):
    """Extract sentiment scores for specific themes within each major group."""
    if df is None or df.empty:
        return {}
    
    # Find the comments column
    comments_column = None
    possible_comment_columns = ['Comments', 'Comment', 'Text', 'Feedback', 'Response', 'Review']
    
    for col in possible_comment_columns:
        if col in df.columns:
            comments_column = col
            break
    
    if comments_column is None:
        print(f"❌ No comments column found in extract_theme_sentiments")
        return {}
    
    theme_sentiments = defaultdict(lambda: defaultdict(list))
    
    for _, row in df.iterrows():
        comment = str(row[comments_column]).lower()
        sentiment = row['Sentiment_Score']
        group = row['Major_Group']
        
        for theme in themes:
            if theme in comment:
                theme_sentiments[group][theme].append(sentiment)
    
    # Aggregate and calculate statistics
    theme_summary = {}
    for group, theme_dict in theme_sentiments.items():
        theme_summary[group] = {}
        for theme, scores in theme_dict.items():
            if len(scores) >= 2:  # Only include themes with at least 2 mentions
                avg_sentiment = np.mean(scores)
                count = len(scores)
                min_score = min(scores)
                max_score = max(scores)
                theme_summary[group][theme] = {
                    'average_sentiment': avg_sentiment,
                    'comment_count': count,
                    'sentiment_range': (min_score, max_score),
                    'sentiment_std': np.std(scores)
                }
    
    return theme_summary

def get_representative_quotes(df, group, theme, sentiment_threshold=0.1, n=2):
    """Get representative quotes for a specific theme and group."""
    if df is None or df.empty:
        return []
    
    # Find the comments column
    comments_column = None
    possible_comment_columns = ['Comments', 'Comment', 'Text', 'Feedback', 'Response', 'Review']
    
    for col in possible_comment_columns:
        if col in df.columns:
            comments_column = col
            break
    
    if comments_column is None:
        print(f"❌ No comments column found in get_representative_quotes")
        return []
    
    # Filter comments by group and theme
    filtered = df[
        (df['Major_Group'] == group) & 
        (df[comments_column].str.lower().str.contains(theme, na=False))
    ]
    
    if filtered.empty:
        return []
    
    # Get quotes with significant sentiment (positive or negative)
    significant_quotes = filtered[abs(filtered['Sentiment_Score']) >= sentiment_threshold]
    
    if len(significant_quotes) >= n:
        # Sample from significant quotes
        quotes = significant_quotes.sample(n=min(n, len(significant_quotes)))
    else:
        # If not enough significant quotes, take from all filtered quotes
        quotes = filtered.sample(n=min(n, len(filtered)))
    
    # Return quotes with their sentiment scores
    result = []
    for _, row in quotes.iterrows():
        sentiment_icon = "😊" if row['Sentiment_Score'] > 0.1 else "😐" if row['Sentiment_Score'] > -0.1 else "😟"
        # Increase text length limit to 300 characters
        comment_text = str(row[comments_column])
        if len(comment_text) > 300:
            comment_text = comment_text[:300] + "..."
        result.append({
            'text': comment_text,
            'sentiment_score': row['Sentiment_Score'],
            'icon': sentiment_icon
        })
    
    return result

# NOW CALCULATE THEMATIC SENTIMENTS (after functions are defined) with caching
@st.cache_data(ttl=3600)
def get_cached_theme_sentiments(processed_df):
    """Cache thematic sentiment analysis results."""
    if processed_df is not None:
        themes = ['mentorship', 'job search', 'work-life', 'funding', 'networking', 'career', 'research', 'teaching', 'salary', 'benefits']
        return extract_theme_sentiments(processed_df, themes)
    return None

theme_sentiments = get_cached_theme_sentiments(processed_sentiment_df)

# ADD CONSOLE OUTPUT HERE (after theme_sentiments is defined)
if sentiment_df is not None:
    sarcasm_count, sarcasm_percent = sarcasm_macro_count(sentiment_df)
    print(f"\n{'='*60}")
    print(f"🎭 SENTIMENT ANALYSIS CONSOLE OUTPUT")
    print(f"{'='*60}")
    print(f"📊 Sarcastic Comments Detected: {sarcasm_count} ({sarcasm_percent:.1f}% of all comments)")
    
    if model_evaluation:
        print(f"\n📈 Model Performance Analysis:")
        print(f"{'─'*40}")
        for model_name, stats in model_evaluation['model_stats'].items():
            print(f"  🔹 {model_name}:")
            print(f"    • Mean Sentiment: {stats['mean_sentiment']:.3f}")
            print(f"    • Coverage: {stats['coverage']:.1%}")
            print(f"    • Confidence: {stats['confidence']:.3f}")
            print(f"    • Positive: {stats['positive_ratio']:.1%}")
            print(f"    • Negative: {stats['negative_ratio']:.1%}")
            print(f"    • Neutral: {stats['neutral_ratio']:.1%}")
        
        if model_evaluation['model_consistency']:
            print(f"\n🤝 Model Agreement:")
            print(f"{'─'*20}")
            for consistency_name, agreement_rate in model_evaluation['model_consistency'].items():
                print(f"    • {consistency_name}: {agreement_rate:.1%}")
        
        print(f"\n📈 Total Comments Analyzed: {model_evaluation['total_comments']}")
        
        if VADER_AVAILABLE:
            print("✅ Using Weighted Ensemble model (VADER 60% + TextBlob 40%)")
        else:
            print("⚠️ Using TextBlob only. Install VADER for better accuracy")
    
    if sentiment_scores:
        print(f"\n🎯 Sentiment Scores by Major Academic Group:")
        print(f"{'─'*45}")
        for group, data in sentiment_scores.items():
            sentiment_emoji = "😊" if data['average_sentiment'] > 0.1 else "😐" if data['average_sentiment'] > -0.1 else "😟"
            print(f"    {sentiment_emoji} {group}: {data['average_sentiment']:.3f} ({data['comment_count']} comments)")
    
    if theme_sentiments:
        print(f"\n🎯 Thematic Sentiment Analysis:")
        print(f"{'─'*30}")
        total_themes_found = 0
        for group, themes in theme_sentiments.items():
            if themes:  # Only show groups that have themes
                print(f"    📊 {group}:")
                for theme, data in themes.items():
                    sentiment_emoji = "😊" if data['average_sentiment'] > 0.2 else "😐" if data['average_sentiment'] > -0.2 else "😟"
                    print(f"      • {theme.title()}: {sentiment_emoji} {data['average_sentiment']:.3f} ({data['comment_count']} comments)")
                    total_themes_found += 1
        
        if total_themes_found == 0:
            print("    ⚠️ No thematic sentiment data found. This might be due to:")
            print("      - No comments containing the specified themes")
            print("      - Comments not matching the theme keywords")
            print("      - Data processing issues")
        else:
            print(f"    ✅ Found {total_themes_found} theme categories with sentiment data")
    
    print(f"\n{'='*60}")
    print(f"🏁 END SENTIMENT ANALYSIS")
    print(f"{'='*60}")

def get_sentiment_insights(selected_division, sentiment_scores):
    """Generate sentiment insights for career exploration."""
    if sentiment_scores is None:
        return ""
    
    # Map the selected division to major group
    major_group = map_division_to_major_group(selected_division)
    
    if major_group in sentiment_scores:
        sentiment_data = sentiment_scores[major_group]
        avg_sentiment = sentiment_data['average_sentiment']
        count = sentiment_data['comment_count']
        
        # Generate insights based on sentiment score
        if avg_sentiment > 0.2:
            sentiment_insight = f"Positive sentiment ({avg_sentiment:.2f}) in {major_group} suggests strong satisfaction with career outcomes."
        elif avg_sentiment < -0.2:
            sentiment_insight = f"Negative sentiment ({avg_sentiment:.2f}) in {major_group} indicates potential challenges in career transitions."
        else:
            sentiment_insight = f"Neutral sentiment ({avg_sentiment:.2f}) in {major_group} suggests mixed career experiences."
        
        # Add model information
        model_info = "**Sentiment Analysis Models Used:**\n"
        if VADER_AVAILABLE:
            model_info += "• **Ensemble Model**: Weighted combination of VADER (60%) and TextBlob (40%)\n"
            model_info += "• **VADER**: Optimized for social media and informal text\n"
            model_info += "• **TextBlob**: General-purpose sentiment analysis\n"
        else:
            model_info += "• **TextBlob**: General-purpose sentiment analysis\n"
            model_info += "• **Recommendation**: Install VADER for improved accuracy\n"
        
        return f"""
{model_info}
**Sentiment Analysis for {major_group}:**
• **Average Sentiment Score**: {avg_sentiment:.3f} (based on {count} comments)
• **Sentiment Range**: {sentiment_data['sentiment_range'][0]:.3f} to {sentiment_data['sentiment_range'][1]:.3f}
• **Career Insight**: {sentiment_insight}
        """
    
    return f"No sentiment data available for {major_group}."



# -----------------------------------------------------------------------------
# Thematic Sentiment Analysis Functions ---------------------------------------
# -----------------------------------------------------------------------------

def get_theme_insights_for_division(selected_division, theme_sentiments, processed_sentiment_df):
    """Generate insights about themes for a specific division."""
    if theme_sentiments is None or processed_sentiment_df is None:
        return ""
    
    major_group = map_division_to_major_group(selected_division)
    
    if major_group not in theme_sentiments:
        return f"No thematic data available for {major_group}."
    
    insights = []
    quotes_section = []
    
    # Define themes to analyze
    themes = ['mentorship', 'job search', 'work-life', 'funding', 'networking', 'career', 'research', 'teaching', 'salary', 'benefits']
    
    for theme in themes:
        if theme in theme_sentiments[major_group]:
            theme_data = theme_sentiments[major_group][theme]
            avg_sentiment = theme_data['average_sentiment']
            count = theme_data['comment_count']
            
            # Generate insight based on sentiment
            if avg_sentiment > 0.2:
                insight = f"**{theme.title()}** has positive sentiment ({avg_sentiment:.2f}) - students are generally satisfied."
            elif avg_sentiment < -0.2:
                insight = f"**{theme.title()}** has negative sentiment ({avg_sentiment:.2f}) - this is a common concern area."
            else:
                insight = f"**{theme.title()}** has neutral sentiment ({avg_sentiment:.2f}) - mixed experiences reported."
            
            insights.append(f"- {insight} (based on {count} comments)")
            
            # Get representative quotes
            quotes = get_representative_quotes(processed_sentiment_df, major_group, theme, n=1)
            if quotes:
                quote = quotes[0]
                quotes_section.append(f"**{theme.title()}**: {quote['icon']} \"{quote['text']}\"")
    
    # Combine insights and quotes
    result = f"**Thematic Analysis for {major_group}:**\n"
    result += "\n".join(insights)
    
    if quotes_section:
        result += "\n\n**Representative Quotes:**\n"
        result += "\n".join(quotes_section)
    
    return result

# Define the custom pipeline class at module level
class LabelEncodingPipeline(Pipeline):
    def __init__(self, steps, label_encoder):
        super().__init__(steps)
        self.label_encoder = label_encoder

    def predict(self, X):
        # Get numerical predictions
        y_pred = super().predict(X)
        # Convert back to original labels
        return self.label_encoder.inverse_transform(y_pred)



# Update the train_text_sector_model function to use local files
def train_text_sector_model():
    try:
        # Use local file instead of Google Sheets
        training_data = load_training_data()
        if training_data is None:
            st.error("Could not load training data from local file")
            return None
        
        # Verify the required columns exist
        if 'profile' not in training_data.columns or 'target_sector' not in training_data.columns:
            st.error(f"""
            Training dataset must contain 'profile' and 'target_sector' columns!
            Found columns: {list(training_data.columns)}
            """)
            return None
            
        # Clean the data
        original_count = len(training_data)
        training_data = training_data.dropna(subset=['profile', 'target_sector'])
        training_data['profile'] = training_data['profile'].astype(str)
        training_data['target_sector'] = training_data['target_sector'].astype(str)
        
        # Remove empty profiles
        training_data = training_data[training_data['profile'].str.strip() != '']
        
        final_count = len(training_data)
        if final_count < original_count:
            print(f"⚠️ Removed {original_count - final_count} rows with missing data")
        
        if len(training_data) == 0:
            st.error("Training data is empty after cleaning. Please check your local file.")
            return None
        
        # Store dataset info for debugging (only show if needed)
        sector_counts = training_data['target_sector'].value_counts()
        unique_sectors = training_data['target_sector'].unique()
        
        X = training_data["profile"]
        y = training_data["target_sector"]

        # Add label encoding for the target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Store the label encoder for later use
        label_encoder = label_encoder

        # Define the vectorizer with improved parameters
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

        # Transform the text data
        try:
            X_transformed = vectorizer.fit_transform(X)
        except Exception as e:
            st.error(f"Error during vectorization: {e}")
            return None

        # Define different models to compare with optimized parameters
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softmax',
                num_class=len(label_encoder.classes_),
                random_state=42
            ),
            'SVM': SVC(
                kernel='linear',
                C=1.0,
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Naive Bayes': MultinomialNB(
                alpha=0.1
            )
        }

        # Determine appropriate cross-validation folds based on data size
        n_samples = len(y_encoded)
        unique_classes = len(np.unique(y_encoded))
        min_samples_per_class = min([np.sum(y_encoded == i) for i in np.unique(y_encoded)])
        
        # Use fewer folds if we have limited data
        if min_samples_per_class < 5:
            cv_folds = min(3, min_samples_per_class)
            print(f"⚠️ Limited data: using {cv_folds}-fold CV instead of 5-fold")
        else:
            cv_folds = 5
        
        # Perform cross-validation for each model
        cv_scores = {}
        best_model = None
        best_score = -1
        model_details = {}

        # Debug information
        print("### Model Training Progress")

        for i, (name, model) in enumerate(models.items()):
            try:
                print(f"Training {name}...")
                # Perform cross-validation with appropriate number of folds
                scores = cross_val_score(model, X_transformed, y_encoded, cv=cv_folds, scoring='accuracy')
                mean_score = scores.mean()
                std_score = scores.std()
                
                # Store detailed scores
                cv_scores[name] = {
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'scores': scores.tolist(),
                    'min_score': scores.min(),
                    'max_score': scores.max()
                }
                
                # Store model details
                model_details[name] = {
                    'parameters': model.get_params(),
                    'type': model.__class__.__name__
                }
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                
                # Update progress
                progress = (i + 1) / len(models)
                print(f"Completed {name}: {mean_score:.3f} ± {std_score:.3f}")
                
            except Exception as e:
                print(f"Error in {name} model: {str(e)}")
                continue

        print("Model training complete!")

        if best_model is None:
            print("All models failed to train. Falling back to Random Forest.")
            try:
                best_model = models['Random Forest']
                best_model.fit(X_transformed, y_encoded)
            except Exception as e:
                print(f"Random Forest model failed to train: {e}")
                # Try with simpler parameters for very small datasets
                try:
                    simple_model = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=5,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42
                    )
                    simple_model.fit(X_transformed, y_encoded)
                    best_model = simple_model
                    print("⚠️ Using simplified Random Forest model for small dataset")
                except Exception as e2:
                    print(f"Even simplified model failed: {e2}")
                    return None
        else:
            # Train the best model on the full dataset
            try:
                best_model.fit(X_transformed, y_encoded)
            except Exception as e:
                print(f"Best model failed to fit: {e}")
                return None

        # Create the pipeline with label encoding support
        try:
            pipeline = LabelEncodingPipeline(
                steps=[
                    ("tfidf", vectorizer),
                    ("clf", best_model)
                ],
                label_encoder=label_encoder
            )
        except Exception as e:
            print(f"Failed to create pipeline: {e}")
            return None

        # Store comprehensive model comparison results
        model_comparison = {
            'cv_scores': cv_scores,
            'model_details': model_details,
            'best_model': best_model.__class__.__name__,
            'best_score': best_score,
            'label_encoder': label_encoder,
            'vectorizer_params': vectorizer.get_params(),
            'training_samples': len(training_data),
            'sector_distribution': sector_counts.to_dict(),
            'dataset_path': 'Local File - Training Data'
        }
        


        
        return pipeline
        
    except Exception as e:
        st.error(f"Error loading training dataset: {str(e)}")
        return None

# Initialize the model at the start of the app
# Model will be loaded fresh each time

# -----------------------------------------------------------------------------
# ALL REMAINING ORIGINAL LOGIC starts here (events dict, Adzuna helpers, class,
# tabs, main). Paste unchanged below this comment.
# -----------------------------------------------------------------------------

# ... (original script continues) ...

# ----------------------------------------------------------------------------
# Handshake Events with Images (Unmodified)
# ----------------------------------------------------------------------------
def get_handshake_events_for_division(division, events_df):
    if events_df is None or events_df.empty:
        return []
    # Filter by division (case-insensitive)
    filtered = events_df[events_df['Division'].str.lower() == division.lower()]
    # Convert to list of dicts
    return filtered.to_dict(orient='records')

# ----------------------------------------------------------------------------
# Adzuna Real-Time Job Fetch
# ----------------------------------------------------------------------------
def fetch_adzuna_jobs(query="data scientist", location="New York"):
    url = "https://api.adzuna.com/v1/api/jobs/us/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": 5,
        "what": query,
        "where": location,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Adzuna API error: {response.text}")
        return None

def display_adzuna_jobs(job_data):
    if not job_data or 'results' not in job_data:
        st.write("No opportunities found in this area.")
        return
    for job in job_data['results']:
        title = job.get("title", "N/A")
        company_dict = job.get("company", {})
        company = company_dict.get("display_name", "N/A")
        location_dict = job.get("location", {})
        job_location = location_dict.get("display_name", "N/A")
        description = job.get("description", "")
        redirect_url = job.get("redirect_url", "#")
        salary_min = job.get("salary_min", "")
        salary_max = job.get("salary_max", "")
        if salary_min and salary_max:
            salary_range = f"${int(salary_min):,} - ${int(salary_max):,}"
        else:
            salary_range = "N/A"

        st.markdown(f"""
        <div class='opportunity-card'>
            <h4>{title}</h4>
            <p><strong>Company:</strong> {company}</p>
            <p><strong>Location:</strong> {job_location}</p>
            <p><strong>Compensation:</strong> {salary_range}</p>
            <p>{description[:200]}...</p>
            <a href='{redirect_url}' target='_blank'>Research This Opportunity</a>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# NYUCareerAdvisor Class
# ----------------------------------------------------------------------------
class NYUCareerAdvisor:
    def __init__(self, df):
        self.df = df
        self._cache = {}  # Simple cache for API responses

    def query_openai(self, messages, max_tokens=1000, temperature=0.5):
        if not use_openai:
            return "[OpenAI API call skipped for testing.]"
        
        # Create cache key from messages
        cache_key = str(messages) + str(max_tokens) + str(temperature)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Add timeout to prevent hanging
            import time
            start_time = time.time()
            
            # Use the OpenAI API format (compatible with both old and new versions)
            try:
                # Try new format first
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            except AttributeError:
                # Fallback to old format
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            # Check if we're taking too long
            if time.time() - start_time > 10:  # Warning at 10 seconds
                st.warning("⚠️ API call is taking longer than expected...")
            
            result = response.choices[0].message.content
            # Cache the result
            self._cache[cache_key] = result
            return result
        except Exception as e:
            st.error(f"❌ Error querying OpenAI: {e}")
            # Return a fallback response instead of None
            return "[OpenAI API call failed - using fallback data]"

    def summarize_text_column(self, series, sample_size=3):
        if not use_openai:
            return "[OpenAI summary skipped for testing.]"
        text_data = series.dropna().astype(str).tolist()
        if not text_data:
            return "No data found."
        limited = text_data[:sample_size]
        combined = "\n".join(limited)
        prompt = f"""
        You are a data analyst. Summarize key themes in these text samples:
        ---
        {combined}
        ---
        Keep it brief.
        """
        messages = [
            {"role": "system", "content": "You are a concise data summarizer."},
            {"role": "user", "content": prompt}
        ]
        return self.query_openai(messages, max_tokens=300) or "No summary returned."

    def find_relevant_columns(self, df):
        relevant_keywords = [
            "funding", "scholarship", "certification", "company", "job",
            "salary", "recommendation", "skills", "industry", "position"
        ]
        relevant_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in relevant_keywords):
                relevant_cols.append(col)
        return relevant_cols

    # Method to infer approximate RIASEC scores from user text
    def infer_riasec_scores_via_gpt(self, user_text):
        if not use_openai:
            # Return a dummy RIASEC score for testing
            return {
                "Realistic": 3,
                "Investigative": 5,
                "Artistic": 2,
                "Social": 4,
                "Enterprising": 3,
                "Conventional": 2
            }
        
        # Use a simpler, faster prompt for Streamlit Cloud
        prompt = f"""
        Based on this text, estimate RIASEC scores (1-7):
        {user_text[:500]}  # Limit text length for speed
        
        Return JSON: {{"Realistic":3,"Investigative":5,"Artistic":2,"Social":4,"Enterprising":3,"Conventional":2}}
        """
        messages = [
            {"role": "system", "content": "Vocational psychologist. Return only JSON."},
            {"role": "user", "content": prompt}
        ]
        raw = self.query_openai(messages, max_tokens=100)  # Further reduced tokens for speed
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            riasec_dims = ["Realistic","Investigative","Artistic","Social","Enterprising","Conventional"]
            scores = {}
            for d in riasec_dims:
                val = data.get(d, 1)  # default 1 if missing
                if isinstance(val, (int,float)):
                    scores[d] = max(1, min(7, int(val)))  # clamp 1-7
                else:
                    scores[d] = 1
            return scores
        except json.JSONDecodeError:
            return {}

    def generate_career_advice(
        self,
        selected_division,
        citizenship,
        user_goals,
        user_tech_skills,
        work_env,
        research_interests,
        desired_industry,
        work_life_balance,
        cohort_stage,
        family_location,
        job_function=None,
        desirability=None,
        user_riasec=None,
        sentiment_scores=None,
        theme_sentiments=None,
        processed_sentiment_df=None
    ):
        if self.df is None:
            return "No dataset loaded. Please upload an Excel file."

        # Original comprehensive logic
        filtered_df = self.df.copy()
        if "academic_division" in filtered_df.columns:
            if selected_division != "(No academic_division column)":
                filtered_df = filtered_df[filtered_df["academic_division"].astype(str) == selected_division]
                if filtered_df.empty:
                    return f"No rows found for academic_division = {selected_division}"

        data_insights = []
        all_columns = filtered_df.columns.tolist()

        for col in all_columns:
            col_data = filtered_df[col].dropna()
            if col_data.empty:
                continue
            if pd.api.types.is_numeric_dtype(col_data):
                mean_val = col_data.mean()
                data_insights.append(f"Column '{col}': average={round(mean_val, 2)}")
            else:
                avg_len = col_data.astype(str).map(len).mean()
                if avg_len > 40:
                    summary = self.summarize_text_column(col_data, sample_size=3)
                    data_insights.append(f"Column '{col}' (text summary): {summary}")
                else:
                    top_vals = col_data.value_counts().head(3).index.tolist()
                    data_insights.append(f"Column '{col}' top responses: {top_vals}")

        relevant_cols = self.find_relevant_columns(self.df)
        relevant_summaries = []
        for rcol in relevant_cols:
            rdata = self.df[rcol].dropna()
            if rdata.empty:
                continue
            if pd.api.types.is_numeric_dtype(rdata):
                mean_val = rdata.mean()
                relevant_summaries.append(f"Relevant Column '{rcol}': average={round(mean_val, 2)}")
            else:
                avg_len = rdata.astype(str).map(len).mean()
                if avg_len > 40:
                    summary = self.summarize_text_column(rdata, sample_size=2)
                    relevant_summaries.append(f"Relevant Column '{rcol}' (text summary): {summary}")
                else:
                    top_vals = rdata.value_counts().head(2).index.tolist()
                    relevant_summaries.append(f"Relevant Column '{rcol}' top: {top_vals}")

        relevant_cols_str = "\n".join(relevant_summaries) or "No additional relevant columns found."

        crucial_cols = [
            "Job Relatedness", "Salary", "Benefits", "Job security",
            "Job location", "Opportunity_for_advancement", "Intellectual_challenge",
            "Level_of_responsibility", "Degree_of_independence", "Contribution_to_society",
            "Work-life_balance", "Prestige_of_employer",
            "Prestige_of_position/job title", "Change_in_career"
        ]
        factor_insights = []
        for col in crucial_cols:
            if col in filtered_df.columns:
                col_data = filtered_df[col].dropna()
                if not col_data.empty:
                    if pd.api.types.is_numeric_dtype(col_data):
                        avg_val = round(col_data.mean(), 2)
                        factor_insights.append(f"{col}: average rating = {avg_val}")
                    else:
                        top_vals = col_data.value_counts().head(2).index.tolist()
                        factor_insights.append(f"{col}: top responses = {top_vals}")
        factor_insights_str = "\n".join(factor_insights) or "No crucial column data available in the filtered set."

        numeric_cols = filtered_df.select_dtypes(include=['int','float']).columns
        if len(numeric_cols) > 1:
            stds = filtered_df[numeric_cols].std().abs().sort_values(ascending=False)
            top_cols = stds.head(5).index
            corr_matrix = filtered_df[top_cols].corr().round(2)
            data_insights.append("Limited Correlation Matrix:\n" + corr_matrix.to_string())

        insights_str = "\n".join(data_insights) or "No data insights found after filtering."

        # Income field removed - all students are on fellowship

        dataset_certifications = []
        dataset_companies = []
        if "recommended_certifications" in filtered_df.columns:
            dataset_certifications = list(filtered_df["recommended_certifications"].dropna().unique())
        if "target_companies" in filtered_df.columns:
            dataset_companies = list(filtered_df["target_companies"].dropna().unique())

        if dataset_certifications:
            cert_prompt = f"\nFrom the dataset, recommended certifications: {dataset_certifications}"
        else:
            cert_prompt = "\nNo certification data in the dataset. Provide typical industry certifications."

        if dataset_companies:
            comp_prompt = f"\nFrom the dataset, potential target companies: {dataset_companies}"
        else:
            comp_prompt = "\nNo company data in the dataset. Provide top industry companies."

        predicted_sector, ml_method = predict_sector_from_text(user_goals, desired_industry, work_env)
        # Store the ML method prediction for CSV export
        st.session_state.ml_model_prediction = ml_method

        user_profile_str = (
            f"PhD Cohort Stage: {cohort_stage}\n"
            f"Citizenship: {citizenship}\n"
            f"Career Goals: {user_goals}\n"
            f"Predicted Sector (via ML): {predicted_sector}\n"
            f"Soft Skills: {user_tech_skills}\n"
            f"Roles You'd Like to Explore: {job_function}\n"
            f"Work Environment: {work_env}\n"
            f"Research Interests: {research_interests}\n"
            f"Where do you see your career after your PhD?: {desired_industry}\n"
            f"Work-Life Balance (1-5): {work_life_balance}\n"
            f"Selected academic_division: {selected_division}\n"
            f"Market Demand: {desirability}\n"
            f"Family Location: {family_location}\n"
        )

        

        # If we do have RIASEC info, add it
        if user_riasec:
            user_profile_str += "\nRIASEC Personality Profile:\n"
            # Sort by score and find top 3
            riasec_list = sorted(user_riasec.items(), key=lambda x: x[1], reverse=True)
            top_riasec = [c for (c,score) in riasec_list[:3]]
            
            for dim,score in riasec_list:
                user_profile_str += f"{dim}: {score}/7\n"
            user_profile_str += f"\nTop RIASEC Codes: {', '.join(top_riasec)}\n"
            
            # Enhanced RIASEC-based career insights
            riasec_insights = []
            for dimension, score in riasec_list:
                if score >= 6.5:
                    riasec_insights.append(f"Very Strong {dimension} preference (score: {score}) - EXCELLENT for {dimension.lower()}-oriented careers")
                elif score >= 5.5:
                    riasec_insights.append(f"Strong {dimension} preference (score: {score}) - highly recommended for {dimension.lower()}-oriented careers")
                elif score >= 4.5:
                    riasec_insights.append(f"Moderate {dimension} preference (score: {score}) - well-suited for {dimension.lower()}-oriented roles")
                elif score >= 3.5:
                    riasec_insights.append(f"Developing {dimension} preference (score: {score}) - potential for growth in {dimension.lower()}-oriented areas")
                else:
                    riasec_insights.append(f"Emerging {dimension} preference (score: {score}) - area for development and exploration")
            
            user_profile_str += "\nRIASEC Career Insights:\n" + "\n".join(riasec_insights) + "\n"
            
            # Add specific career pathway recommendations based on top RIASEC codes
            user_profile_str += "\nRecommended Career Pathways Based on RIASEC:\n"
            career_mapping = create_riasec_career_mapping()
            for i, (dim, score) in enumerate(riasec_list[:3], 1):
                if dim in career_mapping:
                    mapping = career_mapping[dim]
                    user_profile_str += f"{i}. {dim} ({score}/7): {', '.join(mapping['careers'][:3])} in {', '.join(mapping['sectors'])}\n"

        if cohort_stage.lower() == "first year":
            year_instructions = (
                "If the user is in their first year, they must receive relevant course recommendations."
            )
        else:
            year_instructions = (
                "If the user is in All but Dissertation (ABD), do NOT recommend new courses. "
                "Focus on certifications, finishing the dissertation, and bridging to the next career steps."
            )
    # Inject feedback learning (avoid mistakes)
            avoid_instructions = ""
            if q_agent and selected_division in q_agent:
                bad_actions = q_agent[selected_division]
                low_quality_areas = [area for area, count in bad_actions.items() if count >= 1]
                if low_quality_areas:
                    avoid_instructions = (
                        "IMPORTANT: Based on prior user feedback, avoid the following mistakes: "
                        + ", ".join(low_quality_areas) + ".\n\n"
                    )

            # Prepend avoid instructions before normal prompt
            prompt = avoid_instructions + f"""
            You are an NYU career advisor..."""

        # Get sentiment insights for the selected division
        sentiment_insights = get_sentiment_insights(selected_division, sentiment_scores)

        # Get thematic insights for the selected division
        thematic_insights = get_theme_insights_for_division(selected_division, theme_sentiments, processed_sentiment_df)

        # --- NEW: Summarize most relevant theme sentiment for this division ---
        # Find the most positive and most negative theme for the division
        major_group = map_division_to_major_group(selected_division)
        key_theme_summary = ""
        if theme_sentiments and major_group in theme_sentiments:
            group_themes = theme_sentiments[major_group]
            if group_themes:
                # Sort by sentiment
                sorted_themes = sorted(group_themes.items(), key=lambda x: x[1]['average_sentiment'], reverse=True)
                top_theme, top_data = sorted_themes[0]
                bottom_theme, bottom_data = sorted_themes[-1]
                key_theme_summary = (
                    f"Key Theme Sentiment for {major_group}:\n"
                    f"- Most Positive: '{top_theme.title()}' ({top_data['average_sentiment']:.2f}) with {top_data['comment_count']} comments.\n"
                    f"- Most Negative: '{bottom_theme.title()}' ({bottom_data['average_sentiment']:.2f}) with {bottom_data['comment_count']} comments.\n"
                )
        
        # --- NEW: Add explicit instructions for LLM to reference these scores ---
        prompt = f"""
        You are an NYU career advisor specializing in PhD career pathways. Incorporate the data below into a set of 
        short-term (0-3 months), medium-term (3-12 months), and long-term (1+ years) 
        career insights. 

        **CRITICAL: RIASEC ANALYSIS MUST BE CENTRAL TO RECOMMENDATIONS**
        - Use the RIASEC scores to identify the best career pathways
        - Match high-scoring dimensions to specific career opportunities
        - Consider how RIASEC preferences align with different sectors and roles
        - Provide specific examples of careers that match their top RIASEC codes

        **MUST REFERENCE** the crucial columns (Job location, Salary, Benefits, etc.) 
        with the given average ratings or top responses. Clearly tie these columns 
        to the recommended actions.

        --- USER PROFILE ---
        {user_profile_str}

        --- KEY SENTIMENT/THEME INSIGHTS ---
        {key_theme_summary}

        --- SENTIMENT ANALYSIS INSIGHTS ---
        {sentiment_insights}

        --- THEMATIC SENTIMENT ANALYSIS ---
        {thematic_insights}

        --- FILTERED DATA INSIGHTS ---
        {insights_str}

        --- KEY JOB FACTORS (CRUCIAL COLUMNS) ---
        {factor_insights_str}

        --- ADDITIONAL RELEVANT COLUMNS (FULL DATASET) ---
        {relevant_cols_str}

        --- CERTIFICATIONS INFO ---
        {cert_prompt}

        --- TARGET COMPANIES INFO ---
        {comp_prompt}

        IMPORTANT POINTS:
        - {year_instructions}
        - If the user has family in a specific location, prefer roles or suggestions near that location if relevant.
        - Provide suggestions about skill-building, scholarships/funding, and in-demand certifications.
        - Mention 'Job location', 'Salary', 'Benefits', etc. in short-term, medium-term, and long-term recommendations.
        - **FOR EACH RECOMMENDATION SECTION (Short/Medium/Long-term), YOU MUST INCLUDE:**
          a) **RIASEC Justification**: Explain how the recommendation aligns with their top RIASEC dimensions
          b) **Sentiment Justification**: Reference at least one relevant sentiment or theme score
        - **RIASEC-CAREER MAPPING**: Provide specific career examples for their top RIASEC codes:
          * Realistic: Engineering, construction, agriculture, technical trades
          * Investigative: Research, data science, academia, scientific consulting
          * Artistic: Design, creative writing, media, arts administration
          * Social: Teaching, counseling, healthcare, community development
          * Enterprising: Entrepreneurship, sales, management, politics
          * Conventional: Finance, accounting, administration, compliance
        - **INCORPORATE SENTIMENT ANALYSIS**: Use the sentiment insights to provide more nuanced career advice. If sentiment is positive, emphasize opportunities and success stories. If negative, address potential challenges and provide strategies to overcome them.
        - **INCORPORATE THEMATIC ANALYSIS**: Use the thematic insights to address specific concerns or highlight positive aspects mentioned by students in the same division. Reference the representative quotes when relevant to make advice more relatable and data-driven.
        - Keep it concise, structured, and actionable.

        Now produce a bullet-point style recommendation with three sections (Short-term, Medium-term, Long-term). For each section, include a 'Sentiment Justification' bullet that references the most relevant sentiment or theme score for this division.
        """

        messages = [
            {
                "role": "system", 
                "content": "You are a helpful NYU career advisor who references the user data and dataset insights accurately."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        return self.query_openai(messages)

# ----------------------------------------------------------------------------
# RIASEC Career Recommendation Model
# ----------------------------------------------------------------------------
def train_riasec_career_model(riasec_data_path):
    """
    Train a model to predict career pathways based on RIASEC scores.
    This function can be enhanced with your actual RIASEC dataset.
    """
    try:
        # Load RIASEC data if available
        if os.path.exists(riasec_data_path):
            riasec_df = pd.read_excel(riasec_data_path)
            print(f"✅ Loaded {len(riasec_df)} RIASEC records")
            
            # 🔧 ADD YOUR RIASEC DATASET PROCESSING HERE
            # Example of what you can do with your dataset:
            # 1. Analyze survey responses for each dimension
            # 2. Train ML models to predict career pathways
            # 3. Create custom career mappings based on your data
            # 4. Extract specific skills and requirements
            
            print("📊 RIASEC Dataset Columns:", list(riasec_df.columns))
            print("📊 Sample RIASEC Data:")
            print(riasec_df.head())
            
            # This is where you would implement the actual model training
            # For now, return a simple mapping function
            return create_riasec_career_mapping()
        else:
            print(f"⚠️ RIASEC data file not found: {riasec_data_path}")
            return create_riasec_career_mapping()
            
    except Exception as e:
        print(f"❌ Error training RIASEC model: {e}")
        return create_riasec_career_mapping()

def create_riasec_career_mapping():
    """
    Create a mapping of RIASEC dimensions to career pathways.
    This can be enhanced with your actual dataset analysis.
    """
    career_mapping = {
        "Realistic": {
            "careers": ["Engineering", "Construction", "Agriculture", "Technical Trades", "Manufacturing", "Architecture"],
            "skills": ["Technical skills", "Hands-on work", "Problem-solving", "Attention to detail"],
            "sectors": ["Industry", "Government", "Technical Services"]
        },
        "Investigative": {
            "careers": ["Research Scientist", "Data Scientist", "Academic Professor", "Consultant", "Analyst", "Policy Researcher"],
            "skills": ["Analytical thinking", "Research methods", "Data analysis", "Critical thinking"],
            "sectors": ["Academia", "Research Institutions", "Consulting", "Government"]
        },
        "Artistic": {
            "careers": ["Designer", "Creative Director", "Content Creator", "Arts Administrator", "Media Producer", "Writer"],
            "skills": ["Creativity", "Design thinking", "Communication", "Innovation"],
            "sectors": ["Creative Industries", "Media", "Arts", "Marketing"]
        },
        "Social": {
            "careers": ["Teacher", "Counselor", "Healthcare Professional", "Community Developer", "Trainer", "Social Worker"],
            "skills": ["Communication", "Empathy", "Teaching", "Interpersonal skills"],
            "sectors": ["Education", "Healthcare", "Non-profit", "Social Services"]
        },
        "Enterprising": {
            "careers": ["Entrepreneur", "Business Leader", "Sales Manager", "Policy Maker", "Consultant", "Executive"],
            "skills": ["Leadership", "Communication", "Strategic thinking", "Networking"],
            "sectors": ["Business", "Entrepreneurship", "Politics", "Consulting"]
        },
        "Conventional": {
            "careers": ["Financial Analyst", "Accountant", "Administrator", "Compliance Officer", "Project Manager", "Data Manager"],
            "skills": ["Organization", "Attention to detail", "Analytical skills", "Process management"],
            "sectors": ["Finance", "Administration", "Government", "Corporate"]
        }
    }
    return career_mapping

def get_riasec_career_recommendations(riasec_scores, career_mapping, riasec_analysis=None):
    """
    Generate enhanced career recommendations based on RIASEC scores and analysis.
    """
    recommendations = []
    
    # Sort dimensions by score
    sorted_dimensions = sorted(riasec_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 3 dimensions for better analysis
    top_dimensions = sorted_dimensions[:3]
    
    for dimension, score in top_dimensions:
        if dimension in career_mapping:
            mapping = career_mapping[dimension]
            
            # Enhanced recommendation based on score strength and confidence
            if score >= 6.5:
                strength = "very strong"
                confidence = "excellent match"
                priority = "primary"
            elif score >= 5.5:
                strength = "strong"
                confidence = "highly recommended"
                priority = "primary"
            elif score >= 4.5:
                strength = "moderate"
                confidence = "well-suited"
                priority = "secondary"
            elif score >= 3.5:
                strength = "developing"
                confidence = "potential for growth"
                priority = "developmental"
            else:
                strength = "emerging"
                confidence = "area for development"
                priority = "developmental"
            
            # Add confidence analysis if available
            confidence_analysis = ""
            if riasec_analysis and dimension in riasec_analysis:
                analysis = riasec_analysis[dimension]
                if analysis['confidence'] == 'high':
                    confidence_analysis = " (high consistency in responses)"
                elif analysis['confidence'] == 'medium':
                    confidence_analysis = " (moderate consistency in responses)"
                else:
                    confidence_analysis = " (low consistency - consider retaking assessment)"
            
            recommendation = {
                "dimension": dimension,
                "score": score,
                "strength": strength,
                "confidence": confidence + confidence_analysis,
                "priority": priority,
                "careers": mapping["careers"],
                "skills": mapping["skills"],
                "sectors": mapping["sectors"],
                "raw_scores": list(riasec_analysis[dimension].get('raw_scores', [])) if riasec_analysis and dimension in riasec_analysis else []
            }
            recommendations.append(recommendation)
    
    return recommendations

# Initialize RIASEC career model after functions are defined
riasec_career_model = train_riasec_career_model(RIASEC_DATASET_PATH)
print(f"✅ RIASEC career model initialized with dataset: {RIASEC_DATASET_PATH}")

# ----------------------------------------------------------------------------
# Explore Your Options Tab
# ----------------------------------------------------------------------------
# Sample Data Generation Functions (when AI is disabled)
# ----------------------------------------------------------------------------

def generate_sample_riasec_scores(user_profile_text):
    """Generate sample RIASEC scores based on user profile text."""
    # Simple keyword-based scoring
    text = user_profile_text.lower()
    
    scores = {
        "Realistic": 3,      # Default scores
        "Investigative": 4,
        "Artistic": 3,
        "Social": 4,
        "Enterprising": 3,
        "Conventional": 3
    }
    
    # Adjust based on keywords
    if any(word in text for word in ['research', 'analysis', 'data', 'technical']):
        scores["Investigative"] += 1
    if any(word in text for word in ['teaching', 'mentoring', 'helping', 'community']):
        scores["Social"] += 1
    if any(word in text for word in ['leadership', 'management', 'entrepreneur', 'business']):
        scores["Enterprising"] += 1
    if any(word in text for word in ['creative', 'design', 'art', 'writing']):
        scores["Artistic"] += 1
    if any(word in text for word in ['organization', 'planning', 'detail', 'structure']):
        scores["Conventional"] += 1
    if any(word in text for word in ['hands-on', 'practical', 'technical', 'building']):
        scores["Realistic"] += 1
    
    # Ensure scores stay within 1-7 range
    for key in scores:
        scores[key] = max(1, min(7, scores[key]))
    
    return scores

def generate_sample_career_advice(**kwargs):
    """Generate sample career advice when AI is disabled."""
    predicted_sector = kwargs.get('predicted_sector', 'Industry')
    user_goals = kwargs.get('user_goals', '')
    desired_industry = kwargs.get('desired_industry', 'Industry')
    work_env = kwargs.get('work_env', 'Collaborative')
    
    # Generate sector-specific advice
    sector_advice = {
        'Academic': f"""
        <h4>🎓 Academic Career Path</h4>
        <p>Based on your goals: <em>"{user_goals}"</em></p>
        <ul>
            <li><strong>Research Focus:</strong> Develop a strong publication record in your field</li>
            <li><strong>Teaching Experience:</strong> Gain experience as a TA or instructor</li>
            <li><strong>Networking:</strong> Attend conferences and build academic connections</li>
            <li><strong>Funding:</strong> Apply for grants and fellowships early</li>
            <li><strong>Timeline:</strong> Start job applications 12-18 months before graduation</li>
        </ul>
        """,
        'Industry': f"""
        <h4>🏢 Industry Career Path</h4>
        <p>Based on your goals: <em>"{user_goals}"</em></p>
        <ul>
            <li><strong>Skills Development:</strong> Focus on practical, industry-relevant skills</li>
            <li><strong>Internships:</strong> Gain industry experience through summer internships</li>
            <li><strong>Networking:</strong> Connect with alumni in your target industry</li>
            <li><strong>Portfolio:</strong> Build a portfolio of relevant projects</li>
            <li><strong>Job Search:</strong> Start networking 6-12 months before graduation</li>
        </ul>
        """,
        'Government': f"""
        <h4>🏛️ Government Career Path</h4>
        <p>Based on your goals: <em>"{user_goals}"</em></p>
        <ul>
            <li><strong>Policy Knowledge:</strong> Stay informed about current policy issues</li>
            <li><strong>Public Service:</strong> Consider internships in government agencies</li>
            <li><strong>Networking:</strong> Connect with policy professionals</li>
            <li><strong>Skills:</strong> Develop analytical and communication skills</li>
            <li><strong>Applications:</strong> Government hiring can take 6+ months</li>
        </ul>
        """,
        'Non-profit': f"""
        <h4>❤️ Non-profit Career Path</h4>
        <p>Based on your goals: <em>"{user_goals}"</em></p>
        <ul>
            <li><strong>Mission Alignment:</strong> Find organizations that match your values</li>
            <li><strong>Volunteer Experience:</strong> Gain relevant experience through volunteering</li>
            <li><strong>Networking:</strong> Connect with non-profit professionals</li>
            <li><strong>Skills:</strong> Develop grant writing and program management skills</li>
            <li><strong>Funding:</strong> Understand the organization's funding sources</li>
        </ul>
        """
    }
    
    # Get advice for predicted sector, fallback to desired industry
    sector = predicted_sector if predicted_sector in sector_advice else desired_industry
    advice = sector_advice.get(sector, sector_advice['Industry'])
    
    # Add general advice
    general_advice = f"""
    <h4>💡 General Career Development Tips</h4>
    <ul>
        <li><strong>Work Environment:</strong> Your preference for {work_env} work suggests focusing on roles that offer this flexibility</li>
        <li><strong>Skills Gap Analysis:</strong> Identify skills needed for your target roles and develop them</li>
        <li><strong>Mentorship:</strong> Seek mentors in your target field</li>
        <li><strong>Professional Development:</strong> Attend workshops and training sessions</li>
        <li><strong>Online Presence:</strong> Build a professional LinkedIn profile</li>
    </ul>
    
    <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h5>🔍 Next Steps</h5>
        <ol>
            <li>Refine your career goals based on this analysis</li>
            <li>Identify 3-5 target organizations or roles</li>
            <li>Develop a timeline for your job search</li>
            <li>Build relevant skills and experience</li>
            <li>Start networking in your target field</li>
        </ol>
    </div>
    
            <p><em>💡 <strong>Note:</strong> This is sample advice. Enable personalization in the sidebar for personalized analysis using your actual data.</em></p>
    """
    
    return advice + general_advice

# ----------------------------------------------------------------------------
def explore_options_tab(advisor):
    st.markdown("<h1 class='header'>Explore Your Options</h1>", unsafe_allow_html=True)
    if df is None:
        st.error("Error loading NYU dataset. Please check if the file 'data/imputed_data_NYU copy 3.xlsx' exists in the correct location.")
        return

    # --- RECOMMENDATION GENERATION ---
    if not use_openai:
        st.warning("🔓 **Personalization Disabled**")
        st.info("💡 Check the sidebar to enable personalization for enhanced insights (uses API tokens)")
        st.markdown("""
        **What You Can Explore:**
        - Pathway insights from alumni data
        - A sample RIASEC profile
        - Exploration mode with optional personalization
        """)
    else:
        st.success("🔓 **Personalization Enabled**")
    
    st.info("Click Explore Your Options to see insights informed by alumni patterns.")

    if "academic_division" not in df.columns:
        possible_divisions = ["(No academic_division column)"]
    else:
        possible_divisions = df["academic_division"].dropna().unique().tolist()

    selected_division = st.selectbox("Academic Division", possible_divisions, help="Choose your division as listed in the dataset.")

    st.markdown("<h2 class='subheader'>Your Profile</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        user_goals = st.text_area("Your Career Goals:", help="Describe your ideal career pathway after your PhD")
    with col2:
        user_tech_skills = st.text_area("Your Soft Skills:", help="E.g., communication, teamwork, leadership")
        user_citizenship = st.selectbox("Citizenship Status", [
            "U.S. Citizen", 
            "Permanent Resident", 
            "International Student", 
            "Other"
        ], help="Select your current citizenship status.")

    cohort_stage = st.radio("PhD Cohort Stage", ["First Year", "All but Dissertation (ABD)"], index=0, help="Select your current stage in the PhD journey.")

    st.markdown("#### Additional Preferences")
    work_env = st.selectbox("Preferred Work Environment", ["Collaborative", "Independent", "Hybrid"], help="What type of work environment do you prefer?")
    research_interests = st.text_input("Your Primary Research Interests", help="E.g., machine learning, public health, etc.")
    desired_industry = st.selectbox("Where do you see your career after your PhD?", ["Academia", "Industry", "Government", "Non-profit", "Entrepreneurial", "I'm not sure", "I'm open"], help="Where do you see your career after your PhD?")
    work_life_balance = st.selectbox("How important is work-life balance to you?", [1,2,3,4,5], index=2, help="How important is work-life balance to you?")


    job_function = st.text_input("Roles You'd Like to Explore (e.g., Data Scientist, R&D, Product Manager)", help="What job function are you targeting?")


    desirability = st.text_input("Market Demand (e.g., how much demand there is for this role or pathway)", help="How much demand there is for this role or pathway?")

    # --- RIASEC Personality Assessment ---
    st.markdown("#### 🧠 RIASEC Personality Assessment")
    st.markdown("*Rate how well each statement describes you (1=Not at all, 7=Very much)*")
    
    # RIASEC questions based on your dataset (10 focused questions)
    riasec_questions = {
        "Realistic": [
            "I enjoy working with tools and machines",
            "I prefer practical, hands-on work"
        ],
        "Investigative": [
            "I enjoy solving complex problems",
            "I like to analyze data and research"
        ],
        "Artistic": [
            "I enjoy creative expression and design",
            "I prefer non-conformity and originality"
        ],
        "Social": [
            "I enjoy teaching and helping others",
            "I prefer working with people"
        ],
        "Enterprising": [
            "I enjoy leadership and persuasion",
            "I like to take charge and make decisions"
        ],
        "Conventional": [
            "I enjoy organization and attention to detail",
            "I prefer following rules and procedures"
        ]
    }
    
    # Collect RIASEC scores with enhanced scoring model
    riasec_scores = {}
    riasec_analysis = {}
    
    for dimension, questions in riasec_questions.items():
        st.markdown(f"**{dimension}**")
        
        # Display both questions for this dimension
        raw_scores = []
        for i, question in enumerate(questions):
            score = st.slider(f"{question}", 1, 7, 4, key=f"{dimension}_{i}")
            raw_scores.append(score)
        
        # Enhanced scoring model with weighted calculations
        if raw_scores:
            # Calculate weighted average (first question slightly more important)
            weighted_score = (raw_scores[0] * 0.6) + (raw_scores[1] * 0.4)
            
            # Apply confidence adjustment based on score consistency
            score_variance = abs(raw_scores[0] - raw_scores[1])
            if score_variance <= 1:
                confidence_boost = 0.2  # High consistency
            elif score_variance <= 2:
                confidence_boost = 0.1  # Medium consistency
            else:
                confidence_boost = 0.0  # Low consistency
            
            final_score = min(7.0, weighted_score + confidence_boost)
            riasec_scores[dimension] = round(final_score, 1)
            
            # Store analysis for recommendations
            riasec_analysis[dimension] = {
                'raw_scores': raw_scores.copy(),  # Make a copy to ensure it's a list
                'weighted_score': weighted_score,
                'confidence': 'high' if score_variance <= 1 else 'medium' if score_variance <= 2 else 'low',
                'score_variance': score_variance
            }
        
        # Show the score with confidence indicator
        if dimension in riasec_analysis:
            confidence_emoji = "🟢" if riasec_analysis[dimension]['confidence'] == 'high' else "🟡" if riasec_analysis[dimension]['confidence'] == 'medium' else "🔴"
            st.markdown(f"**{dimension} Score: {riasec_scores[dimension]}/7** {confidence_emoji} *{riasec_analysis[dimension]['confidence']} confidence*")
        else:
            st.markdown(f"**{dimension} Score: {riasec_scores[dimension]}/7**")
        st.markdown("---")

    # --- Disable button until required fields are filled ---
    can_recommend = user_goals.strip() and user_tech_skills.strip() and all(riasec_scores.values())
    if not can_recommend:
        if not user_goals.strip() or not user_tech_skills.strip():
            st.warning("Please provide at least your Career Goals and Soft Skills.")
        else:
            st.warning("Please complete the RIASEC Personality Assessment.")
    
    if st.button("Explore Your Options", disabled=not can_recommend):
        # Show loading progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Predict sector using ML model and display
        status_text.text("🔍 Analyzing your career goals and predicting sector...")
        progress_bar.progress(25)
        predicted_sector, ml_method = predict_sector_from_text(user_goals, desired_industry, work_env)
        # Store the ML method prediction for CSV export
        st.session_state.ml_model_prediction = ml_method
        sector_label = {
            "Academic": "Academia",
            "Industry": "Industry",
            "Government": "Government",
            "Non-profit": "Non-profit",
            "Entrepreneurial": "Business",
            "I'm not sure": "Exploring Options",
            "I'm open": "Open to Possibilities",
            "Unknown": "Unknown (Model Error)"
        }.get(predicted_sector, predicted_sector)
        
        # Create a beautiful career prediction card
        st.markdown(f"""
        <div style='margin:20px 0; padding:20px; border:2px solid #57068c; background:linear-gradient(135deg, #57068c10, #8850c810); border-radius:12px; box-shadow:0 4px 12px rgba(87,6,140,0.1);'>
            <div style='text-align:center;'>
                <h3 style='color:#57068c; margin-bottom:10px;'>🎯 Career Pathway Prediction</h3>
                <div style='background:#57068c; color:white; padding:15px; border-radius:8px; margin:10px 0;'>
                    <strong style='font-size:1.2em;'>Your predicted career pathway is best aligned with:</strong><br>
                    <span style='font-size:1.5em; font-weight:bold;'>{sector_label}</span>
                </div>
                <small style='color:#666;'>*Based on your inputs and NYU dataset analysis*</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Display RIASEC scores
        status_text.text("🧠 Analyzing your personality profile (RIASEC)...")
        progress_bar.progress(50)
        
        # Show immediate feedback
        st.success("✅ Career prediction complete!")
        
        # Display RIASEC scores
        st.markdown("### 🧠 Your RIASEC Personality Profile")
        st.markdown("*Based on your self-assessment*")
        
        # Create columns for the visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create a radar chart-like visualization using bars
            st.markdown("#### Your Interest Scores (1-7 Scale)")
            
            # Define colors for each dimension
            colors = {
                "Realistic": "#FF6B6B",
                "Investigative": "#4ECDC4", 
                "Artistic": "#45B7D1",
                "Social": "#96CEB4",
                "Enterprising": "#FFEAA7",
                "Conventional": "#DDA0DD"
            }
            
            # Sort by score for better visualization
            sorted_scores = sorted(riasec_scores.items(), key=lambda x: x[1], reverse=True)
            
            for dimension, score in sorted_scores:
                # Create a progress bar with color
                color = colors.get(dimension, "#6C757D")
                
                # Calculate percentage for progress bar
                percentage = (score - 1) / 6 * 100
                
                # Create the progress bar
                st.markdown(f"""
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content-between; margin-bottom: 5px;">
                        <strong style="color: {color};">{dimension}</strong>
                        <span style="font-weight: bold;">{score}/7</span>
                    </div>
                        <div style="background: #f0f0f0; border-radius: 10px; height: 20px; overflow: hidden;">
                            <div style="background: {color}; height: 100%; width: {percentage}%; border-radius: 10px; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Show top 2 dimensions
                top_dimensions = sorted_scores[:2]
                st.markdown("#### 🎯 Your Top Interests")
                for i, (dimension, score) in enumerate(top_dimensions, 1):
                    color = colors.get(dimension, "#6C757D")
                    st.markdown(f"""
                    <div style="background: {color}20; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid {color};">
                        <strong style="color: {color};">#{i} {dimension}</strong><br>
                        <small>Score: {score}/7</small>
                    </div>
                    """, unsafe_allow_html=True)

        # 3. Generate and display personalized recommendations (only if AI is enabled)
        if use_openai:
            status_text.text("📋 Generating personalized insights...")
            progress_bar.progress(75)
            
            with st.spinner("Generating your personalized recommendations..."):
                recommendations = advisor.generate_career_advice(
                    selected_division=selected_division,
                    citizenship=user_citizenship,
                    user_goals=user_goals,
                    user_tech_skills=user_tech_skills,
                    work_env=work_env,
                    research_interests=research_interests,
                    desired_industry=desired_industry,
                    work_life_balance=work_life_balance,
                    cohort_stage=cohort_stage,
                    family_location="",
                    job_function=job_function,
                    desirability=desirability,
                    user_riasec=riasec_scores,
                    sentiment_scores=sentiment_scores,
                    theme_sentiments=theme_sentiments,
                    processed_sentiment_df=processed_sentiment_df
                )
            
            if recommendations:
                # Complete the progress bar
                progress_bar.progress(100)
                status_text.text("✅ Recommendations complete!")
                
                # Set session state to track that recommendations are completed
                st.session_state.recommendations_completed = True
                
                # Store recommendations data with actual user inputs
                recommendation_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'academic_division': selected_division,
                    'cohort_stage': cohort_stage,
                    'citizenship': user_citizenship,
                    'career_goals': user_goals,
                    'soft_skills': user_tech_skills,
                    'work_environment': work_env,
                    'desired_industry': desired_industry,
                    'work_life_balance': work_life_balance,
                    'predicted_career_path': predicted_sector,
                    'ml_model_prediction': st.session_state.get('ml_model_prediction', 'Not available'),
                    'riasec_scores': str(riasec_scores),
                    'top_riasec_codes': ', '.join([f"{dim}:{score}" for dim, score in sorted(riasec_scores.items(), key=lambda x: x[1], reverse=True)[:2]]),
                    'recommendations': recommendations
                }
                
                st.session_state.recommendations_data.append(recommendation_entry)
                
                st.markdown("### 📋 Your Personalized Career Insights")
                st.markdown("*Based on your profile and NYU data*")
                
                # Create a beautiful recommendations container
                st.markdown(f"""
                <div style='background:white; border:2px solid #57068c; border-radius:12px; padding:25px; margin:20px 0; box-shadow:0 4px 12px rgba(87,6,140,0.1);'>
                    {recommendations}
                </div>
                """, unsafe_allow_html=True)
                
                # Show recommendations summary
                st.markdown("---")
                st.subheader("📊 Recommendations Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Profile Information:**")
                    st.write(f"• Academic Division: {selected_division}")
                    st.write(f"• Cohort Stage: {cohort_stage}")
                    st.write(f"• Career Goals: {user_goals[:100]}..." if len(user_goals) > 100 else f"• Career Goals: {user_goals}")
                    st.write(f"• Work Environment: {work_env}")
                
                with col2:
                    st.write("**Analysis Results:**")
                    st.write(f"• Predicted Career Pathway: {predicted_sector}")
                    st.write(f"• ML Model Method: {st.session_state.get('ml_model_prediction', 'Not available')}")
                    st.write(f"• Total Recommendations: {len(st.session_state.recommendations_data)}")
                    st.write(f"• Desired Industry: {desired_industry}")
                    
                    # Add comprehensive RIASEC summary
                    st.write("**🧠 RIASEC Personality Profile:**")
                    top_3_riasec = sorted(riasec_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    # Create a beautiful RIASEC summary
                    for i, (dim, score) in enumerate(top_3_riasec, 1):
                        # Determine strength and color
                        if score >= 6.5:
                            strength = "Very Strong"
                            color = "#28a745"  # Green
                        elif score >= 5.5:
                            strength = "Strong"
                            color = "#17a2b8"  # Blue
                        elif score >= 4.5:
                            strength = "Moderate"
                            color = "#ffc107"  # Yellow
                        elif score >= 3.5:
                            strength = "Developing"
                            color = "#fd7e14"  # Orange
                        else:
                            strength = "Emerging"
                            color = "#dc3545"  # Red
                        
                        # Get confidence indicator
                        confidence = riasec_analysis[dim]['confidence'] if dim in riasec_analysis else 'unknown'
                        confidence_emoji = "🟢" if confidence == 'high' else "🟡" if confidence == 'medium' else "🔴"
                        
                        st.markdown(f"""
                        <div style="background: {color}15; border-left: 4px solid {color}; padding: 10px; margin: 5px 0; border-radius: 4px;">
                            <strong style="color: {color};">#{i} {dim}</strong> ({score}/7) - {strength} {confidence_emoji}
                            <br><small>Confidence: {confidence}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add RIASEC career insights
                    st.write("**🎯 RIASEC Career Insights:**")
                    career_mapping = create_riasec_career_mapping()
                    riasec_recommendations = get_riasec_career_recommendations(riasec_scores, career_mapping, riasec_analysis)
                    
                    for rec in riasec_recommendations:
                        priority_emoji = "🎯" if rec['priority'] == 'primary' else "📋" if rec['priority'] == 'secondary' else "🌱"
                        st.write(f"{priority_emoji} **{rec['dimension']} ({rec['score']}/7)**: {rec['confidence']}")
                        st.write(f"   → Careers: {', '.join(rec['careers'][:3])}")
                        st.write(f"   → Skills: {', '.join(rec['skills'][:2])}")
                
                # Download current recommendations
                if st.button("📥 Download These Recommendations (CSV)", key="download_current_recommendations"):
                    current_recommendation_df = pd.DataFrame([recommendation_entry])
                    csv = current_recommendation_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"career_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # --- Feedback section: only show after recommendations ---
                st.markdown("<h3 style='margin-top: 30px;'> Feedback</h3>", unsafe_allow_html=True)
                st.markdown("We'd love to hear from you. Your feedback helps us improve future insights!")

                # Simple feedback form without session state
                feedback = st.radio(
                    "Did you find these insights helpful?",
                    ["Yes", "Somewhat", "No"],
                    horizontal=True
                )

                feedback_comments = st.text_area(
                    "Additional Comments (optional)",
                    placeholder="What could be improved? Any specific feedback is helpful."
                )

                # Only show the multiselect if "No" is selected
                if feedback == "No":
                    feedback_issues = st.multiselect(
                        "What didn't work well? (Select all that apply)",
                        [
                            "Not relevant to my goals",
                            "Too generic / vague",
                            "Didn't use my data effectively",
                            "Too academic / not practical",
                            "Wrong assumptions made",
                            "Formatting / structure",
                            "Missing key suggestions",
                            "Other"
                        ],
                        help="Choose all that apply"
                    )

                if st.button("Submit Feedback"):
                    st.success("✅ Thanks for your feedback!")
            else:
                st.warning("No insights could be generated. Please check your inputs.")
        else:
            # When AI is disabled, complete progress and show message
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Set session state to track that recommendations are completed (even if AI is disabled)
            st.session_state.recommendations_completed = True
            
            # Store sample recommendations data
            sample_recommendations = """**Sample Career Insights**

**Short-term Goals (0-3 months):**
- Build your professional network
- Update your resume and LinkedIn profile
- Research potential employers in your field

**Medium-term Goals (3-12 months):**
- Gain relevant experience through internships or projects
- Develop technical skills needed in your target industry
- Attend industry conferences and events

**Long-term Goals (1+ years):**
            - Secure a position in your desired career pathway
- Establish yourself as a professional in your field
- Plan for career advancement and growth"""
            
            recommendation_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'academic_division': selected_division,
                'cohort_stage': cohort_stage,
                'citizenship': user_citizenship,
                'career_goals': user_goals,
                'soft_skills': user_tech_skills,
                'work_environment': work_env,
                'desired_industry': desired_industry,
                'work_life_balance': work_life_balance,
                'predicted_career_path': predicted_sector,
                'ml_model_prediction': st.session_state.get('ml_model_prediction', 'Not available'),
                'riasec_scores': str(riasec_scores),
                'top_riasec_codes': ', '.join(sorted(riasec_scores.items(), key=lambda x: x[1], reverse=True)[:2]),
                'recommendations': sample_recommendations
            }
            
            st.session_state.recommendations_data.append(recommendation_entry)
            
            st.info("💡 **Personalization Disabled**")
            st.markdown("""
            To get personalized career insights:
            1. ✅ Check the "Enable personalization" checkbox in the sidebar
            2. 🔄 Click "Explore Your Options" again
            3. 🤖 AI will generate personalized advice based on your profile
            """)

    # --- Feedback section: always visible below recommendations button ---
    st.markdown("---")
    st.markdown("<h3 style='margin-top: 30px;'>📝 Feedback</h3>", unsafe_allow_html=True)
    st.markdown("We'd love to hear from you. Your feedback helps us improve future insights!")

    # Simple feedback form without session state
    feedback = st.radio(
        "Did you find this app helpful so far?",
        ["Yes", "Somewhat", "No"],
        horizontal=True
    )

    feedback_comments = st.text_area(
        "Additional Comments (optional)",
        placeholder="What could be improved? Any specific feedback is helpful."
    )

    # Only show the multiselect if "No" is selected
    if feedback == "No":
        feedback_issues = st.multiselect(
            "What didn't work well? (Select all that apply)",
            [
                "Not relevant to my goals",
                "Too generic / vague",
                "Didn't use my data effectively",
                "Too academic / not practical",
                "Wrong assumptions made",
                "Formatting / structure",
                "Missing key suggestions",
                "Other"
            ],
            help="Choose all that apply"
        )

    if st.button("Submit Feedback"):
        # Validate feedback submission
        if feedback:
            # Store feedback in session state
            if 'feedback_data' not in st.session_state:
                st.session_state.feedback_data = []
            
            feedback_entry = {
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'helpfulness': feedback,
                'comments': feedback_comments if feedback_comments else 'None',
                'issues': ', '.join(feedback_issues) if feedback == "No" and 'feedback_issues' in locals() else 'None'
            }
            st.session_state.feedback_data.append(feedback_entry)
            
            # Show success message with feedback summary
            st.success("✅ Thanks for your feedback!")
            st.info("**Your feedback has been recorded:**")
            st.markdown(f"""
            - **Helpfulness**: {feedback}
            - **Comments**: {feedback_comments if feedback_comments else 'None'}
            """)
            
            if feedback == "No" and 'feedback_issues' in locals():
                st.markdown(f"- **Issues**: {', '.join(feedback_issues) if feedback_issues else 'None'}")
            
            # Show export option
            if len(st.session_state.feedback_data) > 0:
                st.info("💡 **Export Feedback**: You can download all collected feedback data.")
                # Convert feedback data to DataFrame
                feedback_df = pd.DataFrame(st.session_state.feedback_data)
                # Create CSV for download
                csv = feedback_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Feedback CSV",
                    data=csv,
                    file_name=f"nyu_career_feedback_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ Please select a feedback option before submitting.")

    st.markdown("---")

    if handshake_events_df is not None and not handshake_events_df.empty:
        event_types = handshake_events_df['Event Type Name'].unique().tolist()
        selected_type = st.selectbox("Filter by Event Type", ["All"] + event_types)
        if selected_type != "All":
            filtered_df = handshake_events_df[handshake_events_df['Event Type Name'] == selected_type]
        else:
            filtered_df = handshake_events_df

        # --- Sort by date and show only top 5 ---
        try:
            filtered_df = filtered_df.copy()
            filtered_df['Events Start Date Date'] = pd.to_datetime(filtered_df['Events Start Date Date'], errors='coerce')
            filtered_df = filtered_df.sort_values('Events Start Date Date')
        except Exception:
            pass
        filtered_df = filtered_df.head(5)

        # Only the title is purple, rest is neutral
        st.markdown("""
        <h3 style='color:#57068c; margin-top:32px; margin-bottom:8px;'>Upcoming Handshake Events</h3>
        """, unsafe_allow_html=True)

        for _, event in filtered_df.iterrows():
            st.markdown(f"""
            <div class='event-card' style='margin-bottom:24px; box-shadow:0 2px 8px rgba(0,0,0,0.07); border-left: none;'>
                <div style='display:flex; flex-direction:column; gap:8px;'>
                    <span style='font-size:1.1em; font-weight:600;'>{event['Events Name']}</span>
                    <span><b>Date:</b> {event['Events Start Date Date'].strftime('%b %d, %Y') if pd.notnull(event['Events Start Date Date']) else event['Events Start Date Date']}</span>
                    <span><b>Type:</b> {event['Event Type Name']}</span>
                    <a href='{event['Link to Event in Handshake']}' target='_blank'
                       style='margin-top:8px; display:inline-block; background:#f5f5f5; color:#57068c; padding:8px 18px; border-radius:6px; text-decoration:none; font-weight:600; border:1px solid #e1e1e1; transition:background 0.2s;'>
                       View on Handshake
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No upcoming Handshake events found.")

    st.markdown("""
    <hr style='border:1px solid #e1e1e1; margin-top:32px; margin-bottom:16px;'>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Research Opportunities Tab - Explore the Career Landscape
# ----------------------------------------------------------------------------
def research_opportunities_tab():
    st.markdown("<h1 class='header'>Research Opportunities</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    **🔬 Explore the Career Landscape**
    
    This section helps you research current opportunities and understand what's available in different sectors, 
    locations, and roles. Think of it as your personal research lab for exploring career possibilities.
    """)

    st.info("Complete the 'Explore Your Options' tab first to see your personalized insights and understand your career landscape.")

    st.markdown("<h2 class='subheader'>Understanding Your Career Landscape</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    What factors would be most important to you when exploring career opportunities beyond your PhD field? This helps us understand your research interests and preferences.
    """)
    
    career_factors = {
        "Pay": "Salary and compensation considerations",
        "Promotion opportunities": "Career advancement and growth potential",
        "Working conditions": "Work environment and job conditions",
        "Job location": "Geographic location preferences",
        "Change in career": "Desire for career change",
        "Professional interests": "Alignment with professional goals",
        "Family-related reasons": "Family considerations and responsibilities",
        "Job in doctoral degree field not available": "Limited opportunities in PhD field",
        "Some other reason": "Other factors not listed above"
    }
    
    selected_factor = st.selectbox(
        "Select the most important factor:",
        options=list(career_factors.keys()),
        help="Choose the factor that most influenced your career decision"
    )
    
    # Selected factor stored for this session
    
    # Additional questions based on selected factor
    if selected_factor == "Pay":
        previous_salary = st.text_input(
            "What was your previous salary range? (e.g., '$40,000/year' or '40k-50k')",
            help="This helps us explore opportunities with better compensation"
        )
        # Previous salary stored for this session
        
    elif selected_factor == "Job location":
        previous_location = st.text_input(
            "Where did you work before? (City, State)",
            help="This helps us suggest alternative locations"
        )
        location_preference = st.text_area(
            "What aspects of the previous location did you not like?",
            help="This helps us avoid similar locations"
        )
        # Location preferences stored for this session
        
    elif selected_factor == "Working conditions":
        work_conditions = st.text_area(
            "What specific working conditions were problematic? (e.g., long hours, toxic environment)",
            help="This helps us explore opportunities with better working conditions"
        )
        # Work conditions stored for this session
        
    elif selected_factor == "Professional interests":
        new_interests = st.text_area(
            "What are your new professional interests or goals?",
            help="This helps us align opportunity insights with your interests"
        )
        # New interests stored for this session
        
    elif selected_factor == "Family-related reasons":
        family_considerations = st.text_area(
            "What family-related factors are important? (e.g., need for flexible hours, proximity to family)",
            help="This helps us explore opportunities that accommodate your family needs"
        )
        # Family considerations stored for this session

    st.markdown("<h2 class='subheader'>Explore Current Opportunities</h2>", unsafe_allow_html=True)

    # Modify search location based on career decision factors
    desired_industry = "Industry"  # Default value
    
    # Define industry mapping
    industry_map = {
        "Academia": "professor",
        "Industry": "engineer",
        "Government": "government",
        "Non-profit": "non profit"
    }
    
    # Define alternative cities for opportunity exploration
    alternative_cities = {
        "New York": ["Boston", "Chicago", "Seattle", "San Francisco", "Austin"],
        "Boston": ["New York", "Chicago", "Seattle", "San Francisco", "Austin"],
        "Chicago": ["Boston", "New York", "Seattle", "San Francisco", "Austin"],
        "Los Angeles": ["San Francisco", "Seattle", "Portland", "Denver", "Austin"],
        "San Francisco": ["Seattle", "Portland", "Austin", "Denver", "Boston"],
        "Seattle": ["Portland", "San Francisco", "Austin", "Denver", "Boston"],
        "Austin": ["Seattle", "Denver", "Portland", "Boston", "Chicago"],
        "Denver": ["Seattle", "Portland", "Austin", "Boston", "Chicago"],
        "Portland": ["Seattle", "San Francisco", "Austin", "Denver", "Boston"]
    }
    
    # Define city aliases for better detection
    city_aliases = {
        "new york": ["nyc", "ny", "new york city"],
        "los angeles": ["la", "l.a.", "los angeles"],
        "san francisco": ["sf", "san fran", "bay area"],
        "boston": ["beantown", "boston ma"],
        "chicago": ["chi", "chicago il"],
        "seattle": ["seattle wa"],
        "austin": ["austin tx"],
        "denver": ["denver co"],
        "portland": ["portland or", "pdx"]
    }
    
    # Determine search location based on career decision factors
    if selected_factor == "Job location":
        # Use the actual user inputs from the form
        prev_loc = previous_location.lower() if previous_location else ""
        location_pref = location_preference.lower() if location_preference else ""
        
        # Find the city the user wants to avoid
        avoid_city = None
        for city, aliases in city_aliases.items():
            # Check if the city or any of its aliases are mentioned in either previous location or preferences
            if (city in prev_loc or 
                any(alias in prev_loc for alias in aliases) or
                city in location_pref or 
                any(alias in location_pref for alias in aliases)):
                avoid_city = city.title()  # Convert to title case to match alternative_cities keys
                break
        
        if avoid_city and avoid_city in alternative_cities:
            # Get alternative cities, excluding the one they want to avoid
            alternatives = alternative_cities[avoid_city]
            # Use the first alternative that isn't the city they want to avoid
            search_location = alternatives[0]  # Always use the first alternative city
            st.write(f"🎯 Based on your preferences, searching in **{search_location}** instead of **{avoid_city}**")
            # Show other alternatives, excluding the avoided city
            other_alternatives = alternatives[1:3]  # Show next 2 alternatives
            if other_alternatives:
                st.write(f"📍 Other potential cities to consider: {', '.join(other_alternatives)}")
        else:
            # If we can't determine a specific city to avoid, use a default set of cities
            # excluding any city mentioned in preferences
            default_cities = ["Chicago", "Seattle", "Austin", "Denver", "Portland"]  # Removed NYC and Boston from defaults
            # Filter out any city mentioned in preferences or previous location
            available_cities = [city for city in default_cities 
                              if not any(alias in (location_pref + " " + prev_loc)
                                       for alias in city_aliases.get(city.lower(), []) + [city.lower()])]
            search_location = available_cities[0] if available_cities else "Chicago"
            st.write(f"🔍 Searching in **{search_location}** and other major cities")
            
        # Debug information for location filtering
        if prev_loc or location_pref:
            st.info(f"🔍 Location Analysis:")
            if prev_loc:
                st.write(f"   • Previous location: {previous_location}")
            if location_pref:
                st.write(f"   • Location preferences: {location_preference}")
            if avoid_city:
                st.write(f"   • Avoiding: {avoid_city}")
            st.write(f"   • Searching in: {search_location}")

    else:
        # For non-location factors, use a default set of cities
        search_location = "Chicago"  # Changed from Boston to Chicago as default
        st.write("Searching in Chicago and other major cities")

    # Adjust research query based on career decision factors
    base_query = industry_map.get(desired_industry, "research")
    
    # Enhance query based on user inputs
    if selected_factor == "Pay" and previous_salary:
        st.write(f"💰 Looking for positions with higher compensation than {previous_salary}")
        base_query = f"{base_query} high salary"
    
    elif selected_factor == "Working conditions" and work_conditions:
        conditions_lower = work_conditions.lower()
        if "flexible" in conditions_lower or "remote" in conditions_lower:
            base_query = f"{base_query} remote flexible"
            st.write("🏠 Searching for remote/flexible positions")
        elif "work-life" in conditions_lower or "balance" in conditions_lower:
            base_query = f"{base_query} work-life balance"
            st.write("⚖️ Searching for positions with better work-life balance")
        elif "toxic" in conditions_lower or "stress" in conditions_lower:
            base_query = f"{base_query} positive culture"
            st.write("😊 Searching for positions with positive work culture")
    
    elif selected_factor == "Professional interests" and new_interests:
        # Extract key terms from interests
        key_terms = new_interests.split()[:3]  # Take first 3 words as key terms
        base_query = f"{base_query} {' '.join(key_terms)}"
        st.write(f"🎯 Incorporating your interests: {', '.join(key_terms)}")
    
    elif selected_factor == "Family-related reasons" and family_considerations:
        considerations_lower = family_considerations.lower()
        if "flexible" in considerations_lower or "remote" in considerations_lower:
            base_query = f"{base_query} remote flexible"
            st.write("👨‍👩‍👧‍👦 Searching for family-friendly positions")
        elif "proximity" in considerations_lower or "near" in considerations_lower:
            base_query = f"{base_query} local"
            st.write("🏠 Searching for local positions")

    user_query = st.text_input("Additional Research Keywords (optional)", value=base_query, 
                               help="Add specific terms to explore opportunities that match your interests")

    if st.button("Research Opportunities"):
        with st.spinner("Exploring the current landscape..."):
            job_data = fetch_adzuna_jobs(
                query=user_query,
                location=search_location
            )
            if job_data:
                # Filter and sort opportunities based on research preferences
                if selected_factor == "Pay" and previous_salary:
                    try:
                        # Extract numeric value from salary string
                        salary_num = float(''.join(filter(str.isdigit, previous_salary)))
                        # Filter opportunities with higher compensation
                        original_count = len(job_data['results'])
                        job_data['results'] = [
                            job for job in job_data['results']
                            if job.get('salary_min', 0) > salary_num
                        ]
                        filtered_count = len(job_data['results'])
                        if filtered_count < original_count:
                            st.info(f"💰 Found {filtered_count} opportunities with higher compensation than {previous_salary}")
                    except ValueError:
                        st.warning("⚠️ Could not parse compensation for filtering")
                
                st.success(f"🔍 Found {len(job_data['results'])} opportunities to explore!")
                display_adzuna_jobs(job_data)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    try:
        st.title("🎓 NYU PhD Career Advisor")
        st.write("Explore with confidence, guided by research on PhD career pathways.")
        
        advisor = NYUCareerAdvisor(df)
        tab1, tab2 = st.tabs(["Explore Your Options", "Research Opportunities"])
        with tab1:
            explore_options_tab(advisor)
        with tab2:
            research_opportunities_tab()
    except Exception as e:
        st.error(f"🚨 App Error: {e}")
        st.info("Please check the logs for more details.")

# Handshake events are now loaded in the main data loading section



# Add caching to prevent data reloading and model retraining on every request
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_cached_data():
    """Load data once and cache it."""
    print("🔄 Loading data from local files...")
    loaded_data = load_local_data()
    print("✅ Data loading complete!")
    return loaded_data

@st.cache_data(ttl=3600)  # Cache for 1 hour
def train_cached_model():
    """Train the model once and cache it."""
    print("🔄 Training sector prediction model...")
    model = train_text_sector_model()
    if model is not None:
        print("✅ Model training complete!")
    return model

# Load data globally with caching
try:
    print("🔍 DEBUG: Starting data loading process...")
    print(f"🔍 DEBUG: Current working directory: {os.getcwd()}")
    print(f"🔍 DEBUG: Checking if data folder exists: {os.path.exists('data')}")
    if os.path.exists('data'):
        print(f"🔍 DEBUG: Data folder contents: {os.listdir('data')}")
    
    # Check if we have Streamlit secrets
    try:
        if 'data_files' in st.secrets:
            print("🔍 DEBUG: Found Streamlit secrets with data_files")
            print(f"🔍 DEBUG: Available secret keys: {list(st.secrets['data_files'].keys())}")
        else:
            print("🔍 DEBUG: No data_files in Streamlit secrets")
    except Exception as e:
        print(f"🔍 DEBUG: Error checking secrets: {e}")
    
    # Use cached data loading
    all_data = load_cached_data()
    df = all_data.get('nyu_data', None)
    handshake_events_df = all_data.get('handshake_events', pd.DataFrame())
    sentiment_df = all_data.get('sentiment_data', None)
    
    print(f"🔍 DEBUG: Data loading results:")
    print(f"  - NYU data: {'✅' if df is not None else '❌'} ({len(df) if df is not None else 0} records)")
    print(f"  - Handshake events: {'✅' if not handshake_events_df.empty else '❌'} ({len(handshake_events_df)} records)")
    print(f"  - Sentiment data: {'✅' if sentiment_df is not None else '❌'} ({len(sentiment_df) if sentiment_df is not None else 0} records)")
    
    if df is None:
        print("❌ Could not load data from any source")
        print("📋 Please ensure data files are in the 'data' folder or configure Streamlit secrets")
        # Create a minimal placeholder DataFrame to prevent app crash
        df = pd.DataFrame({
            'Academic_Division': ['Computer Science', 'Engineering', 'Business'],
            'School': ['Tandon', 'Tandon', 'Stern'],
            'Citizenship': ['US Citizen', 'International', 'US Citizen'],
            'Income': ['$50,000-$75,000', '$75,000-$100,000', '$100,000+']
        })
        print("✅ Created placeholder data to keep app running")
        
except Exception as e:
    print(f"🚨 Error loading data: {e}")
    # Create minimal placeholder data
    df = pd.DataFrame({
        'Academic_Division': ['Computer Science'],
        'School': ['Tandon'],
        'Citizenship': ['US Citizen'],
        'Income': ['$50,000-$75,000']
    })
    handshake_events_df = pd.DataFrame()
    sentiment_df = None
    print("✅ Created minimal placeholder data to keep app running")

if __name__ == "__main__":
    main()
