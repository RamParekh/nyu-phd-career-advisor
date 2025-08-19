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
st.set_page_config(page_title="NYU PhD Career Advisor", layout="wide", initial_sidebar_state="expanded")

# --- API Keys from Streamlit Secrets ---

# API Keys - Add your actual keys here
# Get free keys from: https://developer.adzuna.com/

# Replace these with your actual Adzuna API keys
ADZUNA_APP_ID = "5a2ee7fb"  # Replace this
ADZUNA_APP_KEY = "cf15da52bcd7e5585e5cd2d7fea154e5"  # Replace this

# OpenAI API Key - Add your actual key here
# Get your key from: https://platform.openai.com/api-keys

# Replace this with your actual OpenAI API key
openai.api_key = "your-actual-openai-api-key-here"  # Replace this

# Fallback to environment variables if needed
import os
if openai.api_key == "your-actual-openai-api-key-here":
    openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

# Check if OpenAI API key is configured
if openai.api_key != "your-actual-openai-api-key-here":
    # Add checkbox to control OpenAI usage
    use_openai = st.sidebar.checkbox(
        "🤖 Enable Career Recommendations", 
        value=False,
        help="Check this to use OpenAI for personalized career advice (uses API tokens)"
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

.job-card {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  padding: 24px;
  margin-bottom: 24px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.job-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.job-card h4 {
  margin: 0 0 12px;
  color: var(--violet);
  font-size: 1.3rem;
  font-weight: 600;
}

.job-card a {
  color: var(--violet);
  text-decoration: none;
  font-weight: 600;
  transition: color 0.2s ease;
}

.job-card a:hover {
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
  
  .job-card, .recommendations {
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
  <span>NYU PhD Career Advisor</span>
</div>

<div class='nyu-hero'>
  <div class='nyu-hero-text'>
    Empowering PhD Students for Career Success
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(NYU_CSS, unsafe_allow_html=True)

# --- Sidebar: How to Use and Contact Info ---
st.sidebar.markdown("""
### ℹ️ How to Use This App
1. Fill in your academic and career information.
2. Click **Get Career Recommendations** (max 3 per session).
3. Review the personalized advice and job postings.
4. Use the feedback section to help us improve!

""")



# Function to ensure model is loaded
def ensure_model_loaded():
    """Ensure the sector prediction model is loaded."""
    # Use cached model training
    print("🔄 Loading sector prediction model...")
    model = train_cached_model()
    if model is not None:
        print("✅ Sector prediction model loaded successfully!")
        return model
    else:
        print("❌ Failed to load sector prediction model")
        return None

def predict_sector_from_text(user_goals, desired_industry, work_env):
    if not user_goals:
        st.warning("No user goals provided for sector prediction")
        return "Unknown"
    
    # Create comprehensive input text for analysis
    input_text = f"{user_goals} {desired_industry} {work_env}".lower()
    
    # Enhanced keyword-based prediction with better logic
    def analyze_text_for_sector(text):
        # Academic/Research indicators
        academic_keywords = [
            'academic', 'research', 'professor', 'university', 'college', 'school',
            'teaching', 'lecture', 'publish', 'paper', 'journal', 'conference',
            'phd', 'doctoral', 'thesis', 'dissertation', 'scholar', 'faculty',
            'postdoc', 'postdoctoral', 'tenure', 'academia', 'higher education'
        ]
        
        # Industry/Corporate indicators
        industry_keywords = [
            'industry', 'company', 'business', 'corporate', 'startup', 'tech',
            'product', 'development', 'engineering', 'software', 'data', 'analytics',
            'consulting', 'finance', 'banking', 'investment', 'marketing', 'sales',
            'operations', 'management', 'leadership', 'entrepreneur', 'founder'
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
        
        # Count keyword matches for each sector
        scores = {
            'Academic': sum(1 for word in academic_keywords if word in text),
            'Industry': sum(1 for word in industry_keywords if word in text),
            'Government': sum(1 for word in government_keywords if word in text),
            'Non-profit': sum(1 for word in nonprofit_keywords if word in text),
            'Healthcare': sum(1 for word in healthcare_keywords if word in text),
            'Education': sum(1 for word in education_keywords if word in text)
        }
        
        # Find the sector with the highest score
        max_score = max(scores.values())
        if max_score == 0:
            return "Industry"  # Default only if no clear indicators
        
        # Return the sector with the highest score
        best_sector = max(scores, key=scores.get)
        return best_sector
    
    # Try ML model first if available
    if ML_AVAILABLE:
        text_sector_model = ensure_model_loaded()
        if text_sector_model is not None:
            try:
                prediction = text_sector_model.predict([input_text])[0]
                return prediction
            except Exception as e:
                print(f"ML model prediction failed: {e}, falling back to keyword analysis")
                # Fall through to keyword analysis
    
    # Use enhanced keyword analysis as fallback
    sector = analyze_text_for_sector(input_text)
    print(f"🔍 Sector prediction for '{input_text[:100]}...': {sector}")
    return sector



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
    """Generate sentiment insights for career recommendations."""
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
        st.write("No jobs found.")
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
        <div class='job-card'>
            <h4>{title}</h4>
            <p><strong>Company:</strong> {company}</p>
            <p><strong>Location:</strong> {job_location}</p>
            <p><strong>Salary Range:</strong> {salary_range}</p>
            <p>{description[:200]}...</p>
            <a href='{redirect_url}' target='_blank'>View Job Posting</a>
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
        school,
        citizenship,
        income,
        user_goals,
        user_tech_skills,
        work_env,
        research_interests,
        desired_industry,
        work_life_balance,
        mentorship_preference,
        cohort_stage,
        family_location,
        skill_levels=None,
        job_function=None,
        financial_scholarships=None,
        funding=None,
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

        if not income.strip():
            income = "Approx. range from web sources: $60k–$80k"

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

        predicted_sector = predict_sector_from_text(user_goals, desired_industry, work_env)

        user_profile_str = (
            f"School: {school}\n"
            f"PhD Cohort Stage: {cohort_stage}\n"
            f"Citizenship: {citizenship}\n"
            f"Income: {income}\n"
            f"Career Goals: {user_goals}\n"
            f"Predicted Sector (via ML): {predicted_sector}\n"
            f"Soft Skills: {user_tech_skills}\n"
            f"Skill Levels: {skill_levels}\n"
            f"Preferred Job Function: {job_function}\n"
            f"Work Environment: {work_env}\n"
            f"Research Interests: {research_interests}\n"
            f"Desired Industry: {desired_industry}\n"
            f"Work-Life Balance (1-5): {work_life_balance}\n"
            f"Mentorship Preference: {mentorship_preference}\n"
            f"Selected academic_division: {selected_division}\n"
            f"Financial Scholarships: {financial_scholarships}\n"
            f"Funding: {funding}\n"
            f"Desirability: {desirability}\n"
            f"Family Location: {family_location}\n"
        )

        

        # If we do have RIASEC info, add it
        if user_riasec:
            user_profile_str += "\nApprox RIASEC Scores:\n"
            # Just show them in the string, plus find top 2
            riasec_list = sorted(user_riasec.items(), key=lambda x: x[1], reverse=True)
            top_riasec = [c for (c,score) in riasec_list[:2]]
            for dim,score in riasec_list:
                user_profile_str += f"{dim}: {score}\n"
            user_profile_str += f"\nTop codes: {', '.join(top_riasec)}\n"

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
        You are an NYU career advisor. Incorporate the data below into a set of 
        short-term (0-3 months), medium-term (3-12 months), and long-term (1+ years) 
        career recommendations. 

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
        - **FOR EACH RECOMMENDATION SECTION (Short/Medium/Long-term), YOU MUST INCLUDE A 'Sentiment Justification' BULLET**: Reference at least one relevant sentiment or theme score (e.g., 'research' theme sentiment is 0.40, which supports academia recommendations for this division). If a theme has negative sentiment, provide strategies to address or mitigate these concerns.
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
# Career Recommendations Tab
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
    
    <p><em>💡 <strong>Note:</strong> This is sample advice. Enable AI recommendations in the sidebar for personalized analysis using your actual data.</em></p>
    """
    
    return advice + general_advice

# ----------------------------------------------------------------------------
def career_recommendations_tab(advisor):
    st.markdown("<h1 class='header'>Career Recommendations</h1>", unsafe_allow_html=True)
    if df is None:
        st.error("Error loading NYU dataset. Please check if the file 'data/imputed_data_NYU copy 3.xlsx' exists in the correct location.")
        return

    # --- RECOMMENDATION GENERATION ---
    if not use_openai:
        st.warning("🤖 **Career Recommendations Disabled**")
        st.info("💡 Check the sidebar to enable personalized career recommendations (uses API tokens)")
        st.markdown("""
        **Available Features:**
        - Career path prediction using ML model
        - Sample RIASEC personality profile
        - No personalized career advice (to save tokens)
        """)
    else:
        st.success("🤖 **Career Recommendations Enabled**")
    
    st.info("Click 'Get Career Recommendations' to generate personalized advice.")

    if "academic_division" not in df.columns:
        possible_divisions = ["(No academic_division column)"]
    else:
        possible_divisions = df["academic_division"].dropna().unique().tolist()

    selected_division = st.selectbox("Select your academic_division from dataset:", possible_divisions, help="Choose your division as listed in the dataset.")

    st.markdown("<h2 class='subheader'>Your Personal Info</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        user_school = st.text_input("Which school at NYU did you attend?", help="E.g., GSAS, Tandon, Steinhardt")
        user_income = st.text_input("Approximate Income (optional)", help="This helps us tailor recommendations")
        user_goals = st.text_area("Your Career Goals:", help="Describe your ideal career path after your PhD")
    with col2:
        user_tech_skills = st.text_area("Your Soft Skills:", help="E.g., communication, teamwork, leadership")
        user_citizenship = st.selectbox("Citizenship Status", [
            "U.S. Citizen", 
            "Permanent Resident", 
            "International Student", 
            "Other"
        ], help="Select your current citizenship status.")

    cohort_stage = st.radio("PhD Cohort Stage", ["First Year", "All but Dissertation (ABD)"], index=0, help="Select your current stage in the PhD journey.")

    st.markdown("#### Additional Preferences (Optional)")
    work_env = st.selectbox("Preferred Work Environment", ["Collaborative", "Independent", "Hybrid"], help="What type of work environment do you prefer?")
    research_interests = st.text_input("Your Primary Research Interests", help="E.g., machine learning, public health, etc.")
    desired_industry = st.selectbox("Desired Career Path", ["Academia", "Industry", "Government", "Non-profit"], help="Where do you see your career after your PhD?")
    work_life_balance = st.selectbox("Work-Life Balance Priority (1=Low, 5=High)", [1,2,3,4,5], index=2, help="How important is work-life balance to you?")
    mentorship_preference = st.text_input("Describe your ideal mentorship style (e.g., hands-on, advisory)", help="What kind of mentorship do you prefer?")
    skill_levels = st.text_input("List your technical skill levels (e.g., 'Python: Advanced, R: Intermediate')", help="List your technical skills and proficiency.")
    job_function = st.text_input("Preferred Job Function (e.g., Data Scientist, R&D, Product Manager)", help="What job function are you targeting?")
    financial_scholarships = st.text_input("Financial Scholarships (e.g., Fulbright, NSF grant)", help="List any scholarships or grants you've received.")
    funding = st.text_input("Funding Status (e.g., departmental funding, self-funded)", help="How is your PhD funded?")
    desirability = st.text_input("Desirability (e.g., how in-demand your field or profile is)", help="How in-demand is your field or profile?")

    # --- Disable button until required fields are filled ---
    can_recommend = user_goals.strip() and user_tech_skills.strip()
    if not can_recommend:
        st.warning("Please provide at least your Career Goals and Soft Skills.")
    
    if st.button("Get Career Recommendations", disabled=not can_recommend):
        # Show loading progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Predict sector using ML model and display
        status_text.text("🔍 Analyzing your career goals and predicting sector...")
        progress_bar.progress(25)
        predicted_sector = predict_sector_from_text(user_goals, desired_industry, work_env)
        sector_label = {
            "Academic": "Academia",
            "Industry": "Industry",
            "Government": "Government",
            "Non-profit": "Non-profit",
            "Entrepreneurial": "Business",
            "Unknown": "Unknown (Model Error)"
        }.get(predicted_sector, predicted_sector)
        
        # Create a beautiful career prediction card
        st.markdown(f"""
        <div style='margin:20px 0; padding:20px; border:2px solid #57068c; background:linear-gradient(135deg, #57068c10, #8850c810); border-radius:12px; box-shadow:0 4px 12px rgba(87,6,140,0.1);'>
            <div style='text-align:center;'>
                <h3 style='color:#57068c; margin-bottom:10px;'>🎯 Career Path Prediction</h3>
                <div style='background:#57068c; color:white; padding:15px; border-radius:8px; margin:10px 0;'>
                    <strong style='font-size:1.2em;'>Your predicted career path is best aligned with:</strong><br>
                    <span style='font-size:1.5em; font-weight:bold;'>{sector_label}</span>
                </div>
                <small style='color:#666;'>*Based on your inputs and NYU dataset analysis*</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Infer and display RIASEC scores
        status_text.text("🧠 Analyzing your personality profile (RIASEC)...")
        progress_bar.progress(50)
        
        user_profile_text = (
            f"Career Goals: {user_goals}\n"
            f"Soft Skills: {user_tech_skills}\n"
            f"Research Interests: {research_interests}\n"
            f"Desired Industry: {desired_industry}\n"
            f"Work Environment: {work_env}\n"
            f"Cohort Stage: {cohort_stage}\n"
        )
        
        # Show immediate feedback
        st.success("✅ Career prediction complete!")
        
        # Generate RIASEC scores based on AI availability
        if use_openai:
            with st.spinner("Analyzing your personality profile with AI..."):
                riasec_scores = advisor.infer_riasec_scores_via_gpt(user_profile_text)
            
            if riasec_scores:
                st.markdown("### 🧠 Your RIASEC Personality Profile")
                st.markdown("*Powered by GPT-4 Analysis*")
            else:
                st.warning("⚠️ AI analysis failed. Using sample profile.")
                riasec_scores = generate_sample_riasec_scores(user_profile_text)
                st.markdown("### 🧠 Your RIASEC Personality Profile")
                st.markdown("*Using sample analysis (AI was unavailable)*")
        else:
            # Use sample RIASEC scores when AI is disabled
            riasec_scores = generate_sample_riasec_scores(user_profile_text)
            st.markdown("### 🧠 Your RIASEC Personality Profile")
            st.markdown("*Using sample analysis (AI disabled to save tokens)*")
            
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
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
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
            status_text.text("📋 Generating personalized career recommendations...")
            progress_bar.progress(75)
            
            with st.spinner("Generating your personalized recommendations..."):
                recommendations = advisor.generate_career_advice(
                    selected_division=selected_division,
                    school=user_school,
                    citizenship=user_citizenship,
                    income=user_income,
                    user_goals=user_goals,
                    user_tech_skills=user_tech_skills,
                    work_env=work_env,
                    research_interests=research_interests,
                    desired_industry=desired_industry,
                    work_life_balance=work_life_balance,
                    mentorship_preference=mentorship_preference,
                    cohort_stage=cohort_stage,
                    family_location="",
                    skill_levels=skill_levels,
                    job_function=job_function,
                    financial_scholarships=financial_scholarships,
                    funding=funding,
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
                
                st.markdown("### 📋 Your Personalized Career Recommendations")
                st.markdown("*Generated with AI based on your profile and NYU data*")
                
                # Create a beautiful recommendations container
                st.markdown(f"""
                <div style='background:white; border:2px solid #57068c; border-radius:12px; padding:25px; margin:20px 0; box-shadow:0 4px 12px rgba(87,6,140,0.1);'>
                    {recommendations.replace('</div>', '')}
                </div>
                """, unsafe_allow_html=True)
                
                # --- Feedback section: only show after recommendations ---
                st.markdown("<h3 style='margin-top: 30px;'> Feedback</h3>", unsafe_allow_html=True)
                st.markdown("We'd love to hear from you. Your feedback helps us improve future recommendations!")

                # Simple feedback form without session state
                feedback = st.radio(
                    "Did you find these career recommendations helpful?",
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
                st.warning("No recommendations could be generated. Please check your inputs.")
        else:
            # When AI is disabled, complete progress and show message
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            st.info("💡 **Career Recommendations Disabled**")
            st.markdown("""
            To get personalized career recommendations:
            1. ✅ Check the "Enable Career Recommendations" checkbox in the sidebar
            2. 🔄 Click "Get Career Recommendations" again
            3. 🤖 AI will generate personalized advice based on your profile
            """)

    # --- Feedback section: always visible below recommendations button ---
    st.markdown("---")
    st.markdown("<h3 style='margin-top: 30px;'>📝 Feedback</h3>", unsafe_allow_html=True)
    st.markdown("We'd love to hear from you. Your feedback helps us improve future recommendations!")

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
        st.success("✅ Thanks for your feedback!")

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
# Job Postings Tab (unchanged)
# ----------------------------------------------------------------------------
def job_postings_tab():
    st.markdown("<h1 class='header'>Job Postings</h1>", unsafe_allow_html=True)

    st.info("Please complete the 'Career Recommendations' tab first to see your personalized recommendations.")

    st.markdown("<h2 class='subheader'>Career Decision Factors</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    How important were the following factors in making a decision to work in a job not closely related to the field of your first PhD?
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
            help="This helps us recommend jobs with better compensation"
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
            help="This helps us find jobs with better working conditions"
        )
        # Work conditions stored for this session
        
    elif selected_factor == "Professional interests":
        new_interests = st.text_area(
            "What are your new professional interests or goals?",
            help="This helps us align job recommendations with your interests"
        )
        # New interests stored for this session
        
    elif selected_factor == "Family-related reasons":
        family_considerations = st.text_area(
            "What family-related factors are important? (e.g., need for flexible hours, proximity to family)",
            help="This helps us find jobs that accommodate your family needs"
        )
        # Family considerations stored for this session

    st.markdown("<h2 class='subheader'>Real-Time Job Search</h2>", unsafe_allow_html=True)

    # Modify search location based on career decision factors
    desired_industry = "Industry"  # Default value
    
    # Define industry mapping
    industry_map = {
        "Academia": "professor",
        "Industry": "engineer",
        "Government": "government",
        "Non-profit": "non profit"
    }
    
    # Define alternative cities for job search
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

    # Adjust job query based on career decision factors
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

    user_query = st.text_input("Additional Job Keywords (optional)", value=base_query)

    if st.button("Find Jobs"):
        with st.spinner("Fetching real-time listings..."):
            job_data = fetch_adzuna_jobs(
                query=user_query,
                location=search_location
            )
            if job_data:
                # Filter and sort jobs based on career decision factors
                if selected_factor == "Pay" and previous_salary:
                    try:
                        # Extract numeric value from salary string
                        salary_num = float(''.join(filter(str.isdigit, previous_salary)))
                        # Filter jobs with higher salary
                        original_count = len(job_data['results'])
                        job_data['results'] = [
                            job for job in job_data['results']
                            if job.get('salary_min', 0) > salary_num
                        ]
                        filtered_count = len(job_data['results'])
                        if filtered_count < original_count:
                            st.info(f"💰 Filtered to {filtered_count} jobs with higher salary than {previous_salary}")
                    except ValueError:
                        st.warning("⚠️ Could not parse salary for filtering")
                
                display_adzuna_jobs(job_data)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    try:
        st.title("🎓 NYU PhD Career Advisor")
        st.write("Welcome! Let's help you find your career path.")
        
        advisor = NYUCareerAdvisor(df)
        tab1, tab2 = st.tabs(["Career Recommendations", "Job Postings"])
        with tab1:
            career_recommendations_tab(advisor)
        with tab2:
            job_postings_tab()
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