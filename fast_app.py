#!/usr/bin/env python3
"""
Super Fast NYU PhD Career Advisor - Streamlined for Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import os
import json

# Set page config
st.set_page_config(
    page_title="NYU PhD Career Advisor - Fast",
    page_icon="🎓",
    layout="wide"
)

# Global variables
use_openai = os.getenv('OPENAI_API_KEY') is not None
df = None

@st.cache_data
def load_data():
    """Load data once and cache it"""
    try:
        df = pd.read_excel("data/imputed_data_NYU copy 3.xlsx")
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

def simple_sector_prediction(user_goals, desired_industry, work_env):
    """Simple keyword-based sector prediction - no ML, no API calls"""
    text = f"{user_goals} {desired_industry} {work_env}".lower()
    
    if any(word in text for word in ["research", "professor", "academic", "university", "teaching"]):
        return "Academic"
    elif any(word in text for word in ["government", "public", "policy", "federal"]):
        return "Government"
    elif any(word in text for word in ["non-profit", "charity", "social"]):
        return "Non-profit"
    else:
        return "Industry"

def simple_riasec_scores(user_text):
    """Simple RIASEC scoring based on keywords - no API calls"""
    text = user_text.lower()
    
    scores = {
        "Realistic": 3,
        "Investigative": 4,
        "Artistic": 3,
        "Social": 4,
        "Enterprising": 3,
        "Conventional": 3
    }
    
    # Adjust based on keywords
    if any(word in text for word in ["data", "analysis", "research", "science"]):
        scores["Investigative"] += 2
    if any(word in text for word in ["creative", "design", "art", "writing"]):
        scores["Artistic"] += 2
    if any(word in text for word in ["team", "collaboration", "helping", "teaching"]):
        scores["Social"] += 2
    if any(word in text for word in ["leadership", "management", "business", "startup"]):
        scores["Enterprising"] += 2
    if any(word in text for word in ["organization", "detail", "process", "structure"]):
        scores["Conventional"] += 2
    
    return scores

def main():
    st.title("🎓 NYU PhD Career Advisor - Fast Version")
    st.markdown("---")
    
    # Load data
    with st.spinner("📊 Loading data..."):
        global df
        df = load_data()
    
    if df is None:
        st.error("❌ Could not load data. Please check if the Excel file exists.")
        return
    
    st.success(f"✅ Data loaded: {len(df)} records")
    
    # Simple form
    st.subheader("📝 Your Career Information")
    
    col1, col2 = st.columns(2)
    with col1:
        user_goals = st.text_area("Career Goals:", 
                                 placeholder="e.g., I want to work in data science and machine learning")
        desired_industry = st.selectbox("Desired Industry:", 
                                      ["Industry", "Academic", "Government", "Non-profit"])
        work_env = st.selectbox("Preferred Work Environment:", 
                               ["Collaborative", "Independent", "Hybrid"])
    
    with col2:
        soft_skills = st.text_area("Soft Skills:", 
                                  placeholder="e.g., communication, teamwork, leadership")
        research_interests = st.text_input("Research Interests:", 
                                         placeholder="e.g., machine learning, public health")
        cohort_stage = st.selectbox("PhD Stage:", ["First Year", "ABD"])
    
    # Simple button
    if st.button("🚀 Get Fast Recommendations", type="primary"):
        st.markdown("---")
        st.subheader("🎯 Your Career Analysis")
        
        # 1. Quick sector prediction
        with st.spinner("🔮 Predicting career sector..."):
            predicted_sector = simple_sector_prediction(user_goals, desired_industry, work_env)
        
        # Display sector prediction
        st.markdown(f"""
        <div style='background:#57068c; color:white; padding:20px; border-radius:10px; margin:20px 0; text-align:center;'>
            <h3>🎯 Career Path Prediction</h3>
            <h2 style='font-size:2em; margin:10px 0;'>{predicted_sector}</h2>
            <p>Based on your inputs and NYU alumni data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Quick RIASEC scores
        with st.spinner("📊 Calculating RIASEC scores..."):
            user_profile = f"{user_goals} {soft_skills} {research_interests}"
            riasec_scores = simple_riasec_scores(user_profile)
        
        # Display RIASEC scores
        st.subheader("📊 RIASEC Interest Profile")
        cols = st.columns(3)
        for i, (category, score) in enumerate(riasec_scores.items()):
            with cols[i % 3]:
                st.metric(category, score, f"{score}/7")
        
        # 3. Quick data insights
        with st.spinner("📈 Analyzing alumni data..."):
            if 'academic_division' in df.columns:
                divisions = df['academic_division'].value_counts().head(5)
                st.subheader("🎓 Top Academic Divisions")
                for div, count in divisions.items():
                    st.write(f"• **{div}**: {count} alumni")
        
        # 4. Quick recommendations
        st.subheader("💡 Quick Recommendations")
        
        if predicted_sector == "Academic":
            st.markdown("""
            **For Academic Careers:**
            • Focus on publishing in top journals
            • Build strong teaching portfolio
            • Network at academic conferences
            • Consider postdoc positions
            """)
        elif predicted_sector == "Industry":
            st.markdown("""
            **For Industry Careers:**
            • Develop technical skills (Python, R, etc.)
            • Build portfolio of projects
            • Network on LinkedIn
            • Consider internships during PhD
            """)
        elif predicted_sector == "Government":
            st.markdown("""
            **For Government Careers:**
            • Look into federal fellowship programs
            • Develop policy analysis skills
            • Network at government events
            • Consider research agencies (NIH, NSF, etc.)
            """)
        else:
            st.markdown("""
            **For Non-profit Careers:**
            • Focus on social impact
            • Develop grant writing skills
            • Network in non-profit sector
            • Consider research organizations
            """)
        
        st.success("✅ Analysis complete! Recommendations generated successfully.")
    
    # Debug info in sidebar
    st.sidebar.subheader("🔍 Debug Info")
    st.sidebar.write(f"Data shape: {df.shape}")
    st.sidebar.write(f"Environment: {'🌐 Cloud' if os.path.exists('/mount/src') else '💻 Local'}")
    st.sidebar.write(f"OpenAI: {'✅ Available' if use_openai else '❌ Not configured'}")

if __name__ == "__main__":
    main() 