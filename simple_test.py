#!/usr/bin/env python3
"""
Simplified test to isolate the recommendation display issue
"""

import streamlit as st
import pandas as pd
import os

def simple_test():
    st.title("🎓 NYU PhD Career Advisor - Simple Test")
    
    # Load data
    try:
        df = pd.read_excel("data/imputed_data_NYU copy 3.xlsx")
        st.success(f"✅ Data loaded: {len(df)} records")
    except Exception as e:
        st.error(f"❌ Data loading failed: {e}")
        return
    
    # Simple form
    st.subheader("📝 Career Information")
    
    career_goals = st.text_input("Career Goals:", "I want to work in data science")
    soft_skills = st.text_input("Soft Skills:", "Communication, teamwork")
    
    # Simple button
    if st.button("🚀 Get Simple Recommendation"):
        st.subheader("🎯 Your Recommendation")
        
        # Simple logic
        if "data" in career_goals.lower() or "science" in career_goals.lower():
            sector = "Industry"
        elif "research" in career_goals.lower() or "professor" in career_goals.lower():
            sector = "Academic"
        elif "government" in career_goals.lower():
            sector = "Government"
        else:
            sector = "Industry"
        
        # Display recommendation
        st.markdown(f"""
        ### 🎯 Career Path Prediction
        **Your predicted career path is best aligned with:**
        
        <div style='background:#57068c; color:white; padding:15px; border-radius:8px; margin:10px 0;'>
            <strong style='font-size:1.5em;'>{sector}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Show some data insights
        st.subheader("📊 Data Insights")
        if 'academic_division' in df.columns:
            divisions = df['academic_division'].value_counts().head(5)
            st.write("Top academic divisions in our data:")
            for div, count in divisions.items():
                st.write(f"- {div}: {count} alumni")
        
        st.success("✅ Recommendation generated successfully!")
    
    # Debug info
    st.sidebar.subheader("🔍 Debug Info")
    st.sidebar.write(f"Data shape: {df.shape}")
    st.sidebar.write(f"Columns: {list(df.columns[:5])}")
    
    if os.path.exists("/mount/src"):
        st.sidebar.info("🌐 Running on Streamlit Cloud")
    else:
        st.sidebar.info("💻 Running locally")

if __name__ == "__main__":
    simple_test() 