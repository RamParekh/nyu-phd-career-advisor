#!/usr/bin/env python3
"""
Test data loading for deployment
"""

import streamlit as st
import pandas as pd
import os

def test_data_loading():
    st.title("🧪 Data Loading Test")
    
    # Test file existence
    st.subheader("📁 File Existence Test")
    data_files = [
        "data/imputed_data_NYU copy 3.xlsx",
        "data/Events for Graduate Students.csv",
        "data/sentiment analysis- Final combined .xlsx",
        "data/phd_career_sector_training_data_final_1200.xlsx"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            st.success(f"✅ {file_path} exists")
            try:
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)
                st.info(f"📊 {file_path}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                st.error(f"❌ Error reading {file_path}: {e}")
        else:
            st.error(f"❌ {file_path} not found")
    
    # Test current directory
    st.subheader("📂 Directory Info")
    st.write(f"Current directory: {os.getcwd()}")
    st.write(f"Files in current directory: {len(os.listdir('.'))}")
    
    if os.path.exists('data'):
        st.write(f"Files in data directory: {len(os.listdir('data'))}")
    else:
        st.error("❌ data directory not found")
    
    # Test basic Streamlit
    st.subheader("🎯 Streamlit Test")
    if st.button("Test Button"):
        st.success("✅ Button works!")
    
    name = st.text_input("Enter your name:", "Test User")
    st.write(f"Hello {name}!")

if __name__ == "__main__":
    test_data_loading() 