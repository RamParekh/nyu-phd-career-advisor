#!/usr/bin/env python3
"""
Simple debug script to test app loading
"""

import streamlit as st
import pandas as pd
import os

def test_basic_app():
    """Test basic app functionality"""
    st.title("🧪 Debug Test App")
    st.write("If you can see this, the app is loading!")
    
    # Test data loading
    st.subheader("📊 Testing Data Loading")
    try:
        if os.path.exists("data/imputed_data_NYU copy 3.xlsx"):
            df = pd.read_excel("data/imputed_data_NYU copy 3.xlsx")
            st.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
            st.write("First few columns:", list(df.columns[:5]))
        else:
            st.error("❌ Data file not found")
    except Exception as e:
        st.error(f"❌ Data loading error: {e}")
    
    # Test basic Streamlit features
    st.subheader("🎯 Testing Basic Features")
    name = st.text_input("Enter your name:", "Test User")
    if st.button("Click me!"):
        st.success(f"Hello {name}! Button works!")
    
    # Test sidebar
    st.sidebar.title("Sidebar Test")
    st.sidebar.write("Sidebar is working!")
    
    # Test tabs
    tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
    with tab1:
        st.write("Tab 1 content")
    with tab2:
        st.write("Tab 2 content")

if __name__ == "__main__":
    test_basic_app() 