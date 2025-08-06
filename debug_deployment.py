#!/usr/bin/env python3
"""
Debug script to test what works locally vs on Streamlit Cloud
"""

import sys
import os

def test_imports():
    """Test all imports used in the app"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except Exception as e:
        print(f"❌ streamlit import failed: {e}")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except Exception as e:
        print(f"❌ pandas import failed: {e}")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except Exception as e:
        print(f"❌ numpy import failed: {e}")
    
    try:
        import openai
        print("✅ openai imported successfully")
    except Exception as e:
        print(f"❌ openai import failed: {e}")
    
    try:
        import contractions
        print("✅ contractions imported successfully")
    except Exception as e:
        print(f"❌ contractions import failed: {e}")
    
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("✅ vaderSentiment imported successfully")
    except Exception as e:
        print(f"❌ vaderSentiment import failed: {e}")
    
    try:
        from textblob import TextBlob
        print("✅ textblob imported successfully")
    except Exception as e:
        print(f"❌ textblob import failed: {e}")
    
    try:
        import openpyxl
        print("✅ openpyxl imported successfully")
    except Exception as e:
        print(f"❌ openpyxl import failed: {e}")

def test_data_loading():
    """Test if data files can be loaded"""
    print("\n📊 Testing data loading...")
    
    data_files = [
        "data/imputed_data_NYU copy 3.xlsx",
        "data/Events for Graduate Students.csv", 
        "data/sentiment analysis- Final combined .xlsx",
        "data/phd_career_sector_training_data_final_1200.xlsx"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
            try:
                if file_path.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(file_path)
                else:
                    import pandas as pd
                    df = pd.read_excel(file_path)
                print(f"✅ {file_path} loaded successfully ({len(df)} rows)")
            except Exception as e:
                print(f"❌ {file_path} failed to load: {e}")
        else:
            print(f"❌ {file_path} not found")

def test_environment():
    """Test environment variables and paths"""
    print("\n🌍 Testing environment...")
    
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    if os.path.exists('data'):
        print(f"Files in data directory: {os.listdir('data')}")
    else:
        print("❌ data directory not found")

if __name__ == "__main__":
    print("🚀 Starting deployment debug...")
    test_imports()
    test_data_loading()
    test_environment()
    print("\n✅ Debug complete!") 