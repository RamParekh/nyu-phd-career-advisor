import base64
import pandas as pd
import json

def file_to_base64(file_path):
    """Convert a file to base64 string."""
    with open(file_path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

def convert_data_files():
    """Convert all data files to base64 for secure storage."""
    
    data_files = {
        'nyu_data': 'data/imputed_data_NYU copy 3.xlsx',
        'handshake_events': 'data/Events for Graduate Students.csv',
        'sentiment_data': 'data/sentiment analysis- Final combined .xlsx',
        'training_data': 'data/phd_career_sector_training_data_final_1200.xlsx'
    }
    
    secrets_data = {}
    
    for key, file_path in data_files.items():
        try:
            base64_data = file_to_base64(file_path)
            secrets_data[key] = base64_data
            print(f"✅ Converted {file_path} to base64")
        except Exception as e:
            print(f"❌ Error converting {file_path}: {e}")
    
    # Save to a JSON file (you can copy this to Streamlit secrets)
    with open('data_secrets.json', 'w') as f:
        json.dump(secrets_data, f, indent=2)
    
    print("\n📋 Copy the following to your Streamlit secrets:")
    print("=" * 50)
    print("[data_files]")
    for key, base64_data in secrets_data.items():
        print(f"{key} = \"{base64_data}\"")
    print("=" * 50)

if __name__ == "__main__":
    convert_data_files() 