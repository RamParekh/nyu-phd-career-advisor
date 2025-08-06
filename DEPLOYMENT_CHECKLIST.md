# ✅ Streamlit Deployment Checklist

## Pre-Deployment Checklist

### 1. Code Files ✅
- [x] `app1.py` - Main application file
- [x] `requirements.txt` - All dependencies included
- [x] `.gitignore` - Properly configured
- [x] `README.md` - Project documentation

### 2. Data Files ✅
- [x] Real NYU career data loaded
- [x] Real Handshake events data loaded
- [x] Real sentiment analysis data loaded
- [x] Real training data loaded

### 3. Secrets Configuration
- [ ] Convert data to base64 (run `python convert_data_to_secrets.py`)
- [ ] Get API keys ready:
  - [ ] OpenAI API key
  - [ ] Adzuna App ID
  - [ ] Adzuna App Key

## Deployment Steps

### Step 1: GitHub Setup
```bash
# Initialize git (if not done)
git init
git add .
git commit -m "Initial commit: NYU PhD Career Advisor"

# Create GitHub repository
# Go to github.com and create new repo: nyu-phd-career-advisor

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/nyu-phd-career-advisor.git
git branch -M main
git push -u origin main
```

### Step 2: Streamlit Cloud Deployment
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `nyu-phd-career-advisor`
5. Set Main file path: `app1.py`
6. Click "Deploy!"

### Step 3: Configure Secrets
In Streamlit Cloud dashboard:
1. Go to your app → Settings → Secrets
2. Add this configuration:

```toml
[api_keys]
openai_key = "your-openai-api-key"
adzuna_app_id = "your-adzuna-app-id"
adzuna_app_key = "your-adzuna-app-key"

[data_files]
nyu_data = "your-base64-encoded-nyu-data"
handshake_events = "your-base64-encoded-events-data"
sentiment_data = "your-base64-encoded-sentiment-data"
training_data = "your-base64-encoded-training-data"
```

## Testing Checklist

### Local Testing ✅
- [x] App runs locally: `streamlit run app1.py`
- [x] All data files load correctly
- [x] All features work (career recommendations, job postings, sentiment analysis)

### Deployment Testing
- [ ] App deploys successfully
- [ ] Secrets are configured correctly
- [ ] All features work on Streamlit Cloud
- [ ] API keys are valid and working

## Troubleshooting

### Common Issues:
1. **"Module not found"** → Check `requirements.txt`
2. **"Data files not found"** → Verify secrets configuration
3. **"API key errors"** → Check API keys in secrets
4. **"Repository not found"** → Ensure GitHub repo is public

### Debug Commands:
```bash
# Test locally
streamlit run app1.py

# Check dependencies
pip install -r requirements.txt

# Convert data to secrets
python convert_data_to_secrets.py
```

## Success Indicators ✅
- [ ] App URL is accessible
- [ ] All tabs load without errors
- [ ] Career recommendations work
- [ ] Job postings display
- [ ] Sentiment analysis functions
- [ ] ML predictions work

Your app will be live at: `https://your-app-name.streamlit.app` 🎉 