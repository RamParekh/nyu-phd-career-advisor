# 🚀 Streamlit Cloud Deployment Guide

## Prerequisites ✅
- ✅ Your app code is ready (`app1.py`)
- ✅ Your real data files are in the `data/` folder
- ✅ You've converted data to secrets (using `convert_data_to_secrets.py`)
- ✅ `requirements.txt` is configured
- ✅ `.gitignore` is set up properly

## Step 1: Push to GitHub

1. **Initialize Git repository** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: NYU PhD Career Advisor"
   ```

2. **Create a new repository on GitHub:**
   - Go to [GitHub.com](https://github.com)
   - Click "New repository"
   - Name it: `nyu-phd-career-advisor`
   - Make it **Public** (required for free Streamlit Cloud)
   - Don't initialize with README (you already have one)

3. **Push your code to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/nyu-phd-career-advisor.git
   git branch -M main
   git push -u origin main
   ```

## Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Deploy your app:**
   - Click "New app"
   - Select your repository: `nyu-phd-career-advisor`
   - Set **Main file path**: `app1.py`
   - Click "Deploy!"

## Step 3: Configure Secrets

1. **In Streamlit Cloud dashboard:**
   - Go to your deployed app
   - Click "Settings" (gear icon)
   - Click "Secrets"

2. **Add your secrets:**
   ```toml
   [api_keys]
   openai_key = "your-openai-api-key-here"
   adzuna_app_id = "your-adzuna-app-id-here"
   adzuna_app_key = "your-adzuna-app-key-here"

   [data_files]
   nyu_data = "your-base64-encoded-nyu-data"
   handshake_events = "your-base64-encoded-events-data"
   sentiment_data = "your-base64-encoded-sentiment-data"
   training_data = "your-base64-encoded-training-data"
   ```

3. **Get your base64 data:**
   - Run: `python convert_data_to_secrets.py`
   - Copy the output to your Streamlit secrets

## Step 4: Test Your Deployment

1. **Check your app URL:**
   - Your app will be available at: `https://your-app-name.streamlit.app`

2. **Test all features:**
   - Career recommendations
   - Job postings
   - Sentiment analysis
   - ML predictions

## Troubleshooting

### Common Issues:

1. **"Module not found" errors:**
   - Check `requirements.txt` includes all dependencies
   - Add missing packages to requirements.txt

2. **"Data files not found" errors:**
   - Verify secrets are configured correctly
   - Check base64 data is complete

3. **"API key errors":**
   - Ensure API keys are in Streamlit secrets
   - Verify keys are valid and have proper permissions

### Debug Commands:

```bash
# Test locally before deploying
streamlit run app1.py

# Check if all dependencies are installed
pip install -r requirements.txt

# Verify data files exist
ls -la data/
```

## Security Notes

- ✅ **Never commit API keys** to GitHub
- ✅ **Use Streamlit secrets** for sensitive data
- ✅ **Keep data files private** (use base64 encoding)
- ✅ **Monitor usage** of API keys

## Next Steps

1. **Customize your app:**
   - Update the title and description
   - Add your branding
   - Customize the UI

2. **Monitor performance:**
   - Check Streamlit Cloud dashboard
   - Monitor API usage
   - Track user engagement

3. **Share your app:**
   - Share the URL with NYU students
   - Add to your portfolio
   - Present at conferences

## Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Test locally first
3. Verify all secrets are configured
4. Check GitHub repository is public

Your app should now be live at: `https://your-app-name.streamlit.app` 🎉 