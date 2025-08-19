# How to Add Your Adzuna API Keys

## Step 1: Get Your Free API Keys
1. Go to [https://developer.adzuna.com/](https://developer.adzuna.com/)
2. Sign up for a free account
3. Create a new application
4. Copy your **App ID** and **App Key**

## Step 2: Add Keys to the Code
1. Open `app1.py` in your editor
2. Find these lines (around line 90-95):
```python
# Replace these with your actual Adzuna API keys
ADZUNA_APP_ID = "your-actual-adzuna-app-id-here"  # Replace this
ADZUNA_APP_KEY = "your-actual-adzuna-app-key-here"  # Replace this
```

3. Replace the placeholder text with your actual keys:
```python
# Replace these with your actual Adzuna API keys
ADZUNA_APP_ID = "12345678"  # Your actual App ID
ADZUNA_APP_KEY = "abcdef123456789"  # Your actual App Key
```

## Step 3: Save and Deploy
1. Save the file
2. Commit and push to GitHub
3. Your app will automatically redeploy
4. The job search will now work!

## What You'll See:
- **Before**: Warning "Job API Keys Not Set"
- **After**: Success "Job API Keys Configured"
- **Job Postings tab**: Will show real job listings instead of sample data

## Example of Working Keys:
```python
ADZUNA_APP_ID = "a1b2c3d4"
ADZUNA_APP_KEY = "e5f6g7h8i9j0k1l2m3n4o5p6"
```

**Note**: Keep your API keys private and don't share them publicly!
