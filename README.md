# NYU PhD Career Advisor

A comprehensive career advisory application for NYU PhD students, providing personalized career recommendations, job postings, and sentiment analysis.

## Features

- **Career Recommendations**: AI-powered personalized career advice based on academic division and user profile
- **RIASEC Personality Analysis**: Vocational interest assessment using GPT-4
- **Real-time Job Postings**: Live job search using Adzuna API
- **Sentiment Analysis**: Advanced sentiment analysis of career-related comments
- **NYU Dataset Integration**: Uses real NYU career data for recommendations

## Deployment

### Streamlit Cloud Deployment

1. **Fork/Clone this repository**
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
3. **Connect your GitHub account**
4. **Select this repository**
5. **Set the main file path to**: `app1.py`
6. **Deploy!**

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app1.py
```

## Data Files

The app requires the following data files in the `data/` folder:

- `imputed_data_NYU copy 3.xlsx` - NYU career dataset
- `Events for Graduate Students.csv` - Handshake events data
- `sentiment analysis- Final combined .xlsx` - Sentiment analysis data
- `phd_career_sector_training_data_final_1200.xlsx` - ML training data

## API Keys

The app uses the following APIs:
- **OpenAI API**: For career advice and RIASEC analysis
- **Adzuna API**: For real-time job postings

API keys are configured in the app code.

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning models
- **OpenAI GPT-4**: Natural language processing
- **TextBlob & VADER**: Sentiment analysis
- **XGBoost**: Advanced ML algorithms 