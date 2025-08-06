# Data Setup Guide for NYU PhD Career Advisor

## Overview

The NYU PhD Career Advisor application requires four main data files to function properly. This guide explains how to set up these files and what data they should contain.

## Required Data Files

### 1. `data/imputed_data_NYU copy 3.xlsx`
**Purpose**: Main NYU career dataset containing alumni career information
**Required Columns**:
- `division`: Academic division (e.g., "Arts and Science", "Engineering")
- `school`: School within NYU (e.g., "GSAS", "Tandon", "Stern")
- `citizenship`: Citizenship status ("US Citizen", "International")
- `income`: Income range (e.g., "$50k-75k", "$75k-100k")
- `career_sector`: Career sector (e.g., "Academia", "Industry", "Government")
- `job_title`: Job title
- `company`: Company name
- `location`: Geographic location
- `satisfaction_score`: Career satisfaction score (1-5)
- `salary`: Salary information

### 2. `data/Events for Graduate Students.csv`
**Purpose**: Handshake events and career development opportunities
**Required Columns**:
- `event_name`: Name of the event
- `date`: Event date
- `time`: Event time
- `location`: Event location
- `division`: Target division(s)
- `description`: Event description
- `registration_required`: Boolean for registration requirement

### 3. `data/sentiment analysis- Final combined .xlsx`
**Purpose**: Sentiment analysis data from career-related feedback
**Required Columns**:
- `division`: Academic division
- `comment`: Text feedback/comments
- `sentiment_score`: Sentiment score (-1 to 1)
- `theme`: Theme category (e.g., "Career Services", "Networking")
- `year`: Year of feedback

### 4. `data/phd_career_sector_training_data_final_1200.xlsx`
**Purpose**: Machine learning training data for career sector prediction
**Required Columns**:
- `profile`: Text description of career goals/interests
- `target_sector`: Target career sector
- `confidence_score`: Model confidence score

## Setup Options

### Option 1: Use Sample Data (Current Setup)
The application currently uses sample data files that were created automatically. These files contain synthetic data that allows the application to run with basic functionality.

**Pros**:
- ✅ Application runs immediately
- ✅ No data preparation required
- ✅ Good for testing and development

**Cons**:
- ❌ Limited functionality with synthetic data
- ❌ Recommendations not based on real NYU data
- ❌ Sentiment analysis uses fake feedback

### Option 2: Use Real NYU Data
Replace the sample files with actual NYU career data for full functionality.

**Steps**:
1. Obtain the required data files from NYU career services
2. Ensure the data matches the required column structure
3. Replace the sample files in the `data/` folder
4. Restart the application

**Pros**:
- ✅ Full functionality with real data
- ✅ Accurate career recommendations
- ✅ Real sentiment analysis insights

**Cons**:
- ❌ Requires access to real NYU data
- ❌ Data preparation and cleaning needed

### Option 3: Use Streamlit Secrets (Recommended for Deployment)
Convert your data files to base64 and store them in Streamlit secrets for secure deployment.

**Steps**:
1. Place your real data files in the `data/` folder
2. Run the conversion script:
   ```bash
   python convert_data_to_secrets.py
   ```
3. Copy the generated base64 strings to your Streamlit secrets
4. Deploy to Streamlit Cloud

**Pros**:
- ✅ Secure data storage
- ✅ No data files in repository
- ✅ Works with Streamlit Cloud deployment

## Data File Formats

### Excel Files (.xlsx)
- Use pandas to read/write
- Supports multiple sheets
- Good for complex data structures

### CSV Files (.csv)
- Simple text format
- Easy to edit manually
- Good for event data

## Data Quality Requirements

### Required Data Quality
- **No missing values** in key columns
- **Consistent formatting** for categorical variables
- **Valid date formats** for event data
- **Appropriate text length** for comments (max 800 characters)

### Data Cleaning Tips
1. **Handle duplicates**: Remove or merge duplicate records
2. **Standardize categories**: Use consistent naming for divisions, sectors
3. **Clean text data**: Remove special characters, standardize formatting
4. **Validate dates**: Ensure all dates are in proper format
5. **Check data types**: Ensure numeric columns contain numbers

## Troubleshooting

### Common Issues

1. **"Missing required data files" error**
   - Ensure all 4 files are in the `data/` folder
   - Check file names match exactly (including spaces)
   - Verify file permissions

2. **"Error loading [file]" error**
   - Check file format (Excel files must be .xlsx, not .xls)
   - Verify file is not corrupted
   - Ensure required columns are present

3. **"Column not found" error**
   - Check column names match exactly
   - Verify no extra spaces in column names
   - Ensure required columns exist

### Debug Data Files
Run the debug function in the app:
```python
# Add this to app1.py temporarily
debug_data_files()
```

## Next Steps

1. **For Development**: Use the current sample data setup
2. **For Production**: Replace with real NYU data
3. **For Deployment**: Use Streamlit secrets method
4. **For Customization**: Modify the data loading functions in `app1.py`

## Support

If you encounter issues with data setup:
1. Check the error messages in the application
2. Verify file formats and column names
3. Use the debug functions to identify issues
4. Refer to the pandas documentation for data loading 