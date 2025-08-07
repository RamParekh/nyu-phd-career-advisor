def map_division_to_major_group(division):
    """Map detailed academic divisions to three major groups: Humanities, Natural Science, Social Science."""
    
    # Define mapping dictionaries for each major group
    humanities_keywords = [
        'humanities', 'philosophy', 'history', 'literature', 'english', 'languages', 
        'linguistics', 'classics', 'religious', 'theology', 'art', 'music', 'drama',
        'theater', 'film', 'media', 'cultural', 'anthropology', 'archaeology'
    ]
    
    natural_science_keywords = [
        'biology', 'chemistry', 'physics', 'mathematics', 'computer science', 
        'engineering', 'medical', 'health', 'neuroscience', 'biochemistry',
        'molecular', 'genetics', 'ecology', 'environmental', 'geology',
        'astronomy', 'biomedical', 'pharmaceutical', 'biotechnology'
    ]
    
    social_science_keywords = [
        'psychology', 'sociology', 'economics', 'political', 'education',
        "social work", 'public health', 'business', 'management', 'marketing',
        'finance', 'accounting', 'law', 'criminal justice', 'urban planning',
        'international relations', 'communication', 'journalism', 'public policy'
    ]
    
    division_lower = str(division).lower()
    
    # Check for matches in each category
    for keyword in humanities_keywords:
        if keyword in division_lower:
            return 'Humanities'
    
    for keyword in natural_science_keywords:
        if keyword in division_lower:
            return 'Natural Science'
    
    for keyword in social_science_keywords:
        if keyword in division_lower:
            return 'Social Science'
    
    # Default mapping for common patterns
    if any(word in division_lower for word in ['basic medical', 'medical science']):
        return 'Natural Science'
    elif any(word in division_lower for word in ['computer', 'cs', 'software']):
        return 'Natural Science'
    elif any(word in division_lower for word in ['bio', 'biological']):
        return 'Natural Science'
    
    # If no match found, return the original division
    return division

# Test the mapping
test_divisions = ['Economics', 'Computer Science', 'Philosophy', 'Biology', 'Psychology', 'Mathematics', 'History', 'Physics']

print("Testing division mapping:")
for division in test_divisions:
    mapped = map_division_to_major_group(division)
    print(f"  {division} -> {mapped}") 