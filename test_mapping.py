def map_division_to_major_group(division):
    social_science_keywords = [
        'psychology', 'sociology', 'economics', 'political', 'education',
        "social work", 'public health', 'business', 'management', 'marketing',
        'finance', 'accounting', 'law', 'criminal justice', 'urban planning',
        'international relations', 'communication', 'journalism', 'public policy'
    ]
    
    division_lower = str(division).lower()
    
    for keyword in social_science_keywords:
        if keyword in division_lower:
            return 'Social Science'
    
    return division

# Test the mapping
print("Economics ->", map_division_to_major_group('Economics'))
print("Computer Science ->", map_division_to_major_group('Computer Science'))
print("Philosophy ->", map_division_to_major_group('Philosophy'))

# Check if mapped groups exist in sentiment data
sentiment_groups = ['Humanities', 'Natural Science', 'Social Science']
print("\nMapped groups in sentiment data:")
print("Social Science:", 'Social Science' in sentiment_groups)
print("Natural Science:", 'Natural Science' in sentiment_groups)
print("Humanities:", 'Humanities' in sentiment_groups) 