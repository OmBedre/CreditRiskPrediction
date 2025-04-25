def calculate_risk(row):
    risk_score = 0
    
    # Monthly payment burden relative to job stability
    monthly_payment = row['Credit amount'] / row['Duration']
    if row['Job'] == 0 and monthly_payment > 300:  # unskilled non-resident
        risk_score += 5
    elif row['Job'] == 1 and monthly_payment > 400:  # unskilled resident
        risk_score += 4
    elif row['Job'] == 2 and monthly_payment > 600:  # skilled
        risk_score += 3
    elif row['Job'] == 3 and monthly_payment > 800:  # highly skilled
        risk_score += 2

    # Savings to credit ratio with duration consideration
    savings_map = {'little': 1000, 'moderate': 5000, 'quite rich': 15000, 'rich': 50000}
    savings_amount = savings_map.get(row['Saving accounts'], 0)
    credit_to_savings_ratio = row['Credit amount'] / (savings_amount if savings_amount > 0 else 1)
    
    if credit_to_savings_ratio > 10 and row['Duration'] > 36:
        risk_score += 6  # Very high ratio with long duration
    elif credit_to_savings_ratio > 5:
        risk_score += 4
    elif credit_to_savings_ratio > 3:
        risk_score += 2

    # Age and duration combined risk
    if row['Age'] < 25 and row['Duration'] > 24:
        risk_score += 4  # Young with long-term commitment
    elif row['Age'] > 60 and row['Duration'] > 36:
        risk_score += 4  # Elderly with long-term commitment

    # Purpose-based risk
    high_risk_purposes = ['business', 'vacation/others']
    medium_risk_purposes = ['car', 'education']
    if row['Purpose'] in high_risk_purposes:
        risk_score += 3
    elif row['Purpose'] in medium_risk_purposes:
        risk_score += 2

    # Housing and savings combination
    if row['Housing'] == 'rent' and row['Saving accounts'] in ['little', 'moderate']:
        risk_score += 4  # Renting with low savings
    elif row['Housing'] == 'free' and row['Saving accounts'] == 'little':
        risk_score += 3  # Free housing with low savings

    # Final threshold for classification
    return 1 if risk_score >= 6 else 0
