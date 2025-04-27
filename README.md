# Credit Risk Prediction System

## Overview
This project predicts the credit risk of loan applicants using the German Credit dataset. It classifies applicants into two categories:
- **Risky (1):** High risk of default
- **Non-Risky (0):** Low risk of default

The system uses a Logistic Regression model with SMOTE for class balancing and hyperparameter tuning.

## Features
- **Data Analysis:** Visualize data distributions, outliers, and risk factors.
- **Model Performance:** Evaluate the model using metrics like classification report and confusion matrix.
- **Risk Calculation:** Custom logic to calculate risk based on applicant features.

## File Structure
- `creditRisk.py`: Main Streamlit application.
- `model.py`: Contains the `train_model` function for training the Logistic Regression model.
- `riskFactor.py`: Contains the `calculate_risk` function for risk scoring.
- `ReadMe.md`: Project documentation.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run creditRisk.py
   ```

## Dataset
The project uses the German Credit dataset. Ensure the dataset is placed in the project directory as `german_credit_data.csv`.

## Dependencies
- Python 3.8+
- pandas
- numpy
- streamlit
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- shap

## Future Enhancements
- Add support for additional machine learning models.
- Improve risk calculation logic with advanced feature engineering.
- Add more visualizations for better insights.

## ScreenShots

![classification report](<screenshots/Screenshot 2025-04-25 103048.png>)
![Confusion matrix](<screenshots/Screenshot 2025-04-25 103105.png>)
![Feature importance](<screenshots/Screenshot 2025-04-25 103121.png>)
![Feature analysis](<screenshots/Screenshot 2025-04-25 103147.png>)

## demo video
https://drive.google.com/file/d/1Jv_7ngVmHd7RJae8ky2UJykCGwVIv5tm/view?usp=drive_link
