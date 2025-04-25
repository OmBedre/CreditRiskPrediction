from model import train_model
from riskFactor import calculate_risk

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap

# Set page configuration
st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

# Data loading and preprocessing
@st.cache_data
def load_data():
    data = pd.read_csv('german_credit_data.csv')
    return data

def analyze_credit_profile(row):
    """Helper function to analyze credit amount relative to age and duration"""
    credit_per_month = row['Credit amount'] / row['Duration']
    return credit_per_month

def remove_outliers(df, columns):
    df_clean = df.copy()
    
    for column in columns:
        # Get the original dtype of the column
        original_dtype = df_clean[column].dtype
        
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers with bounds and explicitly cast to original dtype
        df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound.astype(original_dtype)
        df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound.astype(original_dtype)
    
    return df_clean

def preprocess_data(data):
    # Create copy of data
    df = data.copy()
    
    # Clean outliers for numerical columns
    numerical_columns = ['Age', 'Credit amount', 'Duration']
    df = remove_outliers(df, numerical_columns)
    
    # Calculate risk after cleaning outliers
    df['Risk'] = df.apply(calculate_risk, axis=1)
    
    # Handle categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns)
    
    return df

def plot_class_distribution(y_train, y_train_balanced):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Before balancing
    sns.countplot(x=y_train, ax=ax1)
    ax1.set_title('Class Distribution Before Balancing')
    ax1.set_xlabel('Risk Category')
    
    # After balancing
    sns.countplot(x=y_train_balanced, ax=ax2)
    ax2.set_title('Class Distribution After Balancing')
    ax2.set_xlabel('Risk Category')
    
    return fig

# Streamlit UI
def plot_outliers(data, column):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Before cleaning
    sns.boxplot(data=data, y=column, ax=ax1)
    ax1.set_title(f'{column} Before Cleaning')
    
    # After cleaning
    cleaned_data = remove_outliers(data, [column])
    sns.boxplot(data=cleaned_data, y=column, ax=ax2)
    ax2.set_title(f'{column} After Cleaning')
    
    return fig

def main():
    st.title("Credit Risk Prediction System")
    
    # Load data
    data = load_data()
    
    # Sidebar with removed Prediction page
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Data Analysis", "Model Performance"])
    
    if page == "Data Analysis":
        st.header("Data Analysis")
        st.write("Dataset Overview")
        st.dataframe(data.head())
        
        # Add outlier analysis
        st.subheader("Outlier Analysis")
        numerical_columns = ['Age', 'Credit amount', 'Duration']
        for column in numerical_columns:
            st.write(f"\nOutlier Analysis for {column}")
            fig = plot_outliers(data, column)
            st.pyplot(fig)
        
        # Add risk distribution visualization
        st.subheader("Risk Distribution")
        processed_data = preprocess_data(data)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=processed_data, x='Risk')
        plt.title("Distribution of Risk (0: Low Risk, 1: High Risk)")
        st.pyplot(fig)
        
        # Display sample entries for risky and non-risky applicants
        st.subheader("Sample Entries for Risky and Non-Risky Applicants")
        risky_samples = processed_data[processed_data['Risk'] == 1].head(5)
        non_risky_samples = processed_data[processed_data['Risk'] == 0].head(5)
        
        st.write("Risky Applicants (Risk = 1):")
        st.dataframe(risky_samples)
        
        st.write("Non-Risky Applicants (Risk = 0):")
        st.dataframe(non_risky_samples)
        
        # Additional risk analysis
        st.subheader("Risk Factors Analysis")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Age vs Risk
        sns.boxplot(data=processed_data, x='Risk', y='Age', ax=axes[0,0])
        axes[0,0].set_title('Age Distribution by Risk')
        
        # Credit Amount vs Risk
        sns.boxplot(data=processed_data, x='Risk', y='Credit amount', ax=axes[0,1])
        axes[0,1].set_title('Credit Amount Distribution by Risk')
        
        # Duration vs Risk
        sns.boxplot(data=processed_data, x='Risk', y='Duration', ax=axes[1,0])
        axes[1,0].set_title('Duration Distribution by Risk')
        
        # Job vs Risk
        sns.boxplot(data=processed_data, x='Risk', y='Job', ax=axes[1,1])
        axes[1,1].set_title('Job Distribution by Risk')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    else:  # Model Performance page
        st.header("Model Performance")
        
        # Preprocess data and train model
        df = preprocess_data(data)
        X = df.drop('Risk', axis=1)
        y = df['Risk']
        
        model, scaler, X_test_scaled, y_test, X_train_scaled, y_train, X_train_balanced, y_train_balanced = train_model(X, y)
        
        # Show class distribution
        st.subheader("Class Balance Analysis")
        fig = plot_class_distribution(y_train, y_train_balanced)
        st.pyplot(fig)
        
        # Display metrics
        y_pred = model.predict(X_test_scaled)
        st.write("Classification Report:")
        st.code(classification_report(y_test, y_pred))
        
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(fig)
        
        # Feature importance (using absolute values of coefficients for logistic regression)
        st.subheader("Feature Importance")
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importances.head(10), x='importance', y='feature')
        plt.title("Top 10 Most Important Features")
        st.pyplot(fig)
        
if __name__ == "__main__":
    main()
