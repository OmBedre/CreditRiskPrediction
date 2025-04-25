from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Define model and parameters for tuning
    model = LogisticRegression(random_state=42, max_iter=2000)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'class_weight': [None, 'balanced', {0:1, 1:2}, {0:1, 1:3}],
        'solver': ['liblinear', 'saga']
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    # Fit the model
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    return best_model, scaler, X_test_scaled, y_test, X_train_scaled, y_train, X_train_balanced, y_train_balanced
