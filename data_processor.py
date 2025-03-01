import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Import our custom modules
from data_cleaner import FinancialDataCleaner
from data_loader import FinancialDataLoader

def main():
    """
    Example of how to use the data_cleaner and data_loader modules
    together with a simple machine learning model.
    """
    # Step 1: Clean the data
    print("=== Step 1: Cleaning Data ===")
    cleaner = FinancialDataCleaner("financial_regression.csv", "cleaned_financial_data.csv")
    cleaned_data = cleaner.full_cleaning_pipeline(visualize=False)
    
    # Step 2: Load the cleaned data
    print("\n=== Step 2: Loading Data ===")
    loader = FinancialDataLoader("cleaned_financial_data.csv")
    
    # Step 3: Prepare data for the model
    # Let's try to predict SP500 close price
    target_column = "sp500 close"
    
    # Choose features - remove other asset close prices to avoid leakage
    features = [col for col in cleaned_data.columns 
                if not (col.endswith(" close") and col != target_column) 
                and col != "date"
                and not col.startswith("palladium")  # For demonstration, exclude one asset group
                ]
    
    print(f"\nTarget: {target_column}")
    print(f"Number of features: {len(features)}")
    print(f"Sample features: {features[:5]}")
    
    # Get data splits
    X_train, X_val, X_test, y_train, y_val, y_test = loader.get_data_pipeline(
        target_column=target_column,
        features=features,
        time_series_split=True  # Use time-series split for financial data
    )
    
    # Step 4: Train a simple model
    print("\n=== Step 3: Training Model ===")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train.values.ravel())
    
    # Step 5: Evaluate the model
    print("\n=== Step 4: Evaluating Model ===")
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Convert predictions back to original scale
    y_train_pred_orig = loader.inverse_transform_target(y_train_pred)
    y_val_pred_orig = loader.inverse_transform_target(y_val_pred)
    y_test_pred_orig = loader.inverse_transform_target(y_test_pred)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(loader.inverse_transform_target(y_train.values), y_train_pred_orig))
    val_rmse = np.sqrt(mean_squared_error(loader.inverse_))