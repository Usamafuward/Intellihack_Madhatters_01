import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_and_inspect_data(data_path):
    """Load and perform initial inspection of the data"""
    df = pd.read_csv(data_path)
    
    print("Initial data shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for basic information
    print("\nDataset info:")
    df.info()
    
    print("\nSummary statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    
    return df


def preprocess_data(df):
    """Preprocess the dataset"""
    # Handle date column
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # Create outlier boxplots
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure']):
        plt.subplot(2, 3, i+1)
        sns.boxplot(data=df[column])
        plt.title(f'Boxplot of {column}')
    plt.tight_layout()
    plt.savefig('boxplots.png')
    plt.close()
    
    # Handle outliers
    for column in ['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure']:
        df = handle_outliers(df, column)
    
    # Handle missing values
    numerical_features = ['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure', 'day_of_year', 'month', 'day']
    imputer = SimpleImputer(strategy='median')
    df[numerical_features] = imputer.fit_transform(df[numerical_features])
    
    # Check for any remaining missing values
    print("\nMissing values after imputation:")
    print(df.isnull().sum())
    
    # Save the preprocessed data
    df.to_csv('preprocessed_weather_data.csv', index=False)
    
    return df


def handle_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
    """Handle outliers using percentile-based clipping"""
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)
    df[column] = df[column].clip(lower_bound, upper_bound)
    return df


def perform_eda(df):
    """Perform exploratory data analysis"""
    # Distribution of target variable
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='rain_or_not')
    plt.title('Distribution of Rain vs No Rain Days')
    plt.savefig('target_distribution.png')
    plt.close()
    
    print("\nTarget variable distribution:")
    print(df['rain_or_not'].value_counts(normalize=True) * 100)
    
    # Create a numeric version of rain_or_not for correlation
    df_corr = df.copy()
    df_corr['rain_binary'] = df_corr['rain_or_not'].map({'Rain': 1, 'No Rain': 0})
    
    # Correlation analysis
    plt.figure(figsize=(12, 10))
    correlation_matrix = df_corr[['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure', 'rain_binary']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Feature distributions by target variable
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure']):
        plt.subplot(2, 3, i+1)
        sns.histplot(data=df, x=feature, hue='rain_or_not', kde=True, bins=20, element='step')
        plt.title(f'Distribution of {feature} by Rain Status')
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()
    
    # Pairplot
    sns.pairplot(df[['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure', 'rain_or_not']], 
                 hue='rain_or_not', diag_kind='kde')
    plt.savefig('pairplot.png')
    plt.close()
    
    # Monthly rain patterns
    monthly_rain = df.groupby('month')['rain_or_not'].apply(lambda x: (x == 'Rain').mean() * 100)
    plt.figure(figsize=(10, 6))
    monthly_rain.plot(kind='bar', color='skyblue')
    plt.title('Monthly Rainfall Probability')
    plt.xlabel('Month')
    plt.ylabel('Rainfall Probability (%)')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.savefig('monthly_rain.png')
    plt.close()


def engineer_features(df):
    """Create new features"""
    # Create interaction features
    df['temp_humidity_interaction'] = df['avg_temperature'] * df['humidity']
    df['wind_pressure_interaction'] = df['avg_wind_speed'] * df['pressure']
    
    # Seasonal features using sine and cosine transformations
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    return df


def prepare_data(df):
    """Prepare data for modeling"""
    # Create a copy of the original dataframe
    df_copy = df.copy()
    
    # Encode target variable
    le = LabelEncoder()
    df_copy['rain_encoded'] = le.fit_transform(df_copy['rain_or_not'])
    
    # Store the label encoder classes for later reference
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("\nLabel encoding mapping:")
    print(label_mapping)
    
    # Prepare features and target
    X = df_copy.drop(['date', 'rain_or_not', 'rain_encoded', 'day_of_year'], axis=1)
    y_encoded = df_copy['rain_encoded']
    y_original = df_copy['rain_or_not']
    
    # Split the data
    X_train, X_test, y_train_encoded, y_test_encoded, y_train_original, y_test_original = train_test_split(
        X, y_encoded, y_original, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train, X_test, y_train_encoded, y_test_encoded, 
            y_train_original, y_test_original, X_train_scaled, X_test_scaled, 
            scaler, le, label_mapping)


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, le, label_mapping):
    """Evaluate a model and produce performance metrics"""
    # Train the model
    model.fit(X_train, y_train)
    
    # Get predictions (encoded)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of the positive class
    
    # Convert predictions back to original labels if needed for display
    y_pred_original = le.inverse_transform(y_pred)
    y_test_original = le.inverse_transform(y_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Display metrics
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix (using original labels for better readability)
    cm = confusion_matrix(y_test_original, y_pred_original)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', label=f'{model_name} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f'roc_curve_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # Plot feature importance if available
    if hasattr(model, 'feature_importances_') and X_train.__class__.__name__ == 'DataFrame':
        importance = model.feature_importances_
        feat_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_importance.head(10), x='Importance', y='Feature')
        plt.title(f'Top 10 Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.replace(" ", "_").lower()}.png')
        plt.close()
    
    # Return metrics for comparison
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'model_object': model
    }


def train_models(X_train, y_train, X_test, y_test, X_train_orig, le, label_mapping):
    """Train and evaluate multiple models"""
    # Define models to test
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42)
    }
    
    # Evaluate each model
    results = []
    for model_name, model in models.items():
        result = evaluate_model(model, X_train, y_train, X_test, y_test, model_name, le, label_mapping)
        results.append(result)
    
    # Compile results into a dataframe
    results_df = pd.DataFrame(results).drop(columns='model_object')
    print("\nModel Comparison:")
    print(results_df)
    
    # Create a bar chart to compare models
    plt.figure(figsize=(12, 8))
    results_melted = results_df.melt(id_vars='model', value_vars=['accuracy', 'precision', 'recall', 'f1', 'auc'])
    sns.barplot(data=results_melted, x='model', y='value', hue='variable')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    return results, results_df


def tune_best_model(results, results_df, X_train, y_train, X_test, y_test, X_train_orig, le, label_mapping):
    """Perform hyperparameter tuning on the best model"""
    # Identify the best performing model based on F1 score
    best_model_name = results_df.loc[results_df['f1'].idxmax(), 'model']
    print(f"\nBest performing model: {best_model_name}")
    
    # Define hyperparameter grids for different models
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
        },
        'Decision Tree': {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    }
    
    # Get the best model and its parameter grid
    best_model = [m['model_object'] for m in results if m['model'] == best_model_name][0]
    param_grid = param_grids[best_model_name]
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=best_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )
    
    print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and results
    print("\nBest parameters:")
    print(grid_search.best_params_)
    
    print("\nBest cross-validation score (F1):", grid_search.best_score_)
    
    # Evaluate the tuned model
    tuned_model = grid_search.best_estimator_
    tuned_model_result = evaluate_model(tuned_model, X_train, y_train, X_test, y_test, f"Tuned {best_model_name}", le, label_mapping)
    
    # Compare original vs tuned model
    print("\nPerformance Improvement:")
    original_f1 = [r['f1'] for r in results if r['model'] == best_model_name][0]
    tuned_f1 = tuned_model_result['f1']
    improvement = (tuned_f1 - original_f1) / original_f1 * 100
    print(f"Original {best_model_name} F1 Score: {original_f1:.4f}")
    print(f"Tuned {best_model_name} F1 Score: {tuned_f1:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    
    return tuned_model, grid_search.best_params_, X_train_orig


def analyze_feature_importance(final_model, X_train_orig):
    """Analyze feature importance of the final model"""
    if hasattr(final_model, 'feature_importances_'):
        importance = final_model.feature_importances_
        feat_importance = pd.DataFrame({
            'Feature': X_train_orig.columns,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feat_importance, x='Importance', y='Feature')
        plt.title('Feature Importance - Final Model')
        plt.tight_layout()
        plt.savefig('final_model_feature_importance.png')
        plt.close()
        
        print("\nTop 5 most important features:")
        print(feat_importance.head(5))
    
    return final_model


def generate_forecast(df, final_model, scaler, le):
    """Generate a 21-day forecast"""
    # For demonstration, use the last 21 days of data for forecasting
    recent_data = df.tail(21).copy()
    # Shift dates forward by 21 days
    recent_data['date'] = recent_data['date'] + pd.Timedelta(days=21)
    recent_data.drop('rain_or_not', axis=1, inplace=True)
    
    # Extract the features for prediction
    forecast_features = recent_data.drop(['date', 'day_of_year'], axis=1)
    
    # Scale the features
    forecast_features_scaled = scaler.transform(forecast_features)
    
    # Make predictions
    rain_probabilities = final_model.predict_proba(forecast_features_scaled)[:, 1]
    rain_predictions_encoded = final_model.predict(forecast_features_scaled)
    rain_predictions = le.inverse_transform(rain_predictions_encoded)
    
    # Create a forecast dataframe
    forecast_df = pd.DataFrame({
        'Date': recent_data['date'],
        'Rain_Probability': rain_probabilities * 100,
        'Prediction': rain_predictions
    })
    
    print("\nRain Probability Forecast for the Next 21 Days:")
    print(forecast_df)
    
    # Visualize the forecast
    plt.figure(figsize=(14, 7))
    bars = plt.bar(forecast_df['Date'], forecast_df['Rain_Probability'], color='skyblue')
    
    # Color the bars based on the prediction
    for i, pred in enumerate(forecast_df['Prediction']):
        if pred == 'Rain':
            bars[i].set_color('navy')
    
    plt.axhline(y=50, color='red', linestyle='--', label='50% Threshold')
    plt.title('Rain Probability Forecast for the Next 21 Days')
    plt.xlabel('Date')
    plt.ylabel('Rain Probability (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(['50% Threshold', 'No Rain', 'Rain'])
    plt.savefig('rain_forecast.png')
    plt.close()
    
    return forecast_df


def save_model(final_model, scaler, le):
    """Save the trained model and scaler to disk"""
    # Save the model
    joblib.dump(final_model, 'rain_prediction_model.joblib')
    # Save the scaler
    joblib.dump(scaler, 'feature_scaler.joblib')
    # Save the label encoder
    joblib.dump(le, 'label_encoder.joblib')
    print("\nModel, scaler, and label encoder saved to disk.")


def create_prediction_function():
    """Create and demonstrate a function for making predictions"""
    # Load the model, scaler, and label encoder
    loaded_model = joblib.load('rain_prediction_model.joblib')
    loaded_scaler = joblib.load('feature_scaler.joblib')
    loaded_le = joblib.load('label_encoder.joblib')
    
    def predict_rain_probability(new_data, model_features):
        """
        Make rain predictions for new data
        
        Parameters:
        new_data (DataFrame): DataFrame with the same features as the training data
        model_features (list): List of feature names the model was trained on
        
        Returns:
        DataFrame: Original data with rain probability and prediction
        """
        # Make sure the data has the same features as the model was trained on
        required_features = model_features
        
        # Check if all required features are present
        missing_features = set(required_features) - set(new_data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select only the required features and in the same order
        X_new = new_data[required_features]
        
        # Scale the features
        X_new_scaled = loaded_scaler.transform(X_new)
        
        # Make predictions
        probabilities = loaded_model.predict_proba(X_new_scaled)[:, 1]
        predictions_encoded = loaded_model.predict(X_new_scaled)
        predictions = loaded_le.inverse_transform(predictions_encoded)
        
        # Add predictions to the original data
        result = new_data.copy()
        result['rain_probability'] = probabilities * 100
        result['rain_prediction'] = predictions
        
        return result
    
    print("\nFunction created for making predictions on new data.")
    return predict_rain_probability


def main():
    # Set the data path
    data_path = 'weather_data.csv'
    
    # 1. Load and inspect data
    df = load_and_inspect_data(data_path)
    
    # 2. Preprocess data
    df = preprocess_data(df)
    
    # 3. Exploratory data analysis
    perform_eda(df)
    
    # 4. Feature engineering
    df = engineer_features(df)
    
    # 5. Prepare data for modeling
    (X_train, X_test, y_train_encoded, y_test_encoded, 
     y_train_original, y_test_original, X_train_scaled, X_test_scaled, 
     scaler, le, label_mapping) = prepare_data(df)
    
    X_train_orig = pd.DataFrame(X_train, columns=X_train.columns)
    
    # 6. Train and evaluate models
    results, results_df = train_models(
        X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, 
        X_train_orig, le, label_mapping
    )
    
    # 7. Tune the best model
    final_model, best_params, X_train_orig = tune_best_model(
        results, results_df, X_train_scaled, y_train_encoded, 
        X_test_scaled, y_test_encoded, X_train_orig, le, label_mapping
    )
    
    # 8. Analyze feature importance
    final_model = analyze_feature_importance(final_model, X_train_orig)
    
    # 9. Generate forecast
    _ = generate_forecast(df, final_model, scaler, le)
    
    # 10. Save model
    save_model(final_model, scaler, le)
    
    # 11. Create prediction function
    _ = create_prediction_function()
    
    print("End of program.")


if __name__ == "__main__":
    main()