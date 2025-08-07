import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Tuple, Dict, Any
import joblib
from datetime import datetime
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


class EnhancedWaterPredictionModel:
    """
    Enhanced XGBoost model for water consumption prediction with advanced features
    """

    def __init__(self, data_path: str = 'D:/iot/iot6.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_importance = None
        self.cv_scores = None

    def load_and_explore_data(self) -> pd.DataFrame:
        """Load data and perform comprehensive exploration"""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")

            # Data exploration
            print("=" * 60)
            print("DATA EXPLORATION REPORT")
            print("=" * 60)
            print(f"Dataset shape: {df.shape}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

            # Basic statistics
            print("\n Basic Statistics:")
            print(df.describe())

            # Data types
            print("\n Data Types:")
            print(df.dtypes)

            # Missing values analysis
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                print("\n Missing Values:")
                print(missing_data[missing_data > 0])
            else:
                print("\n No missing values found")

            # Duplicate analysis
            duplicates = df.duplicated().sum()
            print(f"\n Duplicate rows: {duplicates}")

            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def advanced_data_preprocessing(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced data preprocessing with feature engineering"""
        columns = df.columns
        # Handle missing values
        if df.isnull().sum().sum() > 0:
            # Numerical columns
            num_cols = df.select_dtypes(include=[np.number]).columns
            num_imputer = SimpleImputer(strategy='median')
            df[num_cols] = num_imputer.fit_transform(df[num_cols])

            # Categorical columns
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

        # Feature Engineering - Create interaction features
        # These features might be relevant for water consumption prediction
        # 1. Tỷ lệ ẩm đất / độ ẩm không khí (tránh chia cho 0)
        if {'soil_moisture', 'air_humidity'}.issubset(columns):
            df['moisture_humidity_ratio'] = df['soil_moisture'] / (df['air_humidity'].replace(0, np.nan) + 1e-6)
            df['moisture_humidity_ratio'] = df['moisture_humidity_ratio'].fillna(0)

        # 2. Tương tác giữa nhiệt độ và độ ẩm không khí
        if {'air_temp', 'air_humidity'}.issubset(columns):
            df['temp_humidity_interaction'] = df['air_temp'] * df['air_humidity']
        # 3. Tương tác giữa ánh sáng và nhiệt độ không khí
        if {'light_intensity', 'air_temp'}.issubset(columns):
            df['light_temp_interaction'] = df['light_intensity'] * df['air_temp']
        # 4. Hiệu độ ẩm đất - độ ẩm không khí (sự chênh lệch giữa đất và không khí)
        if {'soil_moisture', 'air_humidity'}.issubset(columns):
            df['moisture_difference'] = df['soil_moisture'] - df['air_humidity']

        # 5. Tương tác giữa mưa và độ ẩm đất
        if {'rain_detected', 'soil_moisture'}.issubset(columns):
            df['rain_soil_effect'] = df['rain_detected'] * df['soil_moisture']

        # 6. Tương tác giữa nhiệt độ đất và nhiệt độ không khí
        if {'soil_temp', 'air_temp'}.issubset(columns):
            df['soil_air_temp_diff'] = df['soil_temp'] - df['air_temp']
        # One-hot encoding for categorical variables
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            logger.info(f"Applied one-hot encoding to: {categorical_cols}")

        # Convert boolean to int
        bool_cols = df.select_dtypes(include='bool').columns
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(int)
            logger.info(f"Converted boolean columns to int: {list(bool_cols)}")

        # Outlier detection and handling using IQR method
        if 'water' in df.columns:
            Q1 = df['water'].quantile(0.25)
            Q3 = df['water'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_count = ((df['water'] < lower_bound) | (df['water'] > upper_bound)).sum()
            logger.info(f"Detected {outliers_count} outliers in target variable")

            # Remove extreme outliers for better model performance
            initial_size = len(df)
            df = df[(df['water'] >= lower_bound) & (df['water'] <= upper_bound)]
            logger.info(f"Removed {initial_size - len(df)} outliers")

        # Feature and target separation
        if 'water' not in df.columns:
            raise ValueError("Target column 'water' not found in dataset")

        X = df.drop('water', axis=1)
        y = df['water']

        self.feature_names = X.columns.tolist()
        logger.info(f"Features prepared: {len(self.feature_names)} columns")
        logger.info(f"Feature names: {self.feature_names}")

        return X.values, y.values

    def visualize_data(self, df: pd.DataFrame) -> None:
        """Create comprehensive data visualizations"""

        plt.figure(figsize=(20, 15))

        # Target distribution
        plt.subplot(3, 3, 1)
        plt.hist(df['water'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Water Consumption Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Water (liters/day)')
        plt.ylabel('Frequency')

        # Box plot for outlier detection
        plt.subplot(3, 3, 2)
        plt.boxplot(df['water'])
        plt.title('Water Consumption Box Plot', fontsize=12, fontweight='bold')
        plt.ylabel('Water (liters/day)')
        #
        # # Correlation heatmap (for numerical features only)
        # plt.subplot(3, 3, 3)
        # numeric_cols = df.select_dtypes(include=[np.number]).columns
        # if len(numeric_cols) > 1:
        #     corr_matrix = df[numeric_cols].corr()
        #     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
        #                 square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        #     plt.title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

        # Statistical summary visualization
        plt.subplot(3, 3, 4)
        stats = df['water'].describe()
        labels = ['Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
        values = [stats['mean'], stats['std'], stats['min'],
                  stats['25%'], stats['50%'], stats['75%'], stats['max']]
        plt.bar(labels, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
                                       '#FFEAA7', '#DDA0DD', '#98D8C8'])
        plt.title(' Water Consumption Statistics', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)

        # Data quality overview
        plt.subplot(3, 3, 5)
        quality_metrics = {
            'Total Rows': len(df),
            'Complete Rows': len(df.dropna()),
            'Missing Values': df.isnull().sum().sum(),
            'Duplicates': df.duplicated().sum(),
            'Unique Waters': df['water'].nunique()
        }
        plt.bar(quality_metrics.keys(), quality_metrics.values(),
                color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC'])
        plt.title(' Data Quality Overview', fontsize=10, fontweight='bold')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters using GridSearchCV with safe settings"""

        logger.info("Starting hyperparameter optimization...")

        # Safe parameter grid to avoid memory issues
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1, 1.5]
        }

        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=1,  # Changed to 1 to avoid multiprocessing issues
            tree_method='hist'  # More stable tree method
        )

        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=3,  # Reduced CV folds
            scoring='neg_mean_squared_error',
            n_jobs=1,  # Sequential processing to avoid crashes
            verbose=1
        )

        try:
            grid_search.fit(X_train, y_train)
            logger.info(f"Best parameters found: {grid_search.best_params_}")
            logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
            return grid_search.best_params_
        except Exception as e:
            logger.warning(f"GridSearchCV failed: {e}. Using default parameters.")
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 4,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }

    def train_model(self, X: np.ndarray, y: np.ndarray, optimize: bool = True) -> None:
        """Train the enhanced XGBoost model"""

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

        # Feature scaling - using RobustScaler for better outlier handling
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Hyperparameter optimization
        if optimize:
            best_params = self.optimize_hyperparameters(X_train_scaled, y_train)
        else:
            best_params = {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.5
            }

        # Train final model
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            **best_params
        )

        logger.info("Training final model...")
        self.model.fit(X_train_scaled, y_train)

        # Cross-validation scores
        self.cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=5, scoring='neg_mean_squared_error'
        )

        # Feature importance
        self.feature_importance = self.model.feature_importances_

        # Model evaluation
        self.evaluate_model(X_test_scaled, y_test, y_train)

        # Store test data for visualization
        self.X_test = X_test_scaled
        self.y_test = y_test

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, y_train: np.ndarray) -> None:
        """Comprehensive model evaluation"""

        y_pred = self.model.predict(X_test)

        # Metrics calculation
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Cross-validation statistics
        cv_rmse_mean = np.sqrt(-self.cv_scores.mean())
        cv_rmse_std = np.sqrt(self.cv_scores.std())

        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE REPORT")
        print("=" * 60)
        print(f"Test Set Metrics:")
        print(f"   RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"   MAE (Mean Absolute Error):      {mae:.4f}")
        print(f"   R² Score:                       {r2:.4f}")
        print(f"   MAPE (Mean Abs Percentage Err): {mape:.2f}%")

        print(f"\n Cross-Validation (5-fold):")
        print(f"   CV RMSE Mean: {cv_rmse_mean:.4f} (±{cv_rmse_std:.4f})")

        print(f"\n Data Statistics:")
        print(f"   Target Mean: {y_train.mean():.4f}")
        print(f"   Target Std:  {y_train.std():.4f}")
        print(f"   Prediction Mean: {y_pred.mean():.4f}")
        print(f"   Prediction Std:  {y_pred.std():.4f}")

    def visualize_results(self) -> None:
        """Visualize model results and feature importance"""

        if self.model is None:
            logger.warning("Model not trained yet!")
            return

        y_pred = self.model.predict(self.X_test)

        plt.figure(figsize=(20, 12))

        # Prediction vs Actual scatter plot
        plt.subplot(2, 3, 1)
        plt.scatter(self.y_test, y_pred, alpha=0.6, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Water Consumption')
        plt.ylabel('Predicted Water Consumption')
        plt.title('Prediction vs Actual')
        plt.grid(True, alpha=0.3)

        # Residuals plot
        plt.subplot(2, 3, 2)
        residuals = self.y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)

        # Feature importance
        plt.subplot(2, 3, 3)
        if self.feature_importance is not None and self.feature_names is not None:
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=True).tail(10)

            plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()

        # Distribution comparison
        plt.subplot(2, 3, 4)
        plt.hist(self.y_test, alpha=0.5, label='Actual', bins=30, color='blue')
        plt.hist(y_pred, alpha=0.5, label='Predicted', bins=30, color='red')
        plt.xlabel('Water Consumption')
        plt.ylabel('Frequency')
        plt.title('Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Cross-validation scores
        plt.subplot(2, 3, 5)
        cv_rmse = np.sqrt(-self.cv_scores)
        plt.bar(range(1, 6), cv_rmse, color='orange', alpha=0.7)
        plt.axhline(y=cv_rmse.mean(), color='red', linestyle='--',
                    label=f'Mean: {cv_rmse.mean():.4f}')
        plt.xlabel('Fold')
        plt.ylabel('RMSE')
        plt.title('Cross-Validation RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Error distribution
        plt.subplot(2, 3, 6)
        plt.hist(residuals, bins=30, alpha=0.7, color='purple')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str = None) -> None:
        """Save the trained model and scaler"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"water_prediction_model_{timestamp}.pkl"

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a pre-trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        logger.info(f"Model loaded from: {filepath}")

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained or loaded!")

        X_new_scaled = self.scaler.transform(X_new)
        return self.model.predict(X_new_scaled)


# Main execution
def main():
    """Main function to run the enhanced water prediction model"""

    # Initialize model
    model = EnhancedWaterPredictionModel()

    # Load and explore data
    df = model.load_and_explore_data()

    # Visualize data
    model.visualize_data(df)

    # Preprocess data
    X, y = model.advanced_data_preprocessing(df)

    # Train model (set optimize=False for faster execution)
    model.train_model(X, y, optimize=True)

    # Visualize results
    model.visualize_results()

    # Save model
    model.save_model()

    logger.info("Model training and evaluation completed successfully!")


if __name__ == "__main__":
    main()