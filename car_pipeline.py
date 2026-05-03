import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Step 1: Custom Feature Adder - CarAttributesAdder Class
# ============================================================================

class CarAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add new features to the dataset.
    Adds: Kms_per_Year = Kms_Driven / Car_Age
    """
    
    def __init__(self, reference_year=2026):
        self.reference_year = reference_year
    
    def fit(self, X, y=None):
        """Fit method (no learning needed)"""
        return self
    
    def transform(self, X):
        """Add new feature: Kms_per_Year"""
        X_copy = X.copy()
        
        # Calculate car age
        car_age = self.reference_year - X_copy['Year']
        
        # Avoid division by zero for brand new cars
        car_age = car_age.replace(0, 1)
        
        # Calculate kilometers per year
        X_copy['Kms_per_Year'] = X_copy['Kms_Driven'] / car_age
        
        return X_copy


# ============================================================================
# Step 2: Load and Prepare Data
# ============================================================================

# Load the dataset
df = pd.read_csv('data/vehicle_data.csv')

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Separate features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

print(f"\n\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")


# ============================================================================
# Step 3: Build the Pipeline
# ============================================================================

# Define numerical and categorical columns
numerical_cols = ['Year', 'Present_Price', 'Kms_Driven']
categorical_cols = ['Fuel_Type']

# Create transformers for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values
    ('scaler', StandardScaler())  # Scale to mean=0, std=1
])

# Create transformers for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create the full pipeline with custom feature adder
full_pipeline = Pipeline(steps=[
    ('feature_adder', CarAttributesAdder(reference_year=2026)),
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])


# ============================================================================
# Step 4: Train the Model
# ============================================================================

print("\n\n" + "=" * 70)
print("TRAINING THE MODEL")
print("=" * 70)

full_pipeline.fit(X, y)

print("✓ Model trained successfully!")


# ============================================================================
# Step 5: Check Transformed Data Shape
# ============================================================================

print("\n\n" + "=" * 70)
print("SHAPE CHECK - AFTER PIPELINE TRANSFORMATION")
print("=" * 70)

# Get the preprocessor part and fit it to see the shape
preprocessor.fit(CarAttributesAdder(reference_year=2026).transform(X))
X_transformed = preprocessor.transform(CarAttributesAdder(reference_year=2026).transform(X))

print(f"Original feature matrix shape: {X.shape}")
print(f"Transformed feature matrix shape: {X_transformed.shape}")
print(f"\nColumns increased from {X.shape[1]} to {X_transformed.shape[1]}")
print(f"Reason: OneHotEncoder converted 'Fuel_Type' categorical column into multiple binary columns")

# Show the mapping
unique_fuels = X['Fuel_Type'].unique()
print(f"\nFuel Types in data: {unique_fuels}")
print(f"OneHotEncoder created {len(unique_fuels)} binary columns for {len(categorical_cols)} categorical column")
print(f"Plus 3 original numerical columns + 1 new engineered feature (Kms_per_Year) = {X_transformed.shape[1]} total features")


# ============================================================================
# Step 6: Model Predictions and Evaluation
# ============================================================================

print("\n\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

# Make predictions
y_pred = full_pipeline.predict(X)

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mse = mean_squared_error(y, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Calculate R² score
from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred)
print(f"R² Score: {r2:.4f}")

print("\nPrediction vs Actual (first 10 samples):")
comparison_df = pd.DataFrame({
    'Actual': y.values[:10],
    'Predicted': y_pred[:10],
    'Error': abs(y.values[:10] - y_pred[:10])
})
print(comparison_df.to_string(index=False))


# ============================================================================
# Step 7: Make a Prediction on New Car Data
# ============================================================================

print("\n\n" + "=" * 70)
print("PREDICTION ON NEW CAR")
print("=" * 70)

# Create a new car data point
new_car = pd.DataFrame({
    'Year': [2020],
    'Present_Price': [8.5],
    'Kms_Driven': [25000],
    'Fuel_Type': ['Petrol']
})

print(f"New car data:\n{new_car.to_string(index=False)}")

# Make prediction
predicted_price = full_pipeline.predict(new_car)[0]

print(f"\nPredicted Selling Price: ₹{predicted_price:.2f} Lakhs")
print(f"\nInterpretation: A {new_car['Year'].values[0]} {new_car['Fuel_Type'].values[0]} car")
print(f"with Present Price ₹{new_car['Present_Price'].values[0]} Lakhs")
print(f"and {new_car['Kms_Driven'].values[0]:,} km driven is predicted to sell at ₹{predicted_price:.2f} Lakhs")

print("\n" + "=" * 70)
print("PROJECT COMPLETE!")
print("=" * 70)