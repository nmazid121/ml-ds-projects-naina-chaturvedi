import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Load the Iris Dataset ---
print("1. Loading the Iris dataset...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=['species_id'])

# The target 'species_id' is numerical (0, 1, 2).
# We need to map it to actual species names for better readability
# and then one-hot encode it if using a model that requires it (e.g., for multi-output classification
# or if we were predicting the one-hot encoded output directly).
# For standard multi-class classifiers like LogisticRegression, it handles integer labels directly,
# but if the prompt specifically asks for one-hot encoding the *labels*, we can do it for the
# demonstration even if the model doesn't strictly require it for its internal training.

# For classification report, it's good to have target names
target_names = iris.target_names
print(f"Iris Features (X) Head:\n{X.head()}")
print(f"\nIris Target (y) Head (original numerical):\n{y.head()}")
print("-" * 50)

# --- 2. Data Preprocessing ---
print("2. Applying One-Hot Encoding, Polynomial Features, and Scaling...")

# Define continuous features
continuous_features = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]

# Split data into training and testing sets before preprocessing to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- One-hot Encoding for Target Variable (y) ---
# Although LogisticRegression handles multi-class directly, if the task explicitly
# requires one-hot encoding the *labels* (e.g., for certain neural networks or
# custom loss functions), here's how you'd apply it.
# For this specific Logistic Regression, we'll keep y_train/y_test as numerical labels
# and demonstrate OHE on a copy for illustration.
print("Illustrative One-Hot Encoding of Target Labels (not strictly needed for Logistic Regression):")
ohe_target_encoder = OneHotEncoder(sparse_output=False)
y_train_ohe = ohe_target_encoder.fit_transform(y_train)
y_test_ohe = ohe_target_encoder.transform(y_test)
print(f"One-Hot Encoded y_train shape: {y_train_ohe.shape}")
print(f"First 5 One-Hot Encoded y_train rows:\n{y_train_ohe[:5]}")
print("-" * 50)


# Create a preprocessing pipeline for features (X)
# We use ColumnTransformer to apply different transformations to different columns.
preprocessor = ColumnTransformer(
    transformers=[
        # Apply StandardScaler for scaling continuous variables
        ('scaler', StandardScaler(), continuous_features),
        # Create Polynomial Features for continuous variables
        # degree=2 adds squared terms and interaction terms (e.g., sepal_length^2, sepal_length*sepal_width)
        ('poly', PolynomialFeatures(degree=2, include_bias=False), continuous_features)
    ],
    remainder='passthrough' # Keep any other columns (if any) as they are
)

# --- 3. Build and Train the Model Pipeline ---
print("3. Building and training the Classification model pipeline...")

# Create a pipeline that first preprocesses the data and then applies Logistic Regression
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42)) # Increased max_iter for convergence
])

# Train the model
model_pipeline.fit(X_train, y_train.values.ravel()) # .values.ravel() converts DataFrame to 1D array

print("\nModel training complete.")
print("-" * 50)

# --- 4. Evaluate the Model ---
print("4. Evaluating the model...")
y_pred = model_pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy:.4f}")

# Display classification report for more detailed metrics
print("\nClassification Report on Test Data:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nIris Flower Classification project complete! 🎉")
