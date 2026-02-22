import pandas as pd
import sys

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import joblib

import pandas as pd

def load_csv_for_decision_tree(csv_path, target_column):
    """
    Loads a CSV file and returns features (X) and target (y) for scikit-learn.

    Args:
        csv_path (str): Path to the CSV file.
        target_column (str): Name of the target column.

    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
    """

    df = pd.read_csv(csv_path).drop(columns=["patientID"])
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def encode_categorical_columns(df):
    """
    Encodes all string (object or category) columns in the DataFrame with unique integer codes.
    Returns a new DataFrame with encoded columns and a dictionary mapping columns to their value mappings.
    """
    encoded_df = df.copy()
    mappings = {}
    for col in encoded_df.select_dtypes(include=['object', 'category']).columns:
        encoded_df[col], uniques = pd.factorize(encoded_df[col])
        mappings[col] = dict(enumerate(uniques))
    return encoded_df, mappings

[raw, expected] = load_csv_for_decision_tree("MP2PRTWT.clinical.csv", "Status")
data, mappings = encode_categorical_columns(raw)

# 2. Split into Training (80%) and Testing (20%)
# We use random_state=42 to ensure the split is the same every time
input_train, input_test, output_train, output_test = train_test_split(data, expected, test_size=0.8, random_state=42)

# 3. Initialize the Tree
# 'max_depth' prevents the tree from becoming a giant, over-complicated mess
model = DecisionTreeClassifier(max_depth=3)

# 4. Train (Fit) the model
model.fit(input_train, output_train)

# 5. Predict and Evaluate
predictions = model.predict(input_test)
accuracy = accuracy_score(output_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# --- Interactive Prediction ---
# Ask user for input values for each feature column
user_input = {}
for col in data.columns:
    # Get mapping if column was encoded
    if col in mappings:
        options = list(mappings[col].values())
        prompt = f"Enter value for '{col}' (options: {options}): "
        val = input(f"{prompt}")
        # Convert string to code
        try:
            code = options.index(val)
        except ValueError:
            print(f"Invalid value for {col}. Using first option.")
            code = 0
        user_input[col] = code
    else:
        val = input(f"Enter value for '{col}': ")
        # Try to convert to float or int
        try:
            val = float(val)
        except ValueError:
            pass
        user_input[col] = val

# Convert to DataFrame
user_df = pd.DataFrame([user_input])

# Predict probability
probs = model.predict_proba(user_df)

# Show probabilities for each class
for i, class_label in enumerate(model.classes_):
    print(f"Probability of '{class_label}': {probs[0][i]*100:.2f}%")
