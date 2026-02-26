import pandas as pd
import sys

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import joblib

import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    X = df.drop(columns=[target_column, "Cause of death"])
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

[raw, expected] = load_csv_for_decision_tree("MP2PRTWT.clinical_v2.csv", "Status")
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

print(f"Training Accuracy: {accuracy * 100:.2f}%")

def ask_user_gui(data, mappings, model):
    root = tk.Tk()
    root.title("Cancer Decision Tree Predictor")
    entries = {}
    input_frame = ttk.Frame(root)
    input_frame.pack(padx=10, pady=10)

    def submit():
        user_input = {}
        for col in data.columns:
            val = entries[col].get()
            if col in mappings:
                options = list(mappings[col].values())
                try:
                    code = options.index(val)
                except ValueError:
                    messagebox.showerror("Input Error", f"Invalid value for {col}. Using first option.")
                    code = 0
                user_input[col] = code
            else:
                try:
                    user_input[col] = float(val)
                except ValueError:
                    user_input[col] = val
        user_df = pd.DataFrame([user_input])

        probs = model.predict_proba(user_df)[0]
        show_pie_chart(probs, model.classes_)

    row = 0
    for col in data.columns:
        ttk.Label(input_frame, text=col).grid(row=row, column=0, sticky="w")
        if col in mappings:
            options = list(mappings[col].values())
            entry = ttk.Combobox(input_frame, values=options)
            entry.grid(row=row, column=1)
            entry.set(options[0])
        else:
            entry = ttk.Entry(input_frame)
            entry.grid(row=row, column=1)
        entries[col] = entry
        row += 1

    submit_btn = ttk.Button(root, text="Predict", command=submit)
    submit_btn.pack(pady=10)

    def show_pie_chart(probs, classes):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(probs, labels=classes, autopct='%1.1f%%', startangle=90)
        ax.set_title("Prediction Probabilities")
        plt.tight_layout()
        chart_win = tk.Toplevel(root)
        chart_win.title("Prediction Results")
        canvas = FigureCanvasTkAgg(fig, master=chart_win)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def on_closing():
        plt.close('all')
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    ask_user_gui(data, mappings, model)