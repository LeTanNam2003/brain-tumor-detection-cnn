import sqlite3
import pandas as pd

# List of class names
class_names = ["Normal", "Meningioma", "Glioma", "Pituitary"]

# Connect to SQLite database
conn = sqlite3.connect("predictions.db")

# Read the 'predictions' table into a DataFrame
df = pd.read_sql_query("SELECT * FROM predictions", conn)
conn.close()

# Convert class indices to string labels
df["Predicted_Label"] = df["predicted_class"].apply(lambda x: class_names[x])
df["Actual_Label"] = df["true_class"].apply(lambda x: class_names[x])

# Reorder columns for better readability (optional)
cols_order = [
    "image_path",
    "Predicted_Label", "Actual_Label",
    "prob_normal", "prob_meningioma", "prob_glioma", "prob_pituitary", 
    "predicted_class", "true_class"
]
df = df[cols_order]

# Export to Excel
df.to_excel("predictions_labeled.xlsx", index=False)

print("Predictions with full labels exported to predictions_labeled.xlsx")
