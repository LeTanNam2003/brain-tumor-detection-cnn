import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect("predictions.db")

# Read the entire predictions table into a Pandas DataFrame
df = pd.read_sql_query("SELECT * FROM predictions", conn)

# Define class names corresponding to label indices
class_names = ["Normal", "Meningioma", "Glioma", "Pituitary"]

# Map predicted and true class indices to readable class names
df["Predicted_Label"] = df["predicted_class"].apply(lambda x: class_names[x])
df["Actual_Label"] = df["true_class"].apply(lambda x: class_names[x])

# Export the DataFrame to an Excel file
df.to_excel("predictions_export.xlsx", index=False)

# Close the database connection
conn.close()

print("Exported predictions_export.xlsx successfully.")
