import argparse
import sqlite3
import pandas as pd

# python export_to_excel.py --input predictions.db --output predictions_labeled.xlsx
# python export_to_excel.py --input "C:/Personal/brain-tumor-detection/brain-tumor-detection-cnn/pytorch_model/data/raw/prediction_128.db" --output predictions_labeled.xlsx
def export_predictions_to_excel(input_db, output_excel):
    # List of class names
    class_names = ["Normal", "Meningioma", "Glioma", "Pituitary"]

    # Connect to the SQLite database
    conn = sqlite3.connect(input_db)

    # Read the 'predictions' table into a DataFrame
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    # Map class indices to class names
    df["Predicted_Label"] = df["predicted_class"].apply(lambda x: class_names[x])
    df["Actual_Label"] = df["true_class"].apply(lambda x: class_names[x])

    # Reorder columns for better readability
    cols_order = [
        "image_path",
        "Predicted_Label", "Actual_Label",
        "prob_normal", "prob_meningioma", "prob_glioma", "prob_pituitary", 
        "predicted_class", "true_class"
    ]
    df = df[cols_order]

    # Export the DataFrame to an Excel file
    df.to_excel(output_excel, index=False)
    print(f"xported predictions to: {output_excel}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export prediction results from SQLite to Excel.")
    parser.add_argument("--input", type=str, required=True, help="Path to the SQLite database file (e.g., predictions.db)")
    parser.add_argument("--output", type=str, required=True, help="Path to the output Excel file (e.g., results.xlsx)")

    args = parser.parse_args()

    export_predictions_to_excel(args.input, args.output)
