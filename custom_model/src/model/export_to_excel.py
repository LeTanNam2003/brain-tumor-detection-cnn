import sqlite3
import pandas as pd
import argparse
#python export_predictions.py -i predictions.db -o results.xlsx
#python export_predictions.py --input predictions.db --output results.xlsx
def export_predictions_to_excel(db_file, output_file):
    """
    Export prediction results from SQLite database to Excel file
    
    Args:
        db_file (str): Path to SQLite database file
        output_file (str): Path to output Excel file
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_file)

        # Read the entire predictions table into a Pandas DataFrame
        df = pd.read_sql_query("SELECT * FROM predictions", conn)

        # Define class names corresponding to label indices
        class_names = ["Normal", "Meningioma", "Glioma", "Pituitary"]

        # Map predicted and true class indices to readable class names
        df["Predicted_Label"] = df["predicted_class"].apply(lambda x: class_names[x])
        df["Actual_Label"] = df["true_class"].apply(lambda x: class_names[x])

        # Calculate if prediction was correct
        df["Correct_Prediction"] = df["predicted_class"] == df["true_class"]

        # Export the DataFrame to an Excel file
        df.to_excel(output_file, index=False, engine='openpyxl')

        print(f"Successfully exported {len(df)} records to {output_file}")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the database connection if it exists
        if 'conn' in locals():
            conn.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Export SQLite predictions to Excel')
    parser.add_argument('--input', '-i', type=str, required=True, 
                       help='Input SQLite database file (e.g., predictions.db)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output Excel file (e.g., predictions.xlsx)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    export_predictions_to_excel(args.input, args.output)