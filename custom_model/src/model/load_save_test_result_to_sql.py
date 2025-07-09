
# python load_save_test_result_to_sql.py --model "C:/Personal/brain-tumor-detection/brain-tumor-detection-cnn/custom_model/models/model_weights_iter4.npz" --img-size 64 64 --output-db results.db
# python load_save_test_result_to_sql.py --input-folder "C:/Personal/final_graduate/Report/dataset/Brain_Tumor_MRI_Dataset/Testing1" --model "C:/Personal/brain-tumor-detection/brain-tumor-detection-cnn/custom_model/models/model_weights_iter4.npz" --img-size 64 64 --output-db results.db

import argparse
import os
import numpy as np
import cv2
import sqlite3
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from custom_model import CNN, Conv2D, ReLU, MaxPool2D, GlobalAvgPool2D, Flatten, Linear, Softmax, Dropout, CrossEntropy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------- Configuration ----------
class_names = ["Normal", "Meningioma", "Glioma", "Pituitary"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, help="Path to dataset folder (with subfolders for classes)")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.npz)")
    parser.add_argument("--img-size", nargs=2, type=int, default=[64, 64], help="Input image size, e.g., --img-size 64 64")
    parser.add_argument("--output-db", type=str, default="results.db", help="Path to output SQLite DB file")
    return parser.parse_args()

# ---------- Model Definition ----------
def build_model(num_classes=4):
    layers = [
        Conv2D(3, 32, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(2, 2),
        Conv2D(32, 64, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(2, 2),
        Conv2D(64, 128, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(2, 2),
        Conv2D(128, 256, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(2, 2),
        Conv2D(256, 512, kernel_size=3, padding=1),
        ReLU(),
        GlobalAvgPool2D(),
        Flatten(),
        Linear(512, 128),
        ReLU(),
        Dropout(0.5),
        Linear(128, num_classes),
        Softmax()
    ]
    model = CNN(layers, CrossEntropy(), lr=0.01)
    return model

# ---------- Dataset Loading ----------
def load_dataset(folder, img_size):
    images, labels, paths = [], [], []
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(folder, class_name.lower())
        if not os.path.exists(class_dir): 
            continue
        for fname in os.listdir(class_dir):
            img_path = os.path.join(class_dir, fname)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, tuple(img_size))
                img = img / 255.0
                img = np.transpose(img, (2, 0, 1))
                images.append(img)
                labels.append(idx)
                paths.append(img_path)
    return np.array(images), np.array(labels), paths

def to_one_hot(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

# ---------- CLI Mode ----------
def run_cli_evaluation(model_path, img_size, db_path, input_folder):
    model = build_model(num_classes=len(class_names))
    model.load_model(model_path)
    
    images, labels, image_paths = load_dataset(input_folder, img_size)
    print(f"Loaded {len(images)} images from {input_folder}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            prob_normal REAL,
            prob_meningioma REAL,
            prob_glioma REAL,
            prob_pituitary REAL,
            predicted_class INTEGER,
            true_class INTEGER
        )
    ''')
    conn.commit()
    cursor.execute("DELETE FROM predictions")
    conn.commit()

    predictions = []
    batch_size = 32
    total_loss = 0

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        true_batch = labels[i:i+batch_size]

        true_one_hot = to_one_hot(true_batch, num_classes=len(class_names))
        probs = model.forward(batch)
        loss = model.loss_fn.forward(true_one_hot, probs)
        total_loss += loss

        batch_preds = np.argmax(probs, axis=1)
        predictions.extend(batch_preds)

        batch_data = []
        for j in range(len(batch)):
            prob = probs[j]
            batch_data.append((
                image_paths[i + j],
                float(prob[0]), float(prob[1]), float(prob[2]), float(prob[3]),
                int(batch_preds[j]),
                int(true_batch[j])
            ))

        cursor.executemany('''
            INSERT INTO predictions 
            (image_path, prob_normal, prob_meningioma, prob_glioma, prob_pituitary, predicted_class, true_class)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', batch_data)
        conn.commit()

    predictions = np.array(predictions)
    acc = accuracy_score(labels[:len(predictions)], predictions)
    avg_loss = total_loss / (len(images) / batch_size)
    report = classification_report(labels[:len(predictions)], predictions, target_names=class_names)
    conf_matrix = confusion_matrix(labels[:len(predictions)], predictions)

    print("\nEvaluation Results:")
    print(f"- Accuracy: {acc * 100:.2f}%")
    print(f"- Average Loss: {avg_loss:.4f}")
    print(f"- Total Processed Images: {len(images)}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"\nResults saved to database: {db_path}")

    conn.close()

# ---------- GUI App ----------
class BatchApp(tk.Tk):
    def __init__(self, model_path, img_size, db_path):
        super().__init__()
        self.title("Batch Brain Tumor Classifier (Custom CNN)")
        self.geometry("800x600")

        self.model_path = model_path
        self.img_size = img_size
        self.db_path = db_path

        self.model = build_model(num_classes=len(class_names))
        self.model.load_model(self.model_path)
        self.conn = sqlite3.connect(self.db_path)
        self.create_table()

        self.create_widgets()

    def __del__(self):
        self.conn.close()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                prob_normal REAL,
                prob_meningioma REAL,
                prob_glioma REAL,
                prob_pituitary REAL,
                predicted_class INTEGER,
                true_class INTEGER
            )
        ''')
        self.conn.commit()

    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        self.btn_load = ttk.Button(main_frame, text="Select Dataset Folder", command=self.load_data_gui)
        self.btn_load.pack(pady=10)

        self.info_label = ttk.Label(main_frame, text="No dataset loaded")
        self.info_label.pack()

        self.btn_eval = ttk.Button(main_frame, text="Evaluate Model", command=self.evaluate, state=tk.DISABLED)
        self.btn_eval.pack(pady=10)

        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack()

        self.result_text = tk.Text(main_frame, height=10, width=70)
        self.result_text.pack(pady=10)

        self.sample_label = ttk.Label(main_frame)
        self.sample_label.pack()

    def load_data_gui(self):
        folder = filedialog.askdirectory()
        if folder:
            self.images, self.labels, self.image_paths = load_dataset(folder, self.img_size)
            
            class_counts = np.bincount(self.labels, minlength=4)
            info = (
                f"Loaded {len(self.images)} images\n"
                f"Normal: {class_counts[0]}\n"
                f"Meningioma: {class_counts[1]}\n"
                f"Glioma: {class_counts[2]}\n"
                f"Pituitary: {class_counts[3]}"
            )
            self.info_label.config(text=info)
            self.btn_eval.config(state=tk.NORMAL)
            self.show_sample_image()

    def show_sample_image(self):
        sample_img = self.images[0].transpose(1, 2, 0)
        sample_img = (sample_img * 255).astype(np.uint8)
        
        img = Image.fromarray(sample_img)
        img.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        self.sample_label.config(image=img_tk)
        self.sample_label.image = img_tk

    def evaluate(self):
        batch_size = 32
        predictions = []
        total_loss = 0

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM predictions")
        self.conn.commit()

        self.progress["maximum"] = len(self.images)

        for i in range(0, len(self.images), batch_size):
            batch = self.images[i:i+batch_size]
            true_batch = self.labels[i:i+batch_size]

            true_one_hot = to_one_hot(true_batch, num_classes=len(class_names))
            probs = self.model.forward(batch)
            loss = self.model.loss_fn.forward(true_one_hot, probs)
            total_loss += loss

            batch_preds = np.argmax(probs, axis=1)
            predictions.extend(batch_preds)

            batch_data = []
            for j in range(len(batch)):
                img_idx = i + j
                image_path = self.image_paths[img_idx]
                prob = probs[j]
                true_class = int(self.labels[img_idx])
                predicted_class = int(batch_preds[j])
                batch_data.append((
                    image_path,
                    float(prob[0]), float(prob[1]), float(prob[2]), float(prob[3]),
                    predicted_class,
                    true_class
                ))

            cursor.executemany('''
                INSERT INTO predictions 
                (image_path, prob_normal, prob_meningioma, prob_glioma, prob_pituitary, predicted_class, true_class)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', batch_data)
            self.conn.commit()

            self.progress["value"] = i + len(batch)
            self.update_idletasks()

        predictions = np.array(predictions)
        acc = accuracy_score(self.labels[:len(predictions)], predictions)
        avg_loss = total_loss / (len(self.images) / batch_size)
        report = classification_report(self.labels[:len(predictions)], predictions, target_names=class_names)
        conf_matrix = confusion_matrix(self.labels[:len(predictions)], predictions)

        result = (
            f"Overall Accuracy: {acc * 100:.2f}%\n"
            f"Average Loss: {avg_loss:.4f}\n"
            f"Total Processed Images: {len(predictions)}\n"
            f"\nClassification Report:\n{report}\n"
            f"\nConfusion Matrix:\n{conf_matrix}\n"
            f"\nResults have been saved to the database!"
        )

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

# ---------- Main Entry ----------
if __name__ == "__main__":
    args = parse_args()

    if args.input_folder:
        run_cli_evaluation(
            model_path=args.model,
            img_size=tuple(args.img_size),
            db_path=args.output_db,
            input_folder=args.input_folder
        )
    else:
        app = BatchApp(
            model_path=args.model,
            img_size=tuple(args.img_size),
            db_path=args.output_db
        )
        app.mainloop()