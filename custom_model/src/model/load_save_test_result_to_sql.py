import tkinter as tk
from tkinter import filedialog, ttk
import os
import numpy as np
import cv2
from PIL import Image, ImageTk
from sklearn.metrics import accuracy_score
from custom_model import CNN, Conv2D, ReLU, MaxPool2D, GlobalAvgPool2D, Flatten, Linear, Softmax, Dropout, CrossEntropy
import sqlite3

# ---------- Configuration ----------
class_names = ["normal", "meningioma", "glioma", "pituitary"]
kept_classes = list(range(len(class_names)))  # [0, 1, 2, 3]

# ---------- Model Definition ----------
def build_model():
    layers = [
        Conv2D(3, 32, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),
        Conv2D(32, 64, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),
        Conv2D(64, 128, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),
        Conv2D(128, 256, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),
        Conv2D(256, 512, kernel_size=3, padding=1),
        ReLU(),
        GlobalAvgPool2D(),
        Flatten(),
        Linear(512, 128),
        ReLU(),
        Dropout(rate=0.5),
        Linear(128, len(kept_classes)),
        Softmax()
    ]
    
    model = CNN(layers, CrossEntropy(), lr=0.01)
    model.load_model("model_weights_iter2.npz")
    return model

# ---------- Load Images from Dataset ----------
def load_dataset(folder):
    images, labels, image_paths = [], [], []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(folder, class_name)
        if not os.path.exists(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (28, 28))
                img = img / 255.0
                img = np.transpose(img, (2, 0, 1))  # Channels first
                images.append(img)
                labels.append(class_idx)
                image_paths.append(img_path)
    return np.array(images), np.array(labels), image_paths

# ---------- One-hot Encoding ----------
def to_one_hot(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

# ---------- GUI Application ----------
class BatchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Batch Brain Tumor Classifier")
        self.geometry("800x600")

        self.model = build_model()
        self.dataset = None

        self.create_widgets()
        self.conn = sqlite3.connect("predictions.db")
        self.create_table()

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

        self.info_label = ttk.Label(main_frame, text="No dataset loaded yet")
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
            self.progress["value"] = 0
            self.update_idletasks()
            self.images, self.labels, self.image_paths = load_dataset(folder)
            self.dataset = (self.images, self.labels, self.image_paths)

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
        img = self.images[0].transpose(1, 2, 0) * 255
        img = Image.fromarray(img.astype("uint8"))
        img.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        self.sample_label.config(image=img_tk)
        self.sample_label.image = img_tk

    def evaluate(self):
        batch_size = 32
        predictions = []
        all_probs = []
        total_loss = 0

        if not hasattr(self, 'image_paths'):
            self.images, self.labels, self.image_paths = self.dataset

        self.progress["maximum"] = len(self.images)

        for i in range(0, len(self.images), batch_size):
            batch = self.images[i:i+batch_size]
            true_labels = self.labels[i:i+batch_size]

            true_one_hot = to_one_hot(true_labels, num_classes=len(class_names))

            probs = self.model.forward(batch)
            loss = self.model.loss_fn.forward(true_one_hot, probs)
            total_loss += loss

            batch_preds = np.argmax(probs, axis=1)

            predictions.extend(batch_preds)
            all_probs.extend(probs)

            batch_data = []
            for j in range(len(batch)):
                img_idx = i + j
                if img_idx >= len(self.image_paths):
                    break
                image_path = self.image_paths[img_idx]
                probs_float = [float(p) for p in probs[j]]
                true_class = int(true_labels[j])
                predicted_class = int(batch_preds[j])
                batch_data.append((image_path, *probs_float, predicted_class, true_class))

            cursor = self.conn.cursor()
            cursor.executemany('''
                INSERT INTO predictions 
                (image_path, prob_normal, prob_meningioma, prob_glioma, prob_pituitary, predicted_class, true_class)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', batch_data)
            self.conn.commit()

            self.progress["value"] = i + batch_size
            self.update_idletasks()

        acc = accuracy_score(self.labels, predictions)
        avg_loss = total_loss / (len(self.images) / batch_size)

        result = (
            f"Overall Accuracy: {acc*100:.2f}%\n"
            f"Average Loss: {avg_loss:.4f}\n"
            f"Total Images Processed: {len(self.images)}\n"
            f"Prediction Distribution:\n"
            f"- Normal: {predictions.count(0)}\n"
            f"- Meningioma: {predictions.count(1)}\n"
            f"- Glioma: {predictions.count(2)}\n"
            f"- Pituitary: {predictions.count(3)}\n"
            f"Results have been saved to the database!"
        )
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

# ---------- Run Application ----------
if __name__ == "__main__":
    app = BatchApp()
    app.mainloop()
