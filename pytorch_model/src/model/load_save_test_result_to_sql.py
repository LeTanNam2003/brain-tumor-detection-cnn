import tkinter as tk
from tkinter import filedialog, ttk
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
from sklearn.metrics import accuracy_score
import sqlite3
from torchvision import transforms
from pytorch_model import BrainTumorCNN

# ---------- Configuration ----------
class_names = ["Normal", "Meningioma", "Glioma", "Pituitary"]
kept_classes = list(range(len(class_names)))

# Same transform used during training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- Load PyTorch model ----------
def build_model_pt():
    model = BrainTumorCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load("brain_tumor_model_1.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# ---------- Load dataset ----------
def load_dataset(folder):
    images, labels, image_paths = [], [], []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(folder, class_name.lower())  # match folder name
        if not os.path.exists(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = transform(img)
                images.append(img)
                labels.append(class_idx)
                image_paths.append(img_path)
    return torch.stack(images), np.array(labels), image_paths

# ---------- GUI App ----------
class BatchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Batch Brain Tumor Classifier (PyTorch)")
        self.geometry("800x600")

        self.model = build_model_pt()
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
        sample_tensor = self.images[0]
        img = sample_tensor.numpy().transpose(1, 2, 0)
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255  # Unnormalize
        img = np.clip(img, 0, 255).astype("uint8")

        img = Image.fromarray(img)
        img.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        self.sample_label.config(image=img_tk)
        self.sample_label.image = img_tk

    def evaluate(self):
        batch_size = 32
        predictions = []
        all_probs = []

        if not hasattr(self, 'image_paths'):
            self.images, self.labels, self.image_paths = self.dataset

        self.progress["maximum"] = len(self.images)

        # Clear old predictions
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM predictions")
        self.conn.commit()

        for i in range(0, len(self.images), batch_size):
            batch = self.images[i:i+batch_size]
            with torch.no_grad():
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1).numpy()
                batch_preds = np.argmax(probs, axis=1)

            predictions.extend(batch_preds)
            all_probs.extend(probs)

            batch_data = []
            for j in range(batch.size(0)):
                img_idx = i + j
                if img_idx >= len(self.image_paths):
                    break
                image_path = self.image_paths[img_idx]
                probs_float = [float(p) for p in probs[j]]
                true_class = int(self.labels[img_idx])
                predicted_class = int(batch_preds[j])
                batch_data.append((image_path, *probs_float, predicted_class, true_class))

            cursor.executemany('''
                INSERT INTO predictions 
                (image_path, prob_normal, prob_meningioma, prob_glioma, prob_pituitary, predicted_class, true_class)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', batch_data)
            self.conn.commit()

            self.progress["value"] = i + batch.size(0)
            self.update_idletasks()

        predictions = np.array(predictions)
        acc = accuracy_score(self.labels[:len(predictions)], predictions)

        pred_counts = [np.sum(predictions == i) for i in range(4)]

        result = (
            f"Overall Accuracy: {acc * 100:.2f}%\n"
            f"Total Processed Images: {len(predictions)}\n"
            f"Prediction Distribution:\n"
            f"- Normal: {pred_counts[0]}\n"
            f"- Meningioma: {pred_counts[1]}\n"
            f"- Glioma: {pred_counts[2]}\n"
            f"- Pituitary: {pred_counts[3]}\n"
            f"Results have been saved to the database!"
        )
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

# ---------- Run App ----------
if __name__ == "__main__":
    app = BatchApp()
    app.mainloop()
