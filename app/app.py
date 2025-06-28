import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import sqlite3
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import io
import cv2
import os
import base64
from model_pytorch import BrainTumorCNN
from model_custom import CNN, Conv2D, ReLU, MaxPool2D, GlobalAvgPool2D, Flatten, Linear, Dropout, Softmax, CrossEntropy

# ========== Cấu hình chung ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Normal", "Meningioma", "Glioma", "Pituitary"]

# ====== Tiện ích chung ======
def save_to_db(data):
    conn = sqlite3.connect("predictions.db")
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
    cursor.execute("DELETE FROM predictions")
    cursor.executemany('''
        INSERT INTO predictions 
        (image_path, prob_normal, prob_meningioma, prob_glioma, prob_pituitary, predicted_class, true_class)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()
    conn.close()

def export_excel():
    conn = sqlite3.connect("predictions.db")
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    if df.empty:
        st.warning("Chưa có dữ liệu để xuất Excel!")
        return

    df["Predicted_Label"] = df["predicted_class"].apply(lambda x: class_names[x])
    df["Actual_Label"] = df["true_class"].apply(lambda x: class_names[x])
    cols_order = [
        "image_path",
        "Predicted_Label", "Actual_Label",
        "prob_normal", "prob_meningioma", "prob_glioma", "prob_pituitary",
        "predicted_class", "true_class"
    ]
    df = df[cols_order]

    excel_path = "predictions_labeled.xlsx"
    df.to_excel(excel_path, index=False)
    st.success(f"Đã xuất dự đoán ra file {excel_path}")

def save_feature_maps(feature_maps, output_dir="feature_maps"):
    os.makedirs(output_dir, exist_ok=True)
    for name, fmap in feature_maps.items():
        if fmap.ndim == 4:
            fmap = fmap[0]
            for i in range(fmap.shape[0]):
                channel_img = fmap[i]
                channel_img = (channel_img - channel_img.min()) / (channel_img.max() - channel_img.min() + 1e-7)
                channel_img = (channel_img * 255).astype(np.uint8)
                fname = f"{name}_channel{i}.png"
                cv2.imwrite(os.path.join(output_dir, fname), channel_img)
        elif fmap.ndim == 2:
            np.save(os.path.join(output_dir, f"{name}.npy"), fmap)
        else:
            np.save(os.path.join(output_dir, f"{name}.npy"), fmap)

def load_img(img_path, target_size=(64, 64)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, ...]
    return img

def file_download_link(filepath, link_text="Tải file"):
    with open(filepath, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{os.path.basename(filepath)}">{link_text}</a>'
    return href

# ========== Tab 1: Ứng dụng Grad-CAM ==========
def grad_cam_app():
    st.title("Brain Tumor Classifier with Grad-CAM & Batch Prediction")

    @st.cache_resource
    def load_model(model_path="brain_tumor_model_1.pth"):
        model = BrainTumorCNN(num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval().to(device)
        return model

    model = load_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    def get_gradcam(model, input_tensor, target_class):
        target_layer = model.features[-2]
        cam = GradCAM(model=model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        return grayscale_cam

    def visualize_feature_maps(model, image_tensor, max_channels=8):
        model.eval()
        x = image_tensor.unsqueeze(0).to(device)
        outputs = []
        names = []
        with torch.no_grad():
            for i, layer in enumerate(model.features):
                x = layer(x)
                if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ReLU):
                    outputs.append(x.clone())
                    names.append(f"Layer_{i}")

        figs = []
        for idx, fmap in enumerate(outputs):
            fmap = fmap.squeeze(0)
            channels = min(max_channels, fmap.shape[0])
            fig, axs = plt.subplots(1, channels, figsize=(15, 4))
            for i in range(channels):
                axs[i].imshow(fmap[i].cpu(), cmap='viridis')
                axs[i].axis('off')
                axs[i].set_title(f"Ch {i}")
            fig.suptitle(f"Feature Maps after {names[idx]}")
            figs.append(fig)
        return figs

    def batch_predict(images, labels, image_names, model):
        batch_size = 32
        predictions = []
        all_probs = []
        data_to_save = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            with torch.no_grad():
                outputs = model(batch)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                batch_preds = np.argmax(probs, axis=1)

            predictions.extend(batch_preds)
            all_probs.extend(probs)

            for j in range(batch.shape[0]):
                img_name = image_names[i + j]
                true_class = labels[i + j] if labels is not None else -1
                pred_class = int(batch_preds[j])
                prob_list = [float(p) for p in probs[j]]
                data_to_save.append((img_name, *prob_list, pred_class, true_class))

        save_to_db(data_to_save)
        return predictions, all_probs

    # Phần upload ảnh đơn
    st.header("Upload 1 ảnh để dự đoán và trích xuất Grad-CAM")
    uploaded_file = st.file_uploader("Chọn ảnh MRI", type=["jpg", "jpeg", "png"], key="single_upload")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh MRI gốc", use_column_width=True)

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))

        st.subheader(f"Dự đoán: **{class_names[pred_class]}**")
        st.bar_chart(probs)

        # Grad-CAM
        st.subheader("Grad-CAM")
        np_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
        grayscale_cam = get_gradcam(model, img_tensor, pred_class)
        cam_img = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)
        st.image(cam_img, caption="Grad-CAM", use_column_width=True)

        # Feature maps
        st.subheader("Feature Maps")
        figs = visualize_feature_maps(model, transform(image).to(device))
        for fig in figs:
            st.pyplot(fig)

    # Phần batch dự đoán
    st.header("Batch dự đoán từ nhiều ảnh")
    uploaded_files = st.file_uploader(
        "Chọn nhiều ảnh cùng lúc (Ctrl+click để chọn nhiều)",
        type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_upload"
    )

    if uploaded_files:
        st.write(f"Đã chọn {len(uploaded_files)} ảnh")
        images = []
        image_names = []
        for f in uploaded_files:
            try:
                img = Image.open(f).convert("RGB")
                img_tensor = transform(img)
                images.append(img_tensor)
                image_names.append(f.name)
            except:
                st.warning(f"Lỗi đọc ảnh: {f.name}")

        if len(images) > 0:
            images_tensor = torch.stack(images)
            labels = None
            if st.button("Chạy dự đoán batch và lưu kết quả", key="batch_predict_btn"):
                preds, probs = batch_predict(images_tensor, labels, image_names, model)

                st.success("Hoàn thành dự đoán batch.")
                st.write(f"Tổng số ảnh: {len(images)}")

                pred_arr = np.array(preds)
                for i, cls in enumerate(class_names):
                    st.write(f"- {cls}: {np.sum(pred_arr == i)}")

                if st.button("Xuất kết quả ra Excel", key="export_excel_btn"):
                    export_excel()

# ========== Tab 2: Ứng dụng Custom CNN ==========
def custom_cnn_app():
    st.title("Brain Tumor MRI Classifier (Custom CNN)")

    @st.cache_resource
    def load_custom_model():
        layers = [
            Conv2D(3, 32, kernel_size=3, padding=1), ReLU(), MaxPool2D(2, 2),
            Conv2D(32, 64, kernel_size=3, padding=1), ReLU(), MaxPool2D(2, 2),
            Conv2D(64, 128, kernel_size=3, padding=1), ReLU(), MaxPool2D(2, 2),
            Conv2D(128, 256, kernel_size=3, padding=1), ReLU(), MaxPool2D(2, 2),
            Conv2D(256, 512, kernel_size=3, padding=1), ReLU(),
            GlobalAvgPool2D(), Flatten(),
            Linear(512, 128), ReLU(), Dropout(0.5),
            Linear(128, 4), Softmax()
        ]
        model = CNN(layers=layers, loss_fn=CrossEntropy(), lr=0.01)
        model.load_model("model_weights_iter3.npz")
        return model

    model = load_custom_model()

    mode = st.radio("Chọn chế độ:", ["Dự đoán ảnh đơn", "Dự đoán batch từ thư mục"], key="custom_cnn_mode")

    if mode == "Dự đoán ảnh đơn":
        uploaded_img = st.file_uploader("Tải ảnh MRI lên:", type=["jpg", "jpeg", "png"], key="custom_single")
        if uploaded_img:
            img = Image.open(uploaded_img).convert("RGB")
            st.image(img, caption="Ảnh đã tải", use_column_width=True)

            # Tiền xử lý ảnh
            img_np = np.array(img)
            temp_path = "temp.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            x = load_img(temp_path)

            # Forward pass + lưu feature maps
            x_input = x.copy()
            feature_maps = {}
            for i, layer in enumerate(model.layers):
                x_input = layer.forward(x_input)
                name = f"{i:02d}_{layer.__class__.__name__}"
                feature_maps[name] = x_input.copy()

            # Lưu feature maps
            save_feature_maps(feature_maps, output_dir="feature_maps_single")

            # Dự đoán
            probs = x_input[0]
            st.subheader("Xác suất dự đoán:")
            for i, prob in enumerate(probs):
                st.write(f"{class_names[i]}: **{prob:.4f}**")

            st.success(f"Đã lưu feature maps tại thư mục: feature_maps_single")

            # Hiển thị feature maps
            st.subheader("Feature maps & dữ liệu:")
            for name, fmap in feature_maps.items():
                if fmap.ndim == 4:
                    fmap = fmap[0]
                    num_maps = min(16, fmap.shape[0])
                    cols = (num_maps + 1) // 2
                    rows = 2
                    fig, axs = plt.subplots(rows, cols, figsize=(15, 8))
                    for i in range(num_maps):
                        row = i // cols
                        col = i % cols
                        axs[row, col].imshow(fmap[i], cmap='viridis')
                        axs[row, col].axis('off')
                        axs[row, col].set_title(f'F{i}')
                    fig.suptitle(f"{name}")
                    st.pyplot(fig)

    elif mode == "Dự đoán batch từ thư mục":
        uploaded_files = st.file_uploader("Chọn nhiều ảnh MRI", type=["jpg", "jpeg", "png"],
                                          accept_multiple_files=True, key="custom_batch")
        if uploaded_files and st.button("Bắt đầu batch dự đoán", key="custom_batch_btn"):
            results = []
            os.makedirs("feature_maps_batch", exist_ok=True)
            for file in uploaded_files:
                try:
                    img = Image.open(file).convert("RGB")
                    img_np = np.array(img)
                    temp_path = "temp_batch.jpg"
                    cv2.imwrite(temp_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                    x = load_img(temp_path)

                    x_input = x.copy()
                    for layer in model.layers:
                        x_input = layer.forward(x_input)

                    probs = x_input[0]
                    pred = np.argmax(probs)
                    results.append((file.name, class_names[pred], probs[pred]))

                    # Lưu feature maps batch từng ảnh
                    feature_maps = {}
                    x_tmp = x.copy()
                    for i, layer in enumerate(model.layers):
                        x_tmp = layer.forward(x_tmp)
                        name = f"{file.name}_{i:02d}_{layer.__class__.__name__}"
                        feature_maps[name] = x_tmp.copy()
                    save_feature_maps(feature_maps, output_dir="feature_maps_batch")

                except Exception as e:
                    results.append((file.name, "ERROR", str(e)))

            st.subheader("Kết quả batch:")
            for fname, pred, prob in results:
                st.write(f" {fname} → **{pred}** ({prob:.4f})")

            save_path = "batch_results.txt"
            with open(save_path, 'w', encoding='utf-8') as f:
                for fname, pred, prob in results:
                    f.write(f"{fname} → {pred} ({prob})\n")

            st.markdown(file_download_link(save_path, " Tải kết quả batch"), unsafe_allow_html=True)

# ========== Main App ==========
def main():
    st.sidebar.title("Chọn chế độ")
    app_mode = st.sidebar.radio("Chọn mô hình", 
                               ["PyTorch Model với Grad-CAM", "Custom CNN Model"])

    if app_mode == "PyTorch Model với Grad-CAM":
        grad_cam_app()
    else:
        custom_cnn_app()

if __name__ == "__main__":
    main()