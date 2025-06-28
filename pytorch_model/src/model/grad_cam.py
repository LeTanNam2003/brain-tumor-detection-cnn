from PIL import Image
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
import matplotlib.pyplot as plt

from pytorch_model import BrainTumorCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = BrainTumorCNN(num_classes=4)
model.load_state_dict(torch.load("brain_tumor_model_1.pth", map_location=device))
model.eval()
model.to(device)

# Chose last convolution class
target_layer = model.features[-2]

# Load and process img
img_path = "C:/Personal/final_graduate/Report/dataset/Brain_Tumor_MRI_Dataset/Testing1/glioma/Te-gl_0037.jpg"
img = Image.open(img_path).convert("RGB")
img_np = np.array(img.resize((224, 224))).astype(np.float32) / 255.0

# Pre-process
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0).to(device)

# Grad-CAM
model.to(device)  # make sure model at device
cam = GradCAM(model=model, target_layers=[target_layer])

targets = [ClassifierOutputTarget(1)]  # For example: "meningioma"

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

# Display
visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.title("Grad-CAM")
plt.axis("off")
plt.savefig("grad_cam.png")
plt.show()
