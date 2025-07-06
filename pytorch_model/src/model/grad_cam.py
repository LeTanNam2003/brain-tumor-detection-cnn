import argparse
from PIL import Image
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
import matplotlib.pyplot as plt

from pytorch_model import BrainTumorCNN  


# python grad_cam.py --input "C:/Personal/final_graduate/Report/dataset/Brain_Tumor_MRI_Dataset/Testing1/glioma/Te-gl_0044.jpg" --model "C:/Personal/brain-tumor-detection/brain-tumor-detection-cnn/pytorch_model/models/brain_tumor_model_128_001_xoay.pth" --output gradcam_auto.png --img-size 224 224

def main(image_path, model_path, output_path, target_class=None, img_size=(128, 128)):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = BrainTumorCNN(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    # Choose the last convolutional layer
    target_layer = model.features[-2]

    # Load and process the input image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img.resize(img_size)).astype(np.float32) / 255.0  # Resize for visualization

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # If target class not provided, use the predicted class
    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax().item()
            print(f"No target class provided. Using predicted class: {target_class}")

    # Set up Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Generate the visualization
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # Save the Grad-CAM image
    plt.imsave(output_path, visualization)
    print(f"Grad-CAM saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for a brain MRI image.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image (e.g., image.jpg)")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pth file (e.g., model.pth)")
    parser.add_argument("--output", type=str, required=True, help="Path to save Grad-CAM image (e.g., cam.png)")
    parser.add_argument("--target", type=int, default=None,
                        help="(Optional) Target class index (0=Normal, 1=Meningioma, 2=Glioma, 3=Pituitary)")
    parser.add_argument("--img-size", type=int, nargs=2, default=[128, 128],
                        help="Resize image to this size, e.g., --img-size 224 224")

    args = parser.parse_args()
    main(args.input, args.model, args.output, args.target, tuple(args.img_size))
