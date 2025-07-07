import importlib.metadata

packages = [
    "pandas", "Pillow", "numpy", "torch",
    "grad-cam", "torchvision", "matplotlib",
    "opencv-python", "scikit-learn", "seaborn"
]

with open("requirements.txt", "w") as f:
    for package in packages:
        try:
            version = importlib.metadata.version(package)
            f.write(f"{package}=={version}\n")
        except importlib.metadata.PackageNotFoundError:
            print(f"Không tìm thấy package: {package}")
