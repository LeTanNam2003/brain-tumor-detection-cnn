import os
import shutil

# Define folder structure and corresponding file extensions
structure = {
    "src/model": [".py"],
    "results/logs": [".txt", ".log"],
    "results/images": [".jpg", ".jpeg", ".png"],
    "models": [".pt", ".pth", ".npz"],
    "data/raw": [".nii", ".nii.gz", ".dcm", ".db", ".xlsx"],  # medical image and data formats
}

# Create folders if they don't exist
for folder in structure.keys():
    os.makedirs(folder, exist_ok=True)

# Iterate through all files in the current directory
for filename in os.listdir():
    # Skip directories
    if os.path.isdir(filename):
        continue

    # Get file extension
    ext = os.path.splitext(filename)[1].lower()

    # Check if the file matches any category
    moved = False
    for folder, extensions in structure.items():
        if ext in extensions:
            shutil.move(filename, os.path.join(folder, filename))
            print(f"Moved: {filename} → {folder}")
            moved = True
            break

    if not moved:
        print(f"Skipped: {filename} (no matching category)")
