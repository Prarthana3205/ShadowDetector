import os

folder = "images"  # your image folder path
for idx, filename in enumerate(sorted(os.listdir(folder))):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        ext = filename.split('.')[-1]
        new_name = f"{idx+1:03d}.{ext}"
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))