# run this once from your project folder
# point it at your original dataset folder (the one with all the disease subfolders)
import os

dataset_path = r"C:\Users\user\Desktop\model_training\data\raw"   # ← change to your folder name

classes = sorted(os.listdir(dataset_path))   # sorted() matches TF's loading order
classes = [c for c in classes if os.path.isdir(os.path.join(dataset_path, c))]

with open("classes.txt", "w") as f:
    f.write("\n".join(classes))

print(f"Written {len(classes)} classes to classes.txt")
print("First 5:", classes[:5])
print("Last 5:", classes[-5:])
