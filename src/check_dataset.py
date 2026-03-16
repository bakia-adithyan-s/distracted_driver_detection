import os

data_path = "dataset/imgs/train"

classes = os.listdir(data_path)

print("Classes:", classes)

for c in classes:
    count = len(os.listdir(os.path.join(data_path, c)))
    print(f"{c}: {count} images")