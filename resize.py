import os
import cv2
from tqdm import tqdm

# 设置你的datasets路径
dataset_folder = 'datasets'  # 请替换为你实际的路径

# 获取文件夹中的所有图像文件
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']  # 可以根据实际情况扩展
image_paths = []

for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(root, file))

# 调整所有图像为512x512
output_folder = 'datasets_resized'  # 输出文件夹
os.makedirs(output_folder, exist_ok=True)

for image_path in tqdm(image_paths, desc="Resizing images"):
    # 读取图像
    img = cv2.imread(image_path)
    
    # 调整图像大小
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    # 保存调整后的图像到输出文件夹
    relative_path = os.path.relpath(image_path, dataset_folder)
    output_path = os.path.join(output_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保输出路径的文件夹存在
    cv2.imwrite(output_path, img_resized)

print("所有图像已调整为512x512并保存至", output_folder)
