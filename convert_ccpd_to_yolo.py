import os
import cv2
import shutil

# 配置路径
ccpd_path = "D:/CCPD2019/ccpd_base"  # 确保正确
splits_path = "D:/CCPD2019/splits_clean"  # 使用清理后的 splits
output_path = "D:/CCPD2019/yolo_format"

# 创建输出目录
os.makedirs(f"{output_path}/images/train", exist_ok=True)
os.makedirs(f"{output_path}/images/val", exist_ok=True)
os.makedirs(f"{output_path}/labels/train", exist_ok=True)
os.makedirs(f"{output_path}/labels/val", exist_ok=True)

# 读取 splits 文件
train_files = []
val_files = []
if os.path.exists(f"{splits_path}/train.txt"):
    with open(f"{splits_path}/train.txt", "r", encoding="utf-8") as f:
        train_files = [line.strip() for line in f if line.strip()]
if os.path.exists(f"{splits_path}/val.txt"):
    with open(f"{splits_path}/val.txt", "r", encoding="utf-8") as f:
        val_files = [line.strip() for line in f if line.strip()]

# 处理图像和标签
for split, split_files in [("train", train_files), ("val", val_files)]:
    for filename in split_files:
        # 移除可能的路径前缀
        filename = filename.replace("ccpd_base/", "").replace("\\", "/")
        img_path = os.path.join(ccpd_path, filename)
        if not os.path.exists(img_path):
            print(f"文件不存在: {img_path}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        height, width = img.shape[:2]

        # 解析文件名获取边界框
        parts = filename.split("-")
        if len(parts) < 3:
            print(f"文件名格式错误: {filename}")
            continue
        bbox = parts[2].split("_")
        try:
            x_min, y_min = map(int, bbox[0].split("&"))
            x_max, y_max = map(int, bbox[1].split("&"))
        except:
            print(f"边界框解析失败: {filename}")
            continue

        # 转换为 YOLO 格式（归一化）
        x_center = (x_min + x_max) / 2 / width
        y_center = (y_min + y_max) / 2 / height
        bbox_width = (x_max - x_min) / width
        bbox_height = (y_max - y_min) / height

        # 保存标签文件
        class_id = 0
        label_filename = filename[:-4].replace("&", "_")  # 替换 & 以避免问题
        label_path = f"{output_path}/labels/{split}/{label_filename}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

        # 复制图像并重命名
        img_output_path = f"{output_path}/images/{split}/{label_filename}.jpg"
        shutil.copy(img_path, img_output_path)

print("转换完成！")