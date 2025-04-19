import os
from PIL import Image
from tqdm import tqdm

def convert_box(size, box):
    """
    将 VisDrone 格式的边界框转换为 YOLO 格式的边界框，并进行归一化
    :param size: 图像的宽和高 (width, height)
    :param box: 边界框坐标 (x_min, y_min, width, height)
    :return: 归一化后的 YOLO 格式边界框 (x_center, y_center, width, height)
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[2] / 2.0) * dw
    y_center = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return x_center, y_center, w, h

def convert_visdrone_to_yolo(images_dir, annotations_dir, output_dir):
    """
    将 VisDrone 数据集的标注文件转换为 YOLO 格式
    :param images_dir: 存放图像的目录
    :param annotations_dir: 存放 VisDrone 标注文件的目录
    :param output_dir: 输出 YOLO 格式标签的目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
    pbar = tqdm(annotation_files, desc='Converting annotations')

    for annotation_file in pbar:
        image_file = annotation_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_dir, image_file)
        annotation_path = os.path.join(annotations_dir, annotation_file)
        output_path = os.path.join(output_dir, annotation_file)

        if not os.path.exists(image_path):
            print(f"Image file {image_file} not found, skipping.")
            continue

        with Image.open(image_path) as img:
            img_size = img.size  # (width, height)

        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        yolo_lines = []
        for line in lines:
            parts = line.strip().split(',')
            x_min, y_min, width, height = map(int, parts[:4])
            class_id = int(parts[5])

            # 过滤掉类别编号大于 9 的标签
            if class_id > 9:
                continue

            # 转换边界框坐标并归一化
            bbox = convert_box(img_size, (x_min, y_min, width, height))
            yolo_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in bbox])
            yolo_lines.append(yolo_line)

        # 保存为 YOLO 格式的标签文件
        with open(output_path, 'w') as f:
            f.write("\n".join(yolo_lines))

    print(f"Conversion completed. YOLO labels are saved in: {output_dir}")

if __name__ == "__main__":
    # 设置路径
    images_dir = r"D:\WKU\2025SPRING\capstone2\code show\PIcamera\VisDrone2019-DET-train\images\train"
    annotations_dir = r"D:\WKU\2025SPRING\capstone2\code show\PIcamera\VisDrone2019-DET-train\annotations"
    output_dir = r"D:\WKU\2025SPRING\capstone2\code show\PIcamera\VisDrone2019-DET-train\labels\train"

    # 转换训练集
    convert_visdrone_to_yolo(images_dir, annotations_dir, output_dir)

