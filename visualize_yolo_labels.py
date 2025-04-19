import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置标签目录路径（YOLO 格式）
labels_dir = r"D:\WKU\2025SPRING\capstone2\code show\PIcamera\VisDrone2019-DET-train\labels\train"

# 标签 -> 类别名映射（0~9）
class_map = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor"
}

# 读取标签数据
data = []
for filename in os.listdir(labels_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(labels_dir, filename), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x, y, w, h = map(float, parts)
                    class_name = class_map.get(int(class_id), f"class_{int(class_id)}")
                    data.append([class_name, x, y, w, h])

# 构建 DataFrame
df = pd.DataFrame(data, columns=['class', 'x', 'y', 'width', 'height'])

# 可视化设置
sns.set(style="whitegrid", font_scale=1.2)

# 只选两个关键变量绘图，例如 x vs y 分布（其他你也可以换）
plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(
    data=df,
    x="x",
    y="y",
    hue="class",
    alpha=0.5,
    palette="tab10"
)
plt.title("YOLO Label Distribution by Class (x vs y)", fontsize=16)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("label_xy_by_class.png", dpi=300)
plt.show()

# ✅ 你也可以再画一个尺寸分布（width vs height）：
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df,
    x="width",
    y="height",
    hue="class",
    alpha=0.5,
    palette="tab10"
)
plt.title("YOLO Box Size Distribution by Class (width vs height)", fontsize=16)
plt.xlim(0, 0.6)
plt.ylim(0, 0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("label_size_by_class.png", dpi=300)
plt.show()
