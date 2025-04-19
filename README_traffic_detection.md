
# Traffic Behavior Detection using YOLOv11 + GOA/WOA

This repository contains the implementation of a real-time traffic behavior detection system optimized with Gazelle Optimization Algorithm (GOA) and Whale Optimization Algorithm (WOA), based on the YOLOv11 object detection framework. The optimized models are trained on the VisDrone2019-DET dataset and deployed to edge devices including Raspberry Pi.

## 📁 Project Structure

- `convert_visdrone_to_yolo.py` – Script to convert VisDrone annotations to YOLO format.
- `main.py` – Main entry point for YOLOv11 detection and optimization.
- `yolo11n.pt`, `yolo11n-seg.pt` – Pretrained YOLOv11 models.
- `*.torchscript` – YOLO models exported for deployment on embedded devices.
- `*_ncnn_model/` – NCNN model files for Raspberry Pi inference.
- `visualize_yolo_labels.py` – Visualize label distribution and data quality.
- `train_yolo_cpu_PIcamera.bat` – Batch script for YOLO training on Raspberry Pi.
- `label_size_by_class.png`, `labels_distribution_clear*.png` – Visual diagnostics of dataset.

## 📦 Dataset

The VisDrone2019-DET dataset is used for training and evaluation.

- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- Converted data should be placed under `/VisDrone2019-DET-train/`

## ⚙️ Requirements

- Python 3.11
- PyTorch 2.0+
- Ultralytics YOLOv11
- OpenCV, NumPy, SciPy, Matplotlib

## 🚀 Deployment

The optimized models can be deployed on:
- Raspberry Pi 4 (using TorchScript or NCNN)
- CPU/GPU platforms

## 🧠 Optimization

GOA and WOA are used to tune YOLOv11 hyperparameters such as:
- Learning rate
- Confidence threshold
- NMS IoU threshold
- Anchor box ratios

## 📜 License

For academic use only.

---
Maintained by `sera94kk`. Contributions welcome!
