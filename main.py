import math
import shutil

import cv2
import threading
import os

import numpy as np
import torch
from flask import Flask, Response, request, send_from_directory, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import io
from PIL import Image
import scipy.stats as stats

# Load the exported NCNN model
ncnn_model = YOLO("/home/pi/program/yolo11n-seg_ncnn_model")

# 设置参数
population_size = 50
max_iterations = 200
# 假设要优化的是分类置信度阈值，范围在0到1之间
lower_bound = np.array([0.001])
upper_bound = np.array([1])

app = Flask(__name__, static_folder='dist')
CORS(app)  # 解决跨域问题

# 用于存储最新帧的变量
latest_frame = None

# 图片上传保存的目录
UPLOAD_FOLDER = '/home/pi/program/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def capture_frames():
    global latest_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    try:
        while True:
            success, frame = cap.read()
            if success:
                latest_frame = frame.copy()
    finally:
        cap.release()


def generate_frames():
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_detect_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            # 保存最新帧为 JPG 图片
            frame_path = os.path.join(UPLOAD_FOLDER, 'latest_frame.jpg')
            cv2.imwrite(frame_path, latest_frame)

            # 进行目标检测和分割
            results = ncnn_model(frame_path)
            # best_params, best_score = goa(results, population_size, max_iterations, lower_bound, upper_bound)

            for result in results:
                result.save(filename=os.path.join(UPLOAD_FOLDER, "resulttest.jpg"))
                # xywh = result.boxes.xywh
                # probs = result.probs
                # 返回识别结果图片
                with open(os.path.join(UPLOAD_FOLDER, "resulttest.jpg"), 'rb') as f:
                    img_data = f.read()
            if img_data is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img_data + b'\r\n')


@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed2')
def video_feed2():
    global latest_frame
    return Response(generate_detect_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        results = ncnn_model(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        all_best_params = []
        all_best_scores = []
        all_woa_best_params = []
        all_woa_best_scores = []

        for result in results:
            converted_results = []
            summary = result.summary(decimals=50)
            for item in summary:
                box = item['box']
                x1 = box['x1']
                y1 = box['y1']
                x2 = box['x2']
                y2 = box['y2']
                width = x2 - x1
                height = y2 - y1
                confidence = item['confidence']
                class_label = item['class']
                converted_results.append((x1, y1, width, height, confidence, class_label))

            # 瞪羚优化算法
            best_params, best_score = gazelle_optimization_algorithm(converted_results, population_size, max_iterations, lower_bound, upper_bound)
            print("瞪羚优化算法最优参数:", best_params)
            print("瞪羚优化算法最优分数:", best_score)

            # 鲸鱼优化算法
            woa_best_params, woa_best_score = whale_optimization_algorithm(converted_results, population_size, max_iterations, lower_bound, upper_bound)
            print("鲸鱼优化算法最优参数:", woa_best_params)
            print("鲸鱼优化算法最优分数:", woa_best_score)

            # 读取原始图像
            original_img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # 在原始图像上绘制所有检测框
            all_boxes_img = original_img.copy()
            for item in summary:
                box = item['box']
                x1 = int(box['x1'])
                y1 = int(box['y1'])
                x2 = int(box['x2'])
                y2 = int(box['y2'])
                confidence = item['confidence']
                class_name = item['name']
                # 绘制更粗的矩形框
                cv2.rectangle(all_boxes_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # 添加类别和置信度文本
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(all_boxes_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # 保存绘制所有检测框的图像
            all_boxes_filename = os.path.join(UPLOAD_FOLDER, "processed_" + filename)
            cv2.imwrite(all_boxes_filename, all_boxes_img)

            # 筛选出置信度低于 0.85 的检测框
            low_confidence_boxes = [item for item in summary if item['confidence'] > best_params[0]]
            low_confidence_img = original_img.copy()
            for item in low_confidence_boxes:
                box = item['box']
                x1 = int(box['x1'])
                y1 = int(box['y1'])
                x2 = int(box['x2'])
                y2 = int(box['y2'])
                confidence = item['confidence']
                class_name = item['name']
                # 绘制更粗的矩形框
                cv2.rectangle(low_confidence_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # 添加类别和置信度文本
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(low_confidence_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # 保存绘制低置信度检测框的图像
            low_confidence_filename = os.path.join(UPLOAD_FOLDER, "filtered_" + filename)
            cv2.imwrite(low_confidence_filename, low_confidence_img)

            all_best_params.append(best_params.tolist())
            all_best_scores.append(best_score)
            all_woa_best_params.append(woa_best_params.tolist())
            all_woa_best_scores.append(woa_best_score)

        response_data = {
            'filename': filename,
            'best_params': str(all_best_params[0]),
            'best_scores': str(all_best_scores[0]),
            'woa_best_params': str(all_woa_best_params[0]),
            'woa_best_scores': str(all_woa_best_scores[0]),
            'processed_filename': "processed_" + filename,
            'filtered_filename': "filtered_" + filename
        }
        return jsonify(response_data)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def calculate_score(params, yolo_results):
    # 这里params是分类相关参数，yolo_results是YOLO识别结果
    # 简单假设params[0]是分类置信度阈值，过滤掉低于阈值的结果再计算分数
    filtered_results = [result for result in yolo_results if result[4] >= params[0]]
    return len(filtered_results)


def levy_flight(beta, size):
    """
    生成Lévy飞行的步长
    """
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    sigma_v = 1
    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, sigma_v, size)
    step = u / (np.abs(v) ** (1 / beta))
    return step


def gazelle_optimization_algorithm(yolo_results, population_size, max_iterations, lower_bound, upper_bound):
    # 初始化种群
    population = np.random.uniform(lower_bound, upper_bound, size=(population_size, len(lower_bound)))

    # 计算初始适应度
    fitness = np.array([calculate_score(individual, yolo_results) for individual in population])

    # 找到初始最优个体
    best_index = np.argmax(fitness)
    best_fitness = fitness[best_index]
    best_solution = population[best_index]

    # 捕食者成功率，可根据需要调整
    psrs = 0.2

    for t in range(max_iterations):
        # 计算控制参数，模拟瞪羚的行为变化
        a = 2 - t * (2 / max_iterations)
        for i in range(population_size):
            # 随机选择两个个体
            r1, r2 = np.random.choice(population_size, 2, replace=False)
            # 计算距离
            d = np.abs(population[r1] - population[r2])
            # 计算概率，决定是开发阶段还是探索阶段
            p = np.random.rand()
            if p < psrs:
                # 开发阶段，模拟布朗运动
                step = np.random.normal(0, 1, size=len(lower_bound))
                population[i] = best_solution + a * d * step
            else:
                # 探索阶段，模拟Lévy飞行
                beta = 1.5  # Lévy分布指数，可根据需要调整
                step = levy_flight(beta, len(lower_bound))
                population[i] = best_solution + a * d * step

            # 边界处理
            population[i] = np.clip(population[i], lower_bound, upper_bound)
            # 计算新的适应度
            new_fitness = calculate_score(population[i], yolo_results)
            # 更新最优解
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_solution = population[i]

    return best_solution, best_fitness


def whale_optimization_algorithm(yolo_results, population_size, max_iterations, lower_bound, upper_bound):
    # 初始化种群
    population = np.random.uniform(lower_bound, upper_bound, size=(population_size, len(lower_bound)))

    # 计算初始适应度
    fitness = np.array([calculate_score(individual, yolo_results) for individual in population])

    # 找到初始最优个体
    best_index = np.argmax(fitness)
    best_fitness = fitness[best_index]
    best_solution = population[best_index]

    for t in range(max_iterations):
        a = 2 - t * (2 / max_iterations)  # 线性递减的控制参数
        a2 = -1 + t * (-1 / max_iterations)  # 用于螺旋更新的控制参数

        for i in range(population_size):
            r1 = np.random.rand()
            r2 = np.random.rand()

            A = 2 * a * r1 - a
            C = 2 * r2

            l = (a2 - 1) * np.random.rand() + 1
            p = np.random.rand()

            if p < 0.5:
                if np.abs(A) < 1:
                    D = np.abs(C * best_solution - population[i])
                    population[i] = best_solution - A * D
                else:
                    rand_index = np.random.randint(0, population_size)
                    D = np.abs(C * population[rand_index] - population[i])
                    population[i] = population[rand_index] - A * D
            else:
                D = np.abs(best_solution - population[i])
                population[i] = D * np.exp(l) * np.cos(2 * np.pi * l) + best_solution

            # 边界处理
            population[i] = np.clip(population[i], lower_bound, upper_bound)

            # 计算新的适应度
            new_fitness = calculate_score(population[i], yolo_results)

            # 更新最优解
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_solution = population[i]

    return best_solution, best_fitness


def clear_folder(folder_path):
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        try:
            # 删除文件夹及其所有内容
            shutil.rmtree(folder_path)
            print(f"已清空文件夹: {folder_path}")
            # 重新创建该文件夹
            os.makedirs(folder_path)
        except Exception as e:
            print(f"清空文件夹 {folder_path} 时出错: {e}")
    else:
        print(f"指定的文件夹 {folder_path} 不存在。")


# 处理根路径，返回 index.html
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


# 处理静态文件请求
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


if __name__ == '__main__':
    clear_folder('/home/pi/program/uploads')

    # 启动线程来持续捕获帧
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=False)