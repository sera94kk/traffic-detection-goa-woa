@echo off
:: 切换到项目目录，注意路径加引号防止空格出错
cd /d "D:\WKU\2025SPRING\capstone2\code show\PIcamera"

:: 使用绝对路径调用 Python，也加引号
"C:\Users\19156\AppData\Local\Programs\Python\Python312\python.exe" yolov11/train.py --img 640 --batch 8 --epochs 50 --data "D:\WKU\2025SPRING\capstone2\code show\PIcamera\dataset\data.yaml" --weights yolov11n.pt --device cpu

:: 保持窗口打开
pause
