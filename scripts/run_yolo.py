import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from src.yolo_detector import process_source  
from src import config


if __name__ == "__main__":
    # 1. 检查模型文件是否存在
    if not os.path.exists(config.MODEL_PATH):
        print(f"❌ 找不到模型文件: {config.MODEL_PATH}，请检查 weights 文件夹！")
        exit()

    # 2. 加载模型
    print(f"🔄 正在加载模型: {config.MODEL_PATH}")
    model = YOLO(config.MODEL_PATH)

    # ================= 测试用例 =================
    # 1. 测试单张图片
    img_path = os.path.join(config.DATA_DIR,'rm_armor.jpg')
    process_source(model, img_path)

    # 2. 测试包含多张图片或视频的文件夹 
    # img_path = config.DATA_DIR
    # process_source(model, 'data/') 
    
    # 3. 测试视频
    # img_path = os.path.join(config.DATA_DIR,'armors.mp4')
    # process_source(model, 'data/armors.mp4')