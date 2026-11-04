import os
import sys
# 强制把项目根目录加入环境变量
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.detector import find_light_bars
from src.geometry import match_armors
from src.utils import draw_light_bars, draw_armors, draw_matched_bars
from src import config

if __name__ == "__main__":
    # ================= 配置区 =================
    # 摄像头：
    # source_path = 0  
    # 视频流：
    source_path = os.path.join(config.DATA_DIR, 'armors.mp4')   
    # ==========================================
    
    # 确保输出文件夹存在
    os.makedirs(config.OUTPUT_CV_DIR, exist_ok=True)

    # 判断输入源类型
    is_camera = isinstance(source_path, int)
    is_video = isinstance(source_path, str) and source_path.lower().endswith(('.mp4', '.avi', '.mov'))

    # 打开输入源
    cap = cv2.VideoCapture(source_path, cv2.CAP_V4L2) if is_camera else cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"❌ 无法打开输入源: {source_path}")
        exit()

    # ================= 视频保存器初始化 =================
    target_width = 1280
    out = None # 默认不保存
    
    if is_video:
        # 获取原视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # 计算缩放后的高度
        scale = target_width / width
        target_height = int(height * scale)
        
        # 生成保存路径
        filename = os.path.basename(source_path)
        save_path = os.path.join(config.OUTPUT_CV_DIR, f"cv_{filename}")
        
        # 只有是视频文件才初始化 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (target_width, target_height))
        print(f"🎬 正在处理视频并保存至: {save_path}")
    else:
        print("📷 正在读取摄像头，按 'q' 退出（不保存视频）...")
    # ====================================================

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 强制缩放
        h, w = frame.shape[:2]
        if w != target_width:
            scale = target_width / w
            frame = cv2.resize(frame, (target_width, int(h * scale)))

        # 调用函数计算
        bars = find_light_bars(frame)
        armors = match_armors(bars)

        # result_img = draw_light_bars(frame, bars)
        result_img = draw_matched_bars(frame, armors)
        result_img = draw_armors(result_img, armors)

        # 如果是视频文件，才写入保存
        if out is not None:
            out.write(result_img)

        cv2.imshow("RM Armor Detector", result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    if out is not None:
        out.release()
        print(f"✅ 视频处理完成，已保存至: {save_path}")
    
    cv2.destroyAllWindows()
