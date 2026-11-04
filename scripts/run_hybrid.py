import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import cv2
from ultralytics import YOLO
from src.hybrid_core import process_frame_hybrid
from src import config


# ================= 配置区 =================
# 输入源设置：

# 1. 读图片：
source_path = 'data/rm_armor.jpg'
# 2. 读文件夹：
# source_path = 'data'
# 3. 读视频：
# source_path = 'data/armors.mp4'
# 4. 读摄像头：
# source_path = 0
# ==========================================

def main():
    # 1. 检查模型
    if not os.path.exists(config.MODEL_PATH):
        print(f"❌ 找不到模型文件: {config.MODEL_PATH}")
        return
        
    print(f"🔄 正在加载模型: {config.MODEL_PATH} ...")
    model = YOLO(config.MODEL_PATH)
    
    # 确保输出文件夹存在
    os.makedirs(config.OUTPUT_HYBRID_DIR, exist_ok=True)
    
    # 判断输入源类型
    is_camera = isinstance(source_path, int)
    is_video = isinstance(source_path, str) and source_path.lower().endswith(('.mp4', '.avi', '.mov'))
    is_dir = isinstance(source_path, str) and os.path.isdir(source_path)   # 判断是否为文件夹
    
    # ================= 视频 / 摄像头分支 =================
    if is_camera or is_video:
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"❌ 无法打开输入源: {source_path}")
            return
            
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # 只有视频文件才保存，摄像头不保存
        out = None
        if is_video:
            filename = os.path.basename(source_path)
            save_path = os.path.join('outputs/hybrid', f"hybrid_{filename}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps_video, (width, height))
            print(f"🎬 正在处理并保存视频至: {save_path}")
        else:
            print("📷 正在读取摄像头，按 'q' 退出...")

        # 逐帧处理
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 核心融合处理
            frame = process_frame_hybrid(frame, model)
            
            # 保存视频
            if out is not None:
                out.write(frame)
                
            # 实时显示
            cv2.imshow("Hybrid Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        if out is not None:
            out.release()
        print("✅ 处理完成！")

    # ================= 文件夹 分支 =================
    elif is_dir:
        # 查找文件夹下所有的 jpg, png, jpeg 图片
        image_paths = glob.glob(os.path.join(source_path, '*.jpg')) + \
                      glob.glob(os.path.join(source_path, '*.png')) + \
                      glob.glob(os.path.join(source_path, '*.jpeg'))
                      
        print(f"📂 发现文件夹，共找到 {len(image_paths)} 张图片，开始批量处理...")
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None: continue
            
            # 核心处理
            result_img = process_frame_hybrid(img, model)
            
            # 保存结果
            filename = os.path.basename(img_path)
            save_path = os.path.join(config.OUTPUT_HYBRID_DIR, f"hybrid_{filename}")
            cv2.imwrite(save_path, result_img)
            print(f"✅ 图片已保存至: {save_path}")

    # ================= 图片分支 =================
    else:
        img = cv2.imread(source_path)
        if img is None:
            print(f"❌ 无法读取图片: {source_path}")
            return
            
        print(f"🖼️ 正在处理图片: {source_path}")
        result_img = process_frame_hybrid(img, model)
        
        # 显示并保存图片
        cv2.imshow("Hybrid Detector", result_img)
        filename = os.path.basename(source_path)
        save_path = os.path.join(config.OUTPUT_HYBRID_DIR, f"hybrid_{filename}")
        cv2.imwrite(save_path, result_img)
        print(f"✅ 图片已保存至: {save_path}")
        
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()