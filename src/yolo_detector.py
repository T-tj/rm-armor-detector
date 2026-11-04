import cv2
import time
import os
import glob
from src import config

def draw_yolo_custom(frame, results):
    """自定义画图函数：画框、画中心准星、写坐标"""
    # 1. 先用 YOLO 默认的画框（画边界框和类别文字）
    annotated_frame = results[0].plot()
    
    # 2. 获取所有的检测框
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            # 提取中心点
            cx = int(box.xywh[0][0])
            cy = int(box.xywh[0][1])
            
            # 画绿色十字准星
            cv2.drawMarker(annotated_frame, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 20, 5)
            
            # 写黄色坐标文字
            text = f"({cx}, {cy})"
            cv2.putText(annotated_frame, text, (cx + 15, cy - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    return annotated_frame


def process_source(model, source_path, output_dir = None):
    """统一的推理入口，支持图片、文件夹、视频"""
    if output_dir is None:
            output_dir = config.OUTPUT_YOLO_DIR

    os.makedirs(output_dir, exist_ok=True)

    # 判断输入源类型
    if os.path.isdir(source_path):
        # 如果是文件夹，处理里面的所有图片
        image_paths = glob.glob(os.path.join(source_path, '*.jpg')) + \
                      glob.glob(os.path.join(source_path, '*.png'))
        print(f"📂 发现文件夹，共 {len(image_paths)} 张图片")
        for img_path in image_paths:
            process_image(model, img_path, output_dir)

    elif source_path.lower().endswith(('.mp4', '.avi', '.mov')):
        # 如果是视频文件
        process_video(model, source_path, output_dir)

    elif source_path.lower().endswith(('.jpg', '.png', '.jpeg')):
        # 如果是单张图片
        process_image(model, source_path, output_dir)
    else:
        print(f"❌ 不支持的输入格式: {source_path}")


def process_image(model, img_path, output_dir):
    """处理单张图片"""
    print(f"🖼️  正在处理图片: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图片: {img_path}")
        return

    # 推理与测速
    start_time = time.time()
    results = model(img, conf=0.5, iou=0.45, verbose=False)
    inference_time = (time.time() - start_time) * 1000
    
    # 画图
    annotated_img = draw_yolo_custom(img, results)
    
    # 叠加推理时间信息
    cv2.putText(annotated_img, f"YOLO Inf: {inference_time:.1f}ms", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 保存
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_dir, f"yolo_{filename}")
    cv2.imwrite(save_path, annotated_img)
    print(f"✅ 结果已保存至: {save_path}")


def process_video(model, video_path, output_dir):
    """处理视频流"""
    print(f"🎬 正在处理视频: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video == 0: fps_video = 30.0 # 防御性：有些视频读不到FPS

    # 设置视频保存器
    filename = os.path.basename(video_path)
    save_path = os.path.join(output_dir, f"yolo_{filename}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps_video, (width, height))

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 推理
        results = model(frame, conf=0.5, iou=0.45, verbose=False)
        
        # 画图
        annotated_frame = draw_yolo_custom(frame, results)
        
        # 计算并显示 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(annotated_frame, f"YOLO FPS: {fps:.1f}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 写入保存
        out.write(annotated_frame)
        
        # 实时显示
        cv2.imshow("YOLO Armor Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 视频处理完成，已保存至: {save_path}")