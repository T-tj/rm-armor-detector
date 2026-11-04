import cv2
import numpy as np
from src import config
from src.detector import find_light_bars
from src.geometry import match_armors

def get_armor_color_by_roi(frame, x1, y1, x2, y2):
    """重点检测两侧灯条区域的颜色，抵抗背景干扰"""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return "unknown"

    h_img, w_img = roi.shape[:2]

    # 核心优化：把框切成三份，取左边 1/3 和 右边 1/3（避开中间的数字）
    left_roi = roi[:, :w_img//3]
    right_roi = roi[:, -w_img//3:]

    def count_colors(img_part):
        hsv = cv2.cvtColor(img_part, cv2.COLOR_BGR2HSV)
        mask_red1 = cv2.inRange(hsv, config.LOWER_RED, config.UPPER_RED)
        mask_red2 = cv2.inRange(hsv, config.LOWER_RED2, config.UPPER_RED2)
        mask_blue = cv2.inRange(hsv, config.LOWER_BLUE, config.UPPER_BLUE)
        
        red_cnt = cv2.countNonZero(mask_red1) + cv2.countNonZero(mask_red2)
        blue_cnt = cv2.countNonZero(mask_blue)
        return red_cnt, blue_cnt

    # 分别计算左右两边的红蓝像素总和
    r1, b1 = count_colors(left_roi)
    r2, b2 = count_colors(right_roi)
    total_red = r1 + r2
    total_blue = b1 + b2

    # 谁的总数多，就判给谁
    if total_red > total_blue:
        return "red"
    elif total_blue > total_red:
        return "blue"
    else:
        return "unknown"


def draw_yolo_boxes(frame, results):
    """画出 YOLO 的装甲板框（红紫蓝青）、中心点、坐标、数字"""
    yolo_boxes = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            yolo_boxes.append((x1, y1, x2, y2))
            
            # 1. 判断颜色
            color_type = get_armor_color_by_roi(frame, x1, y1, x2, y2)
            
            if color_type == "red":
                draw_color = (255, 0, 255)   # 紫色
            elif color_type == "blue":
                draw_color = (255, 255, 0)   # 青色
            else:
                draw_color = (255, 255, 255)     # 白色
                
            # 2. 画装甲板大框
            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 4)
            
            # 3. 画中心点、坐标
            cx = int(box.xywh[0][0])
            cy = int(box.xywh[0][1])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            text = f"({cx}, {cy})"
            cv2.putText(frame, text, (cx + 15, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 4. 提取数字框
            shrink_x = int((x2 - x1) * 0.25)
            rx1, rx2 = x1 + shrink_x, x2 - shrink_x
            if rx2 > rx1:
                roi_digit = frame[y1:y2, rx1:rx2]
                if roi_digit.size > 0:
                    gray = cv2.cvtColor(roi_digit, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_cnt = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest_cnt) > 50:
                            dx, dy, dw, dh = cv2.boundingRect(largest_cnt)
                            cv2.rectangle(frame, (rx1 + dx, y1 + dy), (rx1 + dx + dw, y1 + dy + dh), (0, 255, 255), 2)

    return yolo_boxes


def process_frame_hybrid(frame, model):
    """双引擎融合核心逻辑（干净防污染 + 边界截断）"""
    
    # 备份一张干净的原图（没有画过任何紫框、绿点）
    clean_frame = frame.copy() 

    # 1. YOLO 识别（在 frame 上画图，只用于展示）
    results = model(frame, conf=0.5, iou=0.45, verbose=False)
    yolo_boxes = draw_yolo_boxes(frame, results)

    # 2. 传统 CV 在“干净的原图”上裁剪和识别
    for (x1, y1, x2, y2) in yolo_boxes:
        padding = 0 
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
        
        # 从 clean_frame 里切，不让紫框污染传统 CV
        roi = clean_frame[y1:y2, x1:x2]
        if roi.size == 0: continue
        
        bars = find_light_bars(roi)
        armors = match_armors(bars)
        
        for armor in armors:
            for bar in armor['bars']:
                draw_color = (0, 0, 255) if bar['color'] == 'red' else (255, 0, 0)
                global_box = bar['box'] + [x1, y1]
                
                # 强制把灯条框“压”在 YOLO 框内部，杜绝外扩
                global_box[:, 0] = np.clip(global_box[:, 0], x1, x2)
                global_box[:, 1] = np.clip(global_box[:, 1], y1, y2)
                
                cv2.drawContours(frame, [global_box], 0, draw_color, 2)
                cx, cy = int(bar['center'][0] + x1), int(bar['center'][1] + y1)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    return frame