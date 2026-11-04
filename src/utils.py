import cv2
import math
import numpy as np

# 画灯条
def draw_light_bars(img,light_bars):
    result_img = img.copy()

    for bar in light_bars:

        # 根据标签决定画框的颜色（BGR）
        if bar['color'] == 'red':
            draw_color = (0,0,255)
        elif bar['color'] == 'blue':
            draw_color = (255,0,0)
        else:
            draw_color = (0,255,0)


        # 画灯条外接矩形
        cv2.drawContours(result_img,[bar['box']],0,draw_color,4)

        # 画灯条中心点
        cx,cy = int(bar['center'][0]),int(bar['center'][1])
        cv2.circle(result_img,(cx,cy),4,(0,255,255),-1)

    return result_img

def draw_matched_bars(img, armors):
    """只画出配对成功的灯条"""
    result_img = img.copy()
    
    for armor in armors:
        # armor['bars'] 里刚好包含了一对配对成功的 bar1 和 bar2
        for bar in armor['bars']:
            # 依然根据颜色画框
            if bar['color'] == 'red': 
                draw_color = (0, 0, 255)   # 红框
            elif bar['color'] == 'blue': 
                draw_color = (255, 0, 0)   # 蓝框
            else: 
                draw_color = (0, 255, 0)
            
            # 画出这个灯条的外接框
            cv2.drawContours(result_img, [bar['box']], 0, draw_color, 5)
            
            # 画中心点
            cx, cy = int(bar['center'][0]), int(bar['center'][1])
            cv2.circle(result_img, (cx, cy), 3, (0, 255, 255), -1)
            
    return result_img

# 画装甲板
def draw_armors(img,armors):
    result_img = img.copy()
    for armor in armors:
        cx,cy = armor['center']

        # 1. 画装甲板框
        bar1 = armor['bars'][0]
        bar2 = armor['bars'][1]

        # 提取两根灯条的中心坐标
        p1 = bar1['center']
        p2 = bar2['center']

        # 算出装甲板的中心点和角度
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx))


        # 装甲板真实宽度 = 中心距 + 两边灯条各一半的宽度
        dist = math.sqrt(dx**2 + dy**2)
        total_width = dist + (bar1['size'][0] + bar2['size'][0]) / 2.0 

        # 装甲板真实高度 = 灯条平均高度 * 扩展系数
        # 1.25 是根据真实装甲板比例估算的（如果觉得不够大，可以改成 1.3）
        height_ratio = 1.7
        height = ((bar1['size'][1] + bar2['size'][1]) / 2.0) * height_ratio

        # 生成完美的倾斜矩形
        rect = ((cx, cy), (total_width, height), angle)
        box = cv2.boxPoints(rect).astype(int)

        # 画装甲板大框 
        cv2.drawContours(result_img, [box], 0, (255, 0, 255), 2)

        # 2. 画中心点
        cv2.circle(result_img, (cx, cy), 3, (0, 255, 0), -1)
        
        # 3. 标出坐标文字
        text = f"({cx}, {cy})"
        # 文字位置放在中心点稍微偏左上
        cv2.putText(result_img, text, (cx - 40, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 255), 5)

         # 4. 提取数字框（核心：裁剪中间区域）
        # 计算两根灯条四个角，直接截取中间区域
        # 4. 提取数字框（大收网版）
    try:
        pts1 = bar1['box']
        pts2 = bar2['box']
        all_pts = np.vstack((pts1, pts2))
        rx, ry, rw, rh = cv2.boundingRect(all_pts)
        
        # 【修改 1】砍掉两侧灯条即可，不要切太狠，不然数字被切没了
        shrink_w = int(rw * 0.15) 
        if rw > shrink_w * 2:
            rx = rx + shrink_w
            rw = rw - shrink_w * 2
        
        roi = img[ry:ry+rh, rx:rx+rw]
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 【修改 2】降低阈值到 150，让灰暗一点的笔画也能连上
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 【修改 3】不要遍历所有轮廓画框，找出面积最大的那一个（数字）
            if contours:
                largest_cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_cnt) > 50:
                    x, y, w, h = cv2.boundingRect(largest_cnt)
                    # 画出数字框（黄色）
                    cv2.rectangle(result_img, (rx + x, ry + y), (rx + x + w, ry + y + h), (0, 255, 255), 2)
    except Exception as e:
        pass# 防御性编程，防止数字框报错
            
                
    return result_img 
