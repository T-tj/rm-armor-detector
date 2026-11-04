import cv2
from src import config 

def find_light_bars(img):

    h_img,_ = img.shape[:2]

    # HSV 取掩膜
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    mask_red1 = cv2.inRange(hsv,config.LOWER_RED,config.UPPER_RED)
    mask_red2 = cv2.inRange(hsv,config.LOWER_RED2,config.UPPER_RED2)
    mask_red = cv2.bitwise_or(mask_red1,mask_red2)
    mask_blue = cv2.inRange(hsv,config.LOWER_BLUE,config.UPPER_BLUE)

    # 将颜色名称和掩膜打包在一起，方便循环
    color_masks = [
        ("red",mask_red),
        ("blue",mask_blue)
    ]

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,config.KERNEL_CLOSE_SIZE)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT,config.KERNEL_OPEN_SIZE)
    light_bars = []

    # 循环处理红，蓝二色
    for color_name,mask in color_masks:
        # 形态学运算去噪
        # 1.闭运算(竖向连接断点）
        mask= cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel_close)
        # 开运算去外围光晕
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel_open)

        # 调试查看
        cv2.imwrite(f"output/test_mask_{color_name}.jpg",mask)

        # 找轮廓
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < config.MIN_AREA:
                continue

            rect = cv2.minAreaRect(contour)
            # 如果 rect 的长度是 0，说明提取失败，直接跳过
            if len(rect) == 0:
                continue

            (cx, cy), (w, h), angle = rect
            # 视野屏蔽（ROI）
            # 如果中心点 Y 坐标大于图像高度的 85%（即处于画面最底部），直接枪毙
            if cy > h_img*0.85:
                continue


            # 统一长度，让长边永远为 h
            if w > h:
                w,h = h,w
                angle += 90

            # 几何过滤
            aspect_ratio = h / w
            fill_ratio = area / (w*h) if (w*h) > 0 else 0

            if (config.MIN_RATIO < aspect_ratio <config.MAX_RATIO and
                h > config.MIN_HEIGHT and
                fill_ratio > config.MIN_FILL_RATIO):

                # 存下灯条的信息，方便下一步配对
                light_bars.append({
                    'color': color_name,
                    'center': (cx,cy),
                    'size': (w,h),
                    'angle': angle,
                    'box': cv2.boxPoints(rect).astype(int)
                })
    return light_bars

