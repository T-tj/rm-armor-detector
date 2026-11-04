import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.detector import find_light_bars
from src.geometry import match_armors
from src.utils import draw_matched_bars , draw_armors
from src import config

if __name__ == "__main__":

    in_path = 'data/robot.jpg'

    # 获取输入图片的文件名（比如 armor_08.jpeg）
    filename = os.path.basename(in_path)
    # 确保输出目录存在
    os.makedirs(config.OUTPUT_CV_DIR, exist_ok=True)
    # 拼出完整的输出路径：文件夹路径 + 文件名
    out_path = os.path.join(config.OUTPUT_CV_DIR, f"cv_{filename}")


    img = cv2.imread(in_path)
    if img is None:
        print("读取失败")
        exit()

    # 强制把所有输入图缩放到统一宽度，比如 1280，高度按比例缩放
    target_width = 1280
    h, w = img.shape[:2]
    if w != target_width:
        scale = target_width / w
        img = cv2.resize(img, (target_width, int(h * scale)))
    
    # 调用函数计算    
    bars = find_light_bars(img)
    print(f"找到 {len(bars)} 个疑似灯条")

    armors = match_armors(bars)
    print(f"找到 {len(armors)} 个装甲板")


    # 调用函数画图
    result_img = draw_matched_bars(img, armors)
    result_img = draw_armors(result_img,armors)

    cv2.imwrite(out_path, result_img)
    cv2.imshow(out_path, result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()