import os

# ================= 路径配置区 =================
# 动态获取项目的根目录 (即 src 的上一级)
# __file__ 是 config.py 的路径，dirname 两次就回到了项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 定义各文件夹的绝对路径
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# 定义具体文件/文件夹的输出路径
MODEL_PATH = os.path.join(WEIGHTS_DIR, 'horizon_armor.pt')
OUTPUT_HYBRID_DIR = os.path.join(OUTPUTS_DIR, 'hybrid')
OUTPUT_YOLO_DIR = os.path.join(OUTPUTS_DIR, 'yolo')
OUTPUT_CV_DIR = os.path.join(OUTPUTS_DIR, 'cv')
# ============================================



# ================== 颜色阈值 (HSV) ==================
# 注意：这里数值仅供参考，如果在视频流下太暗，请适当降低 V 的下限
LOWER_RED = (0, 80, 180)
UPPER_RED = (10, 255, 255)
LOWER_RED2 = (170, 80, 180)
UPPER_RED2 = (180, 255, 255)

LOWER_BLUE = (100, 80, 180)
UPPER_BLUE = (124, 255, 255)

# ================== 预处理参数 ==================
# 形态学核大小 (竖向长条)
KERNEL_CLOSE_SIZE = (5,15)
KERNEL_OPEN_SIZE = (7,7)

# ================== 灯条过滤 (放宽初筛) ==================
MIN_AREA = 70       # 下调面积，包容远处的灯条（如果在视频流里杂点太多，可以调回 200）
MIN_RATIO = 1.5     # 下调宽高比，包容光晕胖灯条
MAX_RATIO = 7.0
MIN_FILL_RATIO = 0.5
MIN_HEIGHT = 35     # 灯条的最小像素高度

# ================== 装甲板配对阈值 (相对比例) ==================
# Y轴高度差：不能超过灯条平均高度的 60%
MAX_Y_DIFF_RATIO = 0.5
# 长度差：两根灯条的高度差不能超过平均高度的 40%
MAX_H_DIFF_RATIO = 0.5
# 最小像素距离
MIN_DISTANCE_PIXELS = 30
# 距离比：两根灯条距离 / 平均高度。卡在 1.0 ~ 4.5 之间
MIN_DISTANCE_RATIO = 1.0
MAX_DISTANCE_RATIO = 5.0