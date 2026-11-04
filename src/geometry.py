import math
from src import config

def match_armors(light_bars):
    armors = []
    used_bars = set()
    
    for i in range(len(light_bars)):
        if i in used_bars: 
            continue
        for j in range(i + 1, len(light_bars)):
            if j in used_bars: 
                continue
                
            bar1 = light_bars[i]
            bar2 = light_bars[j]
            
            # 1. 颜色
            if bar1['color'] != bar2['color']: 
                continue
            
            # 计算两灯条平均高度，作为“尺子“
            avg_h = (bar1['size'][1] + bar2['size'][1]) / 2.0
            if avg_h < 15:
                continue

            # 2. Y轴高度差（自适应：不超过平均高度的 60%）
            y_diff = abs(bar1['center'][1] - bar2['center'][1])
            if y_diff > avg_h * config.MAX_Y_DIFF_RATIO: 
                continue
                
            # 3. 两灯条的长度差不能太大（ <= 40% )
            h_diff = abs(bar1['size'][1] - bar2['size'][1])
            if h_diff > avg_h * config.MAX_H_DIFF_RATIO:
                continue
                
            # 4. 距离比例
            dist = math.sqrt((bar1['center'][0] - bar2['center'][0])**2 +
                              (bar1['center'][1] - bar2['center'][1])**2)
            if dist < config.MIN_DISTANCE_PIXELS:
                continue

            ratio = dist / avg_h
        
            if not (config.MIN_DISTANCE_RATIO < ratio < config.MAX_DISTANCE_RATIO):
                print(f"🛑 差点就配上了! 但比例不对: {ratio:.2f}, 距离:{dist:.1f}, 高度:{avg_h:.1f}")
                continue
                
            # 配对成功
            armor_cx = int((bar1['center'][0] + bar2['center'][0]) / 2)
            armor_cy = int((bar1['center'][1] + bar2['center'][1]) / 2)
            
            armors.append({
                'color': bar1['color'],
                'center': (armor_cx, armor_cy),
                'bars': [bar1, bar2]
            })

            used_bars.add(i)
            used_bars.add(j)
            print(f"✅ 成功配对一对 {bar1['color']} 装甲板！")
            break
            
    return armors