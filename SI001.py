import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =====================================================================
#  第一部分：设定去中心化宇宙的物理常数
# =====================================================================
GRID_SIZE = 80          # 空间格子的尺寸 (80x80)
NUM_ANTS = 800          # 粒子数量（增加到800以弥补失去全局感知后的碰撞概率）
EVAPORATION_RATE = 0.04 # 痕迹蒸发率 (每步蒸发 4%)
DIFFUSION_RATE = 0.01   # 痕迹扩散率 (每步向周围蔓延 1%)
PHEROMONE_DROP = 15.0   # 状态回流时的信息素烫印剂量 (自回归强度)

# 核心物理坐标边界
NEST = np.array([40, 40])       # 蚁巢设在沙盒中心
FOODS = [
    np.array([15, 15]),         # 食物点 A (左上方)
    np.array([65, 65])          # 食物点 B (右下方)
]
FOOD_RADIUS = 3

# =====================================================================
#  第二部分：初始化双通道空间与局部状态
# =====================================================================
# 双通道空间：
# 通道 0 -> Home Pheromone (家信息素：白蚂蚁留下，蓝蚂蚁顺着它回家)
# 通道 1 -> Food Pheromone (食物信息素：蓝蚂蚁留下，白蚂蚁顺着它找食物)
pheromone_grid = np.zeros((GRID_SIZE, GRID_SIZE, 2))

# 初始化在蚁巢（巢穴处天然自带最高浓度的家信息素，作为演化的初始火种）
pheromone_grid[NEST[0], NEST[1], 0] = 500.0

# 个体状态 (X坐标, Y坐标, 是否携带食物)
ant_positions = np.full((NUM_ANTS, 2), NEST, dtype=float)
ant_has_food = np.zeros(NUM_ANTS, dtype=bool)

# 移动方向向量池（限制 45 度视角，维持运动的连续时间流）
DIRECTIONS = np.array([
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1],           [0, 1],
    [1, -1],  [1, 0],  [1, 1]
])
ant_last_dir = np.random.randint(0, 8, size=NUM_ANTS)

# =====================================================================
#  第三部分：实施完全局部的因果交叉递归（核心动力学）
# =====================================================================
def update_universe(frame):
    global pheromone_grid, ant_positions, ant_has_food, ant_last_dir
    
    # --- 1. 空间的物理定律：双通道无情蒸发与扩散 ---
    for ch in range(2):
        diffused = pheromone_grid[:, :, ch] * (1.0 - DIFFUSION_RATE)
        diffused[1:-1, 1:-1] += DIFFUSION_RATE * 0.25 * (
            pheromone_grid[:-2, 1:-1, ch] + pheromone_grid[2:, 1:-1, ch] +
            pheromone_grid[1:-1, :-2, ch] + pheromone_grid[1:-1, 2:, ch]
        )
        pheromone_grid[:, :, ch] = diffused * (1.0 - EVAPORATION_RATE)
    
    # 巢穴源源不断维持原初的火种
    pheromone_grid[NEST[0], NEST[1], 0] = 500.0
    
    # --- 2. 遍历每一个完全失去外挂的近视眼个体 ---
    for i in range(NUM_ANTS):
        pos = ant_positions[i]
        has_food = ant_has_food[i]
        last_dir_idx = ant_last_dir[i]
        
        pos_int = np.clip(pos.astype(int), 1, GRID_SIZE - 2)
        
        # 【微观状态翻转监测】
        if not has_food:
            # 找食物状态：踩到绿点就变蓝，并原地调头
            for food in FOODS:
                if np.linalg.norm(pos - food) < FOOD_RADIUS:
                    ant_has_food[i] = True
                    has_food = True
                    last_dir_idx = (last_dir_idx + 4) % 8
                    # 发现食物的地方天然自带最高浓度的食物信息素火种
                    pheromone_grid[pos_int[0], pos_int[1], 1] = 500.0
                    break
        else:
            # 运食物状态：到家了就变白，并原地调头
            if np.linalg.norm(pos - NEST) < 2:
                ant_has_food[i] = False
                has_food = False
                last_dir_idx = (last_dir_idx + 4) % 8
        
        # 【核心对称规则：只看眼前 3 个格子的局部感知】
        valid_indices = [(last_dir_idx - 1) % 8, last_dir_idx, (last_dir_idx + 1) % 8]
        
        # 决定看哪一个通道
        # 白蚂蚁(寻找食物)感知 通道1(食物信息素)；蓝蚂蚁(运送食物)感知 通道0(家信息素)
        target_channel = 0 if has_food else 1
        
        intensities = []
        for idx in valid_indices:
            next_check = pos_int + DIRECTIONS[idx]
            next_check = np.clip(next_check, 0, GRID_SIZE - 1)
            intensities.append(pheromone_grid[next_check[0], next_check[1], target_channel])
            
        sum_int = sum(intensities)
        
        # 概率向心力选择
        if sum_int < 0.05:
            # 眼前一片茫然（0），盲目随机选一个方向探索
            chosen_idx = np.random.choice(valid_indices)
        else:
            # 哪里浓度高，就以哪里为重力概率滑过去（10% 自由扰动防止绝对刚性死锁）
            if np.random.rand() < 0.1:
                chosen_idx = np.random.choice(valid_indices)
            else:
                probs = [int_val / sum_int for int_val in intensities]
                chosen_idx = np.random.choice(valid_indices, p=probs)
                
        move_dir = DIRECTIONS[chosen_idx]
        ant_last_dir[i] = chosen_idx
        
        # --- 状态回流（真递归）：迈步的同时，在相反通道烫下自己的痕迹 ---
        # 白蚂蚁给蓝蚂蚁铺回家的路(留下通道0)；蓝蚂蚁给白蚂蚁铺去食物的路(留下通道1)
        drop_channel = 1 if has_food else 0
        pheromone_grid[pos_int[0], pos_int[1], drop_channel] += PHEROMONE_DROP
        
        # 执行微观位移
        new_pos = pos + move_dir * 0.8
        ant_positions[i] = np.clip(new_pos, 0, GRID_SIZE - 1)

    # --- 第四部分：渲染高维双通道的因果流形 ---
    ax.clear()
    
    # 视觉表现：我们将两个通道融合（红色代表食物信息素，绿色代表家信息素）
    # 混合在一起亮黄色的地方，就是两条因果回路交织产生的稳态“最优路径”
    display_img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    display_img[:, :, 0] = np.clip(pheromone_grid[:, :, 1] * 0.4, 0, 1) # R: 食物痕迹
    display_img[:, :, 1] = np.clip(pheromone_grid[:, :, 0] * 0.4, 0, 1) # G: 家痕迹
    
    ax.imshow(np.transpose(display_img, (1, 0, 2)), origin="lower")
    
    # 绘制基础物理实体边界
    ax.add_patch(plt.Circle((NEST[0], NEST[1]), 2, color='cyan', fill=True, label='Nest'))
    for food in FOODS:
        ax.add_patch(plt.Circle((food[0], food[1]), FOOD_RADIUS, color='magenta', fill=True, label='Food'))
    
    # 绘制粒子（白色为找食物，蓝色为运食物）
    colors = np.where(ant_has_food, '#00ffff', '#ffffff')
    ax.scatter(ant_positions[:, 0], ant_positions[:, 1], c=colors, s=1.5, alpha=0.7)
    
    ax.set_title(f"Symmetric Emergence Epoch: {frame} | No Global Guide", color='white')
    ax.axis('off')

# =====================================================================
#  第四部分：启动数字宇宙
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
fig.patch.set_facecolor('black')
ani = FuncAnimation(fig, update_universe, frames=2000, interval=15, cache_frame_data=False)
plt.show()