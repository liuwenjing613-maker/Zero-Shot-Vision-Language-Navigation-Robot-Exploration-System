# ==============================================================================
# 文件名: topdown_visualizer.py
# 功能: 场景俯视图可视化模块 - 在场景真实俯视图上绘制导航路线
# 支持: v2/v3 导航系统
# ==============================================================================

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class TopdownVisualizer:
    """
    场景俯视图可视化器
    - 从 Habitat pathfinder 获取场景俯视图
    - 在真实场景图上绘制轨迹、路径、目标等
    """

    def __init__(self, sim, meters_per_pixel=0.05, height_for_map=None):
        """
        初始化俯视图可视化器
        
        Args:
            sim: Habitat Simulator 实例
            meters_per_pixel: 每像素代表的米数（越小分辨率越高，但越慢）
            height_for_map: 生成俯视图的高度（None 则自动选择）
        """
        self.sim = sim
        self.meters_per_pixel = meters_per_pixel
        self.pathfinder = sim.pathfinder
        
        # 通过采样可导航点来获取真实边界（比 get_bounds 更准确）
        nav_points = []
        for _ in range(5000):
            pt = self.pathfinder.get_random_navigable_point()
            if pt is not None and not np.isnan(pt).any():
                nav_points.append(pt)
        
        if len(nav_points) < 10:
            # 如果采样失败，回退到 get_bounds
            bounds = self.pathfinder.get_bounds()
            self.lower_bound = np.array(bounds[0])
            self.upper_bound = np.array(bounds[1])
            self.nav_y = (self.lower_bound[1] + self.upper_bound[1]) / 2
        else:
            nav_points = np.array(nav_points)
            # 使用可导航点的实际边界，留一点 margin
            margin = 0.5  # 0.5 米边距
            self.lower_bound = np.array([
                nav_points[:, 0].min() - margin,
                nav_points[:, 1].min(),
                nav_points[:, 2].min() - margin
            ])
            self.upper_bound = np.array([
                nav_points[:, 0].max() + margin,
                nav_points[:, 1].max(),
                nav_points[:, 2].max() + margin
            ])
            self.nav_y = np.median(nav_points[:, 1])  # 使用中位数高度
        
        # 场景在 x 和 z 方向的范围
        self.x_range = self.upper_bound[0] - self.lower_bound[0]
        self.z_range = self.upper_bound[2] - self.lower_bound[2]
        
        # 计算地图尺寸
        self.map_width = max(1, int(self.x_range / meters_per_pixel))
        self.map_height = max(1, int(self.z_range / meters_per_pixel))
        
        # 限制地图大小，避免过大
        max_size = 1200
        if self.map_width > max_size or self.map_height > max_size:
            scale = max_size / max(self.map_width, self.map_height)
            self.map_width = max(1, int(self.map_width * scale))
            self.map_height = max(1, int(self.map_height * scale))
        
        # 分别计算 x 和 z 方向的 meters_per_pixel（可能不同）
        self.meters_per_pixel_x = self.x_range / self.map_width if self.map_width > 0 else 1.0
        self.meters_per_pixel_z = self.z_range / self.map_height if self.map_height > 0 else 1.0
        
        
        # 生成基础俯视图
        self.base_topdown = self._generate_topdown_map(height_for_map)
        
        # 颜色配置（现代风格）
        self.colors = {
            'trajectory': (66, 165, 245),    # 蓝色 - 已走路线
            'plan': (255, 167, 38),          # 橙色 - 规划路径
            'current': (76, 175, 80),        # 绿色 - 当前位置
            'goal': (244, 67, 54),           # 红色 - 目标
            'start': (156, 39, 176),         # 紫色 - 起点
            'fbe': (255, 235, 59),           # 黄色 - FBE 探索点
            'arrow': (255, 255, 255),        # 白色 - 方向箭头
            'gt': (0, 215, 255),             # 金色 - GT 目标位置
        }
        
        # 创建 matplotlib figure
        self.fig = None
        self.ax = None
    
    def _generate_topdown_map(self, height=None):
        """生成场景俯视图"""
        # 直接用网格采样方式生成俯视图，更准确
        return self._generate_grid_topdown()
    
    def _generate_grid_topdown(self):
        """基于网格采样生成俯视图 - 更准确的坐标映射"""
        # 使用较粗的网格采样以提高速度
        sample_step = 2  # 每 2 个像素采样一次
        small_h = (self.map_height + sample_step - 1) // sample_step
        small_w = (self.map_width + sample_step - 1) // sample_step
        
        small_img = np.zeros((small_h, small_w), dtype=np.uint8)
        
        # 使用初始化时计算的导航高度
        y_sample = self.nav_y
        
        
        # 遍历采样网格 - 直接在世界坐标系中采样，避免坐标系转换问题
        for sy in range(small_h):
            for sx in range(small_w):
                # 直接从世界坐标计算，不使用 pixel_to_world（避免 z 轴翻转混乱）
                # 世界坐标: x 从 lower_bound[0] 到 upper_bound[0]
                #          z 从 lower_bound[2] 到 upper_bound[2]
                world_x = self.lower_bound[0] + sx * sample_step * self.meters_per_pixel_x
                world_z = self.lower_bound[2] + sy * sample_step * self.meters_per_pixel_z
                
                # 构造 3D 点
                world_pt = np.array([world_x, y_sample, world_z], dtype=np.float32)
                
                # 检查是否可导航
                try:
                    snapped = self.pathfinder.snap_point(world_pt)
                    if snapped is not None and not np.isnan(snapped).any():
                        # 检查 snap 后的点距离原点是否足够近
                        dist_xz = np.sqrt((snapped[0] - world_x)**2 + (snapped[2] - world_z)**2)
                        if dist_xz < self.meters_per_pixel_x * sample_step * 1.5:
                            # 图像坐标系 y 轴向下，对应世界坐标 z 轴反向
                            # small_img[row, col] 中 row 对应 z，col 对应 x
                            # z 从小到大 -> 图像 row 从下到上 -> 需要翻转
                            img_row = small_h - 1 - sy
                            small_img[img_row, sx] = 255
                except:
                    pass
        
        # 放大到原始尺寸
        nav_mask = cv2.resize(small_img, (self.map_width, self.map_height), interpolation=cv2.INTER_NEAREST)
        
        # 转为彩色图
        img = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        img[:] = [30, 30, 35]  # 深灰色背景
        img[nav_mask > 128] = [180, 180, 175]  # 可导航区域
        
        # 轻微模糊使边缘更平滑
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        return img
    
    def world_to_pixel(self, x, z):
        """世界坐标转像素坐标"""
        # 使用各自方向的缩放比例
        px = int((x - self.lower_bound[0]) / self.meters_per_pixel_x)
        pz = int((z - self.lower_bound[2]) / self.meters_per_pixel_z)
        # 翻转 z 轴使地图上方为 z 正方向
        py = self.map_height - 1 - pz
        # 限制在地图范围内
        px = max(0, min(px, self.map_width - 1))
        py = max(0, min(py, self.map_height - 1))
        return px, py
    
    def pixel_to_world(self, px, py):
        """像素坐标转世界坐标"""
        pz = self.map_height - 1 - py
        x = px * self.meters_per_pixel_x + self.lower_bound[0]
        z = pz * self.meters_per_pixel_z + self.lower_bound[2]
        return x, z
    
    def draw_trajectory(self, img, trajectory, color=None, thickness=1, alpha=0.8):
        """绘制已走轨迹（带渐变效果）"""
        if len(trajectory) < 2:
            return img
        
        color = color or self.colors['trajectory']
        overlay = img.copy()
        
        points = []
        for pt in trajectory:
            if len(pt) >= 2:
                x, z = pt[0], pt[1] if len(pt) == 2 else pt[2]
                px, py = self.world_to_pixel(x, z)
                points.append((px, py))
        
        if len(points) < 2:
            return img
        
        # 绘制带渐变的轨迹
        n = len(points)
        for i in range(n - 1):
            # 颜色渐变：旧的部分更淡
            fade = 0.4 + 0.6 * (i / n)
            c = tuple(int(v * fade) for v in color)
            cv2.line(overlay, points[i], points[i + 1], c, thickness, cv2.LINE_AA)
        
        # 混合
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img
    
    def draw_path(self, img, path_points, color=None, thickness=1, alpha=0.9):
        """绘制规划路径（虚线效果）"""
        if not path_points or len(path_points) < 2:
            return img
        
        color = color or self.colors['plan']
        
        points = []
        for pt in path_points:
            px, py = self.world_to_pixel(pt[0], pt[2])
            points.append((px, py))
        
        # 绘制虚线
        for i in range(len(points) - 1):
            self._draw_dashed_line(img, points[i], points[i + 1], color, thickness)
        
        return img
    
    def _draw_dashed_line(self, img, pt1, pt2, color, thickness, dash_length=10, gap_length=5):
        """绘制虚线"""
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        if dist < 1:
            return
        
        dx = (pt2[0] - pt1[0]) / dist
        dy = (pt2[1] - pt1[1]) / dist
        
        pos = 0
        drawing = True
        while pos < dist:
            if drawing:
                end_pos = min(pos + dash_length, dist)
                start = (int(pt1[0] + pos * dx), int(pt1[1] + pos * dy))
                end = (int(pt1[0] + end_pos * dx), int(pt1[1] + end_pos * dy))
                cv2.line(img, start, end, color, thickness, cv2.LINE_AA)
                pos = end_pos + gap_length
            else:
                pos += gap_length
            drawing = not drawing
    
    def draw_marker(self, img, pos, marker_type='current', size=12, label=None):
        """绘制位置标记"""
        if pos is None:
            return img
        
        # 支持 [x, z] 或 [x, y, z] 格式
        pos = np.array(pos).flatten()
        if len(pos) >= 3:
            x, z = float(pos[0]), float(pos[2])
        elif len(pos) >= 2:
            x, z = float(pos[0]), float(pos[1])
        else:
            return img
        px, py = self.world_to_pixel(x, z)
        
        
        color = self.colors.get(marker_type, (255, 255, 255))
        
        if marker_type == 'current':
            # 当前位置：实心圆 + 外圈
            cv2.circle(img, (px, py), size, color, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), size + 3, (255, 255, 255), 2, cv2.LINE_AA)
        elif marker_type == 'goal':
            # 目标：十字 + 圆圈
            cv2.circle(img, (px, py), size, color, 2, cv2.LINE_AA)
            cv2.line(img, (px - size, py), (px + size, py), color, 2, cv2.LINE_AA)
            cv2.line(img, (px, py - size), (px, py + size), color, 2, cv2.LINE_AA)
        elif marker_type == 'start':
            # 起点：方块
            half = size // 2
            cv2.rectangle(img, (px - half, py - half), (px + half, py + half), color, -1, cv2.LINE_AA)
            cv2.rectangle(img, (px - half, py - half), (px + half, py + half), (255, 255, 255), 1, cv2.LINE_AA)
        elif marker_type == 'fbe':
            # FBE 探索点：三角形
            pts = np.array([
                [px, py - size],
                [px - size, py + size // 2],
                [px + size, py + size // 2]
            ], np.int32)
            cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        elif marker_type == 'gt':
            # GT 目标位置：星形标记（菱形 + 十字）
            # 外圈菱形
            pts = np.array([
                [px, py - size],
                [px + size, py],
                [px, py + size],
                [px - size, py]
            ], np.int32)
            cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
            # 内部十字
            half = size // 2
            cv2.line(img, (px - half, py), (px + half, py), color, 2, cv2.LINE_AA)
            cv2.line(img, (px, py - half), (px, py + half), color, 2, cv2.LINE_AA)
            # 中心点
            cv2.circle(img, (px, py), 3, color, -1, cv2.LINE_AA)
        
        # 添加标签
        if label:
            cv2.putText(img, label, (px + size + 5, py + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return img
    
    def draw_direction_arrow(self, img, pos, yaw, length=25, color=None):
        """绘制方向箭头"""
        if pos is None:
            return img
        
        color = color or self.colors['arrow']
        x, z = pos[0], pos[2] if len(pos) > 2 else pos[1]
        px, py = self.world_to_pixel(x, z)
        
        # 计算箭头终点（注意：像素坐标系 y 轴向下）
        dx = int(length * np.sin(yaw))
        dy = int(-length * np.cos(yaw))  # 负号因为像素 y 轴向下
        
        end_x, end_y = px + dx, py + dy
        
        # 绘制箭头
        cv2.arrowedLine(img, (px, py), (end_x, end_y), color, 2, cv2.LINE_AA, tipLength=0.3)
        
        return img
    
    def render(self, trajectory=None, path_points=None, current_pos=None, 
               goal_pos=None, start_pos=None, current_yaw=None,
               title=None, instruction=None, status=None, fbe_point=None,
               gt_pos=None):
        """
        渲染完整的俯视图
        
        Args:
            trajectory: 已走轨迹 [[x, z], ...]
            path_points: 规划路径 [[x, y, z], ...]
            current_pos: 当前位置 [x, y, z]
            goal_pos: 目标位置 [x, y, z]
            start_pos: 起点位置 [x, y, z]
            current_yaw: 当前朝向（弧度）
            title: 标题
            instruction: 指令
            status: 状态
            fbe_point: FBE 探索点
            gt_pos: GT 目标位置（可选，仅用于可视化）
        
        Returns:
            渲染好的图像 (numpy array, BGR)
        """
        # 复制基础地图
        img = self.base_topdown.copy()
        
        # 绘制 GT 目标位置（最先绘制，在最底层）
        if gt_pos is not None:
            img = self.draw_marker(img, gt_pos, 'gt', size=10, label='GT')
        
        # 绘制起点
        if start_pos is not None:
            img = self.draw_marker(img, start_pos, 'start', size=6, label='Start')
        
        # 绘制轨迹
        if trajectory:
            img = self.draw_trajectory(img, trajectory)
        
        # 绘制规划路径
        if path_points:
            img = self.draw_path(img, path_points)
        
        # 绘制 FBE 探索点
        if fbe_point is not None:
            img = self.draw_marker(img, fbe_point, 'fbe', size=5, label='FBE')
        
        # 绘制目标
        if goal_pos is not None:
            img = self.draw_marker(img, goal_pos, 'goal', size=8, label='Goal')
        
        # 绘制当前位置
        if current_pos is not None:
            img = self.draw_marker(img, current_pos, 'current', size=6)
            if current_yaw is not None:
                img = self.draw_direction_arrow(img, current_pos, current_yaw, length=15)
        
        # 添加信息面板
        img = self._add_info_panel(img, title, instruction, status)
        
        return img
    
    def _add_info_panel(self, img, title=None, instruction=None, status=None):
        """添加信息面板"""
        h, w = img.shape[:2]
        
        # 顶部半透明面板
        panel_height = 80
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        
        # 标题
        y_offset = 25
        if title:
            cv2.putText(img, title, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 25
        
        # 指令
        if instruction:
            cv2.putText(img, f"Goal: {instruction}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (66, 165, 245), 1, cv2.LINE_AA)
            y_offset += 22
        
        # 状态
        if status:
            # 根据状态选择颜色
            if 'Locked' in str(status):
                status_color = (76, 175, 80)  # 绿色
            elif 'FBE' in str(status):
                status_color = (255, 235, 59)  # 黄色
            else:
                status_color = (255, 167, 38)  # 橙色
            cv2.putText(img, f"Status: {status}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)
        
        # 图例（右下角）
        legend_x = w - 150
        legend_y = h - 100
        legend_items = [
            ('Trajectory', self.colors['trajectory']),
            ('Plan', self.colors['plan']),
            ('Current', self.colors['current']),
            ('Goal', self.colors['goal']),
            ('GT', self.colors['gt']),
        ]
        for i, (label, color) in enumerate(legend_items):
            y = legend_y + i * 18
            cv2.circle(img, (legend_x, y), 5, color, -1)
            cv2.putText(img, label, (legend_x + 12, y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        return img
    
    def init_matplotlib_figure(self, figsize=(10, 10)):
        """初始化 matplotlib figure（用于嵌入现有可视化）"""
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=100)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        return self.fig, self.ax
    
    def update_matplotlib(self, trajectory=None, path_points=None, current_pos=None,
                          goal_pos=None, start_pos=None, current_yaw=None,
                          title=None, instruction=None, status=None, fbe_point=None,
                          gt_pos=None):
        """更新 matplotlib 显示"""
        if self.ax is None:
            self.init_matplotlib_figure()
        
        # 渲染图像
        img = self.render(trajectory, path_points, current_pos, goal_pos, 
                         start_pos, current_yaw, title, instruction, status, fbe_point,
                         gt_pos)
        
        # 转换 BGR 到 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 更新显示
        self.ax.clear()
        self.ax.imshow(img_rgb)
        self.ax.axis('off')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        return img


def create_topdown_visualizer(sim, meters_per_pixel=0.05):
    """创建俯视图可视化器的便捷函数"""
    return TopdownVisualizer(sim, meters_per_pixel)
