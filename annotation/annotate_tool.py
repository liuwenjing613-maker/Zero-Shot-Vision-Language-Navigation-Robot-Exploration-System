#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Habitat-sim MP3D 键盘导航标注工具 —— System 2 数据采集
=============================================================================
功能:
  1. 用 habitat-sim 加载 MP3D .glb 场景，渲染第一视角图像
  2. 用键盘控制机器人在场景中自由移动
  3. 按 E 键进入标注模式：冻结当前帧
  4. 在冻结帧上点击目标像素，输入导航指令，回车保存
  5. 自动写入 data.json，支持断点续标，收集满100条自动提示

依赖安装:
  # 安装 habitat-sim（推荐 conda 环境）
  conda install habitat-sim withbullet headless -c conda-forge -c aihabitat

  # 或 pip（需要headless版）
  pip install habitat-sim

  # 其余依赖
  pip install pillow numpy

键盘操作说明:
  W / ↑        前进
  S / ↓        后退
  A            左转
  D            右转
  Q            向左横移 (strafe left)
  E_key        向右横移 (strafe right) [注意: E单独按是横移]
  ←→           左转/右转（同A/D）
  [E] 标注键   切换到 标注模式（进入后鼠标点击目标像素）
  注意: 标注模式用 GUI 按钮进入，避免与横移E冲突

  ─── 标注模式 ───
  点击图片      标记目标像素（红色十字准星）
  回车/确认键   保存当前标注
  Esc/取消      取消标注，返回导航

使用方法:
  python annotate_habitat.py \
      --mp3d_dir ./scenes/MatterPort3D/mp3d \
      --output data.json

  # 指定场景（不随机选）
  python annotate_habitat.py \
      --mp3d_dir ./scenes/MatterPort3D/mp3d \
      --scene 2azQ1b91cZZ \
      --output data.json
=============================================================================
"""

import os
import sys
import json
import argparse
import shutil
import random
import threading
import time
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageTk, ImageFont
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog


# ============================================================
# 关键：GL/Display 环境预设置
# 必须在 import habitat_sim 之前执行，否则 GL context 初始化失败崩溃
# ============================================================
import subprocess, os as _os

def _setup_display():
    """
    确保有可用的 X Display，habitat-sim 的 GL 渲染器依赖 X11 或 EGL。

    策略（按优先级依次尝试）:
      1. DISPLAY 已设置且 X server 可访问 → 直接使用
      2. DISPLAY 未设置 → 尝试 :0, :1 等常用显示器编号
      3. 都不可用 → 启动 Xvfb 虚拟帧缓冲（需安装: sudo apt install xvfb）
    """
    # 已有 DISPLAY 且可连接，直接返回
    display = _os.environ.get("DISPLAY", "")
    if display:
        result = subprocess.run(
            ["xdpyinfo", "-display", display],
            capture_output=True, timeout=3
        )
        if result.returncode == 0:
            print(f"[Display] 使用已有显示: {display}")
            return True

    # 尝试常见 Display 编号
    for d in [":0", ":1", ":99"]:
        result = subprocess.run(
            ["xdpyinfo", "-display", d],
            capture_output=True, timeout=2
        )
        if result.returncode == 0:
            _os.environ["DISPLAY"] = d
            print(f"[Display] 找到可用显示: {d}")
            return True

    # 尝试在 :99 启动 Xvfb 虚拟帧缓冲
    print("[Display] 未找到 X Display，正在启动 Xvfb 虚拟帧缓冲...")
    try:
        subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1280x1024x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        import time; time.sleep(1.5)
        _os.environ["DISPLAY"] = ":99"
        print("[Display] Xvfb 已启动，使用 :99")
        return True
    except FileNotFoundError:
        print("[Display] ⚠️  Xvfb 未安装，请运行: sudo apt install xvfb")
        print("[Display] 或手动设置: export DISPLAY=:0  然后重新运行脚本")
        return False

# 在导入 habitat_sim 前执行显示初始化
_display_ok = _setup_display()

# ============================================================
# Habitat-sim 导入（延迟，方便在没装的环境下查看代码）
# ============================================================
try:
    import habitat_sim
    from habitat_sim.utils import common as utils
    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False
    print("[警告] habitat-sim 未安装，仅可查看界面，无法运行")


# ============================================================
# 常量配置
# ============================================================
TARGET_COUNT    = 100          # 目标标注数量
IMG_W           = 640          # 渲染分辨率宽
IMG_H           = 480          # 渲染分辨率高
DISPLAY_SCALE   = 1.4          # GUI中图片放大倍数（用于更方便点击）
DISPLAY_W       = int(IMG_W * DISPLAY_SCALE)
DISPLAY_H       = int(IMG_H * DISPLAY_SCALE)
MOVE_STEP       = 0.20         # 前进/后退步长（米）
TURN_STEP       = 8.0          # 左右转角度（度）
STRAFE_STEP     = 0.15         # 左右横移步长（米）
BACKUP_INTERVAL = 5            # 每N条自动备份
RENDER_FPS      = 20           # 导航模式渲染帧率

# 颜色定义
COLOR_GT   = "#00ff88"         # 目标点颜色（绿）
COLOR_XHAIR = "#ff3333"        # 十字准星颜色（红）
BG_DARK    = "#1e1e2e"
BG_MID     = "#313244"
BG_PANEL   = "#181825"
FG_TEXT    = "#cdd6f4"
FG_ACCENT  = "#89b4fa"
FG_GREEN   = "#a6e3a1"
FG_RED     = "#f38ba8"
FG_YELLOW  = "#f9e2af"


# ============================================================
# 数据工具函数
# ============================================================

def find_scene_glb(mp3d_dir: str, scene_id: str) -> Optional[str]:
    """在 mp3d 目录下找到对应场景的 .glb 文件"""
    glb_path = os.path.join(mp3d_dir, scene_id, f"{scene_id}.glb")
    if os.path.exists(glb_path):
        return glb_path
    for p in Path(mp3d_dir).rglob("*.glb"):
        if p.stem == scene_id or p.parent.name == scene_id:
            return str(p)
    return None


def list_scenes(mp3d_dir: str) -> list:
    """列出 mp3d_dir 下所有有效场景（存在 .glb 文件）"""
    scenes = []
    mp3d_path = Path(mp3d_dir)
    if not mp3d_path.exists():
        return scenes
    for scene_dir in sorted(mp3d_path.iterdir()):
        if scene_dir.is_dir():
            glb = scene_dir / f"{scene_dir.name}.glb"
            if glb.exists():
                scenes.append(scene_dir.name)
    return scenes


def load_existing_data(output_path: str) -> list:
    """加载已有标注（断点续标）"""
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[断点续标] 已加载 {len(data)} 条已有标注")
        return data
    return []


def save_data(data: list, output_path: str, backup: bool = False):
    """原子写入JSON，防止崩溃时数据损坏"""
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_path)
    if backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bk = output_path.replace(".json", f"_backup_{ts}.json")
        shutil.copy(output_path, bk)
        print(f"[备份] {bk}")


def save_frame_image(rgb_array: np.ndarray, save_dir: str, record_id: int) -> str:
    """将当前帧RGB图像保存到磁盘，返回保存路径"""
    frames_dir = os.path.join(save_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    img_path = os.path.join(frames_dir, f"frame_{record_id:04d}.jpg")
    Image.fromarray(rgb_array).save(img_path, quality=92)
    return img_path


# ============================================================
# Habitat Worker — 专用线程，持有唯一GL context
# ============================================================
import queue as _queue

class HabitatWorker:
    """
    将所有 habitat-sim 调用封装到一个专用线程中。

    根本原因：habitat-sim 的 OpenGL context 是线程绑定的（thread-local）。
    若在线程A中创建 Simulator，在线程B（主线程/Tkinter）中调用
    sim.step() / get_sensor_observations()，GL context 不存在，
    导致 "GL::Context::current(): no current context" + core dump。

    解决方案：
        - 本类在 __init__ 时启动一个专用线程 _worker_thread
        - 所有 sim 操作（初始化、step、get_obs、传送、关闭）都通过
          命令队列 _cmd_q 发送给该线程执行
        - 执行结果通过结果队列 _res_q 返回给调用方
        - 主线程永远不直接调用 habitat-sim API

    公开接口（线程安全，可从任意线程调用）:
        load_scene(glb_path)  → None（阻塞直到加载完成）
        step(action)          → {"rgb":..., "depth":..., "position":..., "rotation":...}
        get_current_obs()     → 同上
        get_agent_state()     → {"position":..., "rotation":...}
        teleport_random()     → None
        close()               → None
    """

    # 命令类型常量
    _CMD_LOAD    = "load"
    _CMD_STEP    = "step"
    _CMD_OBS     = "obs"
    _CMD_STATE   = "state"
    _CMD_TELE    = "teleport"
    _CMD_CLOSE   = "close"

    def __init__(self):
        self._cmd_q = _queue.Queue()   # 主线程 → worker线程
        self._res_q = _queue.Queue()   # worker线程 → 主线程
        self._sim    = None
        self._agent  = None

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="HabitatWorker"
        )
        self._thread.start()

    # ── 公开接口（线程安全）────────────────────────────────────────

    def load_scene(self, glb_path: str):
        """阻塞加载场景，失败时抛出异常"""
        self._cmd_q.put((self._CMD_LOAD, glb_path))
        ok, val = self._res_q.get()
        if not ok:
            raise RuntimeError(val)

    def step(self, action: str) -> dict:
        """执行动作并返回观测"""
        self._cmd_q.put((self._CMD_STEP, action))
        ok, val = self._res_q.get()
        if not ok:
            raise RuntimeError(val)
        return val

    def get_current_obs(self) -> dict:
        """获取当前帧观测（不执行动作）"""
        self._cmd_q.put((self._CMD_OBS, None))
        ok, val = self._res_q.get()
        if not ok:
            raise RuntimeError(val)
        return val

    def get_agent_state(self) -> dict:
        """获取Agent当前位置和旋转"""
        self._cmd_q.put((self._CMD_STATE, None))
        ok, val = self._res_q.get()
        if not ok:
            raise RuntimeError(val)
        return val

    def teleport_random(self):
        """随机传送"""
        self._cmd_q.put((self._CMD_TELE, None))
        ok, val = self._res_q.get()
        if not ok:
            raise RuntimeError(val)

    def close(self):
        """关闭模拟器并结束worker线程"""
        self._cmd_q.put((self._CMD_CLOSE, None))
        self._thread.join(timeout=5)

    # ── Worker线程主循环 ───────────────────────────────────────────

    def _run(self):
        """
        Worker线程入口。
        循环等待命令队列，在本线程内执行所有habitat-sim操作。
        GL context 在 _do_load() 中创建，此后一直绑定在本线程。
        """
        while True:
            cmd, arg = self._cmd_q.get()
            try:
                if cmd == self._CMD_LOAD:
                    self._do_load(arg)
                elif cmd == self._CMD_STEP:
                    self._res_q.put((True, self._do_step(arg)))
                elif cmd == self._CMD_OBS:
                    self._res_q.put((True, self._do_obs()))
                elif cmd == self._CMD_STATE:
                    self._res_q.put((True, self._do_state()))
                elif cmd == self._CMD_TELE:
                    self._do_teleport()
                    self._res_q.put((True, None))
                elif cmd == self._CMD_CLOSE:
                    if self._sim:
                        self._sim.close()
                    break
            except Exception as e:
                import traceback
                self._res_q.put((False, traceback.format_exc()))

    # ── 实际执行函数（只在worker线程内调用）──────────────────────────

    def _do_load(self, glb_path: str):
        """在worker线程内创建Simulator，GL context绑定到本线程"""
        if self._sim:
            self._sim.close()
            self._sim   = None
            self._agent = None

        rgb_sensor  = self._make_camera_sensor("color_sensor", habitat_sim.SensorType.COLOR)
        depth_sensor= self._make_camera_sensor("depth_sensor", habitat_sim.SensorType.DEPTH)

        agent_cfg = habitat_sim.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]
        agent_cfg.action_space = {
            "move_forward":  habitat_sim.ActionSpec("move_forward",
                                habitat_sim.ActuationSpec(amount=MOVE_STEP)),
            "move_backward": habitat_sim.ActionSpec("move_backward",
                                habitat_sim.ActuationSpec(amount=MOVE_STEP)),
            "turn_left":     habitat_sim.ActionSpec("turn_left",
                                habitat_sim.ActuationSpec(amount=TURN_STEP)),
            "turn_right":    habitat_sim.ActionSpec("turn_right",
                                habitat_sim.ActuationSpec(amount=TURN_STEP)),
            "strafe_left":   habitat_sim.ActionSpec("move_left",
                                habitat_sim.ActuationSpec(amount=STRAFE_STEP)),
            "strafe_right":  habitat_sim.ActionSpec("move_right",
                                habitat_sim.ActuationSpec(amount=STRAFE_STEP)),
        }

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id          = glb_path
        sim_cfg.enable_physics    = False
        sim_cfg.load_semantic_mesh= False
        sim_cfg.create_renderer   = True
        sim_cfg.gpu_device_id     = 0

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self._sim   = habitat_sim.Simulator(cfg)
        self._agent = self._sim.initialize_agent(0)

        # 确保Navmesh可用
        self._ensure_navmesh()
        self._reset_to_valid_pos()
        self._res_q.put((True, None))   # 通知加载完成

    def _do_step(self, action: str) -> dict:
        obs   = self._sim.step(action)
        state = self._agent.get_state()
        return {
            "rgb":      obs["color_sensor"][:, :, :3].copy(),
            "depth":    obs["depth_sensor"].copy(),
            "position": np.array(state.position).tolist(),
            "rotation": [float(state.rotation.x), float(state.rotation.y),
                         float(state.rotation.z), float(state.rotation.w)],
        }

    def _do_obs(self) -> dict:
        obs   = self._sim.get_sensor_observations()
        state = self._agent.get_state()
        return {
            "rgb":      obs["color_sensor"][:, :, :3].copy(),
            "depth":    obs["depth_sensor"].copy(),
            "position": np.array(state.position).tolist(),
            "rotation": [float(state.rotation.x), float(state.rotation.y),
                         float(state.rotation.z), float(state.rotation.w)],
        }

    def _do_state(self) -> dict:
        state = self._agent.get_state()
        return {
            "position": np.array(state.position).tolist(),
            "rotation": [float(state.rotation.x), float(state.rotation.y),
                         float(state.rotation.z), float(state.rotation.w)],
        }

    def _do_teleport(self):
        if self._sim.pathfinder.is_loaded:
            pos   = self._sim.pathfinder.get_random_navigable_point()
            state = self._agent.get_state()
            state.position = pos
            angle = random.uniform(0, 2 * np.pi)
            from habitat_sim.utils import common as _hutils
            state.rotation = _hutils.quat_from_angle_axis(angle, np.array([0,1,0]))
            self._agent.set_state(state)

    # ── 辅助函数（在worker线程内使用）────────────────────────────────

    @staticmethod
    def _make_camera_sensor(uuid: str, sensor_type):
        """兼容新旧版本 habitat-sim 的相机传感器构造"""
        if hasattr(habitat_sim, "CameraSensorSpec"):
            spec             = habitat_sim.CameraSensorSpec()
            spec.uuid        = uuid
            spec.sensor_type = sensor_type
            spec.resolution  = [IMG_H, IMG_W]
            spec.position    = [0.0, 0.5, 0.0]
            spec.hfov        = 90.0
            return spec
        spec             = habitat_sim.SensorSpec()
        spec.uuid        = uuid
        spec.sensor_type = sensor_type
        spec.resolution  = [IMG_H, IMG_W]
        spec.position    = [0.0, 0.5, 0.0]
        spec.parameters["hfov"] = "90"
        return spec

    def _ensure_navmesh(self):
        if self._sim.pathfinder.is_loaded:
            print("  [Navmesh] 已预加载 ✓")
            return
        print("  [Navmesh] 未找到预计算文件，正在重新计算（约10~30秒）...")
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius = 0.18
        settings.agent_height = 1.5
        settings.cell_size    = 0.05
        ok = self._sim.recompute_navmesh(self._sim.pathfinder, settings)
        print("  [Navmesh] 重新计算" + ("成功 ✓" if ok else "失败 ✗"))

    def _reset_to_valid_pos(self):
        if self._sim.pathfinder.is_loaded:
            pos = self._sim.pathfinder.get_random_navigable_point()
            state = self._agent.get_state()
            state.position = pos
            self._agent.set_state(state)
            print(f"  [Agent] 初始位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        else:
            state = self._agent.get_state()
            state.position = [0.0, 1.5, 0.0]
            self._agent.set_state(state)
            print("  [Agent] Navmesh不可用，使用原点位置")


# 保留旧名称兼容性（App代码中用到 HabitatScene）
HabitatScene = HabitatWorker


class HabitatAnnotationApp:
    """
    主标注应用。

    两种模式:
        [导航模式] 键盘控制机器人移动，实时渲染第一视角
        [标注模式] 冻结当前帧，鼠标点击目标像素，输入指令后保存

    界面布局:
        +─────────────────────────────+─────────────────+
        │                             │  场景/进度信息    │
        │    Habitat 第一视角渲染       │  操作状态提示    │
        │       (实时/冻结)            │  指令输入框      │
        │                             │  坐标显示        │
        +─────────────────────────────│  功能按钮        │
        │   底部状态栏                  │                 │
        +─────────────────────────────+─────────────────+
    """

    def __init__(self, root: tk.Tk, mp3d_dir: str,
                 scene_id: Optional[str], output_path: str):
        self.root = root
        self.mp3d_dir = mp3d_dir
        self.output_path = output_path
        output_dir = os.path.dirname(os.path.abspath(output_path))
        self.frames_save_dir = output_dir  # 帧图像保存目录

        # 标注数据
        self.data = load_existing_data(output_path)

        # 场景列表
        self.all_scenes = list_scenes(mp3d_dir)
        if not self.all_scenes:
            messagebox.showerror("错误", f"在 {mp3d_dir} 下未找到任何 .glb 场景文件！")
            root.quit()
            return

        # 初始场景
        self.current_scene_id = scene_id or random.choice(self.all_scenes)

        # Habitat Worker（单一专用线程，持有GL context）
        # 只创建一次，生命周期与App相同
        self.hab: Optional[HabitatWorker] = HabitatWorker() if HABITAT_AVAILABLE else None
        self.hab_ready = False

        # 当前帧状态
        self.current_rgb    = None    # np.ndarray (H,W,3)
        self.frozen_rgb     = None    # 标注模式下冻结的帧
        self.frozen_pil     = None    # 冻结帧的PIL图像
        self.frozen_pil_disp = None  # 放大后用于显示
        self.agent_state_snapshot = None  # 冻结时的Agent状态

        # 点击状态（标注模式）
        self.click_disp_x = None
        self.click_disp_y = None
        self.click_orig_x = None
        self.click_orig_y = None

        # 模式: "nav" / "annotate"
        self.mode = "nav"

        # 键盘状态（持续按下）
        self.keys_pressed = set()

        # 渲染循环控制
        self._nav_loop_id  = None  # tkinter after() ID
        self._render_active = True

        self._build_ui()
        self._start_habitat_async()
        self._nav_render_loop()

    # ─────────────────────────────────────────────────────────
    # UI 构建
    # ─────────────────────────────────────────────────────────

    def _build_ui(self):
        """构建整体界面"""
        self.root.title("Habitat-MP3D 导航标注工具 | VLN System2 数据采集")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        # 顶部标题
        title_f = tk.Frame(self.root, bg=BG_MID, pady=5)
        title_f.pack(fill=tk.X)
        tk.Label(
            title_f,
            text="🤖  Habitat-MP3D 键盘导航标注工具  |  System2 像素目标采集",
            font=("Helvetica", 12, "bold"),
            fg=FG_TEXT, bg=BG_MID
        ).pack()

        # 主体
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # 左侧：渲染画布
        self._build_render_panel(body)

        # 右侧：控制面板
        self._build_control_panel(body)

        # 底部状态栏
        self._build_status_bar()

        # 绑定键盘
        self.root.bind("<KeyPress>",   self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.focus_set()

    def _build_render_panel(self, parent):
        """左侧渲染+标注画布"""
        lf = tk.Frame(parent, bg=BG_PANEL, bd=2, relief=tk.SUNKEN)
        lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # 模式提示条
        self.mode_bar = tk.Label(
            lf,
            text="▶ 导航模式  |  WASD移动  |  点击右侧[标注当前帧]进入标注",
            font=("Helvetica", 9, "bold"),
            fg=FG_GREEN, bg="#1a1a2a", pady=3
        )
        self.mode_bar.pack(fill=tk.X)

        # 主渲染 Canvas
        self.canvas = tk.Canvas(
            lf,
            width=DISPLAY_W, height=DISPLAY_H,
            bg="#000000", cursor="crosshair",
            highlightthickness=0
        )
        self.canvas.pack(padx=4, pady=4)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # 键盘映射提示
        help_text = (
            "W/↑前进  S/↓后退  A左转  D右转  Q左移  E右移  "
            "R随机传送  Tab换场景"
        )
        tk.Label(
            lf, text=help_text,
            font=("Courier", 8),
            fg="#6c7086", bg=BG_PANEL
        ).pack(pady=2)

    def _build_control_panel(self, parent):
        """右侧控制面板"""
        rf = tk.Frame(parent, bg=BG_DARK, width=310)
        rf.pack(side=tk.RIGHT, fill=tk.Y)
        rf.pack_propagate(False)

        # ── 进度 ────────────────────────────────────────────
        pf = tk.LabelFrame(rf, text="📊 标注进度",
                           fg=FG_ACCENT, bg=BG_DARK, font=("Helvetica", 9, "bold"))
        pf.pack(fill=tk.X, pady=(0, 6))

        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(pf, variable=self.progress_var,
                        maximum=TARGET_COUNT, length=280).pack(pady=3, padx=4)
        self.progress_label = tk.Label(
            pf, text=f"0 / {TARGET_COUNT}",
            fg=FG_TEXT, bg=BG_DARK, font=("Helvetica", 11, "bold")
        )
        self.progress_label.pack()

        # ── 场景信息 ────────────────────────────────────────
        sf = tk.LabelFrame(rf, text="🏠 当前场景",
                           fg=FG_ACCENT, bg=BG_DARK, font=("Helvetica", 9, "bold"))
        sf.pack(fill=tk.X, pady=(0, 6))

        self.scene_var = tk.StringVar(value=self.current_scene_id)
        scene_cb = ttk.Combobox(
            sf, textvariable=self.scene_var,
            values=self.all_scenes, width=22, state="readonly"
        )
        scene_cb.pack(padx=4, pady=3)
        scene_cb.bind("<<ComboboxSelected>>", self._on_scene_changed)

        tk.Button(
            sf, text="🔀 随机换场景 (Tab)",
            command=self._switch_scene_random,
            bg="#45475a", fg=FG_TEXT, font=("Helvetica", 8), width=24
        ).pack(padx=4, pady=2)

        tk.Button(
            sf, text="📍 随机传送位置 (R)",
            command=self._teleport_random,
            bg="#45475a", fg=FG_TEXT, font=("Helvetica", 8), width=24
        ).pack(padx=4, pady=2)

        # ── 标注操作 ────────────────────────────────────────
        af = tk.LabelFrame(rf, text="📌 标注操作",
                           fg=FG_ACCENT, bg=BG_DARK, font=("Helvetica", 9, "bold"))
        af.pack(fill=tk.X, pady=(0, 6))

        self.annotate_btn = tk.Button(
            af,
            text="📷 标注当前帧  [Space]",
            command=self._enter_annotate_mode,
            bg="#74c7ec", fg="#1e1e2e",
            font=("Helvetica", 10, "bold"), width=26, height=2
        )
        self.annotate_btn.pack(padx=4, pady=4, fill=tk.X)

        # 坐标显示
        self.coord_label = tk.Label(
            af, text="尚未点击目标像素",
            fg=FG_RED, bg=BG_DARK, font=("Helvetica", 10, "bold")
        )
        self.coord_label.pack(pady=2)

        # 指令输入
        tk.Label(
            af, text="导航指令 (标注模式下输入):",
            fg="#a6adc8", bg=BG_DARK, font=("Helvetica", 8)
        ).pack(anchor=tk.W, padx=4)

        self.instruction_text = tk.Text(
            af, height=5, width=30,
            bg=BG_MID, fg=FG_TEXT,
            font=("Helvetica", 9),
            insertbackground=FG_TEXT,
            wrap=tk.WORD, state=tk.DISABLED
        )
        self.instruction_text.pack(padx=4, pady=3)

        # 确认/取消按钮
        btn_row = tk.Frame(af, bg=BG_DARK)
        btn_row.pack(fill=tk.X, padx=4, pady=2)

        self.confirm_btn = tk.Button(
            btn_row, text="✅ 确认保存 (Enter)",
            command=self._confirm_annotation,
            bg=FG_GREEN, fg="#1e1e2e",
            font=("Helvetica", 9, "bold"), width=16,
            state=tk.DISABLED
        )
        self.confirm_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.cancel_btn = tk.Button(
            btn_row, text="✗ 取消 (Esc)",
            command=self._cancel_annotate,
            bg=FG_RED, fg="#1e1e2e",
            font=("Helvetica", 9), width=10,
            state=tk.DISABLED
        )
        self.cancel_btn.pack(side=tk.LEFT)

        # ── 历史预览 ────────────────────────────────────────
        hf = tk.LabelFrame(rf, text="📜 最近标注",
                           fg=FG_ACCENT, bg=BG_DARK, font=("Helvetica", 9, "bold"))
        hf.pack(fill=tk.X, pady=(0, 6))

        self.history_label = tk.Label(
            hf, text="（暂无）",
            fg="#a6adc8", bg=BG_DARK,
            font=("Helvetica", 8), wraplength=290,
            justify=tk.LEFT
        )
        self.history_label.pack(padx=4, pady=2)

        # ── 撤销/退出 ────────────────────────────────────────
        ef = tk.Frame(rf, bg=BG_DARK)
        ef.pack(fill=tk.X, pady=4, padx=4)

        tk.Button(
            ef, text="↩ 撤销上一条 (Z)",
            command=self._undo_last,
            bg="#f38ba8", fg="#1e1e2e",
            font=("Helvetica", 9), width=20
        ).pack(fill=tk.X, pady=2)

        tk.Button(
            ef, text="🚪 保存并退出",
            command=self._quit,
            bg="#6c7086", fg=FG_TEXT,
            font=("Helvetica", 9), width=20
        ).pack(fill=tk.X, pady=2)

        # 更新进度显示
        self._update_progress_display()

    def _build_status_bar(self):
        """底部状态栏"""
        self.status_var = tk.StringVar(value="正在初始化 Habitat-sim...")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            bg=BG_MID, fg=FG_TEXT,
            font=("Helvetica", 9), anchor=tk.W, padx=8, pady=3
        ).pack(side=tk.BOTTOM, fill=tk.X)

    # ─────────────────────────────────────────────────────────
    # Habitat 异步初始化
    # ─────────────────────────────────────────────────────────

    def _start_habitat_async(self):
        """
        在辅助线程中调用 hab.load_scene()（会阻塞直到场景加载完成）。
        load_scene() 内部把真正的工作发给 HabitatWorker 专用线程执行，
        所以 GL context 始终在 HabitatWorker 线程中，不会跨线程访问。
        这个辅助线程只是为了不卡住 Tkinter 主线程。
        """
        self.status_var.set(f"⏳ 正在加载场景: {self.current_scene_id} ...")
        t = threading.Thread(target=self._load_scene_thread, daemon=True)
        t.start()

    def _load_scene_thread(self):
        """辅助线程：发送加载命令并等待完成，然后通知主线程"""
        glb_path = find_scene_glb(self.mp3d_dir, self.current_scene_id)
        if not glb_path:
            self.root.after(0, lambda: messagebox.showerror(
                "错误", f"找不到场景GLB: {self.current_scene_id}"
            ))
            return
        try:
            # 阻塞等待 worker 线程加载完成（worker线程内部创建GL context）
            self.hab.load_scene(glb_path)
            # 获取第一帧（worker线程执行，主线程安全）
            obs = self.hab.get_current_obs()
            self.current_rgb = np.array(obs["rgb"], dtype=np.uint8)
            self.hab_ready = True
            self.root.after(0, self._on_habitat_ready)
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self.root.after(0, lambda: (
                self.status_var.set(f"❌ 加载失败: {e}"),
                messagebox.showerror("Habitat 加载错误", err)
            ))

    def _on_habitat_ready(self):
        """Habitat加载完成后在主线程更新UI"""
        self.status_var.set(
            f"✅ 场景已就绪: {self.current_scene_id}  |  "
            f"WASD移动，Space标注当前帧"
        )
        self._refresh_canvas_nav()

    # ─────────────────────────────────────────────────────────
    # 渲染循环（导航模式）
    # ─────────────────────────────────────────────────────────

    def _nav_render_loop(self):
        """
        导航模式下的连续渲染循环。
        每帧：检查按键 → 执行动作 → 渲染 → 更新Canvas
        使用 tkinter after() 实现，不阻塞UI线程。
        """
        if not self._render_active:
            return

        if self.mode == "nav" and self.hab_ready:
            action = self._keys_to_action()
            if action:
                # hab.step() 是线程安全的：命令发给worker线程执行
                obs = self.hab.step(action)
                self.current_rgb = np.array(obs["rgb"], dtype=np.uint8)
                self._refresh_canvas_nav()

        # 下一帧（1000/FPS 毫秒后再次调用）
        self._nav_loop_id = self.root.after(
            int(1000 / RENDER_FPS), self._nav_render_loop
        )

    def _keys_to_action(self) -> Optional[str]:
        """
        将当前按下的键集合映射到 Habitat 动作。
        优先级：前进 > 后退 > 左转 > 右转 > 横移
        """
        if "w" in self.keys_pressed or "Up" in self.keys_pressed:
            return "move_forward"
        if "s" in self.keys_pressed or "Down" in self.keys_pressed:
            return "move_backward"
        if "a" in self.keys_pressed or "Left" in self.keys_pressed:
            return "turn_left"
        if "d" in self.keys_pressed or "Right" in self.keys_pressed:
            return "turn_right"
        if "q" in self.keys_pressed:
            return "strafe_left"
        if "e" in self.keys_pressed:
            return "strafe_right"
        return None

    def _refresh_canvas_nav(self):
        """将当前RGB帧显示到Canvas（导航模式，无标注标记）"""
        if self.current_rgb is None:
            return
        self._draw_frame_on_canvas(self.current_rgb, overlay=None)

    def _draw_frame_on_canvas(
        self, rgb: np.ndarray,
        overlay: Optional[Tuple[int, int]] = None
    ):
        """
        将 numpy RGB帧绘制到Canvas。

        参数:
            rgb: (H, W, 3) uint8
            overlay: 若非None，在该显示坐标(x,y)画十字准星
        """
        pil_img = Image.fromarray(rgb)
        # 放大到显示尺寸
        pil_disp = pil_img.resize((DISPLAY_W, DISPLAY_H), Image.BILINEAR)

        if overlay:
            draw = ImageDraw.Draw(pil_disp)
            x, y = overlay
            r = 14
            # 外圆
            draw.ellipse([x-r, y-r, x+r, y+r], outline=COLOR_XHAIR, width=3)
            # 十字
            draw.line([x-r-5, y, x+r+5, y], fill=COLOR_XHAIR, width=3)
            draw.line([x, y-r-5, x, y+r+5], fill=COLOR_XHAIR, width=3)
            # 中心点
            draw.ellipse([x-4, y-4, x+4, y+4], fill=COLOR_XHAIR)
            # 坐标标注文字
            ox = int(x * (IMG_W / DISPLAY_W))
            oy = int(y * (IMG_H / DISPLAY_H))
            draw.text((x + r + 3, y - 8), f"({ox}, {oy})",
                      fill=COLOR_XHAIR)

        self._tk_img = ImageTk.PhotoImage(pil_disp)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

    # ─────────────────────────────────────────────────────────
    # 键盘事件
    # ─────────────────────────────────────────────────────────

    def _on_key_press(self, event):
        key = event.keysym.lower()

        # 导航模式下的按键
        if self.mode == "nav":
            # WASD / 方向键 → 加入按键集合（由渲染循环处理）
            if key in ("w", "s", "a", "d", "q", "e",
                       "up", "down", "left", "right"):
                self.keys_pressed.add(key)

            # Space → 进入标注模式
            elif key == "space":
                self._enter_annotate_mode()

            # R → 随机传送
            elif key == "r":
                self._teleport_random()

            # Tab → 切换场景
            elif key == "tab":
                self._switch_scene_random()
                return "break"

            # Z → 撤销
            elif key == "z":
                self._undo_last()

        # 标注模式下的按键
        elif self.mode == "annotate":
            if key == "return":
                self._confirm_annotation()
            elif key == "escape":
                self._cancel_annotate()

    def _on_key_release(self, event):
        key = event.keysym.lower()
        self.keys_pressed.discard(key)

    # ─────────────────────────────────────────────────────────
    # 标注模式
    # ─────────────────────────────────────────────────────────

    def _enter_annotate_mode(self):
        """
        进入标注模式：
        1. 冻结当前帧（暂停渲染循环更新）
        2. 保存Agent状态快照
        3. 切换UI到标注状态
        """
        if not self.hab_ready or self.current_rgb is None:
            self.status_var.set("⚠️ 场景尚未加载，请稍候")
            return

        # 冻结帧（线程安全：通过队列请求worker线程执行）
        obs = self.hab.get_current_obs()
        self.agent_state_snapshot = self.hab.get_agent_state()

        self.frozen_rgb = obs["rgb"].copy()
        self.mode = "annotate"

        # 重置点击状态
        self.click_disp_x = None
        self.click_disp_y = None
        self.click_orig_x = None
        self.click_orig_y = None
        self.coord_label.config(text="请点击目标像素", fg=FG_YELLOW)

        # 渲染冻结帧（无标记）
        self._draw_frame_on_canvas(self.frozen_rgb, overlay=None)

        # 更新UI状态
        self.mode_bar.config(
            text="📌 标注模式  |  ① 点击目标像素  ② 输入指令  ③ 回车保存  |  Esc取消",
            fg=FG_YELLOW
        )
        self.instruction_text.config(state=tk.NORMAL, bg="#2a2a3e")
        self.instruction_text.delete("1.0", tk.END)
        self.instruction_text.focus_set()
        self.confirm_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.NORMAL)
        self.annotate_btn.config(state=tk.DISABLED)

        self.status_var.set(
            "📌 标注模式 | 点击图片中机器人应前往的目标位置，然后输入导航指令"
        )

    def _cancel_annotate(self):
        """取消标注，返回导航模式"""
        self.mode = "nav"
        self.frozen_rgb = None
        self.click_disp_x = None
        self.click_orig_x = None

        self.mode_bar.config(
            text='▶ 导航模式  |  WASD移动  |  Space/点击[标注当前帧]进入标注',
            fg=FG_GREEN
        )
        self.instruction_text.config(state=tk.DISABLED, bg=BG_MID)
        self.confirm_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.DISABLED)
        self.annotate_btn.config(state=tk.NORMAL)
        self.coord_label.config(text="尚未点击目标像素", fg=FG_RED)
        self.status_var.set("已取消标注，返回导航模式")
        self.root.focus_set()

    def _on_canvas_click(self, event):
        """处理Canvas点击事件（仅标注模式有效）"""
        if self.mode != "annotate" or self.frozen_rgb is None:
            return

        # 检查点击在图片范围内
        if not (0 <= event.x < DISPLAY_W and 0 <= event.y < DISPLAY_H):
            return

        # 记录显示坐标
        self.click_disp_x = event.x
        self.click_disp_y = event.y

        # 转换为原始图片坐标
        self.click_orig_x = int(event.x * (IMG_W / DISPLAY_W))
        self.click_orig_y = int(event.y * (IMG_H / DISPLAY_H))

        # 重绘冻结帧 + 十字准星
        self._draw_frame_on_canvas(
            self.frozen_rgb,
            overlay=(self.click_disp_x, self.click_disp_y)
        )

        # 更新坐标显示
        self.coord_label.config(
            text=f"目标像素: ({self.click_orig_x}, {self.click_orig_y})\n"
                 f"(显示坐标: {self.click_disp_x}, {self.click_disp_y})",
            fg=FG_GREEN
        )
        self.status_var.set(
            f"✅ 已选择像素 ({self.click_orig_x}, {self.click_orig_y})  |  "
            f"现在请输入导航指令，然后回车保存"
        )

    def _confirm_annotation(self):
        """确认并保存当前标注"""
        if self.mode != "annotate":
            return

        # 验证点击
        if self.click_orig_x is None:
            messagebox.showwarning("提示", "请先在图片上点击目标像素位置！")
            return

        # 获取指令
        instruction = self.instruction_text.get("1.0", tk.END).strip()
        if not instruction:
            messagebox.showwarning("提示", "请输入导航指令！\n例如: Walk forward and turn right at the door")
            self.instruction_text.focus_set()
            return

        # 保存帧图像到磁盘
        record_id = len(self.data) + 1
        img_path = save_frame_image(self.frozen_rgb, self.frames_save_dir, record_id)

        # 构造标注记录
        record = {
            "id":          record_id,
            "scene_id":    self.current_scene_id,
            "image_path":  img_path,              # 帧图像保存路径
            "image_width": IMG_W,
            "image_height": IMG_H,
            "instruction": instruction,
            "pixel_goal": {
                "x":     self.click_orig_x,
                "y":     self.click_orig_y,
                # 归一化坐标（0~1）方便跨分辨率评估
                "x_norm": round(self.click_orig_x / IMG_W, 6),
                "y_norm": round(self.click_orig_y / IMG_H, 6),
            },
            "agent_state": self.agent_state_snapshot,  # 位置+旋转，供后续分析
            "timestamp": datetime.now().isoformat(),
        }

        # 加入数据列表并保存
        self.data.append(record)
        is_backup = (len(self.data) % BACKUP_INTERVAL == 0)
        save_data(self.data, self.output_path, backup=is_backup)

        count = len(self.data)
        print(
            f"[{count:3d}/{TARGET_COUNT}] ✓ 保存 | 场景:{self.current_scene_id} | "
            f"像素:({self.click_orig_x},{self.click_orig_y}) | "
            f"图片:{img_path} | 指令:{instruction[:50]}..."
        )

        # 更新历史预览
        self._update_history_preview()
        self._update_progress_display()

        # 检查是否完成
        if count >= TARGET_COUNT:
            messagebox.showinfo(
                "🎉 完成！",
                f"已收集满 {TARGET_COUNT} 条标注！\n"
                f"数据: {self.output_path}\n"
                f"帧图像: {self.frames_save_dir}/frames/"
            )
            self._cancel_annotate()
            return

        # 返回导航模式
        self._cancel_annotate()
        self.status_var.set(
            f"✅ 第 {count} 条已保存！继续导航，Space键进入下一次标注"
        )

    # ─────────────────────────────────────────────────────────
    # 场景切换 / 传送
    # ─────────────────────────────────────────────────────────

    def _on_scene_changed(self, event):
        """下拉菜单选择场景后触发"""
        new_scene = self.scene_var.get()
        if new_scene != self.current_scene_id:
            self._load_new_scene(new_scene)

    def _switch_scene_random(self):
        """随机切换到另一个场景"""
        remaining = [s for s in self.all_scenes if s != self.current_scene_id]
        if not remaining:
            return
        new_scene = random.choice(remaining)
        self.scene_var.set(new_scene)
        self._load_new_scene(new_scene)

    def _load_new_scene(self, scene_id: str):
        """切换并加载新场景（通过worker线程，不重建GL context）"""
        if self.mode == "annotate":
            self._cancel_annotate()
        self.hab_ready = False
        self.current_scene_id = scene_id
        self.current_rgb = None
        self.canvas.delete("all")
        self.canvas.create_text(
            DISPLAY_W // 2, DISPLAY_H // 2,
            text=f"⏳ 加载场景 {scene_id} ...",
            fill="white", font=("Helvetica", 16)
        )
        self.status_var.set(f"⏳ 正在切换场景: {scene_id} ...")
        t = threading.Thread(target=self._load_scene_thread, daemon=True)
        t.start()

    def _teleport_random(self):
        """随机传送到场景中另一位置"""
        if not self.hab_ready:
            return
        self.hab.teleport_random()
        obs = self.hab.get_current_obs()
        self.current_rgb = np.array(obs["rgb"], dtype=np.uint8)
        self._refresh_canvas_nav()
        self.status_var.set("📍 已随机传送到新位置")

    # ─────────────────────────────────────────────────────────
    # 其他操作
    # ─────────────────────────────────────────────────────────

    def _undo_last(self):
        """撤销最后一条标注"""
        if self.mode == "annotate":
            return
        if not self.data:
            messagebox.showinfo("提示", "没有可撤销的标注")
            return
        last = self.data.pop()
        # 删除对应的帧图像文件
        if os.path.exists(last.get("image_path", "")):
            os.remove(last["image_path"])
        save_data(self.data, self.output_path)
        self._update_progress_display()
        self._update_history_preview()
        count = len(self.data)
        self.status_var.set(f"↩ 已撤销第 {count+1} 条 | 当前共 {count} 条")
        print(f"[撤销] 已撤销 #{count+1}")

    def _update_progress_display(self):
        count = len(self.data)
        self.progress_var.set(count)
        self.progress_label.config(text=f"{count} / {TARGET_COUNT}")

    def _update_history_preview(self):
        if not self.data:
            self.history_label.config(text="（暂无）")
            return
        last = self.data[-1]
        pg = last["pixel_goal"]
        text = (
            f"#{last['id']}  场景: {last['scene_id']}\n"
            f"像素: ({pg['x']}, {pg['y']})\n"
            f"指令: {last['instruction'][:60]}..."
        )
        self.history_label.config(text=text)

    def _quit(self):
        """保存并退出"""
        if messagebox.askyesno(
            "退出",
            f"确认退出？\n已标注 {len(self.data)} 条\n数据已自动保存到:\n{self.output_path}"
        ):
            self._render_active = False
            if self._nav_loop_id:
                self.root.after_cancel(self._nav_loop_id)
            save_data(self.data, self.output_path)
            if self.hab:
                self.hab.close()   # 发送CLOSE命令，worker线程自行退出
            print(f"\n[退出] 共标注 {len(self.data)} 条，保存到: {self.output_path}")
            self.root.quit()


# ============================================================
# 无Habitat时的演示模式（用于查看UI布局）
# ============================================================

class DemoAnnotationApp(HabitatAnnotationApp):
    """
    演示模式：habitat-sim不可用时，用纯色随机噪声图替代真实渲染。
    仅用于测试UI布局，不产生真实数据。
    """

    def _start_habitat_async(self):
        self.hab_ready = True
        self.current_rgb = np.random.randint(
            50, 200, (IMG_H, IMG_W, 3), dtype=np.uint8
        )
        self.status_var.set(
            "⚠️ [演示模式] habitat-sim未安装，显示随机噪声图。"
            "安装habitat-sim后可使用真实场景渲染。"
        )
        self.root.after(200, self._refresh_canvas_nav)

    def _load_habitat_worker(self):
        self.current_rgb = np.random.randint(
            50, 200, (IMG_H, IMG_W, 3), dtype=np.uint8
        )
        self.hab_ready = True
        self.root.after(0, self._on_habitat_ready)

    def _keys_to_action(self):
        return None  # 演示模式不执行动作

    def _on_habitat_ready(self):
        self.status_var.set("[演示模式] 界面就绪")
        self._refresh_canvas_nav()


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Habitat-MP3D 键盘导航标注工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 随机选择场景开始标注
  python annotate_habitat.py \\
      --mp3d_dir ./scenes/MatterPort3D/mp3d \\
      --output data.json

  # 指定从某个场景开始
  python annotate_habitat.py \\
      --mp3d_dir ./scenes/MatterPort3D/mp3d \\
      --scene 2azQ1b91cZZ \\
      --output data.json

键盘操作:
  W/S/A/D   前进/后退/左转/右转
  Q/E       左移/右移
  R         随机传送到当前场景的另一位置
  Tab       随机切换到另一个场景
  Space     进入标注模式（冻结当前帧）
  Z         撤销最后一条标注
        """
    )
    parser.add_argument("--mp3d_dir", type=str,
                        default="./scenes/MatterPort3D/mp3d",
                        help="MP3D场景目录（含各场景子目录）")
    parser.add_argument("--scene", type=str, default=None,
                        help="指定初始场景ID（默认随机）")
    parser.add_argument("--output", type=str, default="data.json",
                        help="输出JSON文件路径")
    parser.add_argument("--target", type=int, default=TARGET_COUNT,
                        help=f"目标标注数量（默认{TARGET_COUNT}）")
    parser.add_argument("--demo", action="store_true",
                        help="演示模式（不需要habitat-sim，用随机图像）")
    args = parser.parse_args()

    print("=" * 60)
    print("  Habitat-MP3D 导航标注工具")
    print("  InternVLA-N1 System2 数据采集")
    print("=" * 60)
    print(f"  MP3D目录: {args.mp3d_dir}")
    print(f"  输出文件: {args.output}")
    print(f"  目标数量: {args.target}")
    print("=" * 60)

    # 检查目录
    if not args.demo and not os.path.exists(args.mp3d_dir):
        print(f"[错误] 找不到MP3D目录: {args.mp3d_dir}")
        sys.exit(1)

    # 检查已有标注
    existing = load_existing_data(args.output)
    if len(existing) >= args.target:
        print(f"[提示] 已有 {len(existing)} 条，已达目标！若要重标请删除 {args.output}")
        return

    # 启动GUI
    root = tk.Tk()
    root.geometry(f"{DISPLAY_W + 330}x{DISPLAY_H + 120}")

    if args.demo or not HABITAT_AVAILABLE:
        print("[演示模式] 使用随机图像（不需要habitat-sim）")
        app = DemoAnnotationApp(root, args.mp3d_dir, args.scene, args.output)
    else:
        app = HabitatAnnotationApp(root, args.mp3d_dir, args.scene, args.output)

    root.protocol("WM_DELETE_WINDOW", app._quit)
    root.mainloop()

    print(f"\n[完成] 最终共标注 {len(app.data)} 条")
    print(f"[完成] JSON: {args.output}")
    print(f"[完成] 帧图像: {os.path.dirname(os.path.abspath(args.output))}/frames/")


if __name__ == "__main__":
    main()