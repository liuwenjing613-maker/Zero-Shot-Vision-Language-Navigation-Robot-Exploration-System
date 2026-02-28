# ==============================================================================
# 文件名: cloud_s2_server_official.py
# 运行位置: 云端显卡服务器 (有 RTX 3090/4090 的机器)
# 功能: InternVLA-N1 System2 官方推理服务
# 依赖: InternNav 框架 (https://github.com/InternRobotics/InternNav)
# 模型路径: /root/autodl-tmp/.autodl/models/InternVLA-N1-System2
# ==============================================================================
"""
使用方法:
1. 首先确保已安装 InternNav:
   git clone https://github.com/InternRobotics/InternNav
   cd InternNav
   pip install -e .
   git submodule update --init

2. 下载模型:
   huggingface-cli download InternRobotics/InternVLA-N1-System2 --local-dir /root/autodl-tmp/.autodl/models/InternVLA-N1-System2

3. 运行服务:
   python cloud_s2_server_official.py
"""

import os
import sys
import tempfile
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import cv2
import json
import warnings
import gc
from flask import Flask, request, jsonify
from PIL import Image

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ============== 1. 配置 ==============
MODEL_PATH = "/root/autodl-tmp/.autodl/models/InternVLA-N1-System2"
INTERNNAV_PATH = "/root/autodl-tmp/.autodl/InternNav"
DEVICE = "cuda"

# Agent 参数
RESIZE_W, RESIZE_H = 384, 384
NUM_HISTORY = 8
PLAN_STEP_GAP = 4

# 默认相机内参 (Realsense D455)
DEFAULT_INTRINSIC = np.array([
    [386.5, 0.0, 328.9, 0.0],
    [0.0, 386.5, 244.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# ============== 2. 全局状态 ==============
agent = None
step_count = 0
current_instruction = None

# ============== 3. 模型加载 ==============
def load_model():
    """加载 InternVLA-N1 模型"""
    global agent
    
    print("🚀 [云端 S2] 正在加载 InternVLA-N1-System2 模型...")
    print(f"   模型路径: {MODEL_PATH}")
    print(f"   InternNav 路径: {INTERNNAV_PATH}")
    
    # 添加 InternNav 到 path
    sys.path.insert(0, INTERNNAV_PATH)
    sys.path.insert(0, os.path.join(INTERNNAV_PATH, 'third_party', 'diffusion-policy'))
    
    try:
        # 尝试使用官方 Agent
        from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent
        
        class Args:
            def __init__(self):
                self.device = DEVICE
                self.model_path = MODEL_PATH
                self.resize_w = RESIZE_W
                self.resize_h = RESIZE_H
                self.num_history = NUM_HISTORY
                self.camera_intrinsic = DEFAULT_INTRINSIC
                self.plan_step_gap = PLAN_STEP_GAP
        
        args = Args()
        agent = InternVLAN1AsyncAgent(args)
        
        # Warm up
        print("🔥 [云端 S2] 模型预热中...")
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_depth = np.zeros((480, 640), dtype=np.float32) + 2.0
        dummy_pose = np.eye(4)
        agent.reset()
        _ = agent.step(dummy_rgb, dummy_depth, dummy_pose, "hello", intrinsic=args.camera_intrinsic)
        
        print("✅ [云端 S2] InternVLA-N1-System2 Agent 加载完成!")
        return True
        
    except ImportError as e:
        print(f"⚠️ [云端 S2] 无法导入 InternNav: {e}")
        print("   将使用简化模式...")
        return load_simple_model()
    except Exception as e:
        print(f"❌ [云端 S2] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return load_simple_model()


def load_simple_model():
    """
    简化模式：直接使用 Qwen2.5-VL 作为 System2
    不需要完整的 InternNav 框架
    """
    global agent
    
    print("🔄 [云端 S2] 尝试简化模式...")
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        class SimpleS2Agent:
            """
            简化版 System2 Agent
            直接使用 Qwen2.5-VL 进行导航决策
            """
            def __init__(self, model_path, device="cuda"):
                self.device = device
                self.model_path = model_path
                
                print(f"  正在加载模型: {model_path}")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    device_map="auto",
                ).eval()
                self.processor = AutoProcessor.from_pretrained(model_path)
                
                self.history = []
                self.step_idx = 0
                
            def reset(self):
                self.history = []
                self.step_idx = 0
                gc.collect()
                torch.cuda.empty_cache()
            
            def step(self, rgb, depth, pose, instruction, intrinsic=None):
                """
                执行一步 System2 推理
                
                Returns:
                    dict: {
                        'action': 离散动作列表,
                        'trajectory': 轨迹点 (如果有),
                        'pixel_goal': 像素目标 (如果有),
                        'stop': 是否停止
                    }
                """
                self.step_idx += 1
                
                # 保存临时图像
                temp_path = tempfile.mktemp(suffix='.jpg')
                if rgb.shape[-1] == 3:
                    cv2.imwrite(temp_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(temp_path, rgb)
                
                # 调整图像大小
                img = cv2.imread(temp_path)
                if max(img.shape[:2]) > 512:
                    scale = 512 / max(img.shape[:2])
                    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                    img = cv2.resize(img, new_size)
                    cv2.imwrite(temp_path, img)
                h, w = img.shape[:2]
                
                # 构建导航 prompt
                prompt = self._build_prompt(instruction, w, h, depth)
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": temp_path},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                
                text_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                
                del image_inputs, video_inputs
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        use_cache=True,
                    )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True
                )[0]
                
                del inputs, generated_ids, generated_ids_trimmed
                gc.collect()
                torch.cuda.empty_cache()
                
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                print(f"🤖 [S2 Step {self.step_idx}] 输出: {output_text[:100]}...")
                
                return self._parse_output(output_text, w, h)
            
            def _build_prompt(self, instruction, w, h, depth=None):
                """构建导航决策 prompt"""
                front_dist = "unknown"
                if depth is not None:
                    center_depth = depth[depth.shape[0]//2-30:depth.shape[0]//2+30, 
                                        depth.shape[1]//2-30:depth.shape[1]//2+30]
                    valid = center_depth[(center_depth > 0.1) & (center_depth < 10.0)]
                    if len(valid) > 0:
                        front_dist = f"{float(np.median(valid)):.2f}m"
                
                return f"""You are a vision-language navigation agent. Analyze the current view and decide the next action.

INSTRUCTION: {instruction}
IMAGE SIZE: {w}x{h} pixels
FRONT DISTANCE: {front_dist}

AVAILABLE ACTIONS:
- 0 = STOP (target reached, navigation complete)
- 1 = TURN_LEFT_90 (turn left 90 degrees)
- 2 = TURN_LEFT_15 (turn left 15 degrees)  
- 3 = TURN_RIGHT_15 (turn right 15 degrees)
- 4 = TURN_RIGHT_90 (turn right 90 degrees)
- 5 = FORWARD (move forward one step)

OUTPUT FORMAT (use exactly this format):
ACTION: [action_ids]
PIXEL: (u, v) or NONE
STOP: YES or NO

RULES:
- If you clearly see the target object, use STOP: YES
- If target is visible but not reached, output ACTION: [5] with PIXEL coordinates
- If you need to turn to find target, output turning actions like ACTION: [3, 3]
- PIXEL should point to target location or search direction

EXAMPLE OUTPUTS:
- Turning right: ACTION: [3, 3, 3], PIXEL: NONE, STOP: NO
- Moving toward target: ACTION: [5], PIXEL: (320, 240), STOP: NO
- Target reached: ACTION: [0], PIXEL: (300, 200), STOP: YES

Now analyze the image and decide:"""
            
            def _parse_output(self, output_text, w, h):
                """解析模型输出"""
                import re
                
                result = {
                    'action': [5],
                    'trajectory': None,
                    'pixel_goal': None,
                    'stop': False
                }
                
                # 解析 ACTION
                action_match = re.search(r'ACTION:\s*\[([^\]]+)\]', output_text, re.IGNORECASE)
                if action_match:
                    try:
                        actions = [int(x.strip()) for x in action_match.group(1).split(',')]
                        result['action'] = actions
                    except:
                        pass
                
                # 解析 PIXEL
                pixel_match = re.search(r'PIXEL:\s*\((\d+),\s*(\d+)\)', output_text, re.IGNORECASE)
                if pixel_match:
                    try:
                        u, v = int(pixel_match.group(1)), int(pixel_match.group(2))
                        u = max(0, min(u, w - 1))
                        v = max(0, min(v, h - 1))
                        result['pixel_goal'] = [v, u]  # [row, col]
                    except:
                        pass
                
                # 解析 STOP
                if re.search(r'STOP:\s*YES', output_text, re.IGNORECASE):
                    result['stop'] = True
                    if 0 not in result['action']:
                        result['action'] = [0]
                
                return result
        
        agent = SimpleS2Agent(MODEL_PATH, DEVICE)
        
        # Warm up
        print("🔥 [云端 S2] 模型预热中...")
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_depth = np.zeros((480, 640), dtype=np.float32) + 2.0
        dummy_pose = np.eye(4)
        agent.reset()
        _ = agent.step(dummy_rgb, dummy_depth, dummy_pose, "hello", intrinsic=None)
        
        print("✅ [云端 S2] 简化模式 Agent 加载完成!")
        return True
        
    except Exception as e:
        print(f"❌ [云端 S2] 简化模式加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============== 4. API 接口 ==============
@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model': 'InternVLA-N1-System2',
        'step_count': step_count,
        'agent_type': type(agent).__name__ if agent else 'None'
    })


@app.route('/reset', methods=['POST'])
def reset_agent():
    """重置 agent 状态"""
    global step_count, current_instruction
    
    if agent:
        agent.reset()
    step_count = 0
    current_instruction = None
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return jsonify({'status': 'reset', 'message': 'Agent reset successfully'})


@app.route('/step', methods=['POST'])
def step_agent():
    """执行一步 System2 推理"""
    global step_count, current_instruction
    
    try:
        # 获取 RGB 图像
        if 'rgb' not in request.files:
            return jsonify({'error': 'No RGB image provided'}), 400
        
        rgb_file = request.files['rgb']
        rgb_data = np.frombuffer(rgb_file.read(), np.uint8)
        rgb = cv2.imdecode(rgb_data, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # 获取深度图 (可选)
        depth = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32) * 2.0
        if 'depth' in request.files:
            try:
                depth_file = request.files['depth']
                depth_data = np.frombuffer(depth_file.read(), np.float32)
                depth = depth_data.reshape(rgb.shape[:2])
            except:
                pass
        
        # 获取位姿 (可选)
        pose = np.eye(4)
        if 'pose' in request.form:
            try:
                pose = np.array(json.loads(request.form['pose']))
            except:
                pass
        
        # 获取指令
        instruction = request.form.get('instruction', current_instruction or 'navigate to the target')
        if instruction != current_instruction:
            current_instruction = instruction
            if agent:
                agent.reset()
        
        # 获取相机内参 (可选)
        intrinsic = DEFAULT_INTRINSIC
        if 'intrinsic' in request.form:
            try:
                intrinsic = np.array(json.loads(request.form['intrinsic']))
            except:
                pass
        
        # 执行推理
        step_count += 1
        result = agent.step(rgb, depth, pose, instruction, intrinsic=intrinsic)
        
        response = {
            'action': result.get('action', [5]),
            'stop': result.get('stop', False),
            'step_count': step_count
        }
        
        if result.get('trajectory') is not None:
            traj = result['trajectory']
            if isinstance(traj, np.ndarray):
                traj = traj.tolist()
            response['trajectory'] = traj
        
        if result.get('pixel_goal') is not None:
            pg = result['pixel_goal']
            if isinstance(pg, np.ndarray):
                pg = pg.tolist()
            response['pixel_goal'] = pg
        
        print(f"📍 [Step {step_count}] Action: {response['action']}, Stop: {response['stop']}")
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'action': [5], 'stop': False}), 500


@app.route('/plan', methods=['POST'])
def plan_compatible():
    """兼容 v3 的 /plan 接口"""
    global step_count, current_instruction
    
    try:
        # 获取图像
        img_file = request.files.get('rgb') or request.files.get('image')
        if img_file is None:
            return jsonify({'error': 'No image provided'}), 400
        
        img_data = np.frombuffer(img_file.read(), np.uint8)
        rgb = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # 获取指令
        target = request.form.get('target', 'the target')
        instruction = request.form.get('instruction', f'Navigate to {target}')
        
        if instruction != current_instruction:
            current_instruction = instruction
            if agent:
                agent.reset()
        
        depth = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32) * 2.0
        pose = np.eye(4)
        
        step_count += 1
        result = agent.step(rgb, depth, pose, instruction, intrinsic=DEFAULT_INTRINSIC)
        
        action = result.get('action', [5])
        pixel_goal = result.get('pixel_goal')
        stop = result.get('stop', False)
        
        # 转换为 v3 格式
        w, h = rgb.shape[1], rgb.shape[0]
        
        if stop or (action and action[0] == 0):
            status = "Success"
            pixel = (w // 2, h // 2)
        elif pixel_goal is not None:
            status = "Success"
            pixel = (int(pixel_goal[1]), int(pixel_goal[0]))
        else:
            status = "Inferred"
            if action and len(action) > 0:
                first_action = action[0]
                if first_action in [1, 2]:
                    pixel = (w // 4, h // 2)
                elif first_action in [3, 4]:
                    pixel = (3 * w // 4, h // 2)
                else:
                    pixel = (w // 2, h // 2)
            else:
                pixel = (w // 2, h // 2)
        
        response = {
            'pixel': pixel,
            'u': pixel[0],
            'v': pixel[1],
            'status': 'success' if status == "Success" else 'inferred',
            'message': status,
            'confidence': 1.0 if status == "Success" else 0.5,
            'action': action,
            'stop': stop
        }
        
        print(f"📍 [Plan {step_count}] Pixel: {pixel}, Status: {status}, Action: {action}")
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'pixel': None,
            'status': 'error',
            'message': 'SEARCHING',
            'error': str(e)
        }), 500


@app.route('/verify', methods=['POST'])
def verify_target():
    """验证是否到达目标"""
    try:
        img_file = request.files.get('rgb') or request.files.get('image')
        if img_file is None:
            return jsonify({'status': 'error', 'verified': False}), 400
        
        img_data = np.frombuffer(img_file.read(), np.uint8)
        rgb = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        target = request.form.get('target') or request.form.get('instruction', 'target')
        
        depth = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32) * 0.5
        pose = np.eye(4)
        
        result = agent.step(rgb, depth, pose, f"Have you reached {target}?", intrinsic=DEFAULT_INTRINSIC)
        
        verified = result.get('stop', False)
        
        return jsonify({
            'status': 'success' if verified else 'not_reached',
            'verified': verified,
            'action': result.get('action', [])
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'verified': False, 'error': str(e)}), 500


# ============== 5. 启动服务 ==============
if __name__ == '__main__':
    success = load_model()
    
    if not success:
        print("\n❌ 模型加载失败，无法启动服务")
        exit(1)
    
    print("\n" + "=" * 60)
    print("🌟 InternVLA-N1 System2 服务已启动")
    print("=" * 60)
    print(f"模型路径: {MODEL_PATH}")
    print(f"设备: {DEVICE}")
    print(f"Agent 类型: {type(agent).__name__}")
    print("\n可用接口:")
    print("  GET  /health  - 健康检查")
    print("  POST /reset   - 重置 agent")
    print("  POST /step    - 执行一步 S2 推理")
    print("  POST /plan    - 兼容 v3 的推理接口")
    print("  POST /verify  - 验证是否到达目标")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
