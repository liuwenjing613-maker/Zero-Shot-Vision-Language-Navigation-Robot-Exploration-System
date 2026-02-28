# ==============================================================================
# 文件名: cloud_s2_server.py
# 运行位置: 云端显卡服务器 (有 RTX 3090 的机器)
# 功能: InternVLA-N1 System2 + System1 服务端
# 模型路径: /root/autodl-tmp/.autodl/models/InternVLA-N1-System2
# ==============================================================================

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
DEVICE = "cuda"
NUM_HISTORY = 8
RESIZE_W, RESIZE_H = 384, 384

# ============== 2. 全局状态 ==============
agent = None
current_instruction = None
step_count = 0

# ============== 3. 模型加载 ==============
def load_model():
    """加载 InternVLA-N1 模型"""
    global agent
    
    print("🚀 [云端 S2+S1] 正在加载 InternVLA-N1 模型...")
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        # 创建简化的 agent 配置
        class Args:
            def __init__(self):
                self.device = DEVICE
                self.model_path = MODEL_PATH
                self.resize_w = RESIZE_W
                self.resize_h = RESIZE_H
                self.num_history = NUM_HISTORY
                self.camera_intrinsic = np.array([
                    [386.5, 0.0, 328.9, 0.0],
                    [0.0, 386.5, 244.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                self.plan_step_gap = 4
        
        args = Args()
        
        # 尝试导入 InternNav agent
        try:
            sys.path.insert(0, '/root/autodl-tmp/.autodl/InternNav')
            sys.path.insert(0, '/root/autodl-tmp/.autodl/InternNav/third_party/diffusion-policy')
            from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent
            agent = InternVLAN1AsyncAgent(args)
            print("✅ [云端 S2+S1] InternVLA-N1 Agent 加载完成")
        except ImportError as e:
            print(f"⚠️ [云端 S2+S1] 无法导入 InternNav, 尝试简化模式: {e}")
            agent = SimpleInternVLAN1Agent(args)
            print("✅ [云端 S2+S1] 简化模式 Agent 加载完成")
        
        # Warm up
        print("🔥 [云端 S2+S1] 模型预热中...")
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_depth = np.zeros((480, 640), dtype=np.float32)
        dummy_pose = np.eye(4)
        agent.reset()
        agent.step(dummy_rgb, dummy_depth, dummy_pose, "hello", intrinsic=args.camera_intrinsic)
        print("✅ [云端 S2+S1] 模型预热完成，服务已就绪!")
        
    except Exception as e:
        print(f"❌ [云端 S2+S1] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


class SimpleInternVLAN1Agent:
    """
    简化版 InternVLA-N1 Agent
    直接使用 Qwen2.5-VL 模型进行推理，模拟 System2+System1 行为
    """
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.model_path = args.model_path
        self.resize_w = args.resize_w
        self.resize_h = args.resize_h
        self.num_history = args.num_history
        self.intrinsic = args.camera_intrinsic
        
        # 加载模型
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        print(f"  正在从 {self.model_path} 加载模型...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # 历史状态
        self.history_rgb = []
        self.history_depth = []
        self.history_pose = []
        self.current_instruction = None
        self.step_idx = 0
        
    def reset(self):
        """重置 agent 状态"""
        self.history_rgb = []
        self.history_depth = []
        self.history_pose = []
        self.current_instruction = None
        self.step_idx = 0
        gc.collect()
        torch.cuda.empty_cache()
        
    def step(self, rgb, depth, pose, instruction, intrinsic=None):
        """
        执行一步推理
        
        Args:
            rgb: RGB 图像 (H, W, 3), uint8
            depth: 深度图 (H, W), float32, 单位米
            pose: 位姿矩阵 (4, 4)
            instruction: 导航指令
            intrinsic: 相机内参 (4, 4)
            
        Returns:
            dict: {
                'action': 离散动作列表,
                'trajectory': 轨迹点列表 (可选),
                'pixel_goal': 像素目标 (可选),
                'stop': 是否停止
            }
        """
        self.step_idx += 1
        self.current_instruction = instruction
        
        # 更新历史
        self.history_rgb.append(rgb.copy())
        self.history_depth.append(depth.copy())
        self.history_pose.append(pose.copy())
        
        # 保持历史长度
        if len(self.history_rgb) > self.num_history:
            self.history_rgb.pop(0)
            self.history_depth.pop(0)
            self.history_pose.pop(0)
        
        # 预处理图像
        rgb_resized = cv2.resize(rgb, (self.resize_w, self.resize_h))
        
        # 构建推理 prompt
        result = self._infer(rgb_resized, depth, instruction)
        
        return result
    
    def _infer(self, rgb, depth, instruction):
        """执行模型推理"""
        from qwen_vl_utils import process_vision_info
        
        # 保存临时图像
        temp_path = tempfile.mktemp(suffix='.jpg')
        cv2.imwrite(temp_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        # 构建 System2 prompt (导航决策)
        prompt = f"""You are a navigation agent. Given the current view and instruction, decide what action to take.

Instruction: {instruction}

Output format:
ACTION: [action_id, action_id, ...]
PIXEL_GOAL: (u, v) or NONE
STOP: YES or NO

Action IDs:
0 = STOP
1 = TURN_LEFT_LARGE (90°)
2 = TURN_LEFT_SMALL (15°)
3 = TURN_RIGHT_SMALL (15°)
4 = TURN_RIGHT_LARGE (90°)
5 = MOVE_FORWARD

Examples:
- If you need to turn right, output: ACTION: [3, 3, 3]
- If you see the target, output: ACTION: [5], PIXEL_GOAL: (320, 240), STOP: NO
- If you've reached the target, output: ACTION: [0], STOP: YES

Analyze the image and decide the next action:"""

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
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        del inputs, generated_ids, generated_ids_trimmed
        gc.collect()
        torch.cuda.empty_cache()
        
        # 清理临时文件
        try:
            os.remove(temp_path)
        except:
            pass
        
        print(f"🤖 [S2] 原始输出: {output_text}")
        
        # 解析输出
        return self._parse_output(output_text)
    
    def _parse_output(self, output_text):
        """解析模型输出"""
        import re
        
        result = {
            'action': [5],  # 默认前进
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
        
        # 解析 PIXEL_GOAL
        pixel_match = re.search(r'PIXEL_GOAL:\s*\((\d+),\s*(\d+)\)', output_text, re.IGNORECASE)
        if pixel_match:
            try:
                u, v = int(pixel_match.group(1)), int(pixel_match.group(2))
                result['pixel_goal'] = [v, u]  # [row, col] format
            except:
                pass
        
        # 解析 STOP
        if re.search(r'STOP:\s*YES', output_text, re.IGNORECASE):
            result['stop'] = True
            result['action'] = [0]
        
        return result


# ============== 4. API 接口 ==============
@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model': 'InternVLA-N1-System2',
        'step_count': step_count
    })


@app.route('/reset', methods=['POST'])
def reset_agent():
    """重置 agent 状态"""
    global current_instruction, step_count
    
    agent.reset()
    current_instruction = None
    step_count = 0
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return jsonify({'status': 'reset', 'message': 'Agent reset successfully'})


@app.route('/step', methods=['POST'])
def step_agent():
    """执行一步推理"""
    global current_instruction, step_count
    
    try:
        # 获取图像
        if 'rgb' not in request.files:
            return jsonify({'error': 'No RGB image provided'}), 400
        
        rgb_file = request.files['rgb']
        rgb_data = np.frombuffer(rgb_file.read(), np.uint8)
        rgb = cv2.imdecode(rgb_data, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # 获取深度图 (可选)
        depth = None
        if 'depth' in request.files:
            depth_file = request.files['depth']
            depth_data = np.frombuffer(depth_file.read(), np.float32)
            h, w = rgb.shape[:2]
            depth = depth_data.reshape(h, w)
        else:
            depth = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32) * 2.0
        
        # 获取位姿 (可选)
        pose = np.eye(4)
        if 'pose' in request.form:
            try:
                pose = np.array(json.loads(request.form['pose']))
            except:
                pass
        
        # 获取指令
        instruction = request.form.get('instruction', current_instruction or "navigate to the target")
        current_instruction = instruction
        
        # 获取相机内参 (可选)
        intrinsic = None
        if 'intrinsic' in request.form:
            try:
                intrinsic = np.array(json.loads(request.form['intrinsic']))
            except:
                pass
        
        # 执行推理
        step_count += 1
        result = agent.step(rgb, depth, pose, instruction, intrinsic=intrinsic)
        
        # 格式化返回
        response = {
            'action': result.get('action', [5]),
            'stop': result.get('stop', False),
            'step_count': step_count
        }
        
        if result.get('trajectory') is not None:
            response['trajectory'] = result['trajectory']
        
        if result.get('pixel_goal') is not None:
            response['pixel_goal'] = result['pixel_goal']
        
        print(f"📍 [Step {step_count}] Action: {response['action']}, Stop: {response['stop']}")
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/plan', methods=['POST'])
def plan_compatible():
    """
    兼容 v3 的 /plan 接口
    将 System2 的离散动作转换为 v3 期望的格式
    """
    global current_instruction, step_count
    
    try:
        # 获取图像
        if 'rgb' not in request.files:
            return jsonify({'error': 'No image'}), 400
        
        rgb_file = request.files['rgb']
        rgb_data = np.frombuffer(rgb_file.read(), np.uint8)
        rgb = cv2.imdecode(rgb_data, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # 获取指令和其他参数
        target = request.form.get('target', 'the target')
        instruction = request.form.get('instruction', f"Navigate to {target}")
        scan_mode = request.form.get('scan_mode', 'false').lower() == 'true'
        verify_mode = request.form.get('verify', 'false').lower() == 'true'
        
        if instruction != current_instruction:
            current_instruction = instruction
            agent.reset()
            step_count = 0
        
        # 构建深度图和位姿
        depth = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32) * 2.0
        pose = np.eye(4)
        
        # 执行推理
        step_count += 1
        result = agent.step(rgb, depth, pose, instruction, intrinsic=agent.args.camera_intrinsic if hasattr(agent, 'args') else None)
        
        # 转换为 v3 兼容格式
        action = result.get('action', [5])
        pixel_goal = result.get('pixel_goal')
        stop = result.get('stop', False)
        
        # 根据动作类型确定状态
        if stop or (action and action[0] == 0):
            status = "Success"
            pixel = (rgb.shape[1] // 2, rgb.shape[0] // 2)  # 图像中心
        elif pixel_goal is not None:
            status = "Success"
            pixel = (pixel_goal[1], pixel_goal[0])  # 转换为 (u, v)
        else:
            status = "Inferred"
            # 根据动作推断方向
            if action and len(action) > 0:
                first_action = action[0]
                w, h = rgb.shape[1], rgb.shape[0]
                if first_action in [1, 2]:  # 左转
                    pixel = (w // 4, h // 2)
                elif first_action in [3, 4]:  # 右转
                    pixel = (3 * w // 4, h // 2)
                elif first_action == 5:  # 前进
                    pixel = (w // 2, h // 2)
                else:
                    pixel = (w // 2, h // 2)
            else:
                pixel = (rgb.shape[1] // 2, rgb.shape[0] // 2)
        
        response = {
            'pixel': pixel,
            'status': status,
            'confidence': 1.0 if status == "Success" else 0.5,
            'reason': f"Action: {action}",
            'action': action,
            'stop': stop,
            's2_output': result  # 原始 S2 输出供调试
        }
        
        print(f"📍 [Plan {step_count}] Pixel: {pixel}, Status: {status}, Action: {action}")
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'pixel': None,
            'status': 'SEARCHING',
            'confidence': 0.0,
            'error': str(e)
        }), 500


@app.route('/verify', methods=['POST'])
def verify_target():
    """验证是否到达目标"""
    try:
        if 'rgb' not in request.files:
            return jsonify({'verified': False, 'reason': 'No image'}), 400
        
        rgb_file = request.files['rgb']
        rgb_data = np.frombuffer(rgb_file.read(), np.uint8)
        rgb = cv2.imdecode(rgb_data, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        target = request.form.get('target', 'the target')
        
        # 使用模型进行验证
        depth = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32) * 0.5
        pose = np.eye(4)
        
        result = agent.step(rgb, depth, pose, f"Have I reached {target}?", intrinsic=None)
        
        verified = result.get('stop', False)
        
        return jsonify({
            'verified': verified,
            'action': result.get('action', []),
            'reason': 'Target reached' if verified else 'Not yet'
        })
        
    except Exception as e:
        return jsonify({'verified': False, 'error': str(e)}), 500


# ============== 5. 启动服务 ==============
if __name__ == '__main__':
    load_model()
    print("\n" + "=" * 60)
    print("🌟 InternVLA-N1 System2+System1 服务已启动")
    print("=" * 60)
    print(f"模型路径: {MODEL_PATH}")
    print(f"设备: {DEVICE}")
    print("可用接口:")
    print("  GET  /health  - 健康检查")
    print("  POST /reset   - 重置 agent")
    print("  POST /step    - 执行一步推理")
    print("  POST /plan    - 兼容 v3 的推理接口")
    print("  POST /verify  - 验证是否到达目标")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
