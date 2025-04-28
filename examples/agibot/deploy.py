import dataclasses
import numpy as np
import sys
sys.path.append("/mnt/nas/wangjunbo/code/pi0/openpi/src")
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi.policies import Agi_policy
import base64
import cv2
import time
import threading

BASE_URL = "http://192.168.100.126:8000"

timer = {"last": time.time(), "cnt": 0}

latest_image = None
def get_latest_image():
    global latest_image
    while True:
        latest_image, timestamp = requests.get(f"{BASE_URL}/image").json()
        # print(f"image delay: {(time.time_ns() - timestamp) / 1e9} s")

latest_joint_states = None
abnormal = False
abnormal_threshold = 30  # 大拇指旋转角度
def get_latest_joint_states():
    global latest_joint_states, abnormal
    while True:
        positions = requests.get(f"{BASE_URL}/joint_states").json()["positions"]
        positions = np.array(positions)
    
        # 夹爪模型
        little_flex = positions[-2]
        
        # 根据大拇指判断异常
        thumb_flex = positions[-6]
        abnormal = thumb_flex < 0.33
        
        # print(f"thumb_flex: {thumb_flex:.2f}, threshold: {(-19.48056 + 10) * np.pi / 180}")

        positions[:7] = 0.
        positions[16:] = 0.
        positions[15] = -0.16546693444252014 + (1.0560191869735718 + 0.16546693444252014) * (little_flex - 3.0519) / (1.7251 - 3.0519)
        
        # if positions[15] < 1.0:
        #     positions[15] = -0.1654
        # print(positions[15])
        
        # 灵巧手模型
        # positions[:7] = 0.0
        # positions[14:22] = 0.0
        # positions = robot2cloud(positions)
        
        latest_joint_states = positions
        
        # joint 35-40 FPS
        # Image 11-13 FPS
        # time.sleep(0.05)
        
@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 6666
    resize_size: int = 224
    replan_steps: int = 5


    seed: int = 7  # Random Seed (for reproducibility)

hand_ranges = [
    # 左手
    [0.0394, 0.6416],    # 大拇指电机轴与旋转轴夹角
    [1.7492, 3.1131],     # 食指第一节与掌平面夹角
    [1.7071, 3.0728],     # 中指第一节与掌平面夹角
    [1.7694, 3.0812],     # 无名指第一节与掌平面夹角
    [1.7251, 3.0519],     # 小指第一节与掌平面夹角
    [0.0, 1.5708],         # 大拇指旋转角度
    # 右手
    [0.0394, 0.6416],    # 大拇指电机轴与旋转轴夹角
    [1.7492, 3.1131],     # 食指第一节与掌平面夹角
    [1.7071, 3.0728],     # 中指第一节与掌平面夹角
    [1.7694, 3.0812],     # 无名指第一节与掌平面夹角
    [1.7251, 3.0519],     # 小指第一节与掌平面夹角
    [0.0, 1.5708],         # 大拇指旋转角度
]


def map_grippers_to_hands(gripper_values):
    """
    将左右夹爪信号映射到关节角度
    """
    angles = []
    
    for idx, (min_angle, max_angle) in enumerate(hand_ranges):
        gripper_value = gripper_values[0] if idx < 6 else gripper_values[1]
        gripper_value = 1 - gripper_value
        angle = min_angle + gripper_value * (max_angle - min_angle)
        angles.append(angle)
    
    return angles

def map_hands_to_grippers(hands_values):
    """
    将左右夹爪信号映射到关节角度
    """
    min_val_left = min_val_right = max_val_left = max_val_right = left_val = right_val = 0
    
    for idx, (min_angle, max_angle) in enumerate(hand_ranges):
        if idx < 6:
            min_val_left += min_angle
            max_val_left += max_angle
            left_val += hands_values[idx]
        else:
            min_val_right += min_angle
            max_val_right += max_angle
            right_val += hands_values[idx]
    
    angles = [10 * (1 - left_val / (max_val_left - min_val_left)), 10 * (1 - right_val / (max_val_right - min_val_right))]
    
    return angles

def little_finger_to_hand(action):
    """
    将小拇指角度映射到夹爪信号
    action: np.ndarray [28]
    """
    min_val = -0.16546693444252014  # 张开
    max_val = 1.0560191869735718  # 闭合
    
    # little_finger_angle = action[15]
    # for idx in range(-6, -1):
    #     finger_min, finger_max = hand_ranges[idx]
    #     action[idx] = finger_min + (finger_min - finger_max) * (little_finger_angle - max_val) / (max_val - min_val)
    
    # action[-1] = np.pi * 2 / 9  # 0.69  # 大拇指旋转角度 1.57
    # print("hand action:", action[-6:])
    
    # [后端] 实现了 move as gripper, 把action[15] 归一化即可 (0是张开 1是闭合)
    action[15] = (action[15] - min_val) / (max_val - min_val)
    # print(action[15])
    
    return action

import numpy as np
import requests
import time


def pred_action(obs):
    """
    obs: dict
    {
        "image":
        {
            "head": base64,
            "hand_left": base64,
            "hand_right": base64,
        },
        "state": np.ndarray [28]
    }
    """
    
    # omit the model inference code
    # action = ...
    # return action
    
    # random action for testing [0~0.001]
    delta = np.random.rand(28) * 0.01
    delta[:14] = 0
    return delta + obs["state"]

def base64_to_uint8_array(base64_str):
    # 解码Base64字符串得到二进制数据
    decoded_data = base64.b64decode(base64_str)
    # 将二进制数据转换为numpy的uint8数组
    return np.frombuffer(decoded_data, dtype=np.uint8)

def robot2cloud(joint_states: np.array):
    """
    云端角度结果为α； 机器人端侧读取角度为θ；单位均为弧度制。
    大拇指弯曲：α = rad( θ * 180.0 / PI - 19.48056)
    四指：α = rad( 180.0 -  θ * 180.0 / PI - 19.48056)
    大拇指旋转：α =  rad(19.48056 -  θ* 180.0 / PI )
    
    joint_states[28]:
    - [-1] 大拇指旋转
    - [-5: -2] 四指
    - [-6] 大拇指弯曲
    """
    # 大拇指旋转
    # joint_states[-1] = np.pi * 19.48056 / 180.0 - joint_states[-1]
    joint_states[-1] = -0.3581
    # 四指
    joint_states[-5:-2] = np.pi * (180.0 - joint_states[-5:-2] * 180.0 / np.pi - 19.48056) / 180.0
    # 大拇指弯曲
    joint_states[-6] = np.pi * (joint_states[-6] * 180.0 / np.pi - 19.48056) / 180.0
    return joint_states
    
    
def cloud2robot(joint_states: np.array):
    """
    云端角度结果为α； 机器人端侧读取角度为θ；单位均为弧度制。
    大拇指弯曲：α = rad( θ * 180.0 / PI - 19.48056)
    四指：α = rad( 180.0 -  θ * 180.0 / PI - 19.48056)
    大拇指旋转：α =  rad(19.48056 -  θ* 180.0 / PI )
    
    joint_states[28]:
    - [-1] 大拇指旋转
    - [-5: -2] 四指
    - [-6] 大拇指弯曲
    """
    # 大拇指旋转
    # joint_states[-1] = np.pi * 19.48056 / 180.0 + joint_states[-1]
    joint_states[-1] = np.pi * 19.48056 / 180.0 + 0.3581
    # 四指
    joint_states[-5:-2] = (np.pi - 19.48056 * np.pi / 180.0) - joint_states[-5:-2]
    # 大拇指弯曲
    joint_states[-6] = joint_states[-6] - np.pi * 19.48056 / 180.0
    return joint_states
    

def get_observation():
    global latest_image, latest_joint_states
    url = f"{BASE_URL}/image"
    # base64_img_dict = requests.get(url).json()
    base64_img_dict = latest_image
    if base64_img_dict is None:
        base64_img_dict, _ = requests.get(url).json()
    # base64 image to uint8 image
    # # 转为numpy图像，自动reshape    

    image_dict = {
        "head": cv2.imdecode(base64_to_uint8_array(base64_img_dict["/camera/head_color"]), cv2.IMREAD_COLOR),
        "hand_left": cv2.imdecode(base64_to_uint8_array(base64_img_dict["/camera/hand_left_fisheye"]), cv2.IMREAD_COLOR),
        "hand_right": cv2.imdecode(base64_to_uint8_array(base64_img_dict["/camera/hand_right_fisheye"]), cv2.IMREAD_COLOR),
    }
    
    # 保存head
    # cv2.imwrite("/mnt/nas/wangjunbo/code/pi0/openpi/head.jpg", image_dict["head"])
    
    url = f"{BASE_URL}/joint_states"
    joint_states = latest_joint_states
    if joint_states is None:
        print("get joint states from sdk")
        joint_states = requests.get(url).json()["positions"]
        joint_states = np.array(joint_states)
    
        # 夹爪模型
        little_flex = joint_states[-2]
        joint_states[:7] = 0.
        joint_states[16:] = 0.
        joint_states[15] = -0.16546693444252014 + (1.0560191869735718 + 0.16546693444252014) * (little_flex - 3.0519) / (1.7251 - 3.0519)
        
        
        # 灵巧手模型
        # joint_states[:7] = 0.0
        # joint_states[14:22] = 0.0
        # joint_states = robot2cloud(joint_states)
        # print("joint states:", joint_states)
    
    obj = "object"  # wooden cube
    
    return {
        "images": image_dict,
        "state": joint_states,
        "prompt": f"pick up the {obj} and place into the wooden tray",
    }

def send_action(action):
    url = f"{BASE_URL}/move"
    action = action.copy()
    # 左臂
    action[:7] = np.array([-1.07619381,  0.60891777,  0.28577867, -1.281371  ,  0.72612941,
        1.49321389, -0.18383574])
    # 左手
    # action[-12:-6] = np.array([0.45722176,3.16921573,3.16921573,3.16921573,3.16921573,0.6981317])
    
    # 右手映射到 控制
    # action = hand_cmd2data(action)
    
    # hand as gripper

    action = action.tolist()
    # requests.post(url, json={"positions": action[:14] + action[16:]}) # hand
    requests.post(url, json={"positions": action[:16]}) # gripper
    
    timer["cnt"] += 1
    if timer["cnt"] % 10 == 0:
        current_time = time.time()
        print(f"send_action fps: {10 / (current_time - timer['last'])}")
        timer["last"] = current_time

def close_loop_control(args):
    global abnormal
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    while True:
        # get observation
        obs = get_observation()
        
        # print("current state:", obs["state"])
        
        # predict action
        action = client.infer(obs)["actions"]
        
        # print("predicted action:", action)
        
        # send action to robot
        _abnormal = abnormal * 5 * 12
        for i in range(12):
            _action = action[i].copy()
            # 当作夹爪
            _action = little_finger_to_hand(_action)
            
            # 强制开夹爪
            if _abnormal:
                print("abnormal:", _abnormal)
                _action[15] = 0.

            # 灵巧手
            # _action = cloud2robot(_action)
            send_action(_action)
        
        # sleep for a while
        # time.sleep(0.5)

if __name__ == "__main__":
    args = Args()
    
    # init_hand_state = np.array([ 0.27086532, -0.16546694, -0.16546694, -0.16546694, -0.16546694,
    #    -0.35813178,  0.27086532, -0.16546694, -0.16546694, -0.16546694,
    #    -0.16546694, -0.35813178])

    # init_hand_state = hand_data2cmd(init_hand_state)
    # print("init hand state:", init_hand_state)
    
    # init_hand_state = hand_cmd2data(init_hand_state)
    # print("init hand state:", init_hand_state)
    
    # 开一个新线程来获取最新图像
    image_thread = threading.Thread(target=get_latest_image)
    image_thread.start()
    # 开一个新线程来获取最新关节角度
    joint_states_thread = threading.Thread(target=get_latest_joint_states)
    joint_states_thread.start()
    # time.sleep(1)
    close_loop_control(args)