import os, glob
import numpy as np
import gym
from gym import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

def get_sim(env):
    curr = env
    while hasattr(curr, "env"):
        if hasattr(curr, "sim"): return curr.sim
        curr = curr.env
    return getattr(curr, "sim", None)

# ==========================================
# 🛡️ 乾淨無射線版 Wrapper (搭載 800步黃金參數)
# ==========================================
class V11_2_NoAimBotWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=800): #  放寬到 800 步
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
        self.initial_bowl_pos = None
        self.prev_dist = None 
        self.prev_z = None
        self.phase = 0 
        self.hover_steps = 0
        self.phase2_steps = 0
        
        self.observation_space = spaces.Box(0, 255, (128, 128, 6), np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)

    def _process_obs(self, obs):
        # 乾淨俐落，只取 RGB 影像，不畫任何線
        a = obs["agentview_image"]
        w = obs["robot0_eye_in_hand_image"]
        if a.shape[0] == 3: a = np.transpose(a, (1, 2, 0))
        if w.shape[0] == 3: w = np.transpose(w, (1, 2, 0))
        raw_agent_rgb = np.flipud(a).astype(np.uint8)
        raw_wrist_rgb = np.flipud(w).astype(np.uint8)
        return np.concatenate([raw_agent_rgb, raw_wrist_rgb], axis=-1)

    def reset(self, **kwargs):
        self.current_step = 0
        self.initial_bowl_pos = None
        self.prev_dist = None  # 確保回合重置時，距離也會重置
        self.prev_z = None
        self.phase = 0 
        self.hover_steps = 0
        self.phase2_steps = 0
        return self._process_obs(self.env.reset(**kwargs))

    def step(self, action):
        scaled_action = action.copy()
        scaled_action[0:3] *= 0.5  
        scaled_action[3:6] = 0.0   #  鎖死手腕旋轉
        scaled_action[6] = -1.0    #  鎖死夾爪 (保持張開)
        
        base_obs, reward, done, info = self.env.step(scaled_action)
        self.current_step += 1
        obs_img = self._process_obs(base_obs)
        
        task_reward = 0.0
        
        try:
            sim = get_sim(self.env)
            if sim is not None:
                # 獲取夾爪與碗的座標
                g_pos = sim.data.site_xpos[sim.model.site_name2id("gripper0_grip_site")].copy()
                bowl_center = sim.data.site_xpos[sim.model.site_name2id("akita_black_bowl_1_default_site")].copy()
                
                if self.initial_bowl_pos is None: 
                    self.initial_bowl_pos = bowl_center.copy()

                # ---------------------------------------------------------
                #  目標設定：懸停在碗正上方 z1 的安全高度
                # ---------------------------------------------------------
                z1 = bowl_center[2] + 0.12  #  懸停點：碗上方 12cm
                target_xy = np.array([bowl_center[0], bowl_center[1] + 0.035]) 
                target_pos = np.array([target_xy[0], target_xy[1], z1])
                
                # 計算當前距離
                curr_dist = np.linalg.norm(g_pos - target_pos)  # 3D 空間距離
                xy_dist = np.linalg.norm(g_pos[:2] - target_xy) # 水平 (XY) 距離
                xy_tolerance = 0.015
                is_above_bowl = xy_dist <= xy_tolerance

                # 初始化 prev_dist (第一步)
                if self.prev_dist is None: self.prev_dist = curr_dist
                if self.prev_z is None: self.prev_z = g_pos[2]
                
                # --- 同步對齊的三階段獎勵邏輯 ---
                # Phase 0：接近目標
                if not is_above_bowl:
                    self.phase = 0
                    self.hover_steps = 0
                    self.phase2_steps = 0
                    smooth_reward = np.exp(-5.0 * curr_dist)
                    task_reward = smooth_reward - 0.8  #  同步對齊：-0.8
                    
                # Phase 1：對準後懸停
                elif is_above_bowl and scaled_action[2] >= -0.02:
                    self.phase = 1
                    self.hover_steps += 1
                    self.phase2_steps = 0
                    hover_penalty = min(self.hover_steps * 0.05, 2.0)
                    task_reward = -hover_penalty
                    if self.hover_steps >= 60:
                        task_reward = -2.0
                        done = True
                        
                    print(f"對準後準備下移")
                        
                # Phase 2：正確下降中
                else: 
                    self.phase = 2
                    self.hover_steps = 0
                    self.phase2_steps += 1
                    z_progress = self.prev_z - g_pos[2] 
                    z_progress_reward = z_progress * 20.0 
                    xy_action_magnitude = abs(scaled_action[0]) + abs(scaled_action[1])
                    steady_bonus = 1.0 - (xy_action_magnitude * 0.5) #  同步對齊：1.0 基礎分
                    task_reward = z_progress_reward + steady_bonus

                    if g_pos[2] <= z1:
                        task_reward = 100.0  #  同步對齊：100分大獎
                        done = True
                        print(f"🎉🎉到z1🎉🎉")

                # 強制偏離懲罰
                if self.phase == 2 and self.phase2_steps >= 5 and xy_dist > xy_tolerance * 3 and not done:
                    task_reward = -1.0  #  同步對齊：-1.0 懲罰
                    done = True
                    print(f"💥💥下降偏移")                    

                self.prev_dist = curr_dist
                self.prev_z = g_pos[2]

        except Exception:
            task_reward = -0.1

        if self.current_step >= self.max_steps: done = True
        return obs_img, float(task_reward), done, info
        
def make_env(seed=42):
    task_name = benchmark.get_benchmark_dict()["libero_spatial"]().get_task_names()[2]
    bddl = glob.glob(os.path.expanduser(f"~/aimbot-rl/**/bddl_files/libero_spatial/{task_name}.bddl"), recursive=True)[0]
    
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, robots=["Panda"], control_freq=20,
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=128, camera_widths=128, 
        camera_depths=False, #  Baseline 不用算 AimBot，關閉 Depth 省 VRAM 加速！
        reward_shaping=False
    )
    env.seed(seed)
    return Monitor(GymV21CompatibilityV0(env=V11_2_NoAimBotWrapper(env, max_steps=800)))

def main():
    # 專屬的 Baseline 檢查點資料夾
    BASE_DIR = "./checkpoints/SB3_V11_2_NoAimBot_Baseline"
    os.makedirs(BASE_DIR, exist_ok=True)
    
    tb_log_dir = "./tensorboard_logs"
    
    train_env = VecTransposeImage(DummyVecEnv([lambda: make_env(seed=42)]))
    eval_env  = VecTransposeImage(DummyVecEnv([lambda: make_env(seed=50)]))
    
    model = SAC("CnnPolicy", train_env, verbose=1, buffer_size=100000, batch_size=256, learning_rate=1e-4, tensorboard_log=tb_log_dir)
    
    checkpoint_callback = CheckpointCallback(save_freq=25000, save_path=BASE_DIR, name_prefix="v11_2_baseline")
    eval_callback = EvalCallback(eval_env, best_model_save_path=BASE_DIR, eval_freq=25000)
    
    print(" 對照組 (Baseline) 啟動：純視覺探索，無 AimBot 射線，搭載 800 步黃金參數！")
    
    model.learn(total_timesteps=150_000, callback=[checkpoint_callback, eval_callback],
                reset_num_timesteps=True, tb_log_name="V11_2_NoAimBot_Baseline")

if __name__ == "__main__":
    main()
