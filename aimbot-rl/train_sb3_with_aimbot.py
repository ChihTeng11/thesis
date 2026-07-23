import os, glob
import numpy as np
import gym
from gym import spaces
from copy import deepcopy
import cv2

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

# ====== AimBot 所需模組 ======
from crosshair.reticle_builder import ReticleBuilder
from crosshair.config import CONFIG_DICT
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map


def get_sim(env):
    curr = env
    while hasattr(curr, "env"):
        if hasattr(curr, "sim"): return curr.sim
        curr = curr.env
    return getattr(curr, "sim", None)

def is_open(gripper_qpos):
    """判斷夾爪是否為張開狀態"""
    return abs(gripper_qpos[0]) > 0.035 and abs(gripper_qpos[1]) > 0.035

# ==========================================
#  AimBot 強化版 Wrapper (混合視覺+3D Reward)
#  方向 A：Reward 計算用 wrist 視角，與 policy 看到的紅點一致
# ==========================================
class V11_2_AimBotWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=800, resolution=128):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
        self.initial_bowl_pos = None
        self.prev_dist = None
        self.prev_z = None
        self.phase = 0
        self.hover_steps = 0
        self.resolution = resolution
        self.phase2_steps = 0

        # --- AimBot 初始化 ---
        config = deepcopy(CONFIG_DICT["large_dot_dynamic_default_color"])
        self.MAX_EE_TABLE_DIST = 0.4
        self.FIXCAM_TOLERANCE = 18
        self.WSTCAM_TOLERANCE = 8

        if hasattr(config["scope_reticle"], "line_length_cfg"):
            config["scope_reticle"].line_length_cfg.maxdist = self.MAX_EE_TABLE_DIST
        if hasattr(config["scope_reticle"], "circle_radius_cfg"):
            config["scope_reticle"].circle_radius_cfg.maxdist = self.MAX_EE_TABLE_DIST

        self.reticle_builder = ReticleBuilder(
            shooting_line_config=config["shooting_line"],
            scope_reticle_config=config["scope_reticle"]
        )

        self.crosshair_fix_uv = None
        self.crosshair_wst_uv = None    # 🌟 wrist 準星座標（reward 用這個）
        self.agent_ext = None
        self.agent_int = None
        self.wrist_ext = None
        self.wrist_int = None

        self.observation_space = spaces.Box(0, 255, (resolution, resolution, 6), np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)

    def _get_pixel_alignment_reward(self, target_pos):
        """
        🌟 方向 A：用 wrist 視角計算 reward，與 policy 看到的紅點視覺完全一致。
        - wrist 準星座標：來自 render_on_wst_camera 的回傳值
        - 紅點目標座標：把 target_pos 投影到 wrist 相機
        """
        if (self.crosshair_wst_uv is None
            or self.wrist_ext is None
            or self.wrist_int is None):
            return 0.0
        try:
            u_target, v_target, z_target = ReticleBuilder._wld2img(
                target_pos, self.wrist_ext, self.wrist_int
            )
            if z_target <= 0:
                return 0.0
            u_c, v_c = self.crosshair_wst_uv
            pixel_dist = np.sqrt((u_c - u_target) ** 2 + (v_c - v_target) ** 2)

            # wrist 視角解析度更細，半衰距離調為 20px
            # pixel_dist=0  → reward=1.0
            # pixel_dist=20 → reward≈0.37
            # pixel_dist=60 → reward≈0.05
            return float(np.exp(-pixel_dist / 20.0))
        except Exception:
            return 0.0

    def _draw_wrist_marker(self, wrist_img, target_pos, wrist_ext, wrist_int):
        """把目標點（含 XY 偏移）投影到 wrist 相機，畫一個小實心紅點（radius=2）。"""
        try:
            u, v, z = ReticleBuilder._wld2img(target_pos, wrist_ext, wrist_int)
            if z > 0 and 0 <= u < self.resolution and 0 <= v < self.resolution:
                cv2.circle(wrist_img, (u, v), radius=2, color=(255, 0, 0), thickness=-1)
        except Exception:
            pass
        return wrist_img

    def _process_obs(self, obs, target_pos=None):
        sim = get_sim(self.env)

        a_rgb = obs["agentview_image"]
        w_rgb = obs["robot0_eye_in_hand_image"]
        if a_rgb.shape[0] == 3: a_rgb = np.transpose(a_rgb, (1, 2, 0))
        if w_rgb.shape[0] == 3: w_rgb = np.transpose(w_rgb, (1, 2, 0))

        raw_agent_rgb = np.flipud(a_rgb).astype(np.uint8)
        raw_wrist_rgb = np.flipud(w_rgb).astype(np.uint8)

        gripper_pos = deepcopy(obs["robot0_eef_pos"])
        gripper_quat = deepcopy(obs["robot0_eef_quat"])
        gripper_is_open = is_open(obs["robot0_gripper_qpos"])

        # agentview：只畫準星（保留座標但不用於 reward）
        agent_depth = get_real_depth_map(sim, np.flipud(obs["agentview_depth"]).squeeze())
        agent_ext = np.linalg.inv(get_camera_extrinsic_matrix(sim, "agentview"))
        agent_int = get_camera_intrinsic_matrix(sim, "agentview", self.resolution, self.resolution)
        self.agent_ext = agent_ext
        self.agent_int = agent_int

        result = self.reticle_builder.render_on_fix_camera(
            camera_rgb=raw_agent_rgb.copy(),
            camera_depth=agent_depth,
            camera_extrinsics=agent_ext,
            camera_intrinsics=agent_int,
            gripper_pos=gripper_pos, gripper_quat=gripper_quat,
            gripper_open=gripper_is_open,
            image_height=self.resolution, image_width=self.resolution,
            tolerance=self.FIXCAM_TOLERANCE
        )
        if isinstance(result, tuple):
            aimbot_agent_img, self.crosshair_fix_uv = result
        else:
            aimbot_agent_img = result

        # wrist：準星 + 目標點紅點
        wrist_depth = get_real_depth_map(sim, np.flipud(obs["robot0_eye_in_hand_depth"]).squeeze())
        wrist_ext = np.linalg.inv(get_camera_extrinsic_matrix(sim, "robot0_eye_in_hand"))
        wrist_int = get_camera_intrinsic_matrix(sim, "robot0_eye_in_hand", self.resolution, self.resolution)
        self.wrist_ext = wrist_ext
        self.wrist_int = wrist_int

        result_w = self.reticle_builder.render_on_wst_camera(
            wrist_camera_rgb=raw_wrist_rgb.copy(),
            wrist_camera_depth=wrist_depth,
            wrist_camera_extrinsics=wrist_ext,
            wrist_camera_intrinsics=wrist_int,
            gripper_pos=gripper_pos, gripper_quat=gripper_quat,
            gripper_open=gripper_is_open,
            image_height=self.resolution, image_width=self.resolution,
            tolerance=self.WSTCAM_TOLERANCE
        )
        if isinstance(result_w, tuple):
            aimbot_wrist_img, self.crosshair_wst_uv = result_w  # 🌟 取得 wrist 準星座標
        else:
            aimbot_wrist_img = result_w

        # 在 wrist 畫紅點
        if target_pos is not None:
            aimbot_wrist_img = self._draw_wrist_marker(aimbot_wrist_img, target_pos, wrist_ext, wrist_int)

        return np.concatenate([aimbot_agent_img, aimbot_wrist_img], axis=-1)

    def reset(self, **kwargs):
        self.current_step = 0
        self.initial_bowl_pos = None
        self.prev_dist = None
        self.prev_z = None
        self.phase = 0
        self.hover_steps = 0
        self.phase2_steps = 0
        self.crosshair_fix_uv = None
        self.crosshair_wst_uv = None
        self.agent_ext = None
        self.agent_int = None
        self.wrist_ext = None
        self.wrist_int = None
        return self._process_obs(self.env.reset(**kwargs))

    def step(self, action):
        scaled_action = action.copy()
        scaled_action[0:3] *= 0.5
        scaled_action[3:6] = 0.0
        scaled_action[6] = -1.0

        base_obs, reward, done, info = self.env.step(scaled_action)
        self.current_step += 1

        task_reward = 0.0
        target_pos = None

        try:
            sim = get_sim(self.env)
            if sim is not None:
                g_pos = sim.data.site_xpos[sim.model.site_name2id("gripper0_grip_site")].copy()
                bowl_center = sim.data.site_xpos[sim.model.site_name2id("akita_black_bowl_1_default_site")].copy()

                if self.initial_bowl_pos is None:
                    self.initial_bowl_pos = bowl_center.copy()

                z1 = bowl_center[2] + 0.12
                target_xy = np.array([bowl_center[0], bowl_center[1] + 0.035])
                target_pos = np.array([target_xy[0], target_xy[1], z1])

                curr_dist = np.linalg.norm(g_pos - target_pos)
                xy_dist = np.linalg.norm(g_pos[:2] - target_xy)
                xy_tolerance = 0.015
                is_above_bowl = xy_dist <= xy_tolerance

                if self.prev_dist is None: self.prev_dist = curr_dist
                if self.prev_z is None: self.prev_z = g_pos[2]

                obs_img = self._process_obs(base_obs, target_pos=target_pos)

                # Phase 0：接近目標
                if not is_above_bowl:
                    self.phase = 0
                    self.hover_steps = 0
                    self.phase2_steps = 0
                    smooth_reward = np.exp(-5.0 * curr_dist)
                    base_reward = smooth_reward - 0.8
                    pixel_reward = self._get_pixel_alignment_reward(target_pos)
                    task_reward = base_reward * 0.6 + pixel_reward * 0.4

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
                    steady_bonus = 1.0 - (xy_action_magnitude * 0.5)
                    task_reward = z_progress_reward + steady_bonus

                    if g_pos[2] <= z1:
                        task_reward = 100.0
                        done = True
                        print(f"🎉🎉到z1🎉🎉")

                # 強制偏離懲罰
                if self.phase == 2 and self.phase2_steps >= 5 and xy_dist > xy_tolerance * 3 and not done:
                    task_reward = -1.0
                    done = True
                    print(f"💥💥下降偏移")

                self.prev_dist = curr_dist
                self.prev_z = g_pos[2]

        except Exception:
            task_reward = -0.1
            obs_img = self._process_obs(base_obs, target_pos=None)

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
        camera_depths=True,
        reward_shaping=False
    )
    env.seed(seed)
    return Monitor(GymV21CompatibilityV0(env=V11_2_AimBotWrapper(env, max_steps=800, resolution=128)))


def main():
    BASE_DIR = "./checkpoints/SB3_V11_5_AimBot_WristReward"
    os.makedirs(BASE_DIR, exist_ok=True)

    tb_log_dir = "./tensorboard_logs"

    train_env = VecTransposeImage(DummyVecEnv([lambda: make_env(seed=42)]))
    eval_env  = VecTransposeImage(DummyVecEnv([lambda: make_env(seed=50)]))

    model = SAC("CnnPolicy", train_env, verbose=1, buffer_size=100000, batch_size=256, learning_rate=1e-4, tensorboard_log=tb_log_dir)

    checkpoint_callback = CheckpointCallback(save_freq=25000, save_path=BASE_DIR, name_prefix="v11_5_aimbot_wrist")
    eval_callback = EvalCallback(eval_env, best_model_save_path=BASE_DIR, eval_freq=25000)

    print("🎯 AimBot 方向A：Phase0 = 3D距離(60%) + Wrist 準星×紅點像素對齊(40%)")

    model.learn(total_timesteps=150_000, callback=[checkpoint_callback, eval_callback],
                reset_num_timesteps=True, tb_log_name="V11_5_AimBot_WristReward")


if __name__ == "__main__":
    main()
