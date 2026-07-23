import numpy as np
import gym
import cv2
import math
import traceback 

class AimBotWrapper(gym.ObservationWrapper):
    def __init__(self, env, img_size=128, camera_name="agentview"):
        super().__init__(env)
        self.img_size = img_size
        self.camera_name = camera_name
        # 這是碗的名稱，如果換場景可能要改
        self.target_site_name = "akita_black_bowl_1_default_site"
        
        # === 1. 獲取 MuJoCo 核心物件 ===
        self.sim = None
        self.model = None
        self.data = None
        
        # 嘗試從各種路徑挖掘 sim/model/data
        if hasattr(env, "sim"): self.sim = env.sim
        if hasattr(env, "model"): self.model = env.model
        if hasattr(env, "data"): self.data = env.data
            
        if self.sim is None:
            try:
                raw_env = env.unwrapped
                if hasattr(raw_env, "sim"): self.sim = raw_env.sim
                if hasattr(raw_env, "model"): self.model = raw_env.model
                if hasattr(raw_env, "data"): self.data = raw_env.data
            except: pass

        # 舊版 MuJoCo 相容
        if self.sim is not None:
            if self.model is None and hasattr(self.sim, "model"): self.model = self.sim.model
            if self.data is None and hasattr(self.sim, "data"): self.data = self.sim.data

        # === 2. 關鍵修正：鎖定真正會動的夾爪部位 ===
        self.gripper_type = None 
        self.gripper_id = None
        
        if self.model:
            print("🔍 [AimBot] 正在搜尋夾爪部位...")
            # 優先搜尋 Site (定位最準)
            possible_sites = ["robot0_eef_pos", "robot0_grip_site", "gripper0_grip_site", "ee_site"]
            for name in possible_sites:
                try:
                    self.gripper_id = self.model.site_name2id(name)
                    self.gripper_type = 'site'
                    print(f"✅ [AimBot] 成功鎖定夾爪 Site: {name}")
                    break
                except: pass
            
            # 其次搜尋 Body (手腕，絕對會動)
            if self.gripper_id is None:
                possible_bodies = ["robot0_link7", "robot0_hand", "hand", "panda_hand"]
                for name in possible_bodies:
                    try:
                        self.gripper_id = self.model.body_name2id(name)
                        self.gripper_type = 'body'
                        print(f"✅ [AimBot] 成功鎖定夾爪 Body: {name}")
                        break
                    except: pass

        if self.model is None or self.data is None:
            print("❌ [AimBot Init Error] 找不到 model 或 data！")
        else:
            self._cache_intrinsics()

    def _cache_intrinsics(self):
        if self.model is None: return
        cam_id = self.model.camera_name2id(self.camera_name)
        fovy = self.model.cam_fovy[cam_id]
        f = 0.5 * self.img_size / np.tan(fovy * np.pi / 360)
        self.K = np.array([[f, 0, self.img_size / 2], 
                           [0, f, self.img_size / 2], 
                           [0, 0, 1]])
        self.R_gl_to_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    def observation(self, obs):
        image_key = f"{self.camera_name}_image"
        if image_key not in obs: return obs
        
        img = obs[image_key]
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if self.model is not None and self.data is not None:
            img_bgr = self._apply_aimbot(img_bgr)

        img_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        obs[image_key] = img_out
        return obs

    def _apply_aimbot(self, img):
        try:
            # 1. 取得目標座標 (碗)
            try:
                target_id = self.model.site_name2id(self.target_site_name)
                target_pos = self.data.site_xpos[target_id]
            except: return img 

            # 2. 取得夾爪座標 (手)
            if self.gripper_type == 'site':
                gripper_pos = self.data.site_xpos[self.gripper_id]
            elif self.gripper_type == 'body':
                gripper_pos = self.data.xpos[self.gripper_id]
            else: return img

            # 3. 投影 3D -> 2D
            cam_id = self.model.camera_name2id(self.camera_name)
            cam_pos = self.data.cam_xpos[cam_id]
            cam_rot = self.data.cam_xmat[cam_id].reshape(3, 3)
            
            R = cam_rot @ self.R_gl_to_cv
            R_inv = R.T
            t_inv = -R_inv @ cam_pos
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = R_inv
            extrinsics[:3, 3] = t_inv

            u_t, v_t = self._project(target_pos, extrinsics)
            u_g, v_g = self._project(gripper_pos, extrinsics)

            # 4. 繪製視覺效果
            # 黃色夾爪點 (抓取點)
            cv2.circle(img, (u_g, v_g), 5, (0, 255, 255), -1) 
            # 紅色射線 (引導線)
            cv2.line(img, (u_g, v_g), (u_t, v_t), (0, 0, 255), 2)
            # 綠色目標點
            cv2.circle(img, (u_t, v_t), 3, (0, 255, 0), -1)
            
            return img
                
        except Exception:
            return img

    def _project(self, point_3d, extrinsics):
        point_4d = np.append(point_3d, 1)
        cam_coords = extrinsics @ point_4d
        z = cam_coords[2]
        if z <= 0: return (-100, -100)
        pixel_coords = self.K @ cam_coords[:3]
        u = int(pixel_coords[0] / z)
        v = int(pixel_coords[1] / z)
        return (u, v)
