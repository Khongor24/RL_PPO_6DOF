import time
from stable_baselines3 import PPO
from my_robot_env import MyRobotEnv

env = MyRobotEnv(render_mode="human")
model = PPO.load("ppo_my_robot")

obs, info = env.reset()

for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
    time.sleep(0.02)  # Slow down for better visualization

env.close()