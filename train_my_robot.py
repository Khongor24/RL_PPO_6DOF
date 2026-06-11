from stable_baselines3 import PPO
from my_robot_env import MyRobotEnv

env = MyRobotEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.98,
    ent_coef=0.01,
)

model.learn(total_timesteps=300_000)

model.save("ppo_my_robot")

env.close()