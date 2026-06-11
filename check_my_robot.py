from stable_baselines3.common.env_checker import check_env
from my_robot_env import MyRobotEnv

env = MyRobotEnv()
check_env(env)

print("Environment is valid.")