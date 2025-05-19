from ppo_agent import PPO_Agent # 假设你有一个 ppo.py 文件定义了 PPO 类
from env import Environment  # 假设你有一个环境类

def main():
    env = Environment()
    agent = PPO()
    agent.train(env)

if __name__ == "__main__":
    main()