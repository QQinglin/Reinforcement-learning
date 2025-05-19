import time

from gridworld import GridWorldEnv


if __name__ == "__main__":
    # Create the environment
    env = GridWorldEnv()

    # Reset
    obs = env.reset()
    env.render()

    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        # Uncomment this to enable slow motion mode
        # time.sleep(1.0)
        if terminated:
            env.reset()
            env.render()
    env.close()
