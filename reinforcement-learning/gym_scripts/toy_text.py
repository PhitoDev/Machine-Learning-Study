import gymnasium as gym

def create_environment(name="FrozenLake-v1", render_mode='human',):
    match name:
        case "CliffWalking-v1":
            env = gym.make("CliffWalking-v1", render_mode=render_mode)
        case "Taxi-v3":
            env = gym.make("Taxi-v3", render_mode=render_mode)
        case "FrozenLake-v1":
            env = gym.make("FrozenLake-v1", map_name="8x8", render_mode=render_mode, is_slippery=True)
        case _:
            raise ValueError("Unsupported environment name")

    return env