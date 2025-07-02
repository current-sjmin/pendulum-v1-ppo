import argparse
import gymnasium as gym

import torch
import torch.nn as nn

import config
from models import Actor

import numpy as np

def main():
    env = gym.make(config.ENV_NAME, render_mode="human", g=9.81)

    loaded_model = Actor()
    loaded_model.load_state_dict(torch.load(args.model))
    loaded_model.eval()

    while True:
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        while not done and not truncated:
            state_tensor = torch.FloatTensor(state)

            mu, std = loaded_model(state_tensor)
            raw_action, squashed_action, log_prob = loaded_model.sample_action(mu, std)


            next_state, reward, done, truncated, info = env.step(action=squashed_action)
            state = next_state
            total_reward += reward
        print(f"[Scenario] Score : {total_reward:.2f}")

    env.close()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        dest="model",
        type=str,
        default="results/models/best_actor.pth",
        help="set model"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    main()