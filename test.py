import argparse
import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn

import config
from models import Actor
from utils import load_model


def main(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use the {device}.")

    env = gym.make(config.ENV_NAME, render_mode="human", g=config.GRAVITY)

    loaded_model = Actor(config.STATE_DIM, config.ACTION_DIM,
                         config.HIDDEN_DIM, config.ACTION_BOUND).to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()

    while True:
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        while not done and not truncated:
            state_tensor = torch.FloatTensor(state).to(device)

            mu, std = loaded_model(state_tensor)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()

            next_state, reward, done, truncated, info = env.step(action.cpu().numpy())
            state = next_state
            total_reward += reward
        print(f"[Scenario] Score : {total_reward:.2f}")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--episode",
        dest="episode",
        type=int,
        default=1,
        help="set episode number"
    )

    parser.add_argument(
        "-bm", "--best-model",
        dest="best_model",
        action="store_true",
        help="use best model"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    if args.best_model:
        model_path = f"results/models/best_actor.pth"
    else:    
        model_path = f"results/models/actor_{args.episode:05d}.pth"
    main(model_path)