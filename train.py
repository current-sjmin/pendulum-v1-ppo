import gymnasium as gym
import torch, math
import config

from models import Actor, Critic
from buffer import RolloutBuffer
from utils import compute_gae
from train import train_ppo


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use the {device}.")

    env = gym.make(config.ENV_NAME, render_mode=None, g=9.81)

    actor = Actor().to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)

    critic = Critic().to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    rollout_buffer = RolloutBuffer()

    best_reward = float(-math.inf)

    episode_on = True
    curr_episode = 0
    while episode_on:
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0

        for t in range(config.T_HORIZON):
            state_tensor = torch.FloatTensor(state).to(device)

            # Actor
            mu, std = actor(state_tensor)
            raw_action, squashed_action, log_prob = actor.sample_action(mu, std)

            # Critic
            value_tensor = critic(state_tensor)            
            value = critic.get_value(value_tensor)

            next_state, reward, done, truncated, info = env.step(squashed_action)
            episode_done = done or truncated

            # Stack Buffer
            rollout_buffer.add(state, raw_action, squashed_action, reward,
                               episode_done, value, log_prob)
            
            state = next_state
            total_reward += reward

            if done or truncated:
                state, _ = env.reset()
                curr_episode += 1
                print(f"[EPISODE {curr_episode}/{config.EPISODES}] Total Reward : {total_reward}")
                
                if best_reward < total_reward:
                    best_reward = total_reward
                    torch.save(actor.state_dict(), f"results/models/best_actor_{curr_episode:05}.pth")
                    torch.save(actor.state_dict(), f"results/models/best_critic_{curr_episode:05}.pth")
                
                total_reward = 0

                if curr_episode == config.EPISODES:
                    episode_on = False
                    break
        print(f"[INFO] Buffer is stacked.")

        # Advangates, Returns
        advantages, returns = compute_gae(
            rewards = rollout_buffer.rewards,
            dones   = rollout_buffer.dones,
            values  = rollout_buffer.values,
            gamma   = config.GAMMA,
            lam     = config.LAMBDA
        )

        # Get Train Data in Buffer
        train_data = rollout_buffer.get()
        states = train_data["states"]
        raw_actions = train_data["raw_actions"]
        squashed_actions = train_data["squashed_actions"]
        log_probs = train_data["log_probs"]


        # Update the PPO
        train_ppo(states, raw_actions, squashed_actions, log_probs, advantages, returns,
                  actor, actor_optim, critic, critic_optim, device)
        rollout_buffer.clear()


if __name__ == "__main__":
    main()
    