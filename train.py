import gymnasium as gym
import torch
import config
from models import Actor, Critic
from buffer import RolloutBuffer
from train import train_ppo
from utils import save_model, visualize_result

from tqdm import tqdm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use the {device}.")

    env = gym.make(config.ENV_NAME, render_mode=None, g=config.GRAVITY)

    actor = Actor(config.STATE_DIM, config.ACTION_DIM,
                  config.HIDDEN_DIM, config.ACTION_BOUND).to(device)
    critic = Critic(config.STATE_DIM, config.HIDDEN_DIM).to(device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.ACTOR_LR)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.CRITIC_LR)

    buffer = RolloutBuffer()

    avg_rewards, avg_actor_losses, avg_critic_losses, avg_advantages = [], [], [], []
    pbar = tqdm(range(config.MAX_UPDATES), desc="Training", dynamic_ncols=True)
    for update in pbar:
        buffer.clear()

        episode_rewards, last_state, last_done = buffer.rollout(env, actor, critic, config, device)
        avg_actor_loss, avg_critic_loss, avg_advantage = train_ppo(buffer, actor, critic, actor_optim, critic_optim, last_state, last_done, config, device)
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        
        avg_rewards.append(avg_reward)
        avg_actor_losses.append(avg_actor_loss)
        avg_critic_losses.append(avg_critic_loss)
        avg_advantages.append(avg_advantage)

        pbar.set_description(f"Update {update+1} | Reward: {avg_reward:.2f}")
        save_model(actor, critic, update, config.MODEL_DIR)
        visualize_result(avg_rewards, avg_actor_losses, avg_critic_losses, avg_advantages, config.RESULT_DIR)

if __name__ == "__main__":
    main()
    