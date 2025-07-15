import torch
import os
import matplotlib.pyplot as plt

def save_model(actor, critic, num_update, path):
    torch.save(actor.state_dict(), f"{path}/actor_{num_update+1:05}.pth")
    torch.save(critic.state_dict(), f"{path}/critic_{num_update+1:05}.pth")

def load_model(model, path, device='cpu'):
    pass

def visualize_result(episode_rewards, avg_actor_loss, avg_critic_loss, avg_advantage, path):
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.plot(episode_rewards)
    plt.title("Rewards")
    plt.xlabel("Update")
    plt.ylabel("Mean Reward")
    plt.grid(True)


    plt.subplot(1, 4, 2)
    plt.plot(avg_actor_loss)
    plt.title("Actor Losses")
    plt.xlabel("Update")
    plt.ylabel("Mean Loss")
    plt.grid(True)


    plt.subplot(1, 4, 3)
    plt.plot(avg_critic_loss)
    plt.title("Critic Losses")
    plt.xlabel("Update")
    plt.ylabel("Mean Loss")
    plt.grid(True)


    plt.subplot(1, 4, 4)
    plt.plot(avg_advantage)
    plt.title("Advantages")
    plt.xlabel("Update")
    plt.ylabel("Mean Advantage")
    plt.grid(True)

    plt.tight_layout()
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))
    save_path = os.path.join(save_dir, 'training_progress.png')
    plt.savefig(save_path)
    plt.close()
