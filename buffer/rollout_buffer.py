import torch
import numpy as np


class RolloutBuffer(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.dones),
            np.array(self.log_probs),
            np.array(self.values)
        )
    
    def rollout(self, env, actor, critic, config, device):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        episode_reward = 0.0
        episode_rewards = []
        last_done = False

        for step in range(config.N_STEPS):
            with torch.no_grad():
                mu, std = actor(state)
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = critic(state).squeeze()

            next_state, reward, done, truncated, _ = env.step(action.cpu().numpy())
            self.store(
                state.cpu().numpy(),
                action.cpu().numpy(),
                reward,
                done,
                log_prob.cpu().numpy(),
                value.cpu().numpy()
            )

            episode_reward += reward

            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                next_state, _ = env.reset()
                last_done = True
            else :
                last_done = False
            
            state = torch.tensor(next_state, dtype=torch.float32).to(device)

        if episode_reward > 0:
            episode_rewards.append(episode_reward)
        
        return episode_rewards, state, last_done
            

    def clear(self):
        self.__init__()