import torch
import torch.nn as nn

def train_ppo(states, raw_actions, squashed_actions, log_probs, advantages, returns,
              actor, actor_optim, critic, critic_optim, device):
    states_tensor = torch.FloatTensor(states).to(device)
    raw_actions_tensor = torch.FloatTensor(raw_actions).to(device)
    squashed_actions_tensor = torch.FloatTensor(squashed_actions).to(device)
    log_probs_tensor = torch.FloatTensor(log_probs).to(device)
    advantages_tensor = torch.FloatTensor(advantages).to(device)
    returns_tensor = torch.FloatTensor(returns).to(device)

    for _ in range(10):
        mean, std = actor(states_tensor)

        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(raw_actions_tensor).sum(axis=-1)
        ratio = torch.exp(new_log_probs - log_probs_tensor)

        clip_exp = 0.2
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - clip_exp, 1 + clip_exp) * advantages_tensor
        actor_loss = -torch.min(surr1, surr2).mean()

        values = critic(states_tensor)
        critic_loss = nn.MSELoss()(values, returns_tensor)

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()
    return