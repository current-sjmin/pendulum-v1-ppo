import torch
import torch.nn as nn
import torch.nn.functional as F


def train_ppo(buffer, actor, critic, actor_optim, critic_optim, last_state, last_done, config, device):
    actor_losses = []
    critic_losses = []
    advantages_list = []

    # 🔷 버퍼에서 데이터 꺼내기
    states, actions, rewards, dones, log_probs_old, values = buffer.get()
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).to(device)
    values = torch.tensor(values, dtype=torch.float32).to(device)

    # 🔷 마지막 상태 평가
    with torch.no_grad():
        if last_done:
            next_value = torch.tensor(0.0).to(device)
        else:
            next_value = critic(last_state.unsqueeze(0)).squeeze(0)

    # 🔷 GAE + Return 계산
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_v = next_value
        else:
            next_v = values[i+1]
        delta = rewards[i] + config.GAMMA * next_v * (1 - dones[i]) - values[i]
        gae = delta + config.GAMMA * config.LAMBDA * (1 - dones[i]) * gae
        returns.insert(0, (gae + values[i]).view(()))


    returns = torch.stack(returns).detach()
    advantages = returns - values

    # 🔷 어드벤티지 정규화
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 🔷 학습 (여러 epoch + mini-batch)
    dataset_size = len(states)
    for epoch in range(config.PPO_EPOCHS):
        perm = torch.randperm(dataset_size)
        for i in range(0, dataset_size, config.BATCH_SIZE):
            idx = perm[i:i+config.BATCH_SIZE]

            s_batch = states[idx]
            a_batch = actions[idx]
            logp_old_batch = log_probs_old[idx]
            adv_batch = advantages[idx]
            ret_batch = returns[idx]

            # Actor
            mu, std = actor(s_batch)
            dist = torch.distributions.Normal(mu, std)
            logp = dist.log_prob(a_batch).sum(-1)
            ratio = torch.exp(logp - logp_old_batch)

            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - config.CLIP_EPS, 1 + config.CLIP_EPS) * adv_batch

            # Entropy 보너스 -> 탐험을 위해서 추가 높을수록 더욱 탐험 (0.001 ~ 0.1)
            entropy = dist.entropy().sum(-1).mean()
            actor_loss = -torch.min(surr1, surr2).mean() - config.ENTROPY_COEF * entropy

            # Critic
            v_pred = critic(s_batch).squeeze(-1)
            critic_loss = F.mse_loss(v_pred, ret_batch)

            # 합치기
            loss = actor_loss + config.VALUE_COEF * critic_loss

            # 업데이트
            actor_optim.zero_grad()
            critic_optim.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            ###################
            
            actor_optim.step()
            critic_optim.step()

            # 로그 기록
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            advantages_list.append(adv_batch.mean().item())

    avg_actor_loss = sum(actor_losses) / len(actor_losses)
    avg_critic_loss = sum(critic_losses) / len(critic_losses)
    avg_advantage = sum(advantages_list) / len(advantages_list)
    return avg_actor_loss, avg_critic_loss, avg_advantage