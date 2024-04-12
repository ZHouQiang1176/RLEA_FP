import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 dual_clip:float=0.1,
                 risk_seeking = None):
        
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.risk_seeking = risk_seeking

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.dual_clip = dual_clip
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        shape_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy,shape_entropy = self.actor_critic.evaluate_actions(obs_batch,actions_batch)
                action_log_probs = action_log_probs.unsqueeze(dim=1)
                if self.risk_seeking:
                    quantile = torch.quantile(adv_targ, self.risk_seeking)
                    mask = adv_targ>=quantile
                    adv_targ = adv_targ[mask]
                    adv_targ = adv_targ - quantile
                    action_log_probs = action_log_probs[mask]
                    old_action_log_probs_batch = old_action_log_probs_batch[mask]
                    
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                if self.dual_clip>1:
                    clip1 = torch.min(surr1,surr2)
                    clip2 = torch.max(clip1,self.dual_clip*adv_targ)
                    action_loss = -torch.where(adv_targ<0,clip2,clip1).mean()
                else:
                    action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                shape_entropy_epoch += shape_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch,shape_entropy_epoch
