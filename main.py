import copy
import os
import time
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from torch.utils.tensorboard import SummaryWriter 
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

class Config:
	def __init__(self, entries: dict={}):
		for k, v in entries.items():
			if isinstance(v, dict):
				self.__dict__[k] = Config(v)
			else:
				self.__dict__[k] = v   

def main(argss):
    gpu_num = 8
    tt = time.strptime(time.ctime())
    time_ = str(tt.tm_year)+'-'+str(tt.tm_mon)+str(tt.tm_mday)+'-'+str(tt.tm_hour)+str(tt.tm_min)+str(tt.tm_sec)
    log_path = './result/tensorboard/'+argss.block_name+'/'+time_
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    print('Write tensorboard log to: '+log_path)
    writer.add_text('args',str(argss.__dict__))
    figure_dir = './result/pictures/'+str(argss.block_name)+'/'+time_+'/train'
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    print('Save pictures to: '+figure_dir)

    gym.envs.register(id='placement-v0',entry_point='fullplace_env:Placememt')
    #envs = gym.make('placement-v0',block=argss.block_name,log_dir=figure_dir)
    envs = SubprocVecEnv([lambda:gym.make('placement-v0',block=argss.block_name,log_dir=figure_dir) for i in range(argss.num_processes)],context='fork')
    next_state ,ori_state = envs.reset(gpu_num)

    torch.manual_seed(argss.seed)
    torch.cuda.manual_seed(argss.seed)
    torch.cuda.manual_seed_all(argss.seed)
    random.seed(argss.seed)
    np.random.seed(argss.seed)

    if argss.cuda and torch.cuda.is_available() and argss.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(argss.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if argss.cuda else "cpu")

    obs_space = (1, 256, 256)
    action_space = gym.spaces.Discrete(256 * 256)
    actor_critic = Policy(
        obs_space,
        action_space,
        base_kwargs={'recurrent': argss.recurrent_policy})

    actor_critic.to(device)

    if argss.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            argss.value_loss_coef,
            argss.entropy_coef,
            lr=argss.lr,
            eps=argss.eps,
            alpha=argss.alpha,
            max_grad_norm=argss.max_grad_norm)
    elif argss.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            argss.clip_param,
            argss.ppo_epoch,
            argss.num_mini_batch,
            argss.value_loss_coef,
            argss.entropy_coef,
            lr=argss.lr,
            eps=argss.eps,
            max_grad_norm=argss.max_grad_norm,
            dual_clip = argss.dual_clip,
            use_clipped_value_loss = argss.use_clipped_value_loss)
    elif argss.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, argss.value_loss_coef, argss.entropy_coef, acktr=True)
    if argss.gail:
        assert len(obs_space) == 1
        discr = gail.Discriminator(
            obs_space[0] + action_space[1], 100,
            device)
        file_name = os.path.join(
            argss.gail_experts_dir, "trajs_{}.pt".format(
                argss.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > argss.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=argss.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(argss.num_steps, argss.num_processes,
                              obs_space, action_space,
                              actor_critic.recurrent_hidden_state_size,
                              ori_state,argss.replay_weight)

    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        argss.num_env_steps) // argss.num_steps // argss.num_processes

    fig = False
    nums = 0
    for j in range(8000):
        t0 = time.time()
        if argss.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if argss.algo == "acktr" else argss.lr)
        re_step = []
        wl_step = []
        for step in range(argss.num_steps):
            nums += argss.num_processes
            # Sample actions
            # n是macro个数
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(next_state)
            t1 = time.time()
            envs.step_async(action.cpu())
            states, state_list, done, reward, _ = envs.step_wait()
            t2 = time.time()
            #print(value)
            masks = torch.where(torch.tensor(done),0.,1.).unsqueeze(1)
            bad_masks = torch.FloatTensor([[1.0]]*argss.num_processes)

            next_state = states
            #print(states['Current_macro_id'])
            if done.all():   
                wl = (10-reward).squeeze()
                reward = np.exp(reward)
                re = reward.squeeze()
                re_step += list(re)
                wl_step += list(wl)
                episode_rewards.append(torch.tensor(reward))
                next_state,state_list = envs.reset(gpu_num)
            
            rollouts.insert(state_list, action, action_log_prob, value, reward, masks, bad_masks)
        re_step.sort()
        wl_step.sort()
        writer.add_scalar('/result/reward',sum(re_step[3:])/len(re_step[3:]),nums)
        writer.add_scalar('/result/wl',sum(wl_step[3:])/len(wl_step[3:]),nums)
        writer.add_scalar('/result/best_wl',wl.min(),nums)
        with torch.no_grad():
            next_value = actor_critic.get_value(next_state).detach()

        if argss.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = argss.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(argss.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], argss.gamma,
                    rollouts.masks[step])

        t3 = time.time()
        print('time consume:',t3-t0)
        rollouts.compute_returns(next_value, argss.use_gae, argss.gamma, argss.gae_lambda, argss.use_proper_time_limits)

        value_loss, action_loss, dist_entropy,shape_entropy = agent.update(rollouts)
        writer.add_scalar('/loss/value_loss',value_loss,j)
        writer.add_scalar('/loss/policy_loss',action_loss,j)
        writer.add_scalar('/loss/act_entropy',dist_entropy,j)
        writer.add_scalar('/loss/shape_entropy',shape_entropy,j)

        #清空buffer
        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % argss.save_interval == 0
                or j == num_updates - 1) and argss.save_dir != "":
            save_path = os.path.join(argss.save_dir, argss.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save([
                actor_critic,
                None
            ], "./trained_models/placement_" + str(j) + ".pt")

        if j % argss.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * argss.num_processes * argss.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), torch.mean(torch.stack(list(episode_rewards))),
                        torch.median(torch.stack(list(episode_rewards))), torch.min(torch.stack(list(episode_rewards))),
                        torch.max(torch.stack(list(episode_rewards))), dist_entropy, value_loss,
                        action_loss))

if __name__ == "__main__":
    import json
    with open('./config.json','r') as f:
        arg = json.load(f)
    argss = Config(arg)
    main(argss)
