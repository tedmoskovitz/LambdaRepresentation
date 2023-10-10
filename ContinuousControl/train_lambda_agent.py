import random
from argparse import ArgumentParser

import gym
from gym.wrappers import RescaleAction
import numpy as np
import torch

from lambda_sac_1 import LambdaSAC_Agent
from utils import MeanStdevFilter, Transition, make_gif, make_checkpoint
import wandb


def train_agent_model_free(agent, env, params):
    
    update_timestep = params['update_every_n_steps']
    seed = params['seed']
    log_interval = 1000
    gif_interval = 1000000
    n_random_actions = params['n_random_actions']
    n_evals = params['n_evals']
    n_collect_steps = params['n_collect_steps']
    use_statefilter = params['obs_filter']
    save_model = params['save_model']
    total_steps = params['total_steps']

    assert n_collect_steps > agent.batchsize, "We must initially collect as many steps as the batch size!"

    avg_length = 0
    time_step = 0
    cumulative_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    i_episode = 0
    log_episode = 0
    samples_number = 0
    episode_rewards = []
    episode_steps = []

    if use_statefilter:
        state_filter = MeanStdevFilter(env.env.observation_space.shape[0])
    else:
        state_filter = None

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    max_steps = env.spec.max_episode_steps #env._max_episode_steps   #env.spec.max_episode_step

    if params['use_wandb']:
        wandb.login()
        wandb.init(project="lambda-sac", name=params['experiment_name'], config=params)

    prev_episode_reward = 0
    while samples_number < total_steps:
        time_step = 0
        episode_reward = 0
        i_episode += 1
        log_episode += 1
        state = env.reset()
        if state_filter:
            state_filter.update(state)
        done = False
        # sample a value of lambda for this episode
        lambda_ = params['lambda_'] if params['lambda_'] >= 0 else agent.TDC.sample()

        while (not done):
            cumulative_log_timestep += 1
            cumulative_timestep += 1
            time_step += 1
            samples_number += 1
            if samples_number < n_random_actions:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, state_filter=state_filter)
            nextstate, reward, done, _ = env.step(action)
            # if we hit the time-limit, it's not a 'real' done; we don't want to assign low value to those states
            real_done = False if time_step == max_steps else done
            agent.replay_pool.push(Transition(state, action, reward, nextstate, real_done))
            state = nextstate
            if state_filter:
                state_filter.update(state)
            episode_reward += reward
            # update if it's time
            if cumulative_timestep % update_timestep == 0 and cumulative_timestep > n_collect_steps:
                q1_loss, q2_loss, pi_loss, a_loss, lf1_loss, lf2_loss, r_mse, qtarg_mean, qtarg_std = agent.optimize(update_timestep, lambda_=lambda_, state_filter=state_filter)
                n_updates += 1
            # logging
            if cumulative_timestep % log_interval == 0 and cumulative_timestep > n_collect_steps:
                if params['use_wandb']:
                    wandb.log({'Loss/Q-func_1': q1_loss}, step=n_updates)
                    wandb.log({'Loss/Q-func_2': q2_loss}, step=n_updates)
                    wandb.log({'Loss/LambdaF_1': lf1_loss}, step=n_updates)
                    wandb.log({'Loss/LambdaF_2': lf2_loss}, step=n_updates)
                    wandb.log({'Loss/policy': pi_loss}, step=n_updates)
                    wandb.log({'Loss/alpha': a_loss}, step=n_updates)
                    wandb.log({'Loss/r_mse': r_mse}, step=n_updates)
                    wandb.log({'Values/alpha': np.exp(agent.log_alpha.item())}, step=n_updates)
                    wandb.log({'Values/lambda': lambda_}, step=n_updates)
                    wandb.log({'Values/episodes': i_episode}, step=n_updates)
                    wandb.log({'Values/bandit_inverse_temp': agent.TDC.inverse_temp}, step=n_updates)
                    wandb.log({'Values/qtarg_mean': qtarg_mean}, step=n_updates)
                    wandb.log({'Values/qtarg_std': qtarg_std}, step=n_updates)
                    wandb.log({f"Distributions/arm{k}": p for k, p in enumerate(agent.TDC.get_probs())}, step=n_updates)
                avg_length = np.mean(episode_steps)
                running_reward = np.mean(episode_rewards)
                eval_reward = evaluate_agent(env, agent, state_filter, n_starts=n_evals)
                if params['use_wandb']:
                    wandb.log({'Reward/Train': running_reward}, step=cumulative_timestep)
                    wandb.log({'Reward/Test': eval_reward}, step=cumulative_timestep)
                print('Episode {} \t Samples {} \t Avg length: {} \t Test reward: {} \t Train reward: {} \t Number of Policy Updates: {}'.format(i_episode, samples_number, avg_length, eval_reward, running_reward, n_updates))
                episode_steps = []
                episode_rewards = []
            if cumulative_timestep % gif_interval == 0:
                make_gif(agent, env, cumulative_timestep, state_filter)
                if save_model:
                    make_checkpoint(agent, cumulative_timestep, params['env'])

        episode_steps.append(time_step)
        episode_rewards.append(episode_reward)

        if params['lambda_'] < 0:
            # update the bandit distribution
            bandit_feedback = episode_reward - prev_episode_reward
            agent.TDC.update_dists(bandit_feedback)
            prev_episode_reward = episode_reward


def evaluate_agent(env, agent, state_filter, n_starts=1):
    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state = env.reset()
        while (not done):
            action = agent.get_action(state, state_filter=state_filter, deterministic=True)
            nextstate, reward, done, _ = env.step(action)
            reward_sum += reward
            state = nextstate
    return reward_sum / n_starts


def main():
    
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--lambda_', type=float, default=1.0)
    parser.add_argument('--use_obs_filter', dest='obs_filter', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lf_wt', type=float, default=0.0)
    parser.add_argument('--sigmoid_beta', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_layernorm', dest='no_layernorm', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--use_wandb', dest='use_wandb', action='store_true')
    parser.add_argument('--use_one_critic', dest='use_one_critic', action='store_true')
    parser.add_argument('--update_every_n_steps', type=int, default=1)
    parser.add_argument('--n_random_actions', type=int, default=10000)
    parser.add_argument('--temp_annealing_factor', type=float, default=1.0)
    parser.add_argument('--n_collect_steps', type=int, default=1000)
    parser.add_argument('--n_evals', type=int, default=1)
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--make_gif', dest='make_gif', action='store_true')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--total_steps', type=int, default=int(1e6))
    parser.set_defaults(obs_filter=False)
    parser.set_defaults(save_model=False)

    args = parser.parse_args()
    params = vars(args)

    if params['debug']:
        params['n_random_actions'] = 100
        params['n_collect_steps'] = 100
        params['hidden_size'] = 4
        params['batch_size'] = 8

    seed = params['seed']
    all_envs = gym.envs.registry.all()
    available_envs = [env_spec.id for env_spec in all_envs]
    env_name = params['env']
    if env_name in available_envs:
        env = gym.make(params['env'])
    else:
        raise Exception("Invalid environment name")
    env = RescaleAction(env, -1, 1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    feature_dim = state_dim + action_dim

    agent = LambdaSAC_Agent(
        seed,
        state_dim,
        action_dim,
        feature_dim,
        hidden_size=params['hidden_size'],
        batchsize=params['batch_size'],
        bandit_temp_annealing_factor=params['temp_annealing_factor'],
        lf_wt=params['lf_wt'],
        use_one_critic=params['use_one_critic'])

    if len(params['experiment_name']) == 0:
      params['experiment_name'] = f"LambdaSAC_{params['env']}_{params['seed']}_SingleQ{params['use_one_critic']}_lfwt{params['lf_wt']}_lambda{params['lambda_']}_temp{params['temp_annealing_factor']}"

    train_agent_model_free(agent=agent, env=env, params=params)


if __name__ == '__main__':
    main()
