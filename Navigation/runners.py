import numpy as np
from typing import Dict
from utils import * 

from bsuite.baselines import base

import dm_env

def run_experiment_episodic(
    env,
    agent,
    number_of_episodes: int,
    eval_only: bool = False,
    max_ep_len: int = 20,
    display_eps: int = None,
    respect_done: bool = True
    ) -> Dict:
    """
    run an experiment
    """
    return_hist = []
    deltas = []
    trajectories = []
    lambdaR_hist = []
    if hasattr(agent, '_eval'): agent._eval = eval_only; 
    if hasattr(agent, 'r') and hasattr(env, 'r'): agent.r = env.r; 
    if hasattr(agent, 'reset'): agent.reset();

    try:
        action = agent.initial_action()
    except AttributeError:
        action = 0
    for i in range(1, number_of_episodes+1):

        if hasattr(agent, 'reset'): agent.reset();
        reward, discount, next_state, done = env.reset()
        if respect_done: agent._state = next_state; 
        action = agent.step(reward, discount, next_state)
        z = reward
        traj = [env.obs_to_state_coords(next_state)]

        episode_data = [(
            next_state, action, z, agent.Psi, env.goals, env.rewards_remaining(), np.max(agent.Q, axis=1), 0
            )]
        

        for t in range(1, max_ep_len+1):
            
             # effect of action in env
            reward, discount, next_state, done = env.step(action)
            agent.w = env.get_r()
            # agent takes next step
            action = agent.step(reward, discount, next_state)
            z += (discount ** t) * reward
            traj.append(env.obs_to_state_coords(next_state))
            episode_data.append((
                next_state, action, z, agent.Psi, env.goals, env.rewards_remaining(), np.max(agent.Q, axis=1), t
            ))
            if done and respect_done: break; 

        return_hist.append(z)

        # display progress 
        if display_eps is not None and i % display_eps == 0:
            flush_print(f"ep {i}/{number_of_episodes}: mean return = {np.mean(return_hist)}")

        trajectories.append(traj)
        if hasattr(agent, "LR"):
            lambdaR_hist.append(copy.deepcopy(agent.LR))

    results = {
      "return hist": return_hist,
      "trajectory": traj,
      "trajectories": trajectories,
      "deltas": deltas,
      "episode_data": episode_data,
      "lambdaR_hist": lambdaR_hist
    }
    if hasattr(agent, "state_values"): results["state values"] = agent.state_values; 
    if hasattr(agent, "q_values"): results["q values"] = agent.q_values; 
    if hasattr(agent, "SR"): results['SR'] = agent.SR; 
    if hasattr(agent, "FR"): results['FR'] = agent.FR; 


    return results 


def run_nn_eval(env, agent, objects, number_of_episodes, discount, max_ep_len, start_pos, random_start, r_regression_data=None):

    return_hist = []
    agent._eval = True
    agent.eval()
    if hasattr(agent, 'r') and hasattr(env, 'r'): agent.r = env.r; 
    if hasattr(agent, 'reset'): agent.reset();

    try:
        action = agent.initial_action()
    except AttributeError:
        action = 0
    for i in range(1, number_of_episodes+1):
        if hasattr(agent, 'reset'): agent.reset();
        if start_pos is not None:
            env.agent_pos = start_pos
            env.agent_start_pos = start_pos
        next_obs = env.reset(
            objects=objects,
            terminate_on_reward=False,
            random_start=random_start)
        reward, done = 0.0, False
        agent._state = env.get_idx(env.agent_pos); 
        agent_obs = next_obs
        if r_regression_data is None:
            agent.w = env.get_r()
        elif len(r_regression_data) == 1:
            agent.w = r_regression_data[0]
        else:
            Phi, all_states = r_regression_data
            r_vec = env.get_r()
            r = np.array([r_vec[s] for s in all_states])[:, None]  #  [B, 1]
            agent.w = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ r  # [D, 1]
        action = agent.step(reward, discount, agent_obs)
        z = reward

        for t in range(1, max_ep_len+1):

            if r_regression_data is None:
                agent.w = env.get_r()
            elif len(r_regression_data) == 1:
                agent.w = r_regression_data[0]
            else:
                Phi, all_states = r_regression_data
                r_vec = env.get_r()
                r = np.array([r_vec[s] for s in all_states])[:, None]  #  [B, 1]
                agent.w = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ r  # [D, 1]
            
             # effect of action in env
            next_obs, reward, done, _ = env.step(action)
            
            # agent takes next step
            agent_obs = next_obs
            action = agent.step(reward, discount, agent_obs)
            z += (discount ** t) * reward
            if done: break; 

        return_hist.append(z)

    agent._eval = False
    agent.train()

    return np.mean(return_hist)


def run_nn_experiment_episodic(
    env,
    agent,
    number_of_episodes: int,
    objects: dict = None,
    terminate_on_reward: bool = False,
    random_start: bool = False,
    start_pos: tuple = None,
    discount: float = 0.97,
    eval_only: bool = False,
    max_ep_len: int = 20,
    display_eps: int = None,
    respect_done: bool = True,
    use_underlying_pos: bool = False,
    record: bool = False,
    r_regression_data: tuple = None,
    combine_obs_with_state: bool = False,
    env_epsilon: float = 0.0,
    goals_always_available: bool = True,
    eval_every: int = -1,
    ) -> Dict:
    """
    run an experiment
    """
    return_hist = []
    eval_return_hist = []
    deltas = []
    trajectories = []
    state_buff, obs_buff, action_buff, next_state_buff, next_obs_buff = [], [], [], [], []
    if hasattr(agent, '_eval'): agent._eval = eval_only; 
    if hasattr(agent, 'r') and hasattr(env, 'r'): agent.r = env.r; 
    if hasattr(agent, 'reset'): agent.reset();

    try:
        action = agent.initial_action()
    except AttributeError:
        action = 0
    for i in range(1, number_of_episodes+1):
        if eval_every > 0 and i % eval_every == 0:
            eval_return_hist.append(
                run_nn_eval(env, agent, objects, 1, discount, max_ep_len,
                            start_pos, random_start=False, r_regression_data=r_regression_data))
        frames = []
        state_buff_ep, action_buff_ep, next_state_buff_ep = [], [], []
        obs_buff_ep, next_obs_buff_ep = [], []
        value_hist, w_hist, ep_return_hist, ep_reward_hist = [], [], [], []
        if hasattr(agent, 'reset'): agent.reset();
        if start_pos is not None:
            env.agent_pos = start_pos
            env.agent_start_pos = start_pos
        if not goals_always_available:
            possible_goals = list(objects['rewards'].keys())
            goal = possible_goals[np.random.choice(len(possible_goals))]
            objects['rewards'] = {goal: objects['rewards'][goal]}
        next_obs = env.reset(
            objects=objects,
            terminate_on_reward=terminate_on_reward,
            random_start=random_start)
        state_buff_ep.append(env.get_idx(env.agent_pos))
        obs_buff_ep.append(next_obs)
        if record and i == number_of_episodes: 
            frames.append(env.render(provide=True, display=False))
        reward, done = 0.0, False
        ep_reward_hist.append(reward)
        if respect_done: agent._state = env.get_idx(env.agent_pos); 
        agent_obs = next_obs
        if use_underlying_pos:
            agent_obs = env.get_idx(env.agent_pos)
        if combine_obs_with_state:
            agent_obs = (env.get_idx(env.agent_pos), next_obs)
        if r_regression_data is None:
            agent.w = env.get_r()
        elif len(r_regression_data) == 1:
            agent.w = r_regression_data[0]
        else:
            Phi, all_states = r_regression_data
            r_vec = env.get_r()
            r = np.array([r_vec[s] for s in all_states])[:, None]  #  [B, 1]
            agent.w = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ r  # [D, 1]
        w_hist.append(agent.w)
        action = agent.step(reward, discount, agent_obs)
        # take a random action with probability env_epsilon
        if np.random.rand() < env_epsilon:
            action = np.random.randint(0, 5)
        action_buff_ep.append(action)
        z = reward
        ep_return_hist = [z]
        rewards_remaining_hist = [env.rewards_remaining()]
        if hasattr(agent, 'v'): value_hist.append(agent.v)

        for t in range(1, max_ep_len+1):

            if r_regression_data is None:
                agent.w = env.get_r()
            elif len(r_regression_data) == 1:
                agent.w = r_regression_data[0]
            else:
                Phi, all_states = r_regression_data
                r_vec = env.get_r()
                r = np.array([r_vec[s] for s in all_states])[:, None]  #  [B, 1]
                agent.w = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ r  # [D, 1]
            w_hist.append(agent.w)
                

            top_three_rewards = env.rewards_remaining()
            rewards_remaining_hist.append(top_three_rewards)
            
             # effect of action in env
            next_obs, reward, done, _ = env.step(action)
            ep_reward_hist.append(reward)
            next_obs_buff_ep.append(next_obs)
            next_state_buff_ep.append(env.get_idx(env.agent_pos))
            obs_buff_ep.append(next_obs)
            state_buff_ep.append(env.get_idx(env.agent_pos))

            if record and i == number_of_episodes: 
                frames.append(env.render(provide=True, display=False))
            
            # agent takes next step
            agent_obs = next_obs
            if use_underlying_pos:
                agent_obs = env.get_idx(env.agent_pos)
            if combine_obs_with_state:
                agent_obs = (env.get_idx(env.agent_pos), next_obs)
            action = agent.step(reward, discount, agent_obs)
            if np.random.rand() < env_epsilon:
                action = np.random.randint(0, 5)
            action_buff_ep.append(action)
            if hasattr(agent, 'v'): value_hist.append(agent.v)
            z += (discount ** t) * reward
            ep_return_hist.append(z)

            if done and respect_done: break; 

        return_hist.append(z)
        state_buff += state_buff_ep[:-1]
        obs_buff += obs_buff_ep[:-1]
        action_buff += action_buff_ep[:-1]
        next_state_buff += next_state_buff_ep
        next_obs_buff += next_obs_buff_ep

        # display progress 
        if display_eps is not None and i % display_eps == 0:
            flush_print(f"ep {i}/{number_of_episodes}: mean return = {np.mean(return_hist)}")


    results = {
      "return hist": return_hist,
      "eval_return_hist": eval_return_hist,
      "ep_return_hist": ep_return_hist,
      "ep_reward_hist": ep_reward_hist,
      "trajectories": trajectories,
      "rewards_remaining_hist": rewards_remaining_hist,
      "deltas": deltas,
      "frames": frames,
      "state_buff": np.stack(state_buff),
      "obs_buff": np.stack(obs_buff),
      "action_buff": np.stack(action_buff),
      "next_state_buff": np.stack(next_state_buff),
      "next_obs_buff": np.stack(next_obs_buff),
      "value_hist": value_hist,
      "w_hist": w_hist,
    }
    if hasattr(agent, "state_values"): results["state values"] = agent.state_values; 
    if hasattr(agent, "q_values"): results["q values"] = agent.q_values; 
    if hasattr(agent, "SR"): results['SR'] = agent.SR; 
    if hasattr(agent, "FR"): results['FR'] = agent.FR; 


    return results 



def jax_run(agent: base.Agent,
        environment: dm_env.Environment,
        num_episodes: int,
        log_every: int = 10,
        max_episode_len: int = 50) -> None:
  """Runs an agent on an environment.

  Args:
    agent: The agent to train and evaluate.
    environment: The environment to train on.
    num_episodes: Number of episodes to train for.
    verbose: Whether to also log to terminal.
  """

  return_hist = []
  for i in range(num_episodes):
    # Run an episode.
    timestep = environment.reset()
    ep_return = 0.0
    t = 0
    while not timestep.last():
      t += 1
      # Generate an action from the agent's policy.
      action = agent.select_action(timestep)

      # Step the environment.
      new_timestep = environment.step(action)

      # Tell the agent about what just happened.
      agent.update(timestep, action, new_timestep)

      # Book-keeping.
      ep_return += new_timestep.reward * new_timestep.discount**t
      timestep = new_timestep

      if t >= max_episode_len:
        break
    
    return_hist.append(ep_return)
    if i % log_every == 0:
        flush_print(f"Episode {i}/{num_episodes}, Avg. Return: {np.mean(return_hist)}, Avg. Last 10: {np.mean(return_hist[-10:])}")


  results = dict(
      return_hist=return_hist,)

  return results
