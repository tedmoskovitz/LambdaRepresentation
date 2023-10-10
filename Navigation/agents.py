import numpy as np
from numpy.linalg import inv
import torch as tr
from typing import Callable, List
import copy
from tqdm import tqdm
from utils import rgb_to_grayscale_batched, epsilon_greedy, EZGreedy, greedy


class Random(object):

  def __init__(self, number_of_states: int, number_of_actions: int, initial_state: int) -> None:
    """
    A random agent 
    """
    self._number_of_actions = number_of_actions

  def step(self, reward: float, discount: float, next_state: int) -> int:
    del reward, discount, next_state
    next_action = np.random.randint(self._number_of_actions)
    return next_action


class NeuralGPIAgent(object):

    def __init__(self, obs_shape: tuple, number_of_actions: int, z_dim: int, w: np.ndarray, Psi: List, frame_stacking=9):
        """An agent which performs GPI over a fixed set of policies."""
        self._number_of_actions = number_of_actions
        self.w = w  # should be |S|-dim in tabular case
        self.Psi = Psi  # list of neural networks
        self.K = len(Psi)
        self.Q = np.zeros([self.K, number_of_actions])
        self._z_dim = z_dim
        self._frame_stacking = frame_stacking
        C, H, W = obs_shape
        self._num_channels = C
        self._prev_frames = tr.zeros(
            [1, (self._frame_stacking-1) * C, H, W]).float()


    def step(self, reward: float, discount: float, next_state: int) -> int:
        """Take a step using GPE + GPI."""
        del reward, discount

        next_obs = tr.tensor(np.transpose(
            next_state[None, ...], (0, 3, 1, 2)).copy(),
            dtype=tr.float32, device="cpu")
    
        next_obs = rgb_to_grayscale_batched(next_obs)
        # frame stacking
        next_obs = tr.cat([self._prev_frames, next_obs], dim=1)


        # perform GPE: (n x |A| x |S|) @ |S| -> n x |A|
        # Psi[k](next_state) returns a d x |A| matrix
        w = tr.from_numpy(self.w).float()
        # K x |A|
        Q = tr.cat(
            [tr.matmul(w.t(), self.Psi[k](next_obs).squeeze(0).reshape(
            self._z_dim, self._number_of_actions)) for k in range(self.K)], dim=0)
        Q = Q.detach().numpy()
        self.Q = Q
        # perform GPI - max over policies, argmax over actions
        next_action = np.argmax(np.max(Q, axis=0))

        self._prev_frames = next_obs[:, self._num_channels:, :, :].copy()

        return next_action


class GPIAgent(object):

    def __init__(
        self,
        number_of_states: int, 
        number_of_actions: int,
        w: np.ndarray,
        Psi: List) -> None:
        """An agent which performs GPI over a fixed set of policies."""

        self._number_of_actions = number_of_actions
        self._number_of_states = number_of_states
        assert len(w) == number_of_states, "w must be same dimensionality as S"
        self.w = w # should be |S|-dim in tabular case
        # Psi should be a list of n SRs, each is |S| x |A| x |S|
        self.Psi = np.stack(Psi) # n x |S| x |A| x |S|
        n = self.Psi.shape[0]
        self.Q = np.zeros([n, number_of_actions])

    def step(self, reward: float, discount: float, next_state: int) -> int:
        """Take a step using GPI."""
        del reward, discount

        # perform GPE: (n x |A| x |S|) @ |S| -> n x |A| 
        Q = self.Psi[:, next_state, :, :] @ self.w 
        self.Q = Q
        # perform GPI - max over policies, argmax over actions
        next_action = np.argmax(np.max(Q, axis=0)) 

        return next_action


class LambdaR(object):

    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        initial_state: int,
        sa: bool = False,
        policy: Callable = None,
        q: np.ndarray = None,
        lambda_: float = 0.0,
        step_size: float = 0.1) -> None:
        """tabular SR learning

        Args:
            number_of_states (int): size of state space
            number_of_actions (int): size of action space
            initial_state (int): index of initial state
            sa (bool, optional): whether to condition on actions. defaults to False.
            policy (Callable, optional): function defining a policy over Q-values. defaults to None.
            q (np.ndarray, optional): Q-values. defaults to None.
            lambda_ (float, optional): Lambda Rep. parameter. defaults to 0.0.
            step_size (float, optional): learning rate. defaults to 0.1.
        """
        if sa:
            self._Phi = np.zeros(
                [number_of_states, number_of_actions, number_of_states])
            for a in range(number_of_actions):
                self._Phi[:, a, :] = (1 - lambda_) * np.eye(number_of_states)
        else:
            self._Phi = (1 - lambda_) * np.eye(number_of_states)
        self._sa = sa
        self._n = number_of_states
        self._number_of_actions = number_of_actions
        self.state_values = np.zeros([number_of_states])
        self._state = initial_state
        self._step_size = step_size
        self._initial_state = initial_state 
        self._policy = policy 
        self._lambda = lambda_
        self._q = q
        if self._policy is not None and self._q is not None: self._action = self._policy(self._q[initial_state, :]);
        else: self._action = 0; 

    @property
    def LR(self):
        return self._Phi

    @property
    def Q(self):
        return self._q

    @property
    def Psi(self):
        return self._Phi

    def step(self, reward: float, discount: float, next_state: int) -> int:
        """Take a step and update the SR. 

        Args:
            reward (float)
            discount (float)
            next_state (int)

        Returns:
            int: Action in [0, r, ..., |A|-1]
        """
        del reward
            
        # if policy and q-function provided, select action with this
        if self._policy is not None and self._q is not None:
            next_action = self._policy(self._q[next_state, :])
        else:
            # return random action
            next_action = np.random.randint(self._number_of_actions)

        # compute LambdaR update
        one_hot = np.eye(self._n)[self._state]
        if self._sa:
            target = one_hot * (
                1 + self._lambda * discount * self._Phi[next_state, next_action, :])
            target += discount * (
                1 - one_hot) * self._Phi[next_state, next_action, :]
            # target = discount * self._Phi[next_state, next_action, :]
            delta = target - self._Phi[self._state, self._action, :]
            # delta[self._state] = 0 # self._lambda
            self._Phi[self._state, self._action, :] += self._step_size * delta
        else:
            target = one_hot * (
                1 + self._lambda * discount * self._Phi[next_state, :])
            target += discount * (1 - one_hot) * self._Phi[next_state, :]
            delta = target - self._Phi[self._state, :]
            self._Phi[self._state, :] += self._step_size * delta

        # reset current state, action
        self._state = next_state
        self._action = next_action

        return next_action


class Pi1Agent(object):

    def __init__(self, number_of_states, initial_state):
        """
        toy policy 1
        """
        self._state = initial_state
        up_idxs = [25, 26, 19, 20, 13, 14]
        right_idxs = [7, 8, 9]
        down_idxs = [10]
        left_idxs = [15, 16, 21, 22, 27, 28]
        self.q = np.zeros([number_of_states, 4])
        for i in range(number_of_states):
            if i in up_idxs: self.q[i, :] = np.eye(4)[0];
            elif i in right_idxs: self.q[i, :] = np.eye(4)[1];
            elif i in down_idxs: self.q[i, :] = np.eye(4)[2];
            else: self.q[i, :] = np.eye(4)[3];

    def step(self, reward: float, discount: float, next_state: int) -> int:
        return np.argmax(self.q[self._state, :])


class Pi2Agent(object):

    def __init__(self, number_of_states, initial_state):
        """
        toy policy 2
        """
        self._state = initial_state
        right_idxs = [19, 20, 21, 22, 25, 26, 27, 28]
        self.q = np.zeros([number_of_states, 4])
        for i in range(number_of_states):
            if i in right_idxs: self.q[i, :] = np.eye(4)[1];
            else: self.q[i, :] = np.eye(4)[3];

    def step(self, reward: float, discount: float, next_state: int) -> int:
        del reward, discount
        action = np.argmax(self.q[self._state, :])
        self._state = next_state
        return action


class FourRoomsAgent(object):


    def __init__(self, target_room, four_rooms_P, discount, lambda_,
                 max_dp_iters=100, tol=1e-2):

        # rooms are u, r, d, l, numberered clockwise from bottom left        
        W = 4
        u, r, d, l, s = 0, 1, 2, 3, 4

        if target_room == 1:
            policy = np.array([
                    [W, W, W, W, W, W, W, W, W, W, W],
                    [W, r, d, d, d, W, d, d, d, d, W],
                    [W, r, s, l, l, l, l, l, l, l, W], 
                    [W, u, u, u, u, W, u, u, u, u, W], 
                    [W, u, u, u, u, W, u, u, u, u, W], 
                    [W, u, W, W, W, W, W, u, W, W, W],
                    [W, u, l, l, l, W, r, u, l, l, W], 
                    [W, u, l, l, l, W, d, d, d, d, W], 
                    [W, u, l, l, l, l, l, l, l, l, W], 
                    [W, u, l, l, l, W, u, u, u, u, W], 
                    [W, W, W, W, W, W, W, W, W, W, W], 
                ])
        elif target_room == 2:
            policy = np.array([
                    [W, W, W, W, W, W, W, W, W, W, W],
                    [W, d, d, d, d, W, r, r, d, d, W],
                    [W, r, r, r, r, r, r, r, d, d, W], 
                    [W, u, u, u, u, W, r, r, s, l, W], 
                    [W, u, u, u, u, W, r, r, u, l, W], 
                    [W, u, W, W, W, W, W, u, W, W, W],
                    [W, u, l, d, d, W, r, u, l, l, W], 
                    [W, u, l, d, d, W, r, u, l, l, W], 
                    [W, r, r, r, r, r, r, u, l, l, W], 
                    [W, u, u, u, u, W, r, u, l, l, W], 
                    [W, W, W, W, W, W, W, W, W, W, W], 
                ])
        elif target_room == 3:
                policy = np.array([
                    [W, W, W, W, W, W, W, W, W, W, W],
                    [W, d, l, d, d, W, r, d, l, l, W],
                    [W, d, l, r, r, r, r, d, l, l, W], 
                    [W, d, l, u, u, W, r, d, l, l, W], 
                    [W, d, l, u, u, W, r, d, l, l, W], 
                    [W, d, W, W, W, W, W, d, W, W, W],
                    [W, d, d, d, d, W, r, r, r, d, W], 
                    [W, d, d, d, d, W, r, d, d, d, W], 
                    [W, r, r, r, r, r, r, s, l, l, W], 
                    [W, u, u, u, u, W, u, u, l, l, W], 
                    [W, W, W, W, W, W, W, W, W, W, W], 
                ])
        elif target_room == 0:
                policy = np.array([
                    [W, W, W, W, W, W, W, W, W, W, W],
                    [W, d, l, l, l, W, d, d, d, d, W],
                    [W, d, l, l, l, l, l, l, l, l, W], 
                    [W, d, l, l, l, W, r, d, l, l, W], 
                    [W, d, l, l, l, W, r, d, l, l, W], 
                    [W, d, W, W, W, W, W, d, W, W, W],
                    [W, r, r, d, d, W, d, d, d, d, W], 
                    [W, r, r, s, l, W, d, d, d, d, W], 
                    [W, r, r, u, l, l, l, l, l, l, W], 
                    [W, r, r, u, l, W, u, u, u, u, W], 
                    [W, W, W, W, W, W, W, W, W, W, W], 
                ])

        policy = policy.flatten()
        S, A, _ = four_rooms_P.shape
        P = four_rooms_P
        self.policy = np.zeros((policy.shape[0], 5))  # S x A
        for s in np.arange(S):
            self.policy[s, :] = np.eye(5)[policy[s]]

        # compute P_pi
        assert A == 5, "FourRooms Policy assumes 5 actions"
        self.P_pi = np.zeros((S, A, S))
        self.LR = np.zeros((S, A, S))
        for a in range(A):
            self.LR[:, a, :] = (1 - lambda_) * np.eye(S)

        self.LR, self.deltas = self.compute_LR(
            self.LR, P, self.policy, lambda_, discount, max_dp_iters, tol=tol)


    def compute_LR(self, LR, P, pi, lambda_, gamma, max_iters, tol=1e-2):
        S, A, _ = P.shape
        # converged = False
        deltas = []
        for _ in tqdm(range(max_iters)):
            delta = 0.0
            for s in range(S):
                one_hot = np.eye(S)[s]
                for a in range(A):
                    # lr = np.eye(S)[s]
                    lr = np.zeros(S)
                    for stp1 in range(S):
                        for atp1 in range(A):
                            lr += pi[stp1, atp1] * P[s, a, stp1] * one_hot * (
                                1 + lambda_ * gamma * self.LR[stp1, atp1, :])
                            lr += pi[stp1, atp1] * P[s, a, stp1] * gamma * (
                                1 - one_hot) * self.LR[stp1, atp1, :]
                            # lr += gamma * pi[stp1, atp1] * P[s, a, stp1] * LR[stp1, atp1, :]
                    delta = max(np.max(delta), np.max(np.abs(LR[s, a, :] - lr)))
                    LR[s, a, :] = lr
            deltas.append(delta)
            if delta < tol:
                break
        return LR, deltas


class LambdaQ(object):

  def __init__(self, number_of_states, number_of_actions, initial_state,
               method, double, step_size=0.1, epsilon=0.25, decay_explore=int(2e5),
               lambda_=1.0, use_ez_greedy=False, optimistic_init=False):
    # Settings.
    self._number_of_actions = number_of_actions
    self._number_of_states = number_of_states
    self.use_ez_greedy = use_ez_greedy
    self._step_size = step_size
    self._behaviour_policy, self._target_policy = self.off_policy_methods(method, epsilon=epsilon)
    self._double = double
    self._eval = False
    self._episodes = -1
    self.lambda_ = lambda_
    self.w = np.zeros(self._number_of_states)
    # Initial state.
    self._state = initial_state
    self._initial_state = initial_state
    # Tabular q-estimates.
    self._lambda_rep = np.zeros((number_of_states, number_of_actions, number_of_states))
    if optimistic_init:
        self._lambda_rep += .2
    if double:
      self._q2 = np.zeros((number_of_states, number_of_actions))
      self._lambda_rep2 = np.zeros((number_of_states, number_of_actions, number_of_states))
    self._last_action = 0
    self._epsilon = epsilon
    self._steps = 0
    self._decay_explore = decay_explore

  def off_policy_methods(self, method, epsilon=0.25):
      """
      return list of [behavior policy, target_policy]
      """
      def qgreedy(q, a):
          del a
          return np.eye(len(q))[np.argmax(q)]
      def sarsa_target(q, a):
          return np.eye(len(q))[a]
      def expected_sarsa_target(q, a):
          del a
          greedy = np.eye(len(q))[np.argmax(q)]
          return greedy - greedy * epsilon + epsilon / 4 
      def doubleq_target(q, a):
          # Place equal probability on all actions that achieve the `max` value.
          # This is equivalent to `return np.eye(len(q))[np.argmax(q)]` for Q-learning
          # But results in slightly lower variance updates for double Q.
          del a
          max_q = np.max(q)
          pi = np.array([1. if qi == max_q else 0. for qi in q])
          return pi / sum(pi)

      if self.use_ez_greedy:
          explore_method = EZGreedy(epsilon=epsilon, mu=2.0)
      else:
          explore_method = epsilon_greedy


      if method == "q":
          return [explore_method, qgreedy]
      elif method == "sarsa":
          return [explore_method, sarsa_target]
      elif method == "expected sarsa":
          return [explore_method, expected_sarsa_target]
      elif method == "double q":
          return [explore_method, doubleq_target]
      else:
          raise ValueError("Invalid method: {} is not q, sarsa, expected sarsa, or double q.".format(method))


  @property
  def q_values(self):
    q = self._lambda_rep @ self.w
    if self._double:
        q2 = self._lambda_rep2 @ self.w
        return (q + q2)/2
    return q
  
  def eval(self):
    self._eval = True

  def train(self):
    self._eval = False

  @property
  def state_values(self):
    q = self.q_values
    return np.max(q, axis=1)

  @property
  def v(self):
    return [self.state_values[self._state]]

  @property
  def episodes(self):
    return self._episodes

  def step(self, reward, discount, next_state):
    # increment episode count when return to start state s
    if self._state == self._initial_state:
        self._episodes += 1

    # target policy generates distribution over states
    if self._double:
        # if doing a "double" algorithm, pick the target and behavior q-functions randomly
        if np.random.rand() < 0.5:
            behav_lambda_rep = self._lambda_rep; targ_lambda_rep = self._lambda_rep2;
        else:
            behav_lambda_rep = self._lambda_rep2; targ_lambda_rep = self._lambda_rep;
    else:
        behav_lambda_rep = self._lambda_rep; targ_lambda_rep = self._lambda_rep;
    
    targ_q = targ_lambda_rep @ self.w
    behav_q = behav_lambda_rep @ self.w
        
    if not self._eval:
        # compute target action 
        target_action_probs = self._target_policy(targ_q[next_state,:], self._last_action)
        # if this is regular q-learning, the target_action_probs should be one-hot from the target_policy, so equiv to argmax 
        target_action = np.random.choice(self._number_of_actions, p=target_action_probs) 
        one_hot = np.eye(self._number_of_states)[self._state]
        lambda_target = one_hot * (1 + self.lambda_ * discount * behav_lambda_rep[next_state, target_action, :])
        lambda_target += discount * (1 - one_hot) * behav_lambda_rep[next_state, target_action, :]
        # compute error
        lambda_delta = lambda_target - behav_lambda_rep[self._state, self._last_action, :]
        # update lambda rep
        behav_lambda_rep[self._state, self._last_action, :] += self._step_size * lambda_delta
    else:
        self._behaviour_policy = greedy
    
    # reset state, action
    self._state = next_state
    action = self._behaviour_policy(behav_q[next_state, :], self._epsilon)
    self._last_action = action
    self._steps += 1 
    if self._decay_explore is not None and self._steps % self._decay_explore == 0 and self._steps > 0:
        self._epsilon *= 0.5 # reduce exploration
        print("\nnew epsilon:", self._epsilon)
    return action    


