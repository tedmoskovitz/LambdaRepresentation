import copy
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_actions
from copy import deepcopy
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable


class FourRooms(object):

  def __init__(self, start_state=100, reset_goal=False, lambda_=1.0, seed=7697, discount=0.95, goals=None, goal_rewards=None):
    # -1: wall
    # 0: empty, episode continues
    # other: number indicates reward
    W = -1
    G = 1
    np.random.seed(seed)
    self._W = W  # wall
    self._G = G  # goal
    self.discount = discount
    self.lambda_ = lambda_

    self._layout = np.array([
        [W, W, W, W, W, W, W, W, W, W, W],
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W],
        [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, 0, W, W, W, W, W, 0, W, W, W],
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, W, W, W, W, W, W, W, W, W, W], 
    ])
    # reward
    flat_layout = self._layout.flatten()
    wall_idxs = np.stack(np.where(flat_layout == W)).T
    # possible reward states are those where there isn't a wall
    self._possible_reward_states = np.array([s for s in range(len(flat_layout)) if s not in wall_idxs])
    self._idx_layout = np.arange(self._layout.size).reshape(self._layout.shape)

    self._reset_goal = reset_goal
    self._random_start_state = start_state < 0
    if self._random_start_state:
      start_state = np.random.choice(self._possible_reward_states)
    self._start_state = self.obs_to_state_coords(start_state)
    self._episodes = 0
    self._state = self._start_state
    self._start_obs = self.get_obs()
    self._number_of_states = np.prod(np.shape(self._layout))
    
    flat_layout = self._layout.flatten()
    self.wall_idxs = np.stack(np.where(flat_layout == W)).T
    # room layout:
    # 1 2
    # 0 3
    self._N = self._layout.shape[0]
    self.room0_idxs = list(self._idx_layout[self._N//2:, :self._N//2].flatten())
    self.room1_idxs = list(self._idx_layout[:self._N//2, :self._N//2].flatten())
    self.room2_idxs = list(self._idx_layout[:self._N//2, self._N//2:].flatten())
    self.room3_idxs = list(self._idx_layout[self._N//2:, self._N//2:].flatten())
    self.room0_idxs = [idx for idx in self.room0_idxs if idx not in self.wall_idxs]
    self.room1_idxs = [idx for idx in self.room1_idxs if idx not in self.wall_idxs]
    self.room2_idxs = [idx for idx in self.room2_idxs if idx not in self.wall_idxs]
    self.room3_idxs = [idx for idx in self.room3_idxs if idx not in self.wall_idxs]
    self.room_idxs = [self.room0_idxs, self.room1_idxs, self.room2_idxs, self.room3_idxs]
    
    self.r = deepcopy(flat_layout).astype(float)
    self.goal_rewards = np.array([50]) if goal_rewards is None else np.array(goal_rewards)
    if goals is None:
      goal_state = np.random.choice(self._possible_reward_states)
      self.goals = [goal_state]
      self.r[goal_state] = 50
      self.goal_visits = {goal_state: 0}
    else:
      self.goal_visits = {}
      for i, goal in enumerate(goals):
        self.r[goal] = 50 if goal_rewards is None else goal_rewards[i]
        self.goal_visits[goal] = 0
      self.goals = goals
    
    self._goals_reached = 0
    self._steps = 0

    # transition matrix
    self._R = np.array([-1, 0, 50])
    P = np.zeros([self._number_of_states, 5, self._number_of_states])
    l = self._layout.shape[0]

    p = 1
    for a in range(5): 
      for s in range(self._number_of_states):
        for sp in range(self._number_of_states):
          
          if a == 0: 
            if sp == s - l and flat_layout[sp] != W: P[s, a, sp] =  p; 
            elif sp == s - l and flat_layout[sp] == W: P[s, a, s] = p; 
          elif a == 1: 
            if sp == s + 1 and flat_layout[sp] != W: P[s, a, sp] = p; 
            elif sp == s + 1 and flat_layout[sp] == W: P[s, a, s] = p; 
          elif a == 2: 
            if sp == s + l and flat_layout[sp ] != W: P[s, a, sp] = p; 
            elif sp == s + l and flat_layout[sp] == W: P[s, a, s] = p;
          elif a == 3: 
            if sp == s - 1 and flat_layout[sp] != W: P[s, a, sp] = p;
            elif sp == s - 1 and flat_layout[sp] == W: P[s, a, s] = p;
          else:
            P[s, a, sp] = p if sp == s else 0

    self._P = P
    

  @property
  def number_of_states(self):
      return self._number_of_states

  @property
  def goal_states(self):
      return self._goal_hist

  def get_obs(self, s=None):
    y, x = self._state if s is None else s
    return y*self._layout.shape[1] + x

  def obs_to_state(self, obs):
    x = obs % self._layout.shape[1]
    y = obs // self._layout.shape[1]
    s = np.copy(self._layout)
    s[y, x] = 4
    return s

  def obs_to_state_coords(self, obs):
    x = obs % self._layout.shape[1]
    y = obs // self._layout.shape[1]
    return (y, x)

  def rewards_remaining(self):
    return [self.goal_rewards[i] * self.lambda_ ** self.goal_visits[g] for i,g in enumerate(self.goals)]

  def get_r(self):
    remaining_rewards = self.rewards_remaining()
    r = deepcopy(self.r).astype(float)
    for i, goal in enumerate(self.goals):
      r[goal] = remaining_rewards[i]
    return r
  
  @property
  def episodes(self):
      return self._episodes

  def P(self, s, a, sp, r):
    if r not in self._R: return 0; 
    r_idx = np.where(self._R == r)[0][0]
    return self._P[s, a, sp, r_idx]


  def reset(self):
    if self._random_start_state:
      start_state = np.random.choice(self._possible_reward_states)
      self._start_state = self.obs_to_state_coords(start_state)
    self._state = self._start_state
    y, x = self._state
    self._steps = 0
    for goal in self.goals:
      self.goal_visits[goal] = 0
    return self._layout[y, x], 0.9, self.get_obs(), False

  def step(self, action):
    done = False
    y, x = self._state
    r2d = np.reshape(self.r, self._layout.shape)
    
    if action == 0:  # up
      new_state = (y - 1, x)
    elif action == 1:  # right
      new_state = (y, x + 1)
    elif action == 2:  # down
      new_state = (y + 1, x)
    elif action == 3:  # left
      new_state = (y, x - 1)
    elif action == 4: # stay
      new_state = (y, x)
    else:
      raise ValueError(f"Invalid action: {action} is not 0, 1, 2, 3, or 4.")

    new_y, new_x = new_state
    reward = r2d[new_y, new_x]
    if self._layout[new_y, new_x] == self._W:  # wall
      discount = self.discount
      new_state = (y, x)
    elif r2d[new_y, new_x] == 0:  # empty cell
      discount = self.discount
    else:  # a goal
      discount = self.discount
      goal = new_y * self._layout.shape[1] + new_x
      reward = self.lambda_ ** self.goal_visits[goal] * r2d[new_y, new_x]
      self.goal_visits[goal] += 1
    if np.max(self.rewards_remaining()) < 1e-1:
      done = True

    self._state = new_state
    self._steps += 1
    return reward, discount, self.get_obs(), done

  def plot_grid(
    self,
    traj=None, M=None, ax=None, goals=None, cmap='viridis',
    cbar=False, traj_colorbar=True, title='FourRooms', show_idxs=False,
    s_t=None, t=None
    ):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        
    ims = []
    im = ax.imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    ims.append(im)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=18)
    startx, starty = self._start_state
    goalx, goaly = self.obs_to_state_coords(np.argmax(self.r))
    if s_t is None:
      ax.text(starty, startx, r"$\mathbf{s_0}$", ha='center', va='center', fontsize=10)
    else:
      s_t_x, s_t_y = self.obs_to_state_coords(s_t)
      s_t_label = r"$s_t$" if t is None else r'$s_{{{}}}$'.format(t)
      ax.text(s_t_y, s_t_x, s_t_label, ha='center', va='center', fontsize=11)

    if show_idxs:
      for i in range(self._layout.shape[0]):
          for j in range(self._layout.shape[1]):
              ax.text(j, i, f"{self.get_obs(np.array([i, j]))}", ha='center', va='center')
    
    h, w = self._layout.shape
    for y in range(h-1):
      ax.plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
    for x in range(w-1):
      ax.plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)

    if traj is not None:
      # draw trajectories
      traj = np.vstack(traj)
      x = traj[:, 1]
      y = traj[:, 0]

      points = np.array([x, y]).T.reshape(-1, 1, 2)
      segments = np.concatenate([points[:-1], points[1:]], axis=1)

      # Normalize x values to the range [0, 1] for colormap
      indices = np.arange(x.size)
      norm = plt.Normalize(indices.min(), indices.max())
      colors = cmap(norm(indices))

      # Create a LineCollection object
      lc = LineCollection(segments, colors=colors, linewidths=3)

      # Add the LineCollection object to the axes
      ax.add_collection(lc)

      if traj_colorbar:
        # Create a ScalarMappable object for the colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # This is required for the colorbar, but the data is not used.

        # Create a divider for the colorbar and match its size to the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Add the colorbar to the new axis with the label r"$t$"
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label(r"$t$", fontsize=14, rotation=0)

    if goals is not None:
      for i,g in enumerate(goals):
        y, x = self.obs_to_state_coords(g)
        ax.text(x, y, r"$g_{{{}}}$".format(i), ha='center', va='center', fontsize=11, color=f'C{i}')


    if M is not None:
      # M is either a vector of len |S| of a matrix of size |A| x |S|
      if len(M.shape) == 1:
        M_2d = M.reshape(h, w)
      else:
        M_2d = np.mean(M, axis=0).reshape(h, w)

      im2 = ax.imshow(M_2d, cmap=cmap, interpolation='nearest')
      ims.append(im2)
      if cbar: ax.colorbar(); 
    return ims

  def plot_planning_output(self, pi, s_ast, ax=None, colors=None, show_states=False, suptitle=None, Pi=None):
    if ax is None:
        n = 2 if not show_states else 3
        fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
        fig.suptitle(suptitle, fontsize=23)
    
    pi2c = None
    if colors is not None:
      assert len(colors) == len(np.unique(pi)), "incompatible number of colors";
      pi2c = dict(zip(np.unique(pi), colors))

    axs[0].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    axs[1].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    axs[0].set_xticks([]); axs[1].set_xticks([])
    axs[0].set_yticks([]); axs[1].set_yticks([])
    axs[0].set_title(r"$\pi_F(s)$", fontsize=16)
    axs[1].set_title(r"$s_F(s)$", fontsize=16)
    h, w = self._layout.shape
    color_list = []
    for y in range(h):
      for x in range(w):
        pidx = pi[y, x]
        if self._layout[y, x] >= 0:
          c = 'k' if pi2c is None else pi2c[pidx]
          if Pi is None:
            axs[0].text(x, y, r"$\pi_{}$".format(pidx), ha='center', va='center', color=c, fontsize=16)
          row, col = self.obs_to_state_coords(s_ast[y, x])
          axs[1].text(x, y, "{},{}".format(row-1, col-1), ha='center', va='center', color=c, fontsize=10) #.format(s_ast[y, x])

    # plot arrows 
    if Pi is not None:
      # construct "composite" q-function
      cq = np.zeros([self._number_of_states, 4])
      color_list = []
      for s in range(self._number_of_states):
        pidx = np.reshape(pi, [-1])[s]
        cq[s] = Pi[pidx].q[s, :]
        color_list.append(pi2c[pidx])
      
      plot_actions(self._layout, cq.reshape(self._layout.shape + (4,)), ax=axs[0], c=color_list)


    h, w = self._layout.shape
    for y in range(h-1):
      axs[0].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
      axs[1].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
    for x in range(w-1):
      axs[0].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)
      axs[1].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)

    if show_states: 
      axs[2].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
      axs[2].set_xticks([]); axs[1].set_xticks([])
      axs[2].set_yticks([]); axs[1].set_yticks([])
      axs[2].set_title(r"$\mathcal{S}$", fontsize=20)
      for i in range(1, self._layout.shape[0]-1):
          for j in range(1, self._layout.shape[1]-1):
              axs[2].text(j, i, "{},{}".format(i-1, j-1), ha='center', va='center', color='k', fontsize=10) #.format(idx)

      h, w = self._layout.shape
      for y in range(h-1):
        axs[2].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
      for x in range(w-1):
        axs[2].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)


class BasicGrid(object):

  def __init__(self, start_state=25, noisy=False, discount=0.9, lambda_=1.0):
    # -1: wall
    # 0: empty, episode continues
    # other: number indicates reward, episode will terminate
    W = -1
    G = 1
    self._W = W  # wall
    self._G = G  # goal
    self._layout = np.array([
        [W, W, W, W, W, W],
        [W, 0, 0, 0, 0, W],
        [W, 0, 0, G, 0, W], 
        [W, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W], 
        [W, W, W, W, W, W]
    ])
    self.discount = discount
    self.lambda_ = lambda_
    flat_layout = copy.copy(self._layout).flatten()
    self.r = deepcopy(flat_layout).astype(float)
    self.goal_rewards = np.array([1.0])
    goal_state = 15
    self.goals = [goal_state]
    self.r[goal_state] = G
    self.goal_visits = {goal_state: 0}

    self._start_state = self.obs_to_state_coords(start_state)#(4, 1)
    # self.goals = np.argmax(np.reshape(self._layout, (-1,)))
    self._episodes = 0
    self._state = self._start_state
    self._start_obs = self.get_obs()
    self._number_of_states = np.prod(np.shape(self._layout))
    self._number_of_actions = 4
    self._noisy = noisy
    self.rewards_remaining = lambda : G 
    self.get_r = lambda : self.r


    P = np.zeros([self._number_of_states, self._number_of_actions, self._number_of_states])
    l = self._layout.shape[0]

    p = 1
    for a in range(4): 
      for s in range(self._number_of_states):
        for sp in range(self._number_of_states):
          
          if a == 0: 
            if sp == s - l and flat_layout[sp] != W: P[s, a, sp] =  p; 
            elif sp == s - l and flat_layout[sp] == W: P[s, a, s] = p; 
          elif a == 1: 
            if sp == s + 1 and flat_layout[sp] != W: P[s, a, sp] = p; 
            elif sp == s + 1 and flat_layout[sp] == W: P[s, a, s] = p; 
          elif a == 2: 
            if sp == s + l and flat_layout[sp ] != W: P[s, a, sp] = p; 
            elif sp == s + l and flat_layout[sp] == W: P[s, a, s] = p;
          elif a == 3: 
            if sp == s - 1 and flat_layout[sp] != W: P[s, a, sp] = p;
            elif sp == s - 1 and flat_layout[sp] == W: P[s, a, s] = p;
          else:
            P[s, a, sp] = p if sp == s else 0

    self.P = P

  @property
  def number_of_states(self):
      return self._number_of_states

  def rewards_remaining(self):
    return [self.goal_rewards[i] * self.lambda_ ** self.goal_visits[g] for i,g in enumerate(self.goals)]

  def get_r(self):
    remaining_rewards = self.rewards_remaining()
    r = deepcopy(self.r).astype(float)
    for i, goal in enumerate(self.goals):
      r[goal] = remaining_rewards[i]
    return r

  def get_obs(self, s=None):
    y, x = self._state if s is None else s
    return y*self._layout.shape[1] + x

  def obs_to_state(self, obs):
    x = obs % self._layout.shape[1]
    y = obs // self._layout.shape[1]
    s = np.copy(self._layout)
    s[y, x] = 4
    return s

  def obs_to_state_coords(self, obs):
    x = obs % self._layout.shape[1]
    y = obs // self._layout.shape[1]
    return (y, x)

  @property
  def episodes(self):
      return self._episodes

  def reset(self):
    self._state = self._start_state
    self._episodes = 0
    y, x = self._state
    for goal in self.goals:
      self.goal_visits[goal] = 0
    return self._layout[y, x], 0.9, self.get_obs(), False

  def step(self, action):
    done = False
    y, x = self._state
    r2d = np.reshape(self.r, self._layout.shape)
    if action == 0:  # up
      new_state = (y - 1, x)
    elif action == 1:  # right
      new_state = (y, x + 1)
    elif action == 2:  # down
      new_state = (y + 1, x)
    elif action == 3:  # left
      new_state = (y, x - 1)
    # elif action
    else:
      raise ValueError("Invalid action: {} is not 0, 1, 2, or 3.".format(action))

    new_y, new_x = new_state
    reward = self._layout[new_y, new_x]
    if self._layout[new_y, new_x] == self._W:  # wall
      discount = self.discount
      new_state = (y, x)
    elif self._layout[new_y, new_x] == 0:  # empty cell
      discount = self.discount
    else:  # a goal
      self._episodes += 1
      goal = new_y * self._layout.shape[1] + new_x
      reward = self.lambda_ ** self.goal_visits[goal] * r2d[new_y, new_x]
      self.goal_visits[goal] += 1
      done = True
      discount=self.discount

    if self._noisy:
      width = self._layout.shape[1]
      reward += 10*np.random.normal(0, width - new_x + new_y)

    self._state = new_state
    return reward, discount, self.get_obs(), done

  def plot_grid(self, traj=None, M=None, ax=None, cbar=False, cmap='viridis', traj_color=["C2"], title='A Grid', show_states=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    im = ax.imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=22)
    if show_states:
      # ax.text(1, 4, r"$\mathbf{s_0}$", ha='center', va='center', fontsize=22)
      ax.text(3, 2, r"$g$", ha='center', va='center', fontsize=22, color='C0')

    h, w = self._layout.shape
    for y in range(h-1):
      ax.plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
    for x in range(w-1):
      ax.plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)

    if traj is not None:
      # plot trajectory, list of [(y0, x0), (y1, x1), ...]
      traj = np.vstack(traj)
      ax.plot(traj[:, 1], traj[:, 0], c=traj_color[0], lw=3)

    if M is not None:
      # M is either a vector of len |S| of a matrix of size |A| x |S|
      if len(M.shape) == 1:
        M_2d = M.reshape(h, w)
      else:
        M_2d = np.mean(M, axis=0).reshape(h, w)

      im = ax.imshow(M_2d, cmap=cmap, interpolation='nearest')
      if cbar: ax.colorbar(); 

    return im

  def plot_planning_output(self, pi, s_ast, ax=None, colors=None, show_states=False, Pi=None):
    if ax is None:
        n = 2 if not show_states else 3
        fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    
    pi2c = None
    if colors is not None:
      assert len(colors) == len(np.unique(pi)), "incompatible number of colors";
      pi2c = dict(zip(np.unique(pi), colors))

    axs[0].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    axs[1].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    axs[0].set_xticks([]); axs[1].set_xticks([])
    axs[0].set_yticks([]); axs[1].set_yticks([])
    axs[0].set_title(r"$\pi^F(s)$", fontsize=30)
    axs[1].set_title(r"$s^F(s)$", fontsize=30)
    s_ast[1, :] = 15
    s_ast[2, :] = 15
    s_ast[4, 3] = 15
    pi[2, 4] = 0
    pi[1, 3] = 0
    
    for y in range(1, 5):
      for x in range(1, 5):
        pidx = pi[y, x]
        c = 'k' if pi2c is None else pi2c[pidx]
        if Pi is None:
          axs[0].text(y, x, r"$\pi_{}$".format(pidx), ha='center', va='center', color=c, fontsize=22)
        row, col = self.obs_to_state_coords(s_ast[y, x])
        
        axs[1].text(x, y, "{},{}".format(row-1, col-1), ha='center', va='center', color=c, fontsize=22) 


    # plot arrows 
    if Pi is not None:
      # construct "composite" q-function
      cq = np.zeros([self._number_of_states, 4])
      color_list = []
      for s in range(self._number_of_states):
        pidx = np.reshape(pi, [-1])[s]
        cq[s] = Pi[pidx].q[s, :]
        color_list.append(pi2c[pidx])
      
      plot_actions(self._layout, cq.reshape(self._layout.shape + (4,)), ax=axs[0], c=color_list)

    h, w = self._layout.shape
    for y in range(h-1):
      axs[0].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
      axs[1].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
    for x in range(w-1):
      axs[0].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)
      axs[1].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)

    if show_states: 
      axs[2].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
      axs[2].set_xticks([]); axs[1].set_xticks([])
      axs[2].set_yticks([]); axs[1].set_yticks([])
      axs[2].set_title(r"$\mathcal{S}$", fontsize=26)
      for y in range(1, 5):
        for x in range(1, 5):
          idx = 6 * y + x 
          axs[2].text(x, y, "{},{}".format(y-1, x-1), ha='center', va='center', color='k', fontsize=22) #.format(idx)

      h, w = self._layout.shape
      for y in range(h-1):
        axs[2].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
      for x in range(w-1):
        axs[2].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)



