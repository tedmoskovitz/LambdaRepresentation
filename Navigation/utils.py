import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import imageio
from PIL import Image
import io
import copy
import matplotlib.gridspec as gridspec

from collections import Counter
import dm_env
from dm_env import specs

import numpy as np
import random
import torch as tr
from torch.utils.data import Dataset
import pickle 
import sys 
from scipy import stats

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Create a ListedColormap from the extracted colors
# Define the color segments for the colormap
segments = [(i/(len(colors)-1), colors[i]) for i in range(len(colors))]

# Create a LinearSegmentedColormap from the color segments
CMAP = LinearSegmentedColormap.from_list(name='my_colormap', colors=segments)


def epsilon_greedy(q_values, epsilon=0.1):
  """return an action epsilon-greedily from a vector of Q-values Q[s,:]"""
  if epsilon < np.random.random():
    return np.argmax(q_values)
  else:
    return np.random.randint(np.array(q_values).shape[-1])

class EZGreedy:

  def __init__(self, epsilon=0.1, mu=2.0):
    self.epsilon = epsilon
    self.mu = mu
    self.random_action_count = 0
    self.num_random_action_repeats = 0
    self.random_action = 0

  def __call__(self, q_values, epsilon=None):
    del epsilon
    if self.random_action_count < self.num_random_action_repeats:
      self.random_action_count += 1
      return self.random_action
    else:
      if self.epsilon < np.random.random():
        return np.argmax(q_values)
      else:
        # draw a duration from a zeta distribution with parameter mu
        self.num_random_action_repeats = np.random.zipf(self.mu)
        # reset the count
        self.random_action_count = 1
        # draw a random action
        self.random_action = np.random.randint(np.array(q_values).shape[-1])
        return self.random_action


def greedy(q_values, eps=None):
    """return the greedy action from a vector of Q-values"""
    return np.argmax(q_values)

def flush_print(s):
    """
    print updates in a loop that refresh rather than stack 
    """
    print("\r" + s, end="")
    sys.stdout.flush()

def save_results(name, result_dict):
    """
    save experiment results to a pickle file 
    """
    with open('saved/{}_results.pickle'.format(name), 'wb') as file:
        pickle.dump(result_dict, file)

def load_results(file_name):
    """
    load experiment results from a pickle file 
    """
    with open('saved/{}_results.pickle'.format(file_name), 'rb') as file:
        return(pickle.load(file))
    
def set_seed_everywhere(seed):
    tr.manual_seed(seed)
    if tr.cuda.is_available():
        tr.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class TupleDataset(Dataset):
    def __init__(self, data_dict):
        self.data = []
        if 'prev_actions' in data_dict:
          self.prev_actions = data_dict['prev_actions']
          self.data.append(data_dict['prev_actions'])
        self.states = data_dict['states']
        self.data.append(data_dict['states'])
        if 'observations' in data_dict:
          self.observations = data_dict['observations']
          self.data.append(self.observations)
        if 'features' in data_dict:
          self.features = data_dict['features']  
          self.data.append(self.features)
        self.actions = data_dict['actions']
        self.data.append(self.actions)
        self.next_states = data_dict['next_states']
        self.data.append(self.next_states)
        if 'next_observations' in data_dict:
          self.next_observations = data_dict['next_observations']
          self.data.append(self.next_observations)
        if 'next_features' in data_dict:
          self.next_features = data_dict['next_features']
          self.data.append(self.next_features)
        if 'dones' in data_dict:
          self.dones = data_dict['dones']
          self.data.append(self.dones)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return tuple([d[idx] for d in self.data])
    
    
def rgb_to_grayscale_batched(images):
    # images should have shape (batch_size, 3, height, width)
    assert images.shape[1] == 3, "Input images must have 3 channels (R, G, B)"

    # Calculate grayscale values using the luma conversion formula
    gray = 0.2989 * images[:, 0] + 0.5870 * images[:, 1] + 0.1140 * images[:, 2]

    # Add a new axis to convert the shape from (batch_size, height, width) to (batch_size, 1, height, width)
    gray = gray.unsqueeze(1)

    return gray


def stack_frames(observations, T, K=1):
  if K == 0:
    return observations
  # observations is [B*T, C, H, W]
  BT, C, H, W = observations.shape
  num_eps = BT // T
  observations = tr.reshape(observations, (num_eps, T, C, H, W))
  zero_frames = tr.zeros((num_eps, K-1, C, H, W), device=observations.device)
  # [B, T+K, C, H, W]
  observations = tr.cat([zero_frames, observations], dim=1)
  # [B, T, K, C, H, W]
  observations_K = tr.stack([observations[:, t:t+K] for t in range(T)], dim=1)
  # stack the K frames together in the channel dimensions -> [B, T, C*K, H, W]
  observations_K = tr.reshape(observations_K, (num_eps, T, C*K, H, W))
  # stack the episodes together -> [B*T, C*K, H, W]
  observations_K = tr.reshape(observations_K, (BT, C*K, H, W))

  return observations_K

    
def preprocess(dataset, room=0, convert_to_grayscale=True, device='cpu'):
    # dataset[room] is a dictionary with keys "states", "actions", "next_states"
    # each value is a numpy array.
    # states and next_states are shape (num_episodes, episode_len, 128, 128, 3)
    # identify the previous actions (if it's the first step in the episode, the previous action is [1, 0, 0, 0 0])
    room_dataset = dataset[room]
    num_actions = np.max(room_dataset['actions']) + 1
    num_episodes = room_dataset['states'].shape[0]
    # [N, 1, A]
    # prev_initial_actions = tr.tile(tr.eye(num_actions)[0].unsqueeze(0).unsqueeze(0), (num_episodes, 1, 1))
    prev_initial_actions = np.zeros((num_episodes, 1)).astype(int)
    prev_actions = np.concatenate([prev_initial_actions, room_dataset['actions'][:, :-1]], axis=1)

    prev_actions = tr.tensor(
        np.eye(num_actions)[prev_actions.flatten()],
        dtype=tr.float32, device=device)


    # the first step is to convert these to arrays of shape (num_episodes * episode_len, 128, 128, 3)
    observations, next_observations = room_dataset['observations'], room_dataset['next_observations']
    observations = observations.reshape(-1, *observations.shape[2:])
    next_observations = next_observations.reshape(-1, *next_observations.shape[2:])
    # now we want to transpose the last two dimensions so that the channels are first
    observations = tr.tensor(
        np.transpose(observations, (0, 3, 1, 2)), dtype=tr.float32, device=device)
    next_observations = tr.tensor(
        np.transpose(next_observations, (0, 3, 1, 2)), dtype=tr.float32, device=device)
    if convert_to_grayscale:
        observations = rgb_to_grayscale_batched(observations)
        next_observations = rgb_to_grayscale_batched(next_observations)
    # the actions are shaped (num_episodes, episode_len)
    # we want to convert them to one-hot vectors and store in a tensor
    # of shape (num_episodes * episode_len, num_actions)
    actions = tr.tensor(
        np.eye(num_actions)[room_dataset['actions'].flatten()],
        dtype=tr.float32, device=device)
    
    states = tr.tensor(room_dataset['states'].flatten(), dtype=tr.float32, device=device)
    next_states = tr.tensor(room_dataset['next_states'].flatten(), dtype=tr.float32, device=device)
    return dict(
       prev_actions=prev_actions, states=states, observations=observations, actions=actions,
       next_states=next_states, next_observations=next_observations)
    

def gif_from_frames(frames, filename, frame_duration=0.5):
  # Convert numpy arrays to PIL images
  frame_images = [Image.fromarray(frame) for frame in frames]
  # Save the animation as a GIF using imageio
  imageio.mimsave(f'{filename}.gif', frame_images, format='GIF', duration=frame_duration)

    
def plot_fourrooms_gpi_step(env, s_t, a_t, z_t, LRs, goals, rewards_remaining, Qs, t, axs=None):
    if axs is None:
        fig = plt.figure(figsize=(14, 4))

        # Define the overall grid for the figure
        gs = gridspec.GridSpec(2, 5, width_ratios=[4, 2, 4, 2, 2], height_ratios=[4, 4])

        # Define each axis using the overall grid
        ax1 = plt.subplot(gs[:, 0])
        ax2 = plt.subplot(gs[:, 1])
        ax3 = plt.subplot(gs[:, 2])
        ax4 = plt.subplot(gs[0, 3])
        ax5 = plt.subplot(gs[0, 4])
        ax6 = plt.subplot(gs[1, 3])
        ax7 = plt.subplot(gs[1, 4])

        # Adjust spacing between axes
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
    else:
        ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axs
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # goals = [14, 97]
    num_goals = len(goals)

    # axis 1 - show current location and goals
    title = r"$\lambda = 0.5$"
    env_im = env.plot_grid(ax=ax1, title=title, goals=goals, s_t=s_t, t=t)[0]
    ax1.set_xlabel(f"Return = {np.round(z_t, 2)}", fontsize=18)

    # axis 2 - show remaining reward in each goal state
    rew_hist = ax2.bar(np.arange(num_goals), rewards_remaining, width=0.35,
            color=[f"C{i}" for i in range(num_goals)], edgecolor='k', linewidth=5)
    ax2.set_xticks(np.arange(num_goals))
    ax2.set_xticklabels([r"$g_{{{}}}$".format(i) for i in range(num_goals)],
                        fontsize=15)
    ax2.set_ylim([0, np.max(env.goal_rewards)])
    ax2.set_title("Remaining Reward", fontsize=18)

    # axis 3 - show values of each policy in current state
    n_policies = len(Qs)
    val_hist = ax3.bar(np.arange(n_policies), Qs, width=0.35,
            color="C7", edgecolor='k', linewidth=5)
    ax3.set_xticks(np.arange(n_policies))
    ax3.set_xticklabels(
        [r"$v^{{{}}}(s_{{{}}})$".format(i, t) for i in range(n_policies)],
        fontsize=13)
    ax3.set_ylim([0, 0.12 * np.max(env.goal_rewards)/ (1 - env.discount)])
    ax3.set_title("Value Estimates", fontsize=18)

    # axes 4-7, show LRs for each policy
    lr_axes = [ax6, ax4, ax5, ax7]
    LRs = list(LRs)
    lr_ims = []
    for i, ax in enumerate(lr_axes):
        lr_ims.append(ax.imshow(LRs[i][s_t, a_t].reshape(11, 11), cmap=CMAP))
        # env.plot_grid(ax=ax, M=LRs[i][100, 0], cmap=cmap)
        ax.set_title(r"$\varphi^{{{}}}_{{{}}}$".format(i, t))
        ax.set_xticks([])
        ax.set_yticks([])
    return env_im, *rew_hist.patches, *val_hist.patches, *lr_ims 


def calculate_data_weights(dataset, alpha=1.0, use_states=True):
    # Count the frequency of each label in the dataset
    label_counter = Counter()
    for datum in dataset:
        state, obs = datum[1:3]
        if use_states:
           x = state.item()
        else:
           x = obs.numpy()
        label_counter[x] += 1

    # Calculate the total number of samples
    num_samples = len(dataset)

    # Weight each point by the inverse of its frequency
    label_weights = {}
    for label, count in label_counter.items():
        label_weights[label] = alpha * (num_samples / count) + (1 - alpha) #/ num_samples

    # Assign the weight to each sample based on its label
    sample_weights = []
    for datum in dataset:
        state, obs = datum[1:3]
        if use_states:
           x = state.item()
        else:
           x = obs.numpy()
        sample_weights.append(label_weights[x])

    return sample_weights


def record_episode_fourrooms(
  env, 
  episode_data,
  filename: str,
  fps: int = 1):
  """
  Use FuncAnimation to record an episode of the agent interacting with the
  environment.
  """
  fig = plt.figure(figsize=(14, 4))

  # Define the overall grid for the figure
  gs = gridspec.GridSpec(2, 5, width_ratios=[4, 2, 4, 2, 2], height_ratios=[4, 4])

  # Define each axis using the overall grid
  ax1 = plt.subplot(gs[:, 0])
  ax2 = plt.subplot(gs[:, 1])
  ax3 = plt.subplot(gs[:, 2])
  ax4 = plt.subplot(gs[0, 3])
  ax5 = plt.subplot(gs[0, 4])
  ax6 = plt.subplot(gs[1, 3])
  ax7 = plt.subplot(gs[1, 4])
  axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

  def animate(i):

    for ax in axs:
      ax.clear()

    plots = plot_fourrooms_gpi_step(env, *episode_data[i], axs=axs)

    return plots


  anim = FuncAnimation(fig, animate, frames=len(episode_data), blit=True)

  anim.save(filename, writer='ffmpeg', fps=fps)



def create_neuronav_gif(frame_and_map_sequence, rewards_remaining_sequence,  gif_filename, value_sequence=None, return_sequence=None, show_values=True, map_title='', contains_map=True, show_map=False, num_goals=3, max_reward=10, max_value=20, duration=300):
    frames = []
    t = 0
    if value_sequence is None:
        show_values=False
        value_sequence = copy.deepcopy(rewards_remaining_sequence)
    if return_sequence is None:
        return_sequence = [None] * len(frame_and_map_sequence)
    for frame_and_map, rewards_remaining, values, return_ in zip(frame_and_map_sequence, rewards_remaining_sequence, value_sequence, return_sequence):
        fig, _ = plot_neuronav_frame(frame_and_map, rewards_remaining, values, t=t, show_map=show_map, num_goals=num_goals, return_=return_,
                                       max_reward=max_reward, max_value=max_value, map_title=map_title, show_values=show_values, return_axs=True, contains_map=contains_map)

        # Save the figure to a BytesIO buffer and then convert it to a PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        frames.append(im)

        # Close the figure to prevent memory leaks
        plt.close(fig)

        t += 1

    # Save the sequence of PIL Images as a gif
    frames[0].save(gif_filename, save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)


def plot_neuronav_frame(frame_and_map, rewards_remaining, values, t=0, return_=None, show_map=False, num_goals=3, max_reward=10, max_value=20, map_title='', show_values=True, return_axs=False, contains_map=True):
    n_axes = 4 if show_map else 3
    n_axes -= 1 - int(show_values)
    value_width = 2 if len(values) < 3 else 4
    fig, axs = plt.subplots(1, n_axes, figsize=(4 * (n_axes - 1) + 2, 4), gridspec_kw={'width_ratios': [4] * (1 + int(show_map)) + [2] + int(show_values) * [value_width]})
    if contains_map:
      frame, map_ = np.split(frame_and_map, 2, axis=1)
    else:
      assert show_map == False, "if contains_map is False, show_map must be False"
      frame = frame_and_map

    axs[0].imshow(frame)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    if return_ is not None:
        map_title = f"Return = {np.round(return_, 2)}"
    axs[0].set_title(map_title, fontsize=22)

    if show_map:
        axs[1].imshow(map_)
        axs[1].set_xticks([])
        axs[1].set_yticks([])

    r_ax = -2 if show_values else -1
    axs[r_ax].bar(np.arange(len(rewards_remaining)), rewards_remaining, color=[f"C{i}" for i in range(num_goals)], edgecolor='k', linewidth=5, width=0.5)
    axs[r_ax].set_xticks(np.arange(num_goals))
    axs[r_ax].set_xticklabels([r"$g_{{{}}}$".format(i) for i in range(num_goals)], fontsize=15)
    axs[r_ax].set_ylim([0, max_reward])
    axs[r_ax].set_title("Remaining Reward", fontsize=18)

    # axis 3 - show values of each policy in current state
    if show_values:
      n_policies = len(values)
      val_hist = axs[-1].bar(np.arange(n_policies), values, width=0.35,
              color="C7", edgecolor='k', linewidth=5)
      axs[-1].set_xticks(np.arange(n_policies))
      axs[-1].set_xticklabels(
          [r"$v^{{{}}}(x_{{{}}})$".format(i, t) for i in range(n_policies)],
          fontsize=13)
      axs[-1].set_ylim([0, max_value])
      # if only one policy, widen xlimits byt 25% on each each
      if n_policies == 1:
        axs[-1].set_xlim([-0.65, 0.65])
      axs[-1].set_title("Value Estimates", fontsize=18)

    if return_axs:
        return fig, axs
    else:
        plt.show()

map_from_action_to_subplot = lambda a: (2, 6, 8, 4)[a]
map_from_action_to_name = lambda a: ("up", "right", "down", "left")[a]

def plot_values(grid, values, ax=None, colormap='pink', vmin=0, vmax=10, title=None, cbar=True, alpha=1.0, showticks=False, lb_scale=1_000):
  if ax is None:
      im = plt.imshow(values - lb_scale*(grid<0), interpolation="nearest", cmap=colormap, vmin=vmin, vmax=vmax, alpha=alpha)
      if not showticks:
        plt.yticks([])
        plt.xticks([])
      if title is not None: plt.title(title, fontsize=14)
      if cbar: plt.colorbar(ticks=[vmin, vmax], orientation="horizontal"); 
  else:
      im = ax.imshow(values - lb_scale*(grid<0), interpolation="nearest", cmap=colormap, vmin=vmin, vmax=vmax, alpha=alpha)
      if not showticks:
        ax.set_yticks([])
        ax.set_xticks([])
      #ax.grid()
      if title is not None: ax.set_title(title, fontsize=14)
      if cbar: plt.colorbar(im, ticks=[vmin, vmax], ax=ax, orientation="horizontal"); 

  return im


def plot_action_values(grid, action_values, vmin=-5, vmax=5, c=['r']):
  q = action_values
  if len(c) != grid.size:
    #pdb.set_trace()
    colors = [c[0]] * grid.size
  else: colors = c; 
  colors = np.reshape(np.array(colors), grid.shape)
  fig = plt.figure(figsize=(10, 10))
  fig.subplots_adjust(wspace=0.3, hspace=0.3)
  for a in [0, 1, 2, 3]:
    ax = plt.subplot(4, 3, map_from_action_to_subplot(a))
    plot_values(grid, q[..., a], ax=None, vmin=vmin, vmax=vmax)
    action_name = map_from_action_to_name(a)
    plt.title(r"$q(x, \mathrm{" + action_name + r"})$")
    
  plt.subplot(4, 3, 5)
  v = np.max(q, axis=-1)
  plot_values(grid, v, colormap='summer', vmin=vmin, vmax=vmax)
  plt.title("$v(x)$")
  
  # Plot arrows:
  plt.subplot(4, 3, 11)
  plot_values(grid, grid>=0, vmax=1)
  for row in range(len(grid)):
    for col in range(len(grid[0])):
      if grid[row][col] >= 0:
        argmax_a = np.argmax(q[row, col])
        if argmax_a == 0:
          x = col
          y = row + 0.5
          dx = 0
          dy = -0.8
        if argmax_a == 1:
          x = col - 0.5
          y = row
          dx = 0.8
          dy = 0
        if argmax_a == 2:
          x = col
          y = row - 0.5
          dx = 0
          dy = 0.8
        if argmax_a == 3:
          x = col + 0.5
          y = row
          dx = -0.8
          dy = 0
        if argmax_a != 4:
          plt.arrow(
            x, y, dx, dy,width=0.02, head_width=0.3, head_length=0.4,
            length_includes_head=True, fc=colors[row,col], ec=colors[row,col]
            )
        elif argmax_a == 4:
           plt.text(col, row, r"$\cdot$", fontsize=20, color=colors[row,col], ha='center', va='center')


def plot_actions(grid, action_values, ax=None, title=None, vmin=-5, vmax=5, c=['r']):
  if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(4, 4))
  q = action_values
  # Plot arrows:
  plot_values(grid, grid>=0, ax=ax, vmax=1, cbar=False)
  if len(c) != grid.size:
    colors = [c[0]] * grid.size
  else: colors = c; 
  colors = np.reshape(np.array(colors), grid.shape)
  idx = 0
  for row in range(len(grid)):
    for col in range(len(grid[0])):
      idx += 1
      if grid[row][col] >= 0:
        argmax_a = np.argmax(q[row, col])
        if argmax_a == 0:
          x = col
          y = row + 0.5
          dx = 0
          dy = -0.8
        if argmax_a == 1:
          x = col - 0.5
          y = row
          dx = 0.8
          dy = 0
        if argmax_a == 2:
          x = col
          y = row - 0.5
          dx = 0
          dy = 0.8
        if argmax_a == 3:
          x = col + 0.5
          y = row
          dx = -0.8
          dy = 0

        if argmax_a != 4:
          ax.arrow(
            x, y, dx, dy,width=0.02, head_width=0.3, head_length=0.4,
            length_includes_head=True, fc=colors[row,col], ec=colors[row,col]
            )
        elif argmax_a == 4:
           ax.text(col, row, r"$\cdot$", fontsize=20, color=colors[row,col], ha='center', va='center')
  if title is not None: 
    ax.set_title(title, fontsize=30, c=colors[0, 0])

  h, w = grid.shape
  for y in range(h-1):
    ax.plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
  for x in range(w-1):
    ax.plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)


def plot_rewards(xs, rewards, color):
  mean = np.mean(rewards, axis=0)
  p90 = np.percentile(rewards, 90, axis=0)
  p10 = np.percentile(rewards, 10, axis=0)
  plt.plot(xs, mean, color=color, alpha=0.6)
  plt.fill_between(xs, p90, p10, color=color, alpha=0.3)


def colorline(x, y, z):
    """
    Based on:
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=plt.get_cmap('copper_r'),
                              norm=plt.Normalize(0.0, 1.0), linewidth=3)

    ax = plt.gca()
    ax.add_collection(lc)
    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

  

def errorfill(x, y, yerr, color='C0', alpha_fill=0.3, ax=None, label=None, lw=1, marker=None):
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label, lw=lw, marker=marker)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.7)
    #ax.legend(fontsize=13)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


class DmEnvWrapper(dm_env.Environment):
    def __init__(self, env, objects=None, terminate_on_reward=False, random_start=False, discount=0.99):
        self._env = env
        self._objects = objects
        self._terminate_on_reward = terminate_on_reward
        self._random_start = random_start
        self.discount = discount

    def reset(self):
        next_state = self._env.reset(
           objects=self._objects,
           terminate_on_reward=self._terminate_on_reward,
           random_start=self._random_start)
        return dm_env.restart(next_state)

    def step(self, action):
        next_state, reward, done, _ = self._env.step(action)
        if done:
            return dm_env.termination(reward, next_state)
        else:
            return dm_env.transition(reward, next_state, self.discount)

    def observation_spec(self):
        return specs.Array(shape=self._env.obs_space.shape, dtype=np.float32)

    def action_spec(self):
        return specs.DiscreteArray(num_values=self._env.action_space.n, dtype=int)


