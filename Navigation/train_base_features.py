import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import time
import torch as tr
from torch.utils.data.sampler import WeightedRandomSampler
import wandb

from envs import FourRooms
from networks import BaseFeatureNet
from utils import TupleDataset, load_results, preprocess, set_seed_everywhere, calculate_data_weights

def create_parser():
  parser = argparse.ArgumentParser()

  # Basic settings
  parser.add_argument('--base_folder', default='./saved/base_features/', type=str, help='Path to base folder to store runs')
  parser.add_argument('--use_wandb', action='store_true', help="Defaults to false. If specified, log to wandb.")
  parser.add_argument('--run_name', default='run', type=str, help="Run name")
  parser.add_argument('--seed', default=0, type=int, help='Random seed')
  parser.add_argument('--log_every', default=5, type=int, help='Log every log_every steps')
  parser.add_argument('--save_every', default=20, type=int, help='Save every save_every epochs')

  # Model params
  parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
  parser.add_argument('--z_dim', default=20, type=int, help="dimension of feature space")
  parser.add_argument('--hidden_dim', default=32, type=int, help="dimension of hidden layer")
  # parser.add_argument('--train_steps', default=100000, type=int, help="number of training steps")
  parser.add_argument('--num_epochs', default=100, type=int, help="number of epochs")
  parser.add_argument('--batch_size', default=32, type=int, help="batch size")
  parser.add_argument('--final_nl', default='L2', type=str, help="nonlinearity for final layer, supports [L2, tanh, sigmoid]")

  # Dataset params
  parser.add_argument('--grayscale', action='store_true', help="Defaults to false. If specified, convert images to grayscale.")
  parser.add_argument('--weight_samples', action='store_true', help="Defaults to false. If specified, weight samples by inverse frequency.")
  parser.add_argument('--sample_alpha', default=1.0, type=float, help="alpha parameter for weighted sampling")
  parser.add_argument('--obs_type', default='3d', type=str, help="observation type (3d, cropped3d, or visual)")

  return parser



def main(args):
  device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
  set_seed_everywhere(args.seed)
  if args.use_wandb:
    wandb.login()
    wandb.init(
      project="lambda-representation_base-features",
      config=args,
      name=args.run_name)
  
  # Create dataset
  data_dict = load_results(f"fourrooms_{args.obs_type}-obs_dataset")
  data = dict(
    prev_actions=[], states=[], observations=[], actions=[], next_states=[], next_observations=[])
  for room in [0, 1, 2, 3]:
    room_data = preprocess(
      data_dict,
      room=room,
      convert_to_grayscale=args.grayscale,
      device=device)
    data["prev_actions"].append(room_data["prev_actions"])
    data["states"].append(room_data["states"])
    data["observations"].append(room_data["observations"])
    data["actions"].append(room_data["actions"])
    data["next_states"].append(room_data["next_states"])
    data["next_observations"].append(room_data["next_observations"])

  data["prev_actions"] = tr.cat(data["prev_actions"], dim=0)
  data["states"] = tr.cat(data["states"], dim=0)
  data["observations"] = tr.cat(data["observations"], dim=0)
  data["actions"] = tr.cat(data["actions"], dim=0)
  data["next_states"] = tr.cat(data["next_states"], dim=0)
  data["next_observations"] = tr.cat(data["next_observations"], dim=0)
  dataset = TupleDataset(data)


  unique_states, unique_states_idxs = np.unique(data['states'], return_index=True)
  unique_states_observations = data['observations'][unique_states_idxs]
  unique_states_to_observations = {s: o for s, o in zip(unique_states, unique_states_observations)}

  # Create dataloader
  if args.weight_samples:
    weights = calculate_data_weights(dataset, alpha=args.sample_alpha)
    dataloader = tr.utils.data.DataLoader(
      dataset,
      batch_size=args.batch_size,
      sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True))
  else:
    dataloader = tr.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

  # Create model
  _, _, obs, _, _, _ = dataset[0]
  model = BaseFeatureNet(
    obs.shape, 5, args.z_dim, args.hidden_dim, obs_type='pixels', final_nl=args.final_nl)
  model.to(device)
  optimizer = tr.optim.Adam(model.parameters(), lr=args.lr)

  goals = [80, 24, 41]
  goal_rewards = [5, 10, 5]
  env_lambda_ = 0.5
  discount = 0.95
  og_env = FourRooms(
      goals=goals,
      start_state=-1,  # random start state
      goal_rewards=goal_rewards,
      lambda_=env_lambda_,
      discount=discount)
  
  r_vec = og_env.r
  r = np.array([r_vec[int(s.item())] for s in unique_states.reshape(-1)])[:, None]  #  [B, 1]
  best_r_err = float(np.inf)

  # Train
  for epoch in range(1, args.num_epochs+1):
    print(f"Epoch: {epoch}/{args.num_epochs}")

    for i, batch in enumerate(dataloader):
      optimizer.zero_grad()
      _, _, obs, action, _, next_obs = batch
      loss = model(obs, action, next_obs)
      loss.backward()
      optimizer.step()

      if i % args.log_every == 0:
        Phi = model.get_features(unique_states_observations).detach().numpy()  # [B, D]
        w_opt = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ r  # [D, 1]
        r_pred = r_vec.copy()
        for idx, s in enumerate(unique_states.reshape(-1)):
          r_pred[int(s)] = (Phi[idx] @ w_opt)[0]
        r_err = np.mean((Phi @ w_opt - r)**2)
        print(f"Step {i}, Loss: {loss.item()}, R-err: {r_err}")
        if args.use_wandb:
          r_pred_im = wandb.Image(r_pred.reshape(11, 11, 1), caption="Predicted Reward")
          r_true_im = wandb.Image(r_vec.reshape(11, 11, 1), caption="True Reward")
          wandb.log({'loss': loss.item(), 'r_err': r_err, 'step': epoch + i, 'r_pred': r_pred_im, 'r_true': r_true_im})

        if r_err < 0.95 * best_r_err and args.save_every > 0:
          print("Saving best model and optimizer states...")
          if not os.path.exists(f"{args.base_folder}/{args.run_name}"):
            os.makedirs(f"{args.base_folder}/{args.run_name}")
          tr.save(model.state_dict(), f"{args.base_folder}/{args.run_name}/model.pth")
          tr.save(optimizer.state_dict(), f"{args.base_folder}/{args.run_name}/optimizer.pth")
          np.savetxt(f"{args.base_folder}/{args.run_name}/w_opt.csv", w_opt, delimiter=",")
          print("Done.")
          best_r_err = r_err

if __name__ == "__main__":
  parser = create_parser()
  args = parser.parse_args()
  start = time.time()
  main(args)
  end = time.time()
  print("Total training time: ", (end - start) / 60, " minutes")








