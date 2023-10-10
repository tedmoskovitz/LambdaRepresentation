import numpy as np
import argparse
import copy
import torch as tr
from torch import nn
import torch.nn.functional as F
from nn_utils import weight_init
from utils import VectorTargetDataset, set_seed_everywhere, save_results
from fqi_utils import fit_dataset, construct_bootstrap_target
from envs import BasicGrid
from agents import Pi1Agent
from lfs_dp import compute_LF


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    return parser

class LambdaQNet(nn.Module):

    def __init__(self, obs_dim, num_actions, w, hidden_size=32, feature_dim=20, lr=3e-4, device='cpu', wt_decay=0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_actions * feature_dim),)
        self.device = tr.device(device)
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.w = tr.from_numpy(w).float().to(self.device)
        
        self.apply(weight_init)
        
        self.optimizer = tr.optim.Adam(self.parameters(), lr=lr)

        
    def forward(self, obs):
        return self.model(obs).reshape(-1, self.num_actions, self.feature_dim) @ self.w
    
    def get_features(self, obs):
        return self.model(obs).reshape(-1, self.num_actions, self.feature_dim)
    
    def update(self, obs_batch, target_batch):
        self.optimizer.zero_grad()
        obs_batch = obs_batch.float().to(self.device)
        target_batch = target_batch.float().to(self.device)
        preds_batch = self.forward(obs_batch).squeeze()
        assert preds_batch.shape == target_batch.shape, "pred and target must have same shape"
        
        loss = F.smooth_l1_loss(preds_batch, target_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

def run_fqi0(network_class, network_kwargs, env, policy, lambda_, fit_kwargs, q_true=None, num_iters=10, LFs_true=None):
    # initialize network
    S, A = env._number_of_states, env._number_of_actions
    w = env.r
    network = network_class(S, A, env.r, **network_kwargs)

    losses_hist, q_mse_hist, q_cos_hist, q_norm_mse_hist, q_l2_hist = [], [], [], [], []
    for k in range(num_iters):
        print(f"\nfqi iteration {k+1}/{num_iters} ==========")
        # Phase 1: Dataset Construction
        network.eval()
        LFs = network.get_features(tr.eye(S).float()).detach().numpy()
        q_targ = construct_bootstrap_target(env, policy, LFs, lambda_, env.discount)
        q_targ = np.clip(q_targ, -1, 1)
        dataset = VectorTargetDataset(q_targ)
        network.train()

        # Optionally, evaluate current q wrt true q
        if q_true is not None:
            predicted_q = LFs @ w
            q_err = np.mean((q_true - predicted_q)**2)
            # cosine similarity
            predicted_q_norm = predicted_q.reshape(-1)
            q_pred_l2 = np.linalg.norm(predicted_q_norm)
            q_l2_hist.append(q_pred_l2)
            predicted_q_norm /= q_pred_l2

            q_true_norm = copy.copy(q_true).reshape(-1)
            q_true_l2 = np.linalg.norm(q_true_norm)
            q_true_norm /= q_true_l2

            cos_sim = np.dot(predicted_q_norm, q_true_norm)
            q_norm_mse = np.mean((predicted_q_norm - q_true_norm)**2)
            q_cos_hist.append(cos_sim)
            q_mse_hist.append(q_err)
            q_norm_mse_hist.append(q_norm_mse)
            print(f"q error: {q_err}, cos sim: {cos_sim}, q norm mse: {q_norm_mse}, q l2: {q_pred_l2}")

        # Phase 2: Fit network
        network = network_class(S, A, env.r, **network_kwargs)
        network, losses_k = fit_dataset(network, dataset, **fit_kwargs)
        losses_hist.append(losses_k)

    metrics = dict(
        losses_hist=losses_hist,
        q_mse_hist=q_mse_hist,
        q_norm_mse=q_norm_mse_hist,
        q_l2_hist=q_l2_hist,
        q_cos_hist=q_cos_hist,)

    return network, metrics


def main(args):
    set_seed_everywhere(args.seed)
    env = BasicGrid()
    policy = Pi1Agent(env._number_of_states, env.reset()).q
    S, A = env._number_of_states, env._number_of_actions
    LFs = np.zeros((S, A, S))
    lambda_ = 0.0
    LFs, _, _ = compute_LF(
        LFs, np.eye(S), env.P, policy, lambda_, 0.9, 100, tol=1e-5)
    q_true = LFs @ env.r

    # set up run
    network_class = LambdaQNet
    lambda_ = 0.0
    device = 'cpu' if not tr.cuda.is_available() else 'cuda'
    network_kwargs = dict(hidden_size=128, feature_dim=S, lr=3e-4, device=device)
    fit_kwargs = dict(num_epochs=1_000, batch_size=S * A, log_every=50)

    num_iters = 20
    _, metrics = run_fqi0(
        network_class,
        network_kwargs,
        env,
        policy,
        lambda_,
        fit_kwargs,
        q_true=q_true,
        num_iters=num_iters,
        LFs_true=LFs)

    save_results(f"fqi_{args.seed}", metrics)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)

