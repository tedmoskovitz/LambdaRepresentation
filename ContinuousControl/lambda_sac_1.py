import copy

import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution

from bandit import ExpWeights
from utils import ReplayPool, TanhTransform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLPNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )
    
    def forward(self, x):
        return self.network(x)

class TwoHeadedMLPNetwork(nn.Module):
        
    def __init__(self, input_dim, output_dim_1, output_dim_2, hidden_size=256):
        super(TwoHeadedMLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),)
        self.head1 = nn.Linear(hidden_size, output_dim_1)
        self.head2 = nn.Linear(hidden_size, output_dim_2)
    
    def forward(self, x):
        x = self.network(x)
        return self.head1(x), self.head2(x)



class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        return action, logprob, mean


class DoubleLambdaQFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, feature_dim, hidden_size=256):
        super(DoubleLambdaQFunc, self).__init__()
        self.network1 = TwoHeadedMLPNetwork(
            state_dim + action_dim, 1, feature_dim, hidden_size)
        self.network2 = TwoHeadedMLPNetwork(
            state_dim + action_dim, 1, feature_dim, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)
    
class LambdaQFunc(nn.Module):

    def __init__(self, state_dim, action_dim, feature_dim, hidden_size=256):
        super(LambdaQFunc, self).__init__()
        self.network1 = TwoHeadedMLPNetwork(
            state_dim + action_dim, 1, feature_dim, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q, lambdaf = self.network1(x)
        return (q, lambdaf), (q.detach() + 1, lambdaf.detach() + 1)


class DoubleQFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)
    
class QFunc(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(QFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        out = self.network1(x)
        return out, out


class LambdaFunc(nn.Module):

    def __init__(self, state_dim, action_dim, feature_dim, hidden_size=256):
        super(LambdaFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, feature_dim, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        out = self.network1(x)
        return out, out
    
class DoubleLambdaFunc(nn.Module):

    def __init__(self, state_dim, action_dim, feature_dim, hidden_size=256):
        super(DoubleLambdaFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, feature_dim, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, feature_dim, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


class SeparatedLambdaQFunc(nn.Module):

    def __init__(self, state_dim, action_dim, feature_dim, hidden_size=256):
        super(SeparatedLambdaQFunc, self).__init__()
        self.qnetwork1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.lfnetwork1 = MLPNetwork(state_dim + action_dim, feature_dim, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q = self.qnetwork1(x)
        lf = self.lfnetwork1(x)
        return (q, lf), (q.detach() + 1, lf.detach() + 1)
    
class SeparatedDoubleLambdaQFunc(nn.Module):

    def __init__(self, state_dim, action_dim, feature_dim, hidden_size=256):
        super(SeparatedDoubleLambdaQFunc, self).__init__()
        self.qnetwork1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.lfnetwork1 = MLPNetwork(state_dim + action_dim, feature_dim, hidden_size)
        self.qnetwork2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.lfnetwork2 = MLPNetwork(state_dim + action_dim, feature_dim, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q1 = self.qnetwork1(x)
        lf1 = self.lfnetwork1(x)
        q2 = self.qnetwork2(x)
        lf2 = self.lfnetwork2(x)
        return (q1, lf1), (q2, lf2)

class LambdaSAC_Agent:

    def __init__(
            self,
            seed,
            state_dim,
            action_dim,
            feature_dim,
            use_one_critic=False,
            lr=3e-4,
            gamma=0.99,
            tau=5e-3,
            batchsize=256,
            hidden_size=256,
            update_interval=1,
            buffer_size=int(1e6),
            bandit_temp_annealing_factor=1.0,
            lf_wt=1.0,
            target_entropy=None,
            bandit_lr=0.1,
            **kwargs):
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy if target_entropy else -action_dim
        self.batchsize = batchsize
        self.update_interval = update_interval
        self.lf_wt = lf_wt

        torch.manual_seed(seed)

        # bandit top-down controller
        self.TDC = ExpWeights(
            arms=[0.0, 0.5, 1.0],
            lr=bandit_lr,
            init=0.0,
            use_std=True,
            inverse_temp=1.0,
            temp_annealing_factor=bandit_temp_annealing_factor) 

        # aka critic
        if use_one_critic:
            self.lfq_funcs = SeparatedLambdaQFunc(
                state_dim, action_dim, feature_dim, hidden_size=hidden_size).to(device)
        else:
            self.lfq_funcs = SeparatedDoubleLambdaQFunc(
                state_dim, action_dim, feature_dim, hidden_size=hidden_size).to(device)
        self.target_lfq_funcs = copy.deepcopy(self.lfq_funcs)
        self.target_lfq_funcs.eval()
        self.use_one_critic = use_one_critic
        for p in self.target_lfq_funcs.parameters():
            p.requires_grad = False

        print("Lambda-Q Functions:")
        print(self.lfq_funcs)

        # aka actor
        self.policy = Policy(state_dim, action_dim, hidden_size=hidden_size).to(device)

        print("Policy:")
        print(self.policy)

        # aka temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        self.lf_optimizer = torch.optim.Adam(self.lfq_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.replay_pool = ReplayPool(capacity=buffer_size)
    
    def get_action(self, state, state_filter=None, deterministic=False):
        if state_filter:
            state = state_filter(state)
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(device))

        if deterministic:
            return mean.squeeze().cpu().numpy()
        return np.atleast_1d(action.squeeze().cpu().numpy())

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_lf_param, lf_param in zip(self.target_lfq_funcs.parameters(), self.lfq_funcs.parameters()):
                target_lf_param.data.copy_(self.tau * lf_param.data + (1.0 - self.tau) * target_lf_param.data)

    def update_lfq_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch, lambda_):
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
            (q_t1, lf_t1), (q_t2, lf_t2) = self.target_lfq_funcs(nextstate_batch, nextaction_batch)
            features_batch = torch.cat((state_batch, action_batch), dim=1).detach()  # [batch_size, dim(s) + dim(s)]
            min_feature, max_feature = features_batch.min(), features_batch.max()
            # normalize
            features_batch = (features_batch - min_feature) / (max_feature - min_feature)
            w = torch.pinverse(features_batch) @ reward_batch
            # r_mse for measurement purposes only
            r_mse = F.mse_loss(features_batch @ w, reward_batch).detach().item()
            
            if self.use_one_critic:
                q_target = q_t1
                lf_t = lf_t1
            else:
                q_target = torch.min(q_t1, q_t2)
                q_target_idxs = torch.argmin(torch.cat((q_t1, q_t2), dim=1), dim=1)
                lf_t = torch.where(q_target_idxs == 0, lf_t1, lf_t2)

            value_target = reward_batch + self.gamma * (1.0 - done_batch) * (
                q_target + (lambda_ - 1) * torch.matmul(
                features_batch * lf_t, w) - self.alpha * logprobs_batch)
            
            lf_target = features_batch * (1 + lambda_ * self.gamma * (1.0 - done_batch) * lf_t)
            lf_target += self.gamma * (1 - features_batch) * (1.0 - done_batch) * lf_t

        (q_1, lf_1), (q_2, lf_2) = self.lfq_funcs(state_batch, action_batch)

        q_1_loss = F.mse_loss(q_1, value_target)
        lf_1_loss = F.mse_loss(lf_1, lf_target)
        
        if self.use_one_critic:
            q_2_loss = torch.zeros_like(
                q_1_loss, dtype=q_1_loss.dtype, device=q_1_loss.device)
            lf_2_loss = torch.zeros_like(
                lf_1_loss, dtype=lf_1_loss.dtype, device=lf_1_loss.device)
        else:
            q_2_loss = F.mse_loss(q_2, value_target)
            lf_2_loss = F.mse_loss(lf_2, lf_target)
        
        return q_1_loss, lf_1_loss, q_2_loss, lf_2_loss, r_mse, value_target

    def update_policy_and_temp(self, state_batch):
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        (q_b1, _), (q_b2, _) = self.lfq_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss

    def optimize(self, n_updates, lambda_=1.0, state_filter=None):
        pi_loss, a_loss, lf1_loss, lf2_loss, q1_loss, q2_loss = 0, 0, 0, 0, 0, 0
        r_mse, value_target_mean, value_target_std = 0, 0, 0
        for i in range(n_updates):
            samples = self.replay_pool.sample(self.batchsize)

            if state_filter:
                state_batch = torch.FloatTensor(state_filter(samples.state)).to(device)
                nextstate_batch = torch.FloatTensor(state_filter(samples.nextstate)).to(device)
            else:
                state_batch = torch.FloatTensor(samples.state).to(device)
                nextstate_batch = torch.FloatTensor(samples.nextstate).to(device)
            action_batch = torch.FloatTensor(samples.action).to(device)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(samples.done).to(device).unsqueeze(1)

            
            # update lambda-q-funcs
            q1_loss_step, lf1_loss_step, q2_loss_step, lf2_loss_step, r_mse_step, value_target = self.update_lfq_functions(
                state_batch, action_batch, reward_batch, nextstate_batch, done_batch, lambda_)
            lf_loss_step = q1_loss_step + q2_loss_step + self.lf_wt * (lf1_loss_step + lf2_loss_step)
            self.lf_optimizer.zero_grad()
            lf_loss_step.backward()
            self.lf_optimizer.step()

            # update policy and temperature parameter
            for p in self.lfq_funcs.parameters():
                p.requires_grad = False
            pi_loss_step, a_loss_step = self.update_policy_and_temp(state_batch)
            self.policy_optimizer.zero_grad()
            pi_loss_step.backward()
            self.policy_optimizer.step()
            self.temp_optimizer.zero_grad()
            a_loss_step.backward()
            self.temp_optimizer.step()
            for p in self.lfq_funcs.parameters():
                p.requires_grad = True

            lf1_loss += lf1_loss_step.detach().item()
            lf2_loss += lf2_loss_step.detach().item()
            q1_loss += q1_loss_step.detach().item()
            q2_loss += q2_loss_step.detach().item()
            pi_loss += pi_loss_step.detach().item()
            a_loss += a_loss_step.detach().item()
            r_mse += r_mse_step
            value_target_mean += value_target.mean().detach().item()
            value_target_std += value_target.std().detach().item()
            if i % self.update_interval == 0:
                self.update_target()
        return q1_loss, q2_loss, pi_loss, a_loss, lf1_loss, lf2_loss, r_mse / n_updates, value_target_mean / n_updates, value_target_std / n_updates

    @property
    def alpha(self):
        return self.log_alpha.exp()