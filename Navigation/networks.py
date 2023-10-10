import copy
from torch import nn
import torch as tr
import torch.nn.functional as F
from typing import Any

from nn_utils import weight_init, mlp


class Encoder(nn.Module):
    def __init__(self, obs_shape) -> None:
        super().__init__()

        assert len(obs_shape) == 3
        c, h, w = obs_shape
        self.repr_dim = 32 * (h // 8 - 3) * (w // 8 - 3)

        self.convnet = nn.Sequential(nn.Conv2d(c, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Flatten(),
                                     nn.LayerNorm(self.repr_dim),
                                     nn.Tanh())

        self.apply(weight_init)

    def forward(self, obs) -> Any:
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        return h


class FeatureLearner(nn.Module):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, final_nl="L2") -> None:
        super().__init__()
        del action_dim
        self.feature_net: nn.Module = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, final_nl)            
        self.apply(weight_init)

    def forward(self, obs: tr.Tensor, action: tr.Tensor, next_obs: tr.Tensor, future_obs: tr.Tensor):
        del obs, action, next_obs, future_obs
        return None


class Laplacian(FeatureLearner):
    def forward(self, obs: tr.Tensor, action: tr.Tensor, next_obs: tr.Tensor):
        del action
        phi = self.feature_net(obs)
        next_phi = self.feature_net(next_obs)
        loss = (phi - next_phi).pow(2).mean()
        Cov = tr.matmul(phi.T, phi)  # [d, 1] x [1, d] = [d, d]
        I = tr.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        return loss
    
    
class BaseFeatureNet(nn.Module):
    
    def __init__(self, obs_shape, action_dim, z_dim, hidden_dim, obs_type='pixels', final_nl="L2", feature_type='laplacian'):
        super().__init__()
        if obs_type == 'pixels':
            self.encoder = Encoder(obs_shape)
            obs_dim = self.encoder.repr_dim
        else:
            self.encoder = nn.Identity()
            obs_dim = obs_shape[-1]
        
        if feature_type == 'laplacian':
            self.feature_learner = Laplacian(
                obs_dim, action_dim, z_dim, hidden_dim, final_nl=final_nl)
        else:
            raise NotImplementedError
        
    def get_features(self, obs):
        h = self.encoder(obs)
        return self.feature_learner.feature_net(h)
        
    def forward(self, obs, action, next_obs):
        h = self.encoder(obs)
        next_h = self.encoder(next_obs)
        return self.feature_learner(h, action, next_h)

    
class FeatureNet2(nn.Module):

    def __init__(self, obs_shape, action_dim, z_dim, hidden_dim, obs_type='pixels'):
        super().__init__()

        c, h, w = obs_shape
        self.obs_type = obs_type
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.repr_dim = 32 * (h // 8 - 3) * (w // 8 - 3)
        
        self.obs_encoder = nn.Sequential(nn.Conv2d(c, 32, 3, stride=4),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2),
                                    #  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Flatten(),
                                     nn.Linear(self.repr_dim, hidden_dim),
                                     nn.LayerNorm(hidden_dim),
                                     nn.Tanh(),)
        
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim * action_dim))
        
        self.apply(weight_init)

    def forward(self, obs, prev_action=None):
        if self.obs_type == 'pixels':
            obs = obs / 255.0 - 0.5
        x = self.obs_encoder(obs)
        # pdb.set_trace()
        x = tr.cat([x, prev_action], dim=1)
        return self.feature_net(x).reshape(-1, self.z_dim, self.action_dim)
    

class LambdaFeatureNet(nn.Module):

    def __init__(
            self,
            policy,
            obs_shape,
            action_dim,
            z_dim,
            hidden_dim,
            tau=0.005,
            discount=0.97,
            lambda_=1.0,
            target_update_freq=1,
            obs_type='pixels',
            network='rnn'):
        super().__init__()

        self.obs_type = obs_type
        self.policy = tr.from_numpy(policy).float()


        self.feature_net = FeatureNet(
            obs_shape, action_dim, z_dim, hidden_dim, obs_type)
        print(self.feature_net)

        self.target_net = copy.deepcopy(self.feature_net)
        self.apply(weight_init)
        self.tau = tau
        self.discount = discount
        self.lambda_ = lambda_
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.n_updates = 0
        self.target_update_freq = target_update_freq


    def update_target(self) -> None:
        """moving average update of target networks"""
        with tr.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.feature_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def forward(self, obs, action, next_obs):
        del action, next_obs
        return self.feature_net(obs)
    
    def update(
            self,
            state_batch,
            obs_batch,
            feature_batch,
            action_batch,
            next_state_batch,
            next_obs_batch,
            next_feature_batch,):
        # compute lambda-feature target
        del state_batch, next_feature_batch
        with tr.no_grad():
            # [B, z_dim, A]
            next_lf_batch = self.target_net(
                next_obs_batch).reshape(-1, self.z_dim, self.action_dim)
            next_action_idxs = tr.multinomial(
                self.policy[next_state_batch.long()], num_samples=1)
            # [B, 1] -> [B, z_dim, 1]
            next_action_idxs = tr.tile(
                next_action_idxs, (1, self.z_dim)).unsqueeze(-1)
            # [B, z_dim, 1] -> [B, z_dim]
            next_lf_batch_action = next_lf_batch.gather(
                2, next_action_idxs).squeeze(-1)
            # [B, z_dim]
            target = feature_batch + self.discount * next_lf_batch_action

        # compute lambda-feature loss
        # [B, A] -> [B, 1]
        action_idxs = tr.argmax(action_batch, dim=1, keepdim=True)
        action_idxs_expanded = tr.tile(
            action_idxs, (1, self.z_dim)).unsqueeze(-1)
        # select the lambda-feature for the action taken (still [B, z_dim, A]])
        lf_batch = self.feature_net(obs_batch).reshape(
            -1, self.z_dim, self.action_dim).gather(2, action_idxs_expanded)
        loss = F.smooth_l1_loss(lf_batch.squeeze(), target)

        stats = dict(
            target_mean=target.mean().item(),
            target_std=target.std().item(),
            target_max=target.max().item(),
            target_min=target.min().item(),
            lf_mean=lf_batch.mean().item(),
            lf_std=lf_batch.std().item(),
            lf_max=lf_batch.max().item(),
            lf_min=lf_batch.min().item(),
            max_td_error=(lf_batch.squeeze() - target).abs().max().item(),)

        
        self.n_updates += 1
        if self.n_updates % self.target_update_freq == 0:
            self.update_target()

        return loss, stats



    def update_traj(
            self,
            state_batch,
            obs_batch,
            feature_batch,
            action_batch,
            next_state_batch,
            next_obs_batch,
            next_feature_batch,
            feature_returns_batch):
        # compute lambda-feature target
        del state_batch, next_feature_batch, feature_batch, next_state_batch, next_obs_batch
        # obs_batch is [B, T, C, H, W]
        B, T, C, H, W = obs_batch.shape
        # action_batch is [B, T, A]
        # feature_returns_batch is [B, T, z_dim]

        # compute lambda-feature loss
        # [B, T, A] -> [B, T, 1]
        action_idxs = tr.argmax(action_batch, dim=2, keepdim=True)
        # [B, T, 1] -> [B*T, 1]
        action_idxs = action_idxs.reshape(B*T, 1)
        # [B*T, 1] -> [B*T, z_dim, 1]
        action_idxs_expanded = tr.tile(
            action_idxs, (1, self.z_dim)).unsqueeze(-1)
        # previous actions [B, T, A]
        prev_acts0 = tr.zeros((B, 1, self.action_dim), device=obs_batch.device).float()
        prev_acts0[:, :, 0] = 1.0
        prev_actions = tr.cat(
            [prev_acts0, action_batch[:, :-1, :]], dim=1)  # concatenate
        # reshape to [B*T, A]
        prev_actions = prev_actions.reshape(B*T, self.action_dim)
        # select the lambda-feature for the action taken ([B*T, z_dim]])
        lf_batch = self.feature_net(
            obs_batch, prev_actions,
            )[0].gather(2, action_idxs_expanded).reshape(B, T, self.z_dim)
        
        loss = F.smooth_l1_loss(lf_batch, feature_returns_batch.detach())
        

        stats = dict(
            lf_mean=lf_batch.mean().item(),
            lf_std=lf_batch.std().item(),
            lf_max=lf_batch.max().item(),
            lf_min=lf_batch.min().item(),)

        
        self.n_updates += 1
        if self.n_updates % self.target_update_freq == 0:
            self.update_target()

        return loss, stats


class LambdaValueNet(nn.Module):

    def __init__(
            self,
            policy,
            z,
            q,
            obs_shape,
            action_dim,
            z_dim,
            hidden_dim,
            tau=0.005,
            discount=0.97,
            lambda_=1.0,
            target_update_freq=1,
            use_prev_action=False,
            obs_type='pixels',
            network='rnn'):
        super().__init__()

        
        self.obs_type = obs_type
        self.q = tr.from_numpy(q).float()
        self.policy = tr.from_numpy(policy).float()

        self.feature_net = FeatureNet(
            obs_shape, action_dim, z_dim, hidden_dim, obs_type, use_actions=use_prev_action)
        print(self.feature_net)

        self.target_net = copy.deepcopy(self.feature_net)
        self.apply(weight_init)
        self.tau = tau
        self.discount = discount
        self.lambda_ = lambda_
        self.z = tr.from_numpy(z).float()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.n_updates = 0
        self.target_update_freq = target_update_freq
        self.use_prev_action = use_prev_action


    def update_target(self) -> None:
        """moving average update of target networks"""
        with tr.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.feature_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def forward(self, obs, action, next_obs, prev_action=None):
        del action, next_obs
        return self.feature_net(obs, prev_action=prev_action)
    
    def update(
            self,
            prev_action_batch,
            state_batch,
            obs_batch,
            feature_batch,
            action_batch,
            next_state_batch,
            next_obs_batch,
            next_feature_batch,):
        # compute lambda-feature target
        del next_feature_batch
        with tr.no_grad():
            # [B, z_dim, A]
            next_lf_batch = self.target_net(
                next_obs_batch, action_batch).reshape(-1, self.z_dim, self.action_dim)
            next_action_idxs = tr.multinomial(
                self.policy[next_state_batch.long()], num_samples=1)
            
            # [B, 1] -> [B, z_dim, 1]
            next_action_idxs = tr.tile(
                next_action_idxs, (1, self.z_dim)).unsqueeze(-1)
            # [B, z_dim, 1] -> [B, z_dim]
            next_lf_batch_action = next_lf_batch.gather(
                2, next_action_idxs).squeeze(-1)
            # [B, z_dim]
            target = feature_batch + self.discount * next_lf_batch_action
            # [B]
            target = target @ self.z 

        q_values = self.q[state_batch.long()]

        # compute lambda-feature loss
        # [B, A] -> [B, 1]
        action_idxs = tr.argmax(action_batch, dim=1, keepdim=True)
        frac_stay = tr.sum(action_batch[:, -1]).item() / action_batch.shape[0]
        q_values_a = q_values.gather(1, action_idxs).squeeze()
        action_idxs_expanded = tr.tile(
            action_idxs, (1, self.z_dim)).unsqueeze(-1)
        # select the lambda-feature for the action taken 
        # [B, z_dim, A] -> [B, z_dim, 1]
        lf_batch = self.feature_net(obs_batch, prev_action_batch).reshape(
            -1, self.z_dim, self.action_dim).gather(2, action_idxs_expanded)
        # [B]
        q_batch = lf_batch.squeeze(-1) @ self.z
        loss = F.mse_loss(q_batch, q_values_a)

        stats = dict(
            target_mean=target.mean().item(),
            target_std=target.std().item(),
            target_max=target.max().item(),
            target_min=target.min().item(),
            lf_mean=lf_batch.mean().item(),
            lf_std=lf_batch.std().item(),
            lf_max=lf_batch.max().item(),
            lf_min=lf_batch.min().item(),
            frac_stay=frac_stay,
            max_td_error=(q_batch - target).abs().max().item(),)

        
        self.n_updates += 1
        if self.n_updates % self.target_update_freq == 0:
            self.update_target()

        return loss, stats

    def update_traj(
            self,
            state_batch,
            obs_batch,
            feature_batch,
            action_batch,
            next_state_batch,
            next_obs_batch,
            next_feature_batch,
            feature_returns_batch):
        # compute lambda-feature target
        del state_batch, next_feature_batch, feature_batch, next_state_batch, next_obs_batch
        # obs_batch is [B, T, C, H, W]
        B, T, C, H, W = obs_batch.shape
        # action_batch is [B, T, A]
        # feature_returns_batch is [B, T, z_dim]

        # compute lambda-feature loss
        # [B, T, A] -> [B, T, 1]
        action_idxs = tr.argmax(action_batch, dim=2, keepdim=True)
        # [B, T, 1] -> [B*T, 1]
        action_idxs = action_idxs.reshape(B*T, 1)
        # [B*T, 1] -> [B*T, z_dim, 1]
        action_idxs_expanded = tr.tile(
            action_idxs, (1, self.z_dim)).unsqueeze(-1)
        # previous actions [B, T, A]
        prev_acts0 = tr.zeros((B, 1, self.action_dim), device=obs_batch.device).float()
        prev_acts0[:, :, 0] = 1.0
        prev_actions = tr.cat(
            [prev_acts0, action_batch[:, :-1, :]], dim=1)  # concatenate
        # reshape to [B*T, A]
        prev_actions = prev_actions.reshape(B*T, self.action_dim)
        # select the lambda-feature for the action taken ([B*T, z_dim]])
        lf_batch = self.feature_net(
            obs_batch, prev_actions,
            )[0].gather(2, action_idxs_expanded).reshape(B, T, self.z_dim)
        
        pred_returns = (lf_batch @ self.z).flatten()
        
        returns_batch = (feature_returns_batch @ self.z).flatten()
        loss = F.smooth_l1_loss(pred_returns, returns_batch.detach())

        stats = dict(
            lf_mean=lf_batch.mean().item(),
            lf_std=lf_batch.std().item(),
            lf_max=lf_batch.max().item(),
            lf_min=lf_batch.min().item(),)

        
        self.n_updates += 1
        if self.n_updates % self.target_update_freq == 0:
            self.update_target()

        return loss, stats
    

    def update_traj_td(
            self,
            prev_action_batch,
            state_batch,
            obs_batch,
            feature_batch,
            action_batch,
            next_state_batch,
            next_obs_batch,
            next_feature_batch):
        # compute lambda-feature target
        del state_batch, next_feature_batch
        # obs_batch is [B, T, C, H, W]
        B, T, C, H, W = obs_batch.shape
        done_batch = tr.zeros((B, T, self.z_dim), device=obs_batch.device).float()
        done_batch[:, -1, :] = 1.0
        # action_batch is [B, T, A]
        # feature_returns_batch is [B, T, z_dim]

        # compute next observation LFs
        with tr.no_grad():
            # [B*T, z_dim, A]
            next_lf_batch = self.feature_net(
                next_obs_batch, action_batch.reshape(-1, self.action_dim))[0]
            # [B*T, 1]
            next_action_idxs = tr.multinomial(
                self.policy[next_state_batch.flatten().long()], num_samples=1).reshape(-1, 1)
            # [B*T, 1] -> [B*T, z_dim, 1]
            next_action_idxs = tr.tile(
                next_action_idxs, (1, self.z_dim)).unsqueeze(-1)
            # [B*T, z_dim, 1] -> [B*T, z_dim]
            next_lf_batch_action = next_lf_batch.gather(
                2, next_action_idxs).squeeze(-1)
            # [B*T, z_dim] -> [B, T, z_dim]
            next_lf_batch_action = next_lf_batch_action.reshape(B, T, self.z_dim)
            # [B, T, z_dim]
            targets = feature_batch + self.discount * (1.0 - done_batch) * next_lf_batch_action
            # [B*T]
            targets = (targets @ self.z).flatten()

        # compute lambda-feature loss
        # [B, T, A] -> [B, T, 1]
        action_idxs = tr.argmax(action_batch, dim=2, keepdim=True)
        # [B, T, 1] -> [B*T, 1]
        action_idxs = action_idxs.reshape(B*T, 1)
        # [B*T, 1] -> [B*T, z_dim, 1]
        action_idxs_expanded = tr.tile(
            action_idxs, (1, self.z_dim)).unsqueeze(-1)
        prev_actions = prev_action_batch.reshape(B*T, self.action_dim)
        # select the lambda-feature for the action taken ([B*T, z_dim]])
        lf_batch = self.feature_net(
            obs_batch, prev_actions,
            )[0].gather(2, action_idxs_expanded).reshape(B, T, self.z_dim)
        
        pred_returns = (lf_batch @ self.z).flatten()
        
        loss = F.smooth_l1_loss(pred_returns, targets.detach())

        stats = dict(
            lf_mean=lf_batch.mean().item(),
            lf_std=lf_batch.std().item(),
            lf_max=lf_batch.max().item(),
            lf_min=lf_batch.min().item(),
            max_td_error=(pred_returns - targets).abs().max().item(),)

        
        self.n_updates += 1
        if self.n_updates % self.target_update_freq == 0:
            self.update_target()

        return loss, stats


class FeatureActionNet(nn.Module):

    def __init__(self, obs_shape, action_dim, z_dim, hidden_dim, obs_type='pixels', use_actions=False, use_prev_actions=False):
        super().__init__()

        c, h, w = obs_shape
        self.obs_type = obs_type
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.repr_dim = 32 * (h // 8 - 3) * (w // 8 - 3)


        self.embedding_net = nn.Sequential(nn.Conv2d(c, 32, 3, stride=4),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Flatten(),
                                     nn.LayerNorm(self.repr_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.repr_dim, hidden_dim),
                                     nn.ReLU())
        added_action_dims = (int(use_actions) + int(use_prev_actions)) * action_dim
        self.use_prev_actions = use_prev_actions
        self.use_actions = use_actions
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim + added_action_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, z_dim))
        
        self.apply(weight_init)

    def forward(self, obs, action=None, prev_action=None):
        if self.obs_type == 'pixels':
            obs = obs / 255.0 - 0.5
        x = self.embedding_net(obs)
        if self.use_actions:
            x = tr.cat([x, action], dim=1)
        if self.use_prev_actions:
            x = tr.cat([x, prev_action], dim=1)
        out = self.feature_net(x).reshape(-1, self.z_dim)
        return out


class LambdaVNet(nn.Module):

    def __init__(
            self,
            policy,
            z,
            q,
            lf_true,
            obs_shape,
            action_dim,
            z_dim,
            hidden_dim,
            tau=0.005,
            discount=0.97,
            lambda_=1.0,
            target_update_freq=1,
            use_prev_action=False,
            use_action=True,
            obs_type='pixels',
            network='rnn',
            print_network=True):
        super().__init__()
        del network


        self.obs_type = obs_type
        self.q = tr.from_numpy(q).float()
        self.policy = tr.from_numpy(policy).float()
        self.lf_true = tr.from_numpy(lf_true).float()  # [S, A, z_dim]


        self.feature_net = FeatureActionNet(
            obs_shape, action_dim, z_dim, hidden_dim, obs_type, use_prev_actions=use_prev_action, use_actions=use_action)
        if print_network:
            print(self.feature_net)

        self.target_net = copy.deepcopy(self.feature_net)
        self.apply(weight_init)
        self.tau = tau
        self.discount = discount
        self.lambda_ = lambda_
        self.z = tr.from_numpy(z).float()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.n_updates = 0
        self.target_update_freq = target_update_freq
        self.use_prev_action = use_prev_action


    def update_target(self) -> None:
        """moving average update of target networks"""
        with tr.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.feature_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def forward(self, obs, action=None, prev_action=None):
        out1 = self.feature_net(obs, action=action, prev_action=prev_action)
        return out1
        
    
    def update(
            self,
            prev_action_batch,
            state_batch,
            obs_batch,
            feature_batch,
            action_batch,
            next_state_batch,
            next_obs_batch,
            next_feature_batch,
            done_batch):
        # compute lambda-feature target
        del next_feature_batch
        with tr.no_grad():


            ####### bootstrapping
            # [B, 1]
            next_action_idxs = tr.multinomial(
                self.policy[next_state_batch.long()], num_samples=1)
            # [B, A]
            next_action_batch = F.one_hot(
                next_action_idxs, num_classes=self.action_dim).float().squeeze()
            # [B, z_dim]
            next_lf_batch1 = self.target_net(
                next_obs_batch, next_action_batch, action_batch)
            # [B]
            next_lfq_batch = next_lf_batch1 @ self.z

            target = feature_batch @ self.z + self.discount * (1.0 - done_batch) * next_lfq_batch

        q_values = self.q[state_batch.long()]

        # compute lambda-feature loss
        # [B, A] -> [B, 1]
        action_idxs = tr.argmax(action_batch, dim=1, keepdim=True)
        frac_stay = tr.sum(action_batch[:, -1]).item() / action_batch.shape[0]
        q_values_a = q_values.gather(1, action_idxs).squeeze()
        # [B, z_dim]
        lf_batch1 = self.feature_net(
            obs_batch, action_batch, prev_action_batch)
        # [B]
        q_batch1 = lf_batch1.squeeze() @ self.z
        loss = F.smooth_l1_loss(q_batch1, target)

        stats = dict(
            target_mean=target.mean().item(),
            target_std=target.std().item(),
            target_max=target.max().item(),
            target_min=target.min().item(),
            frac_stay=frac_stay,
            bootstrap_targ_mse=(q_values_a - target).pow(2).mean().item(),
            max_td_error1=(q_batch1 - target).abs().max().item(),)

        
        self.n_updates += 1
        if self.n_updates % self.target_update_freq == 0:
            self.update_target()

        return loss, stats


class FeatureNet(nn.Module):

    def __init__(self, obs_shape, action_dim, z_dim, hidden_dim, obs_type='pixels', use_prev_actions=False):
        super().__init__()

        c, h, w = obs_shape
        self.obs_type = obs_type
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.repr_dim = 32 * (h // 8 - 3) * (w // 8 - 3)


        self.embedding_net = nn.Sequential(nn.Conv2d(c, 32, 3, stride=4),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Flatten(),
                                     nn.LayerNorm(self.repr_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.repr_dim, hidden_dim),
                                     nn.ReLU())
        added_action_dims = action_dim if use_prev_actions else 0
        self.use_prev_actions = use_prev_actions
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim + added_action_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim * action_dim))
        
        self.apply(weight_init)

    def forward(self, obs, prev_action=None):
        if self.obs_type == 'pixels':
            obs = obs / 255.0 - 0.5
        x = self.embedding_net(obs)
        if self.use_prev_actions:
            x = tr.cat([x, prev_action], dim=1)
        out = self.feature_net(x).reshape(-1, self.z_dim, self.action_dim)
        return out


class LambdaQNet(nn.Module):

    def __init__(
            self,
            policy,
            z,
            q,
            obs_shape,
            action_dim,
            z_dim,
            hidden_dim,
            tau=0.005,
            discount=0.97,
            lambda_=1.0,
            target_update_freq=1,
            use_prev_action=False,
            obs_type='pixels',
            network='rnn',
            print_network=True):
        super().__init__()
        del network


        self.obs_type = obs_type
        self.q = tr.from_numpy(q).float()
        self.policy = tr.from_numpy(policy).float()


        self.feature_net = FeatureNet(
            obs_shape, action_dim, z_dim, hidden_dim, obs_type, use_prev_actions=use_prev_action)
        if print_network:
            print(self.feature_net)

        self.target_net = copy.deepcopy(self.feature_net)
        self.apply(weight_init)
        self.tau = tau
        self.discount = discount
        self.lambda_ = lambda_
        self.z = tr.from_numpy(z).float()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.n_updates = 0
        self.target_update_freq = target_update_freq
        self.use_prev_action = use_prev_action


    def update_target(self) -> None:
        """moving average update of target networks"""
        with tr.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.feature_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def forward(self, obs, prev_action=None):
        out1 = self.feature_net(obs, prev_action=prev_action)
        return out1
        
    
    def update(
            self,
            prev_action_batch,
            state_batch,
            obs_batch,
            feature_batch,
            action_batch,
            next_state_batch,
            next_obs_batch,
            next_feature_batch,
            done_batch):
        # compute lambda-feature target
        del next_feature_batch
        action_idxs = tr.argmax(action_batch, dim=1, keepdim=True)
        with tr.no_grad():
            # [B, 1]
            next_action_idxs = tr.multinomial(
                self.policy[next_state_batch.long()],
                num_samples=1)
            # [B, A]
            next_lfq_batch1 = self.q[next_state_batch.long()].float()

            next_lfq_batch = next_lfq_batch1.gather(1, next_action_idxs).squeeze()
            next_lfq_batch1 = self.q[state_batch.long()].float()
            next_lfq_batch = next_lfq_batch1.gather(1, action_idxs).squeeze()

            target = next_lfq_batch

        q_values = self.q[state_batch.long()]

        # compute lambda-feature loss
        # [B, A] -> [B, 1]
        frac_stay = tr.sum(action_batch[:, -1]).item() / action_batch.shape[0]
        q_values_a = q_values.gather(1, action_idxs).squeeze()
        # select the lambda-feature for the action taken 
        # [B, z_dim, A] -> [B, z_dim]
        lf_batch1 = self.feature_net(
            obs_batch, prev_action_batch)
        # [B, A]
        lfq_batch = tr.einsum("bza,z->ba", lf_batch1, self.z)
        # [B]
        q_batch1 = lfq_batch.gather(1, action_idxs).squeeze()

        loss = F.smooth_l1_loss(q_batch1, target)

        stats = dict(
            target_mean=target.mean().item(),
            target_std=target.std().item(),
            target_max=target.max().item(),
            target_min=target.min().item(),
            frac_stay=frac_stay,
            bootstrap_targ_mse=(q_values_a - target).pow(2).mean().item(),
            max_td_error1=(q_batch1 - target).abs().max().item(),)

        
        self.n_updates += 1
        if self.n_updates % self.target_update_freq == 0:
            self.update_target()

        return loss, stats

