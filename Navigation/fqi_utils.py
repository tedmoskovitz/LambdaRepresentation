import numpy as np
import torch as tr


def fit_dataset(network, dataset, num_epochs=10, batch_size=32, log_every=5):
    
    dataloader = tr.utils.data.DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False)
    
    losses = []
    for epoch in range(1, num_epochs+1):
      for input_batch, target_batch in dataloader:
        loss = network.update(input_batch, target_batch)
        losses.append(loss)          

        if epoch % log_every == 0 and log_every > 0:
            print(f"Epoch: {epoch}/{num_epochs}, avg loss = {np.mean(losses[-len(dataset):]):.4f}")

    return network, losses


def construct_bootstrap_target(env, policy, LFs, lambda_, discount):
    S, A, w = env._number_of_states, env._number_of_actions, env.r
    target = np.zeros((S, A, S))
    for s in range(S):
        base_feat = np.eye(S)[s]
        for a in range(A):
            sp = np.argmax(env.P[s, a, :])
            ap = np.argmax(policy[sp, :])
            targ_sa = base_feat * (1 + lambda_ * discount * LFs[sp, ap])
            targ_sa += discount * (1 - base_feat) * LFs[sp, ap]
            target[s, a] = env.P[s, a, sp] * policy[sp, ap] * targ_sa

    return target @ w

