from typing import Any, Callable, NamedTuple, Tuple, Optional

from bsuite.baselines import base
from bsuite.baselines.utils import sequence

import dm_env
from dm_env import specs
import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp
import optax
import rlax
import chex
from rlax._src import multistep
Array = chex.Array
Numeric = chex.Numeric

Logits = jnp.ndarray
Value = jnp.ndarray
LSTMState = Any
RecurrentPolicyValueNet = Callable[[jnp.ndarray, LSTMState],
                                   Tuple[Tuple[Logits, Value], LSTMState]]


def lf_lambda_returns(
    phi_t: Array,
    discount_t: Array,
    lf_t: Array,
    lambda_: Numeric = 1.,
    lf_lambda_: Numeric = 1.,
    stop_target_gradients: bool = True,) -> Array:
  """Estimates a multistep truncated lambda return from a trajectory.

  Args:
    r_t: sequence of rewards rₜ for timesteps t in [1, T].
    discount_t: sequence of discounts γₜ for timesteps t in [1, T].
    v_t: sequence of state values estimates under π for timesteps t in [1, T].
    lambda_: mixing parameter; a scalar or a vector for timesteps t in [1, T].
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.

  Returns:
    Multistep lambda returns.
  """
   # If scalar make into vector.
  lambda_ = jnp.ones_like(discount_t) * lambda_
  lf_lambda_ = jnp.ones_like(discount_t) * lf_lambda_

  # Work backwards to compute `G_{T-1}`, ..., `G_0`.
  def _body(acc, xs):
    features, discounts, lfs, lambda_, lf_lambda_ = xs
    lf_mixed = (1 - lambda_) * lfs + lambda_ * acc
    acc = features * (1 + lf_lambda_ * discounts * lf_mixed) + discounts * (1 - features) * lf_mixed
    return acc, acc

  _, lf_targets = jax.lax.scan(
      _body, lf_t[-1], (phi_t, discount_t, lf_t, lambda_, lf_lambda_),
      reverse=True)

  return jax.lax.select(stop_target_gradients,
                        jax.lax.stop_gradient(lf_targets),
                        lf_targets)


def td_lambda_lambda(
    v_tm1: Array,
    r_t: Array,
    discount_t: Array,
    v_t: Array,
    lf_t: Array, 
    lambda_: Numeric,
    lf_lambda_: Numeric,
    stop_target_gradients: bool = False,
) -> Array:
  """Calculates the TD(lambda) temporal difference error.

  See "Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/book/ebook/node74.html).

  Args:
    v_tm1: sequence of state values at time t-1.
    r_t: sequence of rewards at time t.
    discount_t: sequence of discounts at time t.
    v_t: sequence of state values at time t.
    lambda_: mixing parameter lambda, either a scalar or a sequence.
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.
  Returns:
    TD(lambda) temporal difference error.
  """
  chex.assert_rank([v_tm1, r_t, discount_t, v_t, lambda_], [1, 1, 1, 1, {0, 1}])
  chex.assert_type([v_tm1, r_t, discount_t, v_t, lambda_], float)

  target_tm1 = multistep.lambda_returns(r_t, discount_t, v_t + (lf_lambda_ - 1) * r_t * lf_t, lambda_)
  target_tm1 = jax.lax.select(stop_target_gradients,
                              jax.lax.stop_gradient(target_tm1), target_tm1)
  return target_tm1 - v_tm1

def compute_lf_errors(
    lf_tm1: Array,
    phi_t: Array,
    discount_t: Array,
    lf_t: Array,
    lambda_: Numeric,
    lf_lambda_: Numeric,
    stop_target_gradients: bool = True,):
  
  target_tm1 = lf_lambda_returns(
    phi_t, discount_t, lf_t, lambda_=lambda_, lf_lambda_=lf_lambda_,
    stop_target_gradients=stop_target_gradients)

  target_tm1 = jax.lax.select(stop_target_gradients,
                              jax.lax.stop_gradient(target_tm1), target_tm1)
  return target_tm1 - lf_tm1

class AgentState(NamedTuple):
  """Holds the network parameters, optimizer state, and RNN state."""
  params: hk.Params
  opt_state: Any
  rnn_state: LSTMState
  rnn_unroll_state: LSTMState


class ActorCriticRNN(base.Agent):
  """Recurrent actor-critic agent."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      network: RecurrentPolicyValueNet,
      initial_rnn_state: LSTMState,
      optimizer: optax.GradientTransformation,
      rng: hk.PRNGSequence,
      sequence_length: int,
      discount: float,
      td_lambda: float,
      entropy_cost: float = 0.,
      lf_wt: float = 1.0,
      lf_lambda: float = 1.0,):

    # Define loss function.
    def loss(trajectory: sequence.Trajectory, rnn_unroll_state: LSTMState,):
      """"Actor-critic loss."""
      (logits, values, lfs), new_rnn_unroll_state = hk.dynamic_unroll(
          network, trajectory.observations[:, None, ...], rnn_unroll_state)
      # logits is [T+1, 1, A]  (1 = batch_size)
      # values is [T+1, 1]
      # lfs is [T+1, 1, feature_dim]
      seq_len = trajectory.actions.shape[0]
      indices = jnp.expand_dims(jnp.argmax(trajectory.observations, axis=1), axis=-1)
      sequence_indices = jnp.expand_dims(jnp.arange(lfs.shape[0]), axis=-1)
      lfs_states = lfs[sequence_indices, 0, indices]
      td_errors = td_lambda_lambda(
          v_tm1=values[:-1, 0],
          r_t=trajectory.rewards,
          discount_t=trajectory.discounts * discount,
          v_t=values[1:, 0],
          lf_t=lfs_states[:-1, 0],
          lambda_=jnp.array(td_lambda),
          lf_lambda_=jnp.array(lf_lambda),)

      critic_loss = jnp.mean(td_errors**2)

      # lf errors is [T, feature_dim]
      lf_errors = compute_lf_errors(
          lf_tm1=lfs[:-1, 0],
          phi_t=trajectory.observations[:-1],
          discount_t=trajectory.discounts * discount,
          lf_t=lfs[1:, 0],
          lambda_=jnp.array(0.0),
          lf_lambda_=jnp.array(lf_lambda),)

      lf_loss = jnp.mean(lf_errors**2)
      actor_loss = rlax.policy_gradient_loss(
          logits_t=logits[:-1, 0],
          a_t=trajectory.actions,
          adv_t=td_errors,
          w_t=jnp.ones(seq_len))
      entropy_loss = jnp.mean(
          rlax.entropy_loss(logits[:-1, 0], jnp.ones(seq_len)))

      combined_loss = actor_loss + critic_loss + entropy_cost * entropy_loss + lf_wt * lf_loss

      return combined_loss, new_rnn_unroll_state

    # Transform the loss into a pure function.
    loss_fn = hk.without_apply_rng(hk.transform(loss)).apply

    # Define update function.
    @jax.jit
    def sgd_step(state: AgentState,
                 trajectory: sequence.Trajectory,) -> AgentState:
      """Does a step of SGD over a trajectory."""
      gradients, new_rnn_state = jax.grad(
          loss_fn, has_aux=True)(state.params, trajectory,
                                 state.rnn_unroll_state,)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      return state._replace(
          params=new_params,
          opt_state=new_opt_state,
          rnn_unroll_state=new_rnn_state)

    # Initialize network parameters and optimiser state.
    init, forward = hk.without_apply_rng(hk.transform(network))
    dummy_observation = jnp.zeros((1, *obs_spec.shape), dtype=obs_spec.dtype)
    initial_params = init(next(rng), dummy_observation, initial_rnn_state)
    initial_opt_state = optimizer.init(initial_params)

    # Internalize state.
    self._state = AgentState(initial_params, initial_opt_state,
                             initial_rnn_state, initial_rnn_state)
    self._forward = jax.jit(forward)
    self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length)
    self._sgd_step = sgd_step
    self._rng = rng
    self._initial_rnn_state = initial_rnn_state
    self._lambda = lf_lambda

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions according to a softmax policy."""
    key = next(self._rng)
    observation = timestep.observation[None, ...]
    (logits, _, _), rnn_state = self._forward(self._state.params, observation,
                                           self._state.rnn_state)
    self._state = self._state._replace(rnn_state=rnn_state)
    action = jax.random.categorical(key, logits).squeeze()
    return int(action)

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    """Adds a transition to the trajectory buffer and periodically does SGD."""
    if new_timestep.last():
      self._state = self._state._replace(rnn_state=self._initial_rnn_state)
    self._buffer.append(timestep, action, new_timestep)
    if self._buffer.full() or new_timestep.last():
      trajectory = self._buffer.drain()
      self._state = self._sgd_step(self._state, trajectory, self._LR)


def default_lambda_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray,
                  lf_lambda: float = 1.0,
                  lf_wt: float = 1.0,
                  feature_dim: int = 32,
                  hidden_size: int = 32,
                  entropy_cost: float = 0.0,
                  seed: int = 0) -> base.Agent:
  """Creates an actor-critic agent with default hyperparameters."""

  initial_rnn_state = hk.LSTMState(
      hidden=jnp.zeros((1, hidden_size), dtype=jnp.float32),
      cell=jnp.zeros((1, hidden_size), dtype=jnp.float32))

  def network(inputs: jnp.ndarray,
              state) -> Tuple[Tuple[Logits, Value], LSTMState]:
    flat_inputs = hk.Flatten()(inputs)
    lstm = hk.LSTM(hidden_size)
    policy_head = hk.Linear(action_spec.num_values)
    value_head = hk.Linear(1)
    lf_head = hk.Linear(feature_dim)

    embedding, state = lstm(flat_inputs, state)
    logits = policy_head(embedding)
    value = value_head(embedding)
    lf = lf_head(embedding)
    return (logits, jnp.squeeze(value, axis=-1), lf), state

  return ActorCriticRNN(
      obs_spec=obs_spec,
      action_spec=action_spec,
      network=network,
      initial_rnn_state=initial_rnn_state,
      optimizer=optax.adam(3e-4),
      rng=hk.PRNGSequence(seed),
      sequence_length=32,
      discount=0.99,
      td_lambda=0.9,
      entropy_cost=entropy_cost,
      lf_wt=lf_wt,
      lf_lambda=lf_lambda,)