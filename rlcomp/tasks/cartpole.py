"""
Toy task (not computation-related): cartpole swingup.

We rely heavily on Christoph Dann's `tdlearn` library here.
"""

import numpy as np
import tensorflow as tf

from tdlearn.examples import PendulumSwingUpCartPole

from rlcomp import util
from rlcomp.dpg import DPG


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("policy_dims", "20", "")
flags.DEFINE_string("critic_dims", "", "")

flags.DEFINE_integer("batch_size", 64, "")
flags.DEFINE_integer("buffer_size", 10 ** 6, "")

flags.DEFINE_integer("num_iter", 1000, "")
flags.DEFINE_integer("eval_interval", 10,
                     "Evaluate policy without exploration every $n$ "
                     "iterations.")
flags.DEFINE_float("policy_lr", 0.0001, "")
flags.DEFINE_float("critic_lr", 0.00001, "")
flags.DEFINE_float("momentum", 0.9, "")
flags.DEFINE_float("gamma", 0.95, "")
flags.DEFINE_float("tau", 0.001, "")


def preprocess_state(state):
  # bounds stolen from chrodan's implementation
  min = np.array([-35, -3, -12, -3])
  max = np.array([35, 4, 12, 3])

  range = max - min
  state = np.clip(state, min, max)
  state -= min + 0.5 * range
  state /= (max - min)
  return state


def preprocess_action(action):
  return action


def run_episode(mdp, dpg, policy, buffer=None, max_len=100):
  sess = tf.get_default_session()
  def policy_fn(state):
    state = preprocess_state(state).reshape((-1, mdp.dim_S))
    action = sess.run(policy, {dpg.inputs: state})
    return action.flatten()

  states, actions, rewards, states_next = [], [], [], []
  trajectory_gen = mdp.sample_transition(max_len, policy_fn)

  for s, a, s_n, r in trajectory_gen:
    states.append(preprocess_state(s))
    actions.append(preprocess_action(a))
    states_next.append(preprocess_state(s_n))
    rewards.append(r)

  if buffer is not None:
    buffer.extend(states, actions, rewards, states_next)
  return states, actions, rewards, states_next


def train_batch(dpg, policy_update, critic_update, buffer):
  sess = tf.get_default_session()

  # Sample a training minibatch.
  try:
    b_states, b_actions, b_rewards, b_states_next = \
        buffer.sample(FLAGS.batch_size)
  except ValueError:
    # Not enough data. Keep collecting trajectories.
    return 0.0

  # Compute targets (TD error backups) given current Q function.
  a_next, q_next = sess.run([dpg.a_pred_track, dpg.critic_on_track],
                            {dpg.inputs: b_states_next})
  b_targets = b_rewards + FLAGS.gamma * q_next.flatten()

  # Policy update.
  sess.run(policy_update, {dpg.inputs: b_states})

  # Critic update.
  cost_t, _, q_pred = sess.run([dpg.critic_objective, critic_update, dpg.critic_off],
                               {dpg.inputs: b_states, dpg.q_targets: b_targets})

  return cost_t


def build_model():
  mdp = PendulumSwingUpCartPole()
  mdp_spec = util.MDPSpec(mdp.dim_S, mdp.dim_A)

  dpg_spec = util.DPGSpec(FLAGS.policy_dims, FLAGS.critic_dims)
  dpg = DPG(mdp_spec, dpg_spec)

  return mdp, dpg


def build_updates(dpg):
  policy_optim = tf.train.MomentumOptimizer(FLAGS.policy_lr, FLAGS.momentum)
  policy_update = policy_optim.minimize(dpg.policy_objective,
                                        var_list=dpg.policy_params)

  critic_optim = tf.train.MomentumOptimizer(FLAGS.critic_lr, FLAGS.momentum)
  critic_update = critic_optim.minimize(dpg.critic_objective,
                                        var_list=dpg.critic_params)

  return policy_update, critic_update


def train(mdp, dpg, policy_update, critic_update, replay_buffer):
  sess = tf.get_default_session()

  for t in xrange(FLAGS.num_iter):
    print t
    # Sample a trajectory off-policy, then update the critic.
    offp_states, offp_actions, offp_rewards, _ = \
        run_episode(mdp, dpg, dpg.a_explore, replay_buffer)
    cost_t = train_batch(dpg, policy_update, critic_update, replay_buffer)

    # Update tracking model.
    sess.run([dpg.track_update], {dpg.tau: [FLAGS.tau]})

    if t % FLAGS.eval_interval == 0:
      # Evaluate actor by sampling a trajectory on-policy.
      states, actions, rewards, _ = run_episode(mdp, dpg, dpg.a_pred)

      print np.mean(rewards)
      # TODO log


def main(unused_args):
  FLAGS.policy_dims = [int(x) for x in filter(None, FLAGS.policy_dims.split(","))]
  FLAGS.critic_dims = [int(x) for x in filter(None, FLAGS.critic_dims.split(","))]

  mdp, dpg = build_model()
  policy_update, critic_update = build_updates(dpg)
  replay_buffer = util.ReplayBuffer(FLAGS.buffer_size, dpg.mdp_spec)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    train(mdp, dpg, policy_update, critic_update, replay_buffer)


if __name__ == "__main__":
  tf.app.run()
