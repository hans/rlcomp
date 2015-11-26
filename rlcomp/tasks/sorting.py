"""
Easy computation task: sorting sequences of distinct discrete elements.
"""

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

from rlcomp import util
from rlcomp.dpg import RecurrentDPG


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("logdir", "/tmp/rlcomp_sorting", "")

# Data parameters
flags.DEFINE_integer("seq_length", 5, "")
flags.DEFINE_integer("vocab_size", 10, "")

# Architecture hyperparameters
flags.DEFINE_string("policy_dims", "20", "")
flags.DEFINE_string("critic_dims", "", "")

# Training hyperparameters
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


class SortingDPG(RecurrentDPG):

  def _loop_function(self):
    """
    The sorter policy uses actions to compute a weighted sum of the sequence
    states.
    """
    def loop_fn(output_t, t):
      # TODO maybe remove recently selected item from input vec?
      return self.inputs

    return loop_fn

  def harden_actions(self, action_list):
    ret = []
    for actions_t in action_list:
#      # Build a one-hot matrix
#      ret_t = np.zeros_like(actions_t)
#      ret_t[np.arange(len(actions_t)), actions_t.argmax(axis=1)] = 1
      # HACK: Just return a vector of ints at each timestep.
      # We are not satisfying action_dim anymore, but that's okay as long as
      # calc_rewards sticks with the same contract as we are.
      ret_t = actions_t.argmax(axis=1)
      ret.append(ret_t)
    return np.array(ret)


def calc_rewards(dpg, xs, inputs, actions):
  rewards = []
  actions = dpg.harden_actions(actions).T
  batch_size, seq_length = actions.shape

  # "Dereference" the predicted sorts, which are index sequences.
  row_idxs = np.arange(batch_size).reshape((-1, 1)).repeat(seq_length, axis=1)
  predicted = xs[row_idxs, actions]

  # Compute per-timestep rewards by evaluating constraint violations.
  rewards = (predicted[:, 1:] > predicted[:, :-1]).astype(np.int)
  # Add reward for t = 0, fixed as 0
  rewards = np.concatenate([np.zeros((batch_size, 1)), rewards], axis=1)

  return rewards


def run_episode(xs, inputs, dpg, policy, buffer=None):
  sess = tf.get_default_session()

  # TODO eventually this should just trigger an assign op on shared data
  # rather than a transfer of data back to TF client
  ret = sess.run(policy + dpg.decoder_states,
                 {dpg.inputs: inputs})
  actions = ret[:dpg.seq_length]
  decoder_states = ret[dpg.seq_length:]

  # Calculate reward for all timesteps.
  # TODO: Run this as part of the TF graph when possible. (i.e., whenever
  # environment dynamics can be feasibly simulated in TF)
  rewards = calc_rewards(dpg, xs, inputs, actions)

  if buffer is not None:
    decoder_states = np.array(decoder_states).swapaxes(0, 1)
    actions = np.array(actions).swapaxes(0, 1)
    for i in range(len(xs)):
      buffer.add_trajectory(inputs[i], decoder_states[i], actions[i], rewards[i])

  return decoder_states, actions, rewards


def train_batch(dpg, policy_update, critic_update, buffer, summary_op=None):
  sess = tf.get_default_session()

  b_inputs, b_states, b_states_next, b_actions, b_rewards = \
      buffer.sample(FLAGS.batch_size)

  # Compute Q_next for all sampled tuples
  q_next = sess.run(dpg.critic_on,
                    {dpg.inputs: b_inputs,
                     dpg.decoder_state_ind: b_states_next})

  b_targets = b_rewards + FLAGS.gamma * q_next.flatten()

  # Policy update.
  sess.run(policy_update, {dpg.inputs: b_inputs,
                           dpg.decoder_state_ind: b_states})

  # Critic update.
  cost_t, _, summary = sess.run(
      [dpg.critic_objective, critic_update,
       (summary_op or tf.constant(0.0))],
      {dpg.inputs: b_inputs, dpg.decoder_state_ind: b_states,
       dpg.q_targets: b_targets})

  return cost_t, summary


def build_updates(dpg):
  policy_optim = tf.train.MomentumOptimizer(FLAGS.policy_lr, FLAGS.momentum)
  policy_update = policy_optim.minimize(dpg.policy_objective,
                                        var_list=dpg.policy_params)

  critic_optim = tf.train.MomentumOptimizer(FLAGS.critic_lr, FLAGS.momentum)
  critic_update = critic_optim.minimize(dpg.critic_objective,
                                        var_list=dpg.critic_params)

  return policy_update, critic_update


def gen_inputs():
  xs = np.random.choice(FLAGS.vocab_size, replace=False, size=FLAGS.seq_length)
  inputs = np.zeros((FLAGS.seq_length * FLAGS.vocab_size))

  for i, x in enumerate(xs):
    inputs[i * FLAGS.vocab_size + x] = 1

  return xs, inputs


def train(dpg, policy_update, critic_update, replay_buffer):
  sess = tf.get_default_session()

  summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)

  for t in xrange(FLAGS.num_iter):
    print t

    # Run a batch of N rollouts.
    N = FLAGS.batch_size
    batch = [gen_inputs() for _ in range(N)]
    xs = np.array([xs for xs, _ in batch])
    inputs = np.array([inputs for _, inputs in batch])
    run_episode(xs, inputs, dpg, dpg.a_explore, replay_buffer)

    # Update the actor and critic.
    cost_t, summary = train_batch(dpg, policy_update, critic_update,
                                  replay_buffer, summary_op=summary_op)
    if summary:
      summary_writer.add_summary(summary, t)

    if t % FLAGS.eval_interval == 0:
      xs, inputs = gen_inputs()
      xs, inputs = xs[np.newaxis, :], inputs[np.newaxis, :]
      _, _, rewards = run_episode(xs, inputs, dpg, dpg.a_pred)
      print "\t", rewards.mean()


def main(unused_args):
  FLAGS.policy_dims = [int(x) for x in filter(None, FLAGS.policy_dims.split(","))]
  FLAGS.critic_dims = [int(x) for x in filter(None, FLAGS.critic_dims.split(","))]

  mdp_spec = util.MDPSpec(FLAGS.policy_dims[-1], FLAGS.seq_length)
  dpg_spec = util.DPGSpec(FLAGS.policy_dims, FLAGS.critic_dims)

  input_dim = FLAGS.seq_length * FLAGS.vocab_size

  dpg = SortingDPG(mdp_spec, dpg_spec, input_dim, FLAGS.vocab_size,
                   FLAGS.seq_length)
  policy_update, critic_update = build_updates(dpg)
  replay_buffer = util.RecurrentReplayBuffer(
      FLAGS.buffer_size, mdp_spec, input_dim, FLAGS.seq_length,
      FLAGS.policy_dims[-1])

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    train(dpg, policy_update, critic_update, replay_buffer)


if __name__ == "__main__":
  tf.app.run()
