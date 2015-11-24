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


class SortingDPG(RecurrentDPG):

  def _loop_function(self):
    """
    The sorter policy uses actions to compute a weighted sum of the sequence
    states.
    """
    def loop_fn(output_t, t):
      # output_t is already a softmax value.
      # TODO: true?
      weighted_inputs = self.encoder_states * tf.expand_dims(output_t, 2)
      processed = tf.reduce_sum(weighted_inputs, 1)
      return processed

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
    return ret


def calc_rewards(dpg, input_seq, actions):
  rewards = []
  seq_length = len(input_seq)
  batch_size = input_seq[0].shape[0]

  actions = np.array(actions)
  assert actions.shape == (batch_size, seq_length) # DEV

  # "Dereference" the predicted sorts, which are index sequences.
  row_idxs = np.arange(batch_size).reshape((-1, 1)).repeat(seq_length, axis=1)
  predicted = input_seq[row_idxs, actions]

  # Compute per-timestep rewards by evaluating constraint violations.
  rewards = (predicted[:, 1:] > predicted[:, :-1]).astype(np.int)
  # Add reward for t = 0, fixed as 0
  rewards = np.concatenate([np.zeros((batch_size, 1)), rewards], axis=1)

  return rewards


def run_episode(input_seq, dpg, policy, buffer=None)
  sess = tf.get_default_session()

  # TODO eventually this should just trigger an assign op on shared data
  # rather than a transfer of data back to TF client
  ret = sess.run(policy + dpg.encoder_states + dpg.decoder_states,
                 {dpg.inputs: input_seq})
  actions = ret[:dpg.seq_length]
  input_enc = ret[dpg.seq_length:dpg.seq_length * 2]
  decoder_states = ret[dpg.seq_length * 2:dpg.seq_length * 3]

  # Calculate reward for all timesteps.
  # TODO: Run this as part of the TF graph when possible. (i.e., whenever
  # environment dynamics can be feasibly simulated in TF)
  rewards = calc_rewards(dpg, input_seq, actions)

  if buffer is not None:
    buffer.add_trajectory(input_seq, decoder_states, actions, rewards)


def train_batch(dpg, policy_update, critic_update, buffer):
  sess = tf.get_default_session()

  # Sample a training minibatch.
  try:
    input_seq, states, actions, rewards = \
        buffer.sample_trajectory()
  except ValueError:
    # Not enough data. Keep collecting trajectories.
    return 0.0

  # Compute targets (TD error backups) given current Q function.

