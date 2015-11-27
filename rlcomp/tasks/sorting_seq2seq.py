"""
Easy computation task: sorting sequences of distinct discrete elements.
"""

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

from rlcomp import util
from rlcomp.dpg import PointerNetDPG


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("logdir", "/tmp/rlcomp_sorting", "")

# Data parameters
flags.DEFINE_integer("seq_length", 5, "")
flags.DEFINE_integer("vocab_size", 10, "")

# Architecture hyperparameters
flags.DEFINE_integer("embedding_dim", 20, "")
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


class SortingDPG(PointerNetDPG):

  """
  Pointer-net DPG with extensions:

  1. Token inputs
  2. Concrete hardening definition
  """

  def __init__(self, mdp, spec, embedding_dim, vocab_size, seq_length,
               **kwargs):
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    super(SortingDPG, self).__init__(mdp, spec, embedding_dim, seq_length,
                                     **kwargs)

  def _make_params(self):
    self.embeddings = tf.get_variable("embeddings",
                                      (self.vocab_size, self.embedding_dim))

  def _make_inputs(self):
    self.input_tokens = [tf.placeholder(tf.int32, (None,))
                         for _ in range(self.seq_length)]
    self.inputs = [tf.nn.embedding_lookup(self.embeddings, tokens_t)
                   for tokens_t in self.input_tokens]
    super(SortingDPG, self)._make_inputs()

  def _make_q_targets(self):
    # Predict rewards for a_explore policy
    # seq_length * batch_size * 1
    rewards = self._calc_rewards(self.a_explore)

    # Compute bootstrap Q(s_next, pi(s_next))
    bootstraps = [self.critic_off[t + 1]
                  for t in range(self.seq_length - 1)]
    bootstraps.append(tf.constant(0.0))

    self.q_targets = [rewards_t + FLAGS.gamma * bootstraps_t
                      for rewards_t, bootstraps_t
                      in zip(rewards, bootstraps)]

  def _calc_rewards(self, action_list):
    action_list = tf.transpose(self.harden_actions(action_list))
    action_list = tf.unpack(action_list, FLAGS.batch_size)

    # batch_size * seq_length
    token_matrix = tf.transpose(tf.pack(self.input_tokens))
    token_matrix = tf.unpack(token_matrix, FLAGS.batch_size)

    # "Dereference" the predicted sorts, which are index sequences.
    predicted = [tf.gather(token_matrix[i], action_list[i])
                 for i in range(FLAGS.batch_size)]
    predicted = tf.concat(0, [tf.expand_dims(predicted_i, 0)
                              for predicted_i in predicted])

    # Compute per-timestep rewards by evaluating constraint violations.
    rewards = (tf.slice(predicted, [0, 1], [-1, -1])
               > tf.slice(predicted, [0, 0], [-1, self.seq_length - 1]))
    rewards = tf.cast(rewards, tf.float32)
    # Add reward for t = 0, fixed as 0
    self.rewards = tf.concat(1, [tf.zeros((FLAGS.batch_size, 1)),
                                 rewards])

    return tf.unpack(tf.transpose(self.rewards), self.seq_length,
                     name="rewards")

  def harden_actions(self, action_list):
    ret = []
    for actions_t in action_list:
      # HACK: Just return a vector of ints at each timestep.
      # We are not satisfying action_dim anymore, but that's okay as long as
      # calc_rewards sticks with the same contract as we are.
      ret_t = tf.argmax(actions_t, 1)
      ret.append(ret_t)
    return tf.pack(ret)


def build_updates(dpg):
  policy_optim = tf.train.MomentumOptimizer(FLAGS.policy_lr, FLAGS.momentum)
  policy_update = policy_optim.minimize(dpg.policy_objective,
                                        var_list=dpg.policy_params)

  critic_optim = tf.train.MomentumOptimizer(FLAGS.critic_lr, FLAGS.momentum)
  critic_update = critic_optim.minimize(dpg.critic_objective,
                                        var_list=dpg.critic_params)

  return policy_update, critic_update


def gen_inputs():
  xs = np.random.choice(FLAGS.vocab_size, replace=False,
                        size=FLAGS.seq_length)
  return xs


def make_batch(batch_size):
  inputs = np.array([gen_inputs() for _ in range(batch_size)])
  return inputs.T


def train(dpg, policy_update, critic_update):
  sess = tf.get_default_session()

  summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)

  for t in xrange(FLAGS.num_iter):
    print t

    # inputs: seq_length * batch_size
    inputs = make_batch(FLAGS.batch_size)
    feed_dict = {dpg.input_tokens[t]: inputs[t]
                 for t in range(FLAGS.seq_length)}

    cost_t, summary, _, _ = sess.run(
        [dpg.critic_objective, summary_op, policy_update, critic_update],
        feed_dict)

    if summary:
      summary_writer.add_summary(summary, t)

    if t % FLAGS.eval_interval == 0:
      inputs = make_batch(FLAGS.batch_size)
      feed_dict = {dpg.input_tokens[t]: inputs[t]
                   for t in range(FLAGS.seq_length)}
      rewards_fetch = tf.reduce_mean(dpg.rewards)
      rewards = sess.run(rewards_fetch, feed_dict)
      print "\t", rewards


def main(unused_args):
  FLAGS.policy_dims = [int(x) for x in filter(None, FLAGS.policy_dims.split(","))]
  FLAGS.critic_dims = [int(x) for x in filter(None, FLAGS.critic_dims.split(","))]

  state_dim = FLAGS.policy_dims[0] * 2
  mdp_spec = util.MDPSpec(state_dim, FLAGS.seq_length)
  dpg_spec = util.DPGSpec(FLAGS.policy_dims, FLAGS.critic_dims)

  dpg = SortingDPG(mdp_spec, dpg_spec, FLAGS.embedding_dim,
                   FLAGS.vocab_size, FLAGS.seq_length)
  policy_update, critic_update = build_updates(dpg)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    train(dpg, policy_update, critic_update)


if __name__ == "__main__":
  tf.app.run()
