"""
Easy computation task: sorting sequences of distinct discrete elements.
"""

import os
import os.path
import pprint

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, seq2seq
from tensorflow.python.framework import ops

from rlcomp import util
from rlcomp.dpg import PointerNetDPG


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "`train` or `test`")
flags.DEFINE_string("logdir", "/tmp/rlcomp_sorting", "")
flags.DEFINE_string("checkpoint_path", None,
                    "Path to model checkpoint. Used only in `test` mode")
flags.DEFINE_boolean("verbose_summaries", False,
                    "Log very detailed summaries of parameter magnitudes, "
                    "activations, etc.")

flags.DEFINE_integer("eval_interval", 9999,
                     "Evaluate policy without exploration every $n$ "
                     "iterations.")
flags.DEFINE_integer("summary_flush_interval", 120, "")

# Data parameters
flags.DEFINE_integer("seq_length", 5, "")
flags.DEFINE_integer("vocab_size", 10, "")
flags.DEFINE_boolean("variable_length", False, "")

# Initialization hyperparameters
flags.DEFINE_float("embedding_init_range", 0.1, "")

# Architecture hyperparameters
flags.DEFINE_integer("embedding_dim", 20, "")
flags.DEFINE_string("policy_dims", "20", "")
flags.DEFINE_string("critic_dims", "", "")
flags.DEFINE_boolean("batch_normalize_actions", False, "")

# Autoencoder
flags.DEFINE_integer("pretrain_autoencoder", 0, "")

# Training hyperparameters
flags.DEFINE_integer("batch_size", 64, "")
flags.DEFINE_integer("buffer_size", 10 ** 6, "")
flags.DEFINE_integer("num_iter", 10000, "")
flags.DEFINE_float("policy_lr", 0.0001, "")
flags.DEFINE_float("critic_lr", 0.00001, "")
flags.DEFINE_float("momentum", 0.9, "")
flags.DEFINE_float("gamma", 0.95, "")
flags.DEFINE_float("tau", 0.001, "")
flags.DEFINE_boolean("cut_lr", True, "")
flags.DEFINE_float("explore_strength", 0.3, "Mean of permute strength")


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

    kwargs["noiser"] = kwargs.get("noiser", self._noise_actions)

    super(SortingDPG, self).__init__(mdp, spec, embedding_dim, seq_length,
                                     **kwargs)

  def _make_params(self):
    super(SortingDPG, self)._make_params()

    embedding_init = tf.random_normal_initializer(
        stddev=FLAGS.embedding_init_range)
    with ops.device("/cpu:0"):
      self.embeddings = tf.get_variable(
          "embedding", (self.vocab_size + 1, self.embedding_dim),
          initializer=embedding_init)

  def _policy_params(self):
    params = super(SortingDPG, self)._policy_params()
    #params.append(self.embeddings)
    return params

  def _make_inputs(self):
    self.input_tokens = [tf.placeholder(tf.int32, (None,))
                         for _ in range(self.seq_length)]
    self.inputs = [tf.nn.embedding_lookup(self.embeddings, tokens_t)
                   for tokens_t in self.input_tokens]
    super(SortingDPG, self)._make_inputs()

  def _make_q_targets(self):
    # Predict rewards for each policy
    # seq_length * batch_size * 1
    self.rewards_pred, _ = self._calc_rewards(self.a_pred, name="rewards_pred")
    self.rewards_explore, rewards_explore_unpacked = \
        self._calc_rewards(self.a_explore, name="rewards_explore")

    reward_normalizer = tf.to_float(self.real_lengths) - 1.0
    tf.scalar_summary("rewards/pred.mean",
                      tf.reduce_mean(
                        tf.reduce_sum(self.rewards_pred, 0) / reward_normalizer))
    tf.scalar_summary("rewards/explore.mean",
                      tf.reduce_mean(
                        tf.reduce_sum(self.rewards_explore, 0) / reward_normalizer))

    tf.scalar_summary("rewards/pred.max_mean",
                      tf.reduce_max(
                        tf.reduce_sum(self.rewards_pred, 0) / reward_normalizer))
    tf.scalar_summary("rewards/explore.max_mean",
                      tf.reduce_max(
                        tf.reduce_sum(self.rewards_explore, 0) / reward_normalizer))

    # Compute bootstrap Q(s_next, pi_off(s_next))
    bootstraps = [self.critic_off_track[t + 1]
                  for t in range(self.seq_length - 1)]
    bootstraps.append(tf.constant(0.0))

    self.q_targets = [rewards_t + FLAGS.gamma * bootstraps_t
                      for rewards_t, bootstraps_t
                      in zip(rewards_explore_unpacked, bootstraps)]

  def _calc_rewards(self, action_list, name="rewards"):
    action_list = tf.transpose(self.harden_actions(action_list))
    action_list = tf.unpack(action_list, FLAGS.batch_size)

    # batch_size * seq_length
    token_matrix = tf.transpose(tf.pack(self.input_tokens))
    token_matrix = tf.unpack(token_matrix, FLAGS.batch_size)

    # "Dereference" the predicted sorts, which are index sequences.
    predicted = [tf.gather(token_matrix[i], action_list[i])
                 for i in range(FLAGS.batch_size)]
#    predicted[0] = tf.Print(predicted[0], [predicted[0]], "predicted_" + name, summarize=100)
    predicted = tf.concat(0, [tf.expand_dims(predicted_i, 0)
                              for predicted_i in predicted])
    #predicted = tf.Print(predicted, [predicted], "predicted_" + name, summarize=100)

    # Compute per-timestep rewards by evaluating constraint violations.
    rewards = (tf.slice(predicted, [0, 1], [-1, -1])
               > tf.slice(predicted, [0, 0], [-1, self.seq_length - 1]))
    rewards = tf.cast(rewards, tf.float32)
    # Add reward for t = 0, fixed as 0
    rewards = tf.concat(1, [tf.zeros((FLAGS.batch_size, 1)),
                            rewards])
    rewards = tf.transpose(rewards)

    # Zero-mask reward values at end of sequence for shorter sequences
    reward_mask = tf.pack([self.real_lengths > t
                           for t in range(self.seq_length)])
    rewards = tf.select(reward_mask, rewards,
                        tf.zeros((self.seq_length, FLAGS.batch_size)))

    rewards_unpacked = tf.unpack(rewards, self.seq_length,
                                 name=name)

    return rewards, rewards_unpacked

  def _noise_actions(self, inputs, actions, name="noiser"):
    # Permute the elements of each softmax at each timestep.
    # Cheap approximation: permute all columns of each softmax at each timestep
    # in the same way.
    assert isinstance(actions, list)
    actions_new = []
    for actions_t in actions:
      # With some weight favor less permutation over more permutation.
      no_permute = tf.range(0, self.seq_length)
      permute = tf.random_shuffle(tf.range(0, self.seq_length))

      permute_strength = tf.maximum(
          0.0, tf.random_normal((1,), mean=FLAGS.explore_strength, stddev=0.2))
      maybe_permute = tf.select(
          tf.random_uniform([self.seq_length]) < permute_strength,
          permute, no_permute)

      actions_new_t = tf.gather(tf.transpose(actions_t), maybe_permute)
      actions_new_t = tf.transpose(actions_new_t)
      actions_new.append(actions_new_t)

    # TODO add some Gaussian noise?
    return actions_new

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
  policy_params = dpg.policy_params
  if FLAGS.pretrain_autoencoder > 0:
    # We already trained the encoder using autoencoder task. Remove encoder
    # weights from optimization.
    policy_params = [p for p in policy_params if "encoder" not in p.name]

  policy_lr = tf.Variable(FLAGS.policy_lr, name="policy_lr")
  policy_optim = tf.train.AdamOptimizer(policy_lr)
  policy_update = policy_optim.minimize(dpg.policy_objective,
                                        var_list=policy_params)

  critic_lr = tf.Variable(FLAGS.critic_lr, name="critic_lr")
  critic_optim = tf.train.AdamOptimizer(critic_lr)
  critic_update = critic_optim.minimize(dpg.critic_objective,
                                        var_list=dpg.critic_params)

  return policy_lr, critic_lr, policy_update, critic_update


def build_autoencoder(dpg):
  hidden_dim = dpg.spec.policy_dims[0]
  dec_cell = util.GRUCell(FLAGS.embedding_dim, hidden_dim)
  dec_cell = rnn_cell.OutputProjectionWrapper(dec_cell,
                                              FLAGS.vocab_size)

  dec_inp = [tf.zeros_like(dpg.input_tokens[0], name="adec_inp%i" % t)
             for t in range(dpg.seq_length)]
  dec_out, _ = util.embedding_rnn_decoder(
      dec_inp, dpg.encoder_states[-1], dec_cell, FLAGS.vocab_size,
      feed_previous=True, embedding=dpg.embeddings, scope="adec")

  labels = [tf.placeholder(tf.int32, shape=(None,), name="labels%i" % t)
            for t in range(dpg.seq_length)]
  weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]

  loss = seq2seq.sequence_loss(dec_out, labels, weights, FLAGS.vocab_size)

  optimizer = tf.train.AdamOptimizer(0.01)
  train_op = optimizer.minimize(loss) # TODO wrt what?

  return labels, loss, train_op


def pretrain_autoencoder(dpg, autoencoder, num_iters):
  labels, loss, train_op = autoencoder

  sess = tf.get_default_session()
  for _ in xrange(num_iters):
    X = [np.random.choice(FLAGS.vocab_size, size=(dpg.seq_length,),
                          replace=False)
         for _ in range(dpg.seq_length)]
    Y = X[:]

    # Dimshuffle to seq_len * batch_size
    X = np.array(X).T
    Y = np.array(Y).T

    feed_dict = {dpg.input_tokens[t]: X[t] for t in range(dpg.seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(dpg.seq_length)})

    _, loss_t = sess.run([train_op, loss], feed_dict)
    print loss_t


def make_batch(batch_size, seq_length, variable_length=False):
  if variable_length:
    length_fn = lambda: seq_length
  else:
    length_fn = lambda: np.random.uniform(2, seq_length + 1)

  batch = np.empty((batch_size, seq_length), dtype=np.int32)
  lengths = np.empty((batch_size,), dtype=np.int32)
  vocab = range(1, FLAGS.vocab_size + 1)

  for i in xrange(batch_size):
    length_i = (np.random.randint(2, seq_length + 1) if variable_length
                else FLAGS.seq_length)
    xs_i = np.random.choice(vocab, replace=False, size=length_i)

    # Pad at left.
    xs_i = np.concatenate([[0] * (FLAGS.seq_length - length_i), xs_i])

    batch[i] = xs_i
    lengths[i] = length_i

  return batch.T, lengths


def train(dpg, policy_lr, critic_lr, policy_update, critic_update):
  sess = tf.get_default_session()

  summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def,
                                          flush_secs=FLAGS.summary_flush_interval)
  saver = tf.train.Saver()

  halved_yet = 0
  for t in xrange(FLAGS.num_iter):
    print t

    # inputs: seq_length * batch_size
    inputs, lengths = make_batch(FLAGS.batch_size, FLAGS.seq_length,
                                 FLAGS.variable_length)
    feed_dict = {dpg.input_tokens[t]: inputs[t]
                 for t in range(FLAGS.seq_length)}
    feed_dict[dpg.real_lengths] = lengths

    # Run a batch of rollouts and execute policy + critic update
    cost_t, summary, _, _ = sess.run(
        [dpg.critic_objective, summary_op, policy_update, critic_update],
        feed_dict)

    # Now update tracking model
    # sess.run(dpg.track_update) DEV: Not necessary

    if summary:
      summary_writer.add_summary(summary, t)

    if t % FLAGS.eval_interval == 0:
      inputs, lengths = make_batch(FLAGS.batch_size, FLAGS.seq_length,
                                   FLAGS.variable_length)
      feed_dict = {dpg.input_tokens[t]: inputs[t]
                   for t in range(FLAGS.seq_length)}
      feed_dict[dpg.real_lengths] = lengths

      rewards_fetch = tf.reduce_mean(dpg.rewards_pred)
      rewards = sess.run(rewards_fetch, feed_dict)

      # DEV
      if FLAGS.cut_lr:
        if rewards > 0.1 and halved_yet < 1:
          sess.run([tf.assign(policy_lr, policy_lr * 0.5),
                    tf.assign(critic_lr, critic_lr * 0.5)])
          halved_yet = max(halved_yet, 1)
        if rewards > 0.2 and halved_yet < 2:
          sess.run([tf.assign(policy_lr, policy_lr * 0.5),
                    tf.assign(critic_lr, critic_lr * 0.5)])
          halved_yet = max(halved_yet, 2)
        if rewards > 0.3 and halved_yet < 3:
          sess.run([tf.assign(policy_lr, policy_lr * 0.5),
                    tf.assign(critic_lr, critic_lr * 0.5)])
          halved_yet = max(halved_yet, 3)

      print "\t", rewards

    if t % FLAGS.eval_interval == 0 or t + 1 == FLAGS.num_iter:
      save_path = os.path.join(FLAGS.logdir, "model.ckpt")
      saver.save(sess, save_path, global_step=t)


def test(dpg):
  sess = tf.get_default_session()

  mean_reward = tf.reduce_mean(
      tf.reduce_sum(dpg.rewards_pred, 0)
      / (tf.to_float(dpg.real_lengths) - 1.0))
  seq_pred = tf.transpose(
      tf.pack([tf.argmax(pred_t, 1) for pred_t in dpg.a_pred]))

  for t in xrange(FLAGS.num_iter):
    inputs, lengths = make_batch(FLAGS.batch_size, FLAGS.seq_length,
                                 FLAGS.variable_length)
    feed_dict = {dpg.input_tokens[t]: inputs[t]
                 for t in range(FLAGS.seq_length)}
    feed_dict[dpg.real_lengths] = lengths

    # Run a batch of rollouts and calculate average reward.
    rewards_t = sess.run(mean_reward, feed_dict)
    #rewards_t, seq_t = sess.run([mean_reward, seq_pred], feed_dict)
    print rewards_t
    #pprint.pprint(zip(inputs.T, seq_t))


def main(unused_args):
  try:
    os.makedirs(FLAGS.logdir)
  except: pass
  with open(os.path.join(FLAGS.logdir, "flags"), "w") as flagfile:
    pprint.pprint(FLAGS.__dict__["__flags"], flagfile)

  FLAGS.policy_dims = [int(x) for x in filter(None, FLAGS.policy_dims.split(","))]
  FLAGS.critic_dims = [int(x) for x in filter(None, FLAGS.critic_dims.split(","))]

  state_dim = FLAGS.embedding_dim#FLAGS.policy_dims[0] + FLAGS.embedding_dim #* 2
  mdp_spec = util.MDPSpec(state_dim, FLAGS.embedding_dim)#FLAGS.policy_dims[0])
  dpg_spec = util.DPGSpec(FLAGS.policy_dims, FLAGS.critic_dims)

  dpg = SortingDPG(mdp_spec, dpg_spec, FLAGS.embedding_dim,
                   FLAGS.vocab_size, FLAGS.seq_length, tau=FLAGS.tau)

  if FLAGS.mode == "train":
    policy_lr, critic_lr, policy_update, critic_update = build_updates(dpg)

    if FLAGS.pretrain_autoencoder > 0:
      autoencoder = build_autoencoder(dpg)

    if FLAGS.verbose_summaries:
      util.add_histogram_summaries(set(dpg.policy_params + dpg.critic_params))

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())

      if FLAGS.pretrain_autoencoder > 0:
        pretrain_autoencoder(dpg, autoencoder, FLAGS.pretrain_autoencoder)

      train(dpg, policy_lr, critic_lr, policy_update, critic_update)

  elif FLAGS.mode == "test":
    with tf.Session() as sess:
      saver = tf.train.Saver()
      saver.restore(sess, FLAGS.checkpoint_path)

      test(dpg)


if __name__ == "__main__":
  util.read_flagfile()
  tf.app.run()
