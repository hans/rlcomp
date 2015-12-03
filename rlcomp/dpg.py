"""
Implementation of a deep deterministic policy gradient RL learner.
Roughly follows algorithm described in Lillicrap et al. (2015).
"""

from functools import partial

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell, seq2seq

from rlcomp.pointer_network import ptr_net_decoder
from rlcomp import util


def policy_model(inp, mdp, spec, name="policy", reuse=None,
                 track_scope=None):
  """
  Predict actions for the given input batch.

  Returns:
    actions: `batch_size * action_dim`
  """

  # TODO remove magic numbers
  with tf.variable_scope(name, reuse=reuse,
                         initializer=tf.truncated_normal_initializer(stddev=0.5)):
    return util.mlp(inp, mdp.state_dim, mdp.action_dim,
                    hidden=spec.policy_dims, track_scope=track_scope)


def noise_gaussian(inp, actions, stddev, name="noiser"):
  # Support list `actions` argument.
  if isinstance(actions, list):
    return [noise_gaussian(inp, actions_t, stddev, name=name)
            for actions_t in actions]

  noise = tf.random_normal(tf.shape(actions), 0, stddev)
  return actions + noise


def critic_model(inp, actions, mdp, spec, name="critic", reuse=None,
                 track_scope=None):
  """
  Predict the Q-value of the given state-action pairs.

  Returns:
    `batch_size` vector of Q-value predictions.
  """

  with tf.variable_scope(name, reuse=reuse):
    output = util.mlp(tf.concat(1, [inp, actions]),
                      mdp.state_dim + mdp.action_dim, 1,
                      hidden=spec.critic_dims, bias_output=True,
                      track_scope=track_scope)

    return tf.squeeze(output)


class DPG(object):

  def __init__(self, mdp, spec, inputs=None, q_targets=None, tau=None,
               noiser=None, name="dpg"):
    """
    Args:
      mdp:
      spec:
      inputs: Tensor of input values
      q_targets: Tensor of Q-value targets
    """

    if noiser is None:
      # TODO remove magic number
      noiser = partial(noise_gaussian, stddev=0.1)
    self.noiser = noiser

    # Hyperparameters
    self.mdp_spec = mdp
    self.spec = spec

    # Inputs
    self.inputs = inputs
    self.q_targets = q_targets
    self.tau = tau

    self.name = name

    with tf.variable_scope(self.name) as vs:
      self._vs = vs

      self._make_params()
      self._make_inputs()
      self._make_graph()
      self._make_objectives()
      self._make_updates()

  def _make_params(self):
    pass

  def _make_inputs(self):
    self.inputs = (self.inputs
                   or tf.placeholder(tf.float32, (None, self.mdp_spec.state_dim),
                                     name="inputs"))
    self.q_targets = (self.q_targets
                      or tf.placeholder(tf.float32, (None,), name="q_targets"))
    self.tau = self.tau or tf.placeholder(tf.float32, (1,), name="tau")

  def _make_graph(self):
    # Build main model: actor
    self.a_pred = policy_model(self.inputs, self.mdp_spec, self.spec,
                               name="policy")
    self.a_explore = self.noiser(self.inputs, self.a_pred)

    # Build main model: critic (on- and off-policy)
    self.critic_on = critic_model(self.inputs, self.a_pred, self.mdp_spec,
                                  self.spec, name="critic")
    self.critic_off = critic_model(self.inputs, self.a_explore, self.mdp_spec,
                                   self.spec, name="critic", reuse=True)

    # Build tracking models.
    self.a_pred_track = policy_model(self.inputs, self.mdp_spec, self.spec,
                                     track_scope="%s/policy" % self.name,
                                     name="policy_track")
    self.critic_on_track = critic_model(self.inputs, self.a_pred, self.mdp_spec,
                                        self.spec, name="critic_track",
                                        track_scope="%s/critic" % self.name)

  def _make_objectives(self):
    # TODO: Hacky, will cause clashes if multiple DPG instances.
    # Can't instantiate a VS cleanly either, because policy params might be
    # nested in unpredictable way by subclasses.
    policy_params = [var for var in tf.all_variables()
                     if "policy/" in var.name]
    critic_params = [var for var in tf.all_variables()
                     if "critic/" in var.name]
    self.policy_params = policy_params
    self.critic_params = critic_params

    # Policy objective: maximize on-policy critic activations
    self.policy_objective = -tf.reduce_mean(self.critic_on)

    # Critic objective: minimize MSE of off-policy Q-value predictions
    q_errors = tf.square(self.critic_off - self.q_targets)
    self.critic_objective = tf.reduce_mean(q_errors)

  def _make_updates(self):
    # Make tracking updates.
    policy_track_update = util.track_model_updates(
         "%s/policy" % self.name, "%s/policy_track" % self.name, self.tau)
    critic_track_update = util.track_model_updates(
        "%s/critic" % self.name, "%s/critic_track" % self.name, self.tau)
    self.track_update = tf.group(policy_track_update, critic_track_update)

    # SGD updates are left to client.


class PointerNetDPG(DPG):

  """
  Sequence-to-sequence pointer network DPG implementation.

  This recurrent DPG encodes an input float sequence `x1...xT` into an encoder
  memory sequence `e1...eT`. Using a recurrent decoder, it computes hidden
  states `d1...dT`. Combining these decoder states with an attention scan over
  the encoder memory at each timestep, it produces an entire rollout `a1...aT`
  (sequence of continuous action representations). The action at timestep `ai`
  is used to compute an input to the decoder for the next timestep.

  A recurrent critic model is applied to the action representation at each
  timestep.
  """

  def __init__(self, mdp, spec, input_dim, seq_length, bn_actions=False,
               **kwargs):
    """
    Args:
      mdp:
      spec:
      input_dim: Dimension of input values provided to encoder (`self.inputs`)
      seq_length:
      bn_actions: If true, batch-normalize action outputs.
    """
    self.input_dim = input_dim
    self.seq_length = seq_length

    self.bn_actions = bn_actions

    # state: decoder hidden state + input value
    assert mdp.state_dim == self.input_dim
    # outputs weighted sum of input memories
    assert mdp.action_dim == self.input_dim

    super(PointerNetDPG, self).__init__(mdp, spec, **kwargs)

  def _make_params(self):
    if self.bn_actions:
      with tf.variable_scope("bn"):
        shape = (self.mdp_spec.action_dim,)
        self.bn_beta = tf.Variable(tf.constant(0.0, shape=shape), name="beta")
        self.bn_gamma = tf.Variable(tf.constant(1.0, shape=shape),
                                    name="gamma")

        # Track avg values of the beta + gamma (scale + shift)
        tf.scalar_summary("bn_beta.mean", tf.reduce_mean(self.bn_beta))
        tf.scalar_summary("bn_gamma.mean", tf.reduce_mean(self.bn_gamma))

  def _make_inputs(self):
    if not self.inputs:
      self.inputs = [tf.placeholder(tf.float32, (None, self.input_dim))
                     for _ in range(self.seq_length)]
    self.tau = self.tau or tf.placeholder(tf.float32, (1,), name="tau")

  def _make_graph(self):
    # Encode sequence.
    # TODO: MultilayerRNN?
    encoder_cell = util.GRUCell(self.input_dim, self.spec.policy_dims[0])
    _, self.encoder_states = rnn.rnn(encoder_cell, self.inputs,
                                     dtype=tf.float32, scope="encoder")
    assert len(self.encoder_states) == self.seq_length # DEV

    # Reshape encoder states into an "attention states" tensor of shape
    # `batch_size * seq_length * policy_dim`.
    attn_states = tf.concat(1, [tf.expand_dims(state_t, 1)
                                for state_t in self.inputs])

    # Build a simple GRU-powered recurrent decoder cell.
    decoder_cell = util.GRUCell(self.input_dim, self.spec.policy_dims[0])

    # Prepare dummy encoder input. This will only be used on the first
    # timestep; in subsequent timesteps, the `loop_function` we provide
    # will be used to dynamically calculate new input values.
    batch_size = tf.shape(self.inputs[0])[0]
    dec_inp_shape = tf.pack([batch_size, decoder_cell.input_size])
    dec_inp_dummy = tf.zeros(dec_inp_shape, dtype=tf.float32)
    dec_inp_dummy.set_shape((None, decoder_cell.input_size))
    dec_inp = [dec_inp_dummy] * self.seq_length

    # Build pointer-network decoder.
    self.a_pred, dec_states, dec_inputs = ptr_net_decoder(
        dec_inp, self.encoder_states[-1], attn_states, decoder_cell,
        loop_function=self._loop_function(), scope="decoder")
    # Store dynamically calculated inputs -- critic may want to use these
    self.decoder_inputs = dec_inputs
    # Again strip the initial state.
    self.decoder_states = dec_states[1:]

    # Use noiser to build exploratory rollouts.
    self.a_explore = self.noiser(self.inputs, self.a_pred)

    # Now "dereference" the soft pointers produced by the policy network.
    a_pred_deref = self._deref_rollout(self.a_pred)
    a_explore_deref = self._deref_rollout(self.a_explore)

    # # Optional batch normalization.
    # if self.bn_actions:
    #   # Compute moments over all timesteps (treat as one big batch).
    #   batch_pred = tf.concat(0, self.a_pred)
    #   mean = tf.reduce_mean(batch_pred, 0)
    #   variance = tf.reduce_mean(tf.square(batch_pred - mean), 0)

    #   # TODO track running mean, avg with exponential averaging
    #   # in order to prepare test-time normalization value

    #   # Resize to make BN op happy. (It is built for 4-dim CV applications.)
    #   batch_pred = tf.expand_dims(tf.expand_dims(batch_pred, 1), 1)
    #   batch_pred = tf.nn.batch_norm_with_global_normalization(
    #       batch_pred, mean, variance, self.bn_beta, self.bn_gamma,
    #       0.001, True)
    #   self.a_pred = tf.split(0, self.seq_length, tf.squeeze(batch_pred))

    # Build main model: recurrently apply a critic over the entire rollout.
    _, self.critic_on, self.critic_on_track = self._critic(a_pred_deref)
    self.critic_off_pre, self.critic_off, self.critic_off_track = \
        self._critic(a_explore_deref, reuse=True)

    self._make_q_targets()

  def _make_q_targets(self):
    if not self.q_targets:
      self.q_targets = [tf.placeholder(tf.float32, (None,))
                        for _ in range(self.seq_length)]

  def _policy_params(self):
    return [var for var in tf.all_variables()
            if "encoder/" in var.name or "decoder/" in var.name]

  def _make_objectives(self):
    # TODO: Hacky, will cause clashes if multiple DPG instances.
    policy_params = self._policy_params()
    critic_params = [var for var in tf.all_variables()
                     if "critic/" in var.name]
    self.policy_params = policy_params
    self.critic_params = critic_params

    if self.bn_actions:
      bn_params = [self.bn_beta, self.bn_gamma]
      self.policy_params += bn_params
      self.critic_params += bn_params

    # Policy objective: maximize on-policy critic activations
    mean_critic_over_time = tf.add_n(self.critic_on) / self.seq_length
    mean_critic = tf.reduce_mean(mean_critic_over_time)
    self.policy_objective = -mean_critic

    # DEV
    tf.scalar_summary("critic(a_pred).mean", mean_critic)

    # Critic objective: minimize MSE of off-policy Q-value predictions
    q_errors = [tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(critic_off_t, q_targets_t))#tf.square(critic_off_t - q_targets_t))
                for critic_off_t, q_targets_t
                in zip(self.critic_off_pre, self.q_targets)]
    self.critic_objective = tf.add_n(q_errors) / self.seq_length
    tf.scalar_summary("critic_objective", self.critic_objective)

    mean_critic_off = tf.reduce_mean(tf.add_n(self.critic_off)) / self.seq_length
    tf.scalar_summary("critic(a_explore).mean", mean_critic_off)

    tf.scalar_summary("a_pred.mean", tf.reduce_mean(tf.add_n(self.a_pred)) / self.seq_length)
    tf.scalar_summary("a_pred.maxabs", tf.reduce_max(tf.abs(tf.pack(self.a_pred))))

  def _make_updates(self):
    critic_updates = util.track_model_updates(
        "%s/critic" % self.name, "%s/critic_track" % self.name, self.tau)
    self.track_update = critic_updates

  def _deref_pointer(self, attn_states, soft_ptr):
    """
    Args:
      attn_states: batch_size * seq_length * model_dim tensor
      soft_ptr: batch_size * seq_length soft pointer (softmax outputs, not
        logits)

    Returns:
      batch_size * model_dim weighted sum of input states
    """
    weighted_mems = attn_states * tf.expand_dims(soft_ptr, 2)
    weighted_out = tf.reduce_sum(weighted_mems, 1)
    return weighted_out

  def _deref_rollout(self, rollout):
    attn_states = tf.concat(1, [tf.expand_dims(states_t, 1)
                                for states_t in self.inputs])
    deref = [self._deref_pointer(attn_states, rollout_t)
             for rollout_t in rollout]
    return deref

  def _loop_function(self):
    """
    Build a function which maps from decoder outputs to decoder inputs.

    Returns:
      A function which accepts two arguments `output_t, t`. `output_t` is a
      `batch_size * action_dim` tensor and `t` is an integer.
    """
    # Use logits from output layer to compute a weighted sum of encoder input
    # elements.
    attn_states = tf.concat(1, [tf.expand_dims(states_t, 1)
                                for states_t in self.inputs])
    loop_fn = lambda output_t, t: self._deref_pointer(attn_states, output_t)

    return loop_fn

  def _critic(self, actions_lst, reuse=None):
    scores_pre, scores, scores_track = [], [], []

    # Fetch scaler parameter (shared across critics).
    with tf.variable_scope("critic", reuse=reuse):
      scaler = tf.get_variable("scaler", (1,))
      if reuse is None: # First time fetching this variable; log its value
        tf.scalar_summary("critic/scaler", scaler[0])

    prev_action = tf.zeros_like(actions_lst[0])

    # Evaluate Q(s, a) at each timestep.
    for t, actions_t in enumerate(actions_lst):
      state_t = prev_action

      reuse_t = (reuse or t > 0) or None
      critic_pre = critic_model(state_t, actions_t, self.mdp_spec,
                                self.spec, name="critic", reuse=reuse_t)
#      critic_out *= scaler
      critic_out = tf.sigmoid(critic_pre)

      # Also build a tracking model
      critic_track = critic_model(state_t, actions_t, self.mdp_spec,
                                  self.spec, name="critic_track",
                                  track_scope="%s/critic" % self.name,
                                  reuse=reuse_t)
#      critic_track *= scaler
      critic_track = tf.sigmoid(critic_track)

      prev_action = actions_t

      scores_pre.append(critic_pre)
      scores.append(critic_out)
      scores_track.append(critic_track)

    return scores_pre, scores, scores_track

  def harden_actions(self, action_list):
    """
    Harden the given sequence of soft actions such that they describe a
    concrete trajectory.

    Args:
      action_list: List of Numpy matrices of shape `batch_size * action_dim`
    """
    # TODO: eventually we'd like to run this within a TF graph when possible.
    # We can probably define hardening solely with TF
    raise NotImplementedError("abstract method")
