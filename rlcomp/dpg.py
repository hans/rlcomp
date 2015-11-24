"""
Implementation of a deep deterministic policy gradient RL learner.
Roughly follows algorithm described in Lillicrap et al. (2015).
"""

from functools import partial

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell, seq2seq

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
    return util.mlp(inp, tf.shape(inp)[1], mdp.action_dim,
                    hidden=spec.policy_dims, track_scope=track_scope)


def noise_gaussian(inp, actions, stddev, name="noiser"):
    noise = tf.random_normal(tf.shape(actions), 0, stddev)
    return actions + noise


def critic_model(inp, actions, mdp, spec, name="critic", reuse=None,
                 track_scope=None):
  """
  Predict the Q-value of the given state-action pairs.

  Returns:
    `batch_size` vector of Q-value predictions.
  """

  # TODO remove magic numbers
  with tf.variable_scope(name, reuse=reuse,
                         initializer=tf.truncated_normal_initializer(stddev=0.25)):
    output = util.mlp(tf.concat(1, [inp, actions]),
                      mdp.state_dim + mdp.action_dim, 1,
                      hidden=spec.critic_dims, bias_output=True,
                      track_scope=track_scope)

    return tf.squeeze(tf.tanh(output))


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

      self._make_inputs()
      self._make_graph()
      self._make_objectives()
      self._make_updates()

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
    with tf.variable_scope("policy", reuse=True) as policy_vs:
      policy_params = [var for var in tf.all_variables()
                       if var.name.startswith(policy_vs.name + "/")]
    with tf.variable_scope("critic", reuse=True) as critic_vs:
      critic_params = [var for var in tf.all_variables()
                       if var.name.startswith(critic_vs.name + "/")]
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


class RecurrentDPG(DPG):

  """
  DPG sequence model.

  This recurrent DPG is recurrent over both an input sequence and the actual
  policy / decision process. It accepts an input token sequence (a sequence of
  `batch_size * input_dim` tensors) and encodes this sequence using an RNN. Let
  $e$ refer to this encoded sequence representation. Then it computes a rollout
  using a recurrent deterministic policy $\pi(e, h_{t-1})$, where $h_{t-1}$ is
  some hidden representation computed in the recurrence.
  """

  def __init__(self, mdp, spec, input_dim, vocab_size, seq_length, **kwargs):
    super(RecurrentDPG, self).__init__(mdp, spec, **kwargs)

    self.input_dim = input_dim
    self.vocab_size = vocab_size
    self.seq_length = seq_length

  def _make_inputs(self):
    self.inputs = (self.inputs
                   or [tf.placeholder(tf.int32, (None, self.input_dim),
                                      name="inputs_%t" % t)
                       for t in range(self.seq_length)])
    self.q_targets = (self.q_targets
                      or [tf.placeholder(tf.float32, (None,),
                                         name="q_targets_%t" % t)
                          for t in range(self.seq_length)])
    # TODO design choice: should we update on an entire trajectory like
    # sketched out above? Or just accept q_targets for particular timesteps
    # to avoid too heavily correlated batches?
    self.tau = self.tau or tf.placeholder(tf.float32, (1,), name="tau")

  class PolicyRNNCell(rnn_cell.RNNCell):

    """
    Simple MLP policy.

    Maps from decoder hidden state to continuous action space using a basic
    feedforward neural network.
    """

    def __init__(self, encoder_seq, cell, dpg):
      self._encoder_seq = encoder_seq

      self._cell = cell
      self._dpg = dpg

    @property
    def input_size(self):
      return self._cell.input_size

    @property
    def output_size(self):
      return self._dpg.mdp_spec.action_dim

    @property
    def state_size(self):
      return self._cell.state_size

    def __call__(self, inputs, state, scope=None):
      # Run the wrapped cell.
      output, res_state = self._cell(inputs, state)

      with tf.variable_scope(scope or type(self).__name__):
        actions = policy_model(output, self._dpg.mdp_spec)

      return actions, res_state

  def _make_graph(self):
    # Encode sequence.
    encoder_cell = rnn_cell.GRUCell(self.spec.policy_dims[0])
    encoder_cell = rnn_cell.EmbeddingWrapper(cell, self.vocab_size)
    _, encoder_states = rnn.rnn(cell, self.inputs)
    # Omit initial state.
    self.encoder_states = encoder_states[1:]
    # DEV
    assert len(self.encoder_states) == self.seq_length

    decoder_cell = rnn_cell.GRUCell(self.spec.policy_dims[0])
    decoder_cell = self._policy_cell(self.encoder_states, decoder_cell)

    # Prepare dummy decoder inputs.
    input_shape = tf.pack([tf.shape(self.inputs[0])[0],
                           decoder_cell.input_size])
    decoder_inputs = [tf.zeros(input_shape, dtype=tf.float32)
                      for _ in range(self.seq_length)]

    # Build decoder loop function which maps from decoder outputs / policy
    # actions to decoder inputs.
    loop_function = self._loop_function(self.encoder_states)

    self.a_pred, self.decoder_states = seq2seq.rnn_decoder(
        decoder_inputs, self.encoder_states[-1], decoder_cell,
        loop_function=loop_function)

    # DEV
    assert len(self.decoder_states) == len(self.encoder_states)

    self.a_explore = self.noiser(self.inputs, self.a_pred)

    # Build main model: critic (on- and off-policy)
    self.critic_on = self._critic(self.decoder_states, self.a_pred)
    self.critic_off = self._critic(self.decoder_states, self.a_explore,
                                   reuse=True)

  def _policy_cell(self, decoder_cell):
    """
    Build a policy RNN cell wrapper around the given decoder cell.

    Args:
      decoder_cell: An `RNNCell` instance which implements the hidden-layer
        recurrence of the decoder / policy

    Returns:
      An `RNNCell` instance which wraps `decoder_cell` and produces outputs in
      action-space.
    """
    # By default, use a simple MLP policy.
    return self.PolicyRNNCell(self.encoder_states, decoder_cell, self)

  def _loop_function(self):
    """
    Build a function which maps from decoder outputs to decoder inputs.

    Returns:
      A function which accepts two arguments `output_t, t`. `output_t` is a
      `batch_size * action_dim` tensor and `t` is an integer.
    """
    raise NotImplementedError("abstract method")

  def _critic(self, states_list, actions_list, reuse=None):
    scores = []
    for t, (states_t, actions_t) in enumerate(zip(states_list, actions_list)):
      reuse_t = (reuse or t > 0) or None
      scores.append(critic_model(self.states_t, actions_t, self.mdp_spec,
                                 self.spec, name="critic", reuse=reuse_t))

    return scores

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
