from collections import namedtuple
import re
import sys

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import linear, rnn_cell, seq2seq
from tensorflow.python.ops.variable_scope import _VariableScope # HACK


# MDP specification
MDPSpec = namedtuple("MDPSpec", ["state_dim", "action_dim"])


# DPG model specification
DPGSpec = namedtuple("DPGSpec", ["policy_dims", "critic_dims"])


def match_variable(name, scope_name):
  """
  Match a variable (initialize with same value) from another variable scope.

  After initialization, the values of the two variables are not tied in any
  way.
  """

  # HACK: Using private _VariableScope API in order to be able to get an
  # absolute-path to the given variable scope name (i.e., not have it treated
  # as a relative path and placed under whatever variable scope might contain
  # this function call)
  with tf.variable_scope(_VariableScope(True, scope_name), reuse=True):
    track_var = tf.get_variable(name)

  # Create a dummy initializer.
  initializer = lambda *args, **kwargs: track_var.initialized_value()

  return tf.get_variable(name, shape=track_var.get_shape(),
                         initializer=initializer)


def track_model_updates(main_name, track_name, tau):
  """
  Build an update op to make parameters of a tracking model follow a main model.

  Call outside of the scope of both the main and tracking model.

  Returns:
    A group of `tf.assign` ops which require no inputs (only parameter values).
  """

  updates = []
  params = [var for var in tf.all_variables()
            if var.name.startswith(main_name + "/")]

  for param in params:
    track_param_name = param.op.name.replace(main_name + "/",
                                             track_name + "/")
    with tf.variable_scope(_VariableScope(True), reuse=True):
      track_param = tf.get_variable(track_param_name)

    # TODO sparse params
    update_op = tf.assign(track_param,
                          tau * param + (1 - tau) * track_param)
    updates.append(update_op)

  return tf.group(*updates)


def mlp(inp, inp_dim, outp_dim, track_scope=None, hidden=None, f=tf.tanh,
        bias_output=False):
  """
  Basic multi-layer neural network implementation, with custom architecture
  and activation function.
  """
  if not hidden:
    hidden = []

  layer_dims = [inp_dim] + hidden + [outp_dim]
  x = inp

  for i, (src_dim, tgt_dim) in enumerate(zip(layer_dims, layer_dims[1:])):
    Wi_name, bi_name = "W%i" % i, "b%i" % i

    Wi = ((track_scope and match_variable(Wi_name, track_scope))
          or tf.get_variable("W%i" % i, (src_dim, tgt_dim)))
    x = tf.matmul(x, Wi)

    final_layer = i == len(layer_dims) - 2
    if not final_layer or bias_output:
      bi = ((track_scope and match_variable(bi_name, track_scope))
            or tf.get_variable("b%i" % i, (tgt_dim,),
                               initializer=tf.zeros_initializer))
      x += bi

    if not final_layer:
      x = f(x)

  return x


class GRUCell(rnn_cell.RNNCell):
  """
  Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Supports inputs of different dimension than state values.
  """

  def __init__(self, input_size, num_units):
    self._input_size = input_size
    self._num_units = num_units

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with tf.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not udpate.
        r, u = tf.split(1, 2, linear.linear([inputs, state],
                                            2 * self._num_units,
                                            True, 1.0))
        r, u = tf.sigmoid(r), tf.sigmoid(u)
      with tf.variable_scope("Candidate"):
        c = tf.tanh(linear.linear([inputs, r * state], self._num_units, True))
      new_h = u * state + (1 - u) * c
    return new_h, new_h


def embedding_rnn_decoder(decoder_inputs, initial_state, cell, num_symbols,
                          output_projection=None, feed_previous=False,
                          scope=None, embedding=None):
  """RNN decoder with embedding and a pure-decoding option.
  Args:
    decoder_inputs: a list of 1D batch-sized int32-Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function.
    num_symbols: integer, how many symbols come into the embedding.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [cell.output_size x num_symbols] and B has
      shape [num_symbols]; if provided and feed_previous=True, each fed
      previous output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
      If False, decoder_inputs are used as given (the standard decoder case).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_decoder".
  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x cell.output_size] containing the generated outputs.
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].
  Raises:
    ValueError: when output_projection has the wrong shape.
  """
  if output_projection is not None:
    proj_weights = tf.convert_to_tensor(output_projection[0], dtype=tf.float32)
    proj_weights.get_shape().assert_is_compatible_with([cell.output_size,
                                                        num_symbols])
    proj_biases = tf.convert_to_tensor(output_projection[1], dtype=tf.float32)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with tf.variable_scope(scope or "embedding_rnn_decoder"):
    if embedding is None:
      with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [num_symbols, cell.input_size])

    def extract_argmax_and_embed(prev, _):
      """Loop_function that extracts the symbol from prev and embeds it."""
      if output_projection is not None:
        prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      return tf.nn.embedding_lookup(embedding, prev_symbol)

    loop_function = None
    if feed_previous:
      loop_function = extract_argmax_and_embed

    emb_inp = [tf.nn.embedding_lookup(embedding, i) for i in decoder_inputs]
    return seq2seq.rnn_decoder(emb_inp, initial_state, cell,
                               loop_function=loop_function)


def add_histogram_summaries(xs):
  for x in xs:
    tf.histogram_summary(x.name, x)


class ReplayBuffer(object):

  """
  Experience replay storage, defined relative to an MDP.

  Stores experience tuples `(s_t, a_t, r_t, s_{t+1})` in a fixed-size cyclic
  buffer and randomly samples tuples from this buffer on demand.
  """

  def __init__(self, buffer_size, mdp):
    self.buffer_size = buffer_size
    self.mdp = mdp

    self.cursor_write_start = 0
    self.cursor_read_end = 0

    self.states = np.empty((buffer_size, mdp.state_dim), dtype=np.float32)
    self.actions = np.empty((buffer_size,), dtype=np.int32)
    self.rewards = np.empty((buffer_size,), dtype=np.int32)
    self.states_next = np.empty_like(self.states)

  def sample(self, batch_size):
    if self.cursor_read_end - 1 < batch_size:
      raise ValueError("Not enough examples in buffer (just %i) to fill a batch of %i."
               % (self.cursor_read_end, batch_size))

    idxs = np.random.choice(self.cursor_read_end, size=batch_size, replace=False)
    return (self.states[idxs], self.actions[idxs], self.rewards[idxs],
            self.states_next[idxs])

  def extend(self, states, actions, rewards, states_next):
    # If the buffer is near full, fit what we can and drop the rest
    remaining_space = self.buffer_size - self.cursor_write_start
    if len(states) >= self.buffer_size - self.cursor_write_start:
      states = states[:remaining_space]
      actions = actions[:remaining_space]
      rewards = rewards[:remaining_space]
      states_next = states_next[:remaining_space]

      # Reset for next time
      self.cursor_write_start = 0

    # Write into buffer.
    self.states[self.cursor_write_start:len(states)] = states
    self.actions[self.cursor_write_start:len(states)] = actions
    self.rewards[self.cursor_write_start:len(states)] = rewards
    self.states_next[self.cursor_write_start:len(states)] = states_next

    self.cursor_read_end = min(self.buffer_size, self.cursor_read_end + len(states))


class RecurrentReplayBuffer(object):

  def __init__(self, buffer_size, mdp, input_dim, seq_length, policy_dim):
    self.buffer_size = buffer_size
    self.mdp = mdp

    self.cursor_write_start = 0
    self.cursor_read_end = 0

    self.inputs = np.empty((buffer_size, input_dim), dtype=np.float32)
    self.states = np.empty((buffer_size, seq_length, policy_dim),
                           dtype=np.float32)
    self.actions = np.empty((buffer_size, seq_length, mdp.action_dim),
                            dtype=np.float32)
    self.rewards = np.empty((buffer_size, seq_length), dtype=np.int32)

  def sample_trajectory(self):
    if self.cursor_read_end == 0:
      raise ValueError("not enough trajectories in buffer (just %i) to fill a "
                       "batch of %i." % (self.cursor_read_end, 1))

    i = np.random.randint(0, self.cursor_read_end)
    return self.inputs[i], self.states[i], self.actions[i], self.rewards[i]

  def add_trajectory(self, inputs, states, actions, rewards):
    self.inputs[self.cursor_write_start] = inputs
    self.states[self.cursor_write_start] = states
    self.actions[self.cursor_write_start] = actions
    self.rewards[self.cursor_write_start] = rewards

    self.cursor_write_start += 1
    self.cursor_write_start = self.cursor_write_start % self.buffer_size

    self.cursor_read_end = min(self.buffer_size, self.cursor_read_end + 1)

  def sample(self, batch_size):
    b_inputs, b_states, b_states_next, b_actions, b_rewards = \
        [], [], [], [], []

    for _ in range(batch_size):
      inputs, states, actions, rewards = self.sample_trajectory()

      # Sample a single timestep.
      t = np.random.randint(0, states.shape[0])
      state, action, reward = states[t], actions[t], rewards[t]
      try:
        state_next = states[t + 1]
      except IndexError:
        # TODO hack.
        state_next = np.zeros((self.mdp.state_dim), dtype=np.float32)#0.0

      b_inputs.append(inputs)
      b_states.append(state)
      b_states_next.append(state_next)
      b_actions.append(action)
      b_rewards.append(reward)

    b_inputs = np.array(b_inputs)
    b_states = np.array(b_states)
    b_states_next = np.array(b_states_next)
    b_actions = np.array(b_actions)
    b_rewards = np.array(b_rewards)

    return b_inputs, b_states, b_states_next, b_actions, b_rewards


def read_flagfile():
  """
  Fake gflag's `flagfile` feature.

  Search for a --flagfile option in `sys.argv`; if it exists; prepend items
  from the flagfile to `sys.argv`.
  """

  flagfile_re = re.compile(r"^--flagfile=?(.*)$", re.I)
  flagfile = None
  remove_slice = None

  # Find a flagfile arg.
  for i, arg in enumerate(sys.argv):
    matches = flagfile_re.findall(arg)
    if matches:
      if matches[0]:
        flagfile = matches[0]
        remove_slice = i, 1
        break
      elif i < len(sys.argv) - 1:
        flagfile = sys.argv[i + 1]
        remove_slice = i, 2
        break

  # Slice out the flagfile arg.
  new_argv = sys.argv
  if flagfile is None:
    return
  elif remove_slice is not None:
    slice_start, slice_len = remove_slice
    new_argv = new_argv[:slice_start] + new_argv[slice_start + slice_len:]

  with open(flagfile, "r") as flagfile_f:
    flags = [line.strip() for line in flagfile_f]

  # Prepend loaded flags directly after script name
  new_argv = new_argv[:1] + flags + new_argv[1:]
  sys.argv = new_argv
