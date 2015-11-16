from collections import namedtuple

import tensorflow as tf
from tensorflow.python.ops.variable_scope import _VariableScope # HACK


# MDP specification
MDP = namedtuple("MDP", ["state_dim", "action_dim"])


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

    Wi = ((track_scope and track_variable(Wi_name, track_scope))
          or tf.get_variable("W%i" % i, (src_dim, tgt_dim)))
    x = tf.matmul(x, Wi)

    final_layer = i == len(layer_dims) - 2
    if not final_layer or bias_output:
      bi = ((track_scope and track_variable(bi_name, track_scope))
            or tf.get_variable("b%i" % i, (tgt_dim,),
                               initializer=tf.zeros_initializer))
      x += bi

    if not final_layer:
      x = f(x)

  return x


class ReplayBuffer(object):

  """
  Experience replay storage, defined relative to an MDP.

  Stores experience tuples `(s_t, a_t, r_t, s_{t+1})` in a fixed-size cyclic
  buffer and randomly samples from this buffer on demand.
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
    if len(self.states) - 1 < batch_size:
      raise ValueError("Not enough examples in buffer (just %i) to fill a batch of %i."
               % (len(self.states), batch_size))

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
