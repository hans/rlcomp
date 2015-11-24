"""
Implementation of a deep deterministic policy gradient RL learner.
Roughly follows algorithm described in Lillicrap et al. (2015).
"""

from functools import partial

import tensorflow as tf

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
