"""Defines a pointer network decoder."""


import tensorflow as tf
from tensorflow.models.rnn import linear


def ptr_net_decoder(decoder_inputs, initial_state, attention_states, cell,
                    num_heads=1, loop_function=None, dtype=tf.float32,
                    real_lengths=None, scope=None):
  """
  Pointer network decoder.

  This is a barely modified version of the attention decoder from
  `tensorflow.models.rnn.seq2seq`.

  Args:
    decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    output_size: size of the output vectors; if None, we use cell.output_size.
    num_heads: number of attention heads that read from attention_states.
    loop_function: if not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x cell.output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x cell.input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    real_lengths: TODO
    scope: VariableScope for the created subgraph; default: "attention_decoder".
  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors of shape
      [batch_size x seq_length]. These represent the generated outputs.
      Output i is computed from input i (which is either i-th decoder_inputs or
      loop_function(output {i-1}, i)) as follows. First, we run the cell
      on a combination of the input and previous attention masks:
        cell_output, new_state = cell(linear(input, prev_attn), prev_state).
      Then, we calculate new attention masks:
        new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
      and then we calculate the output:
        output = linear(cell_output, new_attn).
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].
    seen_inputs: Inputs used at each timestep. When `loop_function` isn't set,
      this is simply a reference to `decoder_inputs`. When `loop_function` is
      set, this is a list of the dynamically computed inputs used at each
      timestep. (Note that `inputs[0] == decoder_inputs[0]` is always true.)
  Raises:
    ValueError: when num_heads is not positive, there are no inputs, or shapes
      of attention_states are not set.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())

  with tf.variable_scope(scope or "ptrnet_decoder"):
    batch_size = tf.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = tf.reshape(attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    attention_vec_size = attn_size    # Size of query vectors for attention.
    for a in xrange(num_heads):
      k = tf.get_variable("AttnW_%d" % a, [1, 1, attn_size, attention_vec_size])
      hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(tf.get_variable("AttnV_%d" % a, [attention_vec_size]))

    states = [initial_state]

    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""

      a = 0
      with tf.variable_scope("Attention_%i" % a):
        y = linear.linear(query, attention_vec_size, True)
        y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
        # Attention mask is a softmax of v^T * tanh(...).
        s = tf.reduce_sum(v[a] * tf.tanh(hidden_features[a] + y), [2, 3])
        a = tf.nn.softmax(s)

        return a

    seen_inputs = []
    outputs = []
    prev = None
    batch_attn_size = tf.pack([batch_size, attn_size])
    attns = [tf.zeros(batch_attn_size, dtype=dtype)
             for _ in xrange(num_heads)]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])

    # We need to mask outputs for short sequences which don't require attention
    # over the entire unrolled input sequence graph.
    if real_lengths is not None:
      output_mask = tf.concat(1, [tf.expand_dims(real_lengths >= attn_length - t, 1)
                                  for t in range(len(decoder_inputs))])
      output_mask_val = tf.zeros(tf.pack([batch_size, attn_length]))

    # Begin recurrence.
    for i in xrange(len(decoder_inputs)):
      if i > 0:
        tf.get_variable_scope().reuse_variables()
      inp = decoder_inputs[i]

      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with tf.variable_scope("loop_function", reuse=True):
          inp = tf.stop_gradient(loop_function(prev, i))
      seen_inputs.append(inp)

      # Merge input and previous attentions into one vector of the right size.
      x = linear.linear([inp] + attns, cell.input_size, True,
                        scope="inp_to_hidden")

      # Run the RNN.
      cell_output, new_state = cell(x, states[-1])
      states.append(new_state)

      # Run the attention mechanism.
      output = attention(cell_output)

      # Mask outputs for shorter sequences.
      if real_lengths is not None:
        output = tf.select(output_mask, output, output_mask_val)
        # Renormalize
        output = output / tf.reduce_sum(output, 1, keep_dims=True)

      if loop_function is not None:
        # We do not propagate gradients over the loop function.
        prev = tf.stop_gradient(output)
      outputs.append(output)

  return outputs, states, seen_inputs
