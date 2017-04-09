import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh, sigmoid
from tensorflow.python.ops import variable_scope as vs


class QRNCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, input_size=None, activation=tanh):
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or 'qrn_cell'):
            with vs.variable_scope('update_gate'):
                z = sigmoid(linear(inputs[0] * inputs[1], self._num_units))
            with vs.variable_scope('candidate'):
                c = self._activation(linear(tf.concat(inputs, 1), self._num_units))
            with vs.variable_scope('reset_gate'):
                r = sigmoid(linear(inputs[0] * inputs[1], self._num_units))
            new_h = z * r * c + (1 - z) * state
        return new_h, new_h


def linear(input_, output_size, scope=None):
    """
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
    args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
    """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term
