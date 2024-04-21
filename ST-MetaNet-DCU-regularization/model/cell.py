import tensorflow as tf
from model.basic_structure import MetaDense
from config import MODEL

class MyGRUCell(tf.keras.layers.Layer):
    """
    Common implementation of a GRU layer for inputs shape [n, b, t, d]
    outputs_reshaped [n, b, t, hidden]
    state_reshaped [n, b, hidden]
    """
    def __init__(self, hidden_size):
        super(MyGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.cell = tf.keras.layers.GRUCell(self.hidden_size)

    def call(self, feature, inputs, states, training=None):
        # Inputs shape: [n, b, t, d]
        n, b, t, d = inputs.shape

        if states is None:
            states = tf.zeros((n, b, self.hidden_size))
            
        # Reshape to [n*b, t, d] for processing with GRUCell
        inputs_reshaped = tf.reshape(inputs, [n * b, t, d])
        states_reshaped = [tf.reshape(state, [n * b, self.hidden_size]) for state in [states]]
        
        # Process the sequence
        outputs, state = tf.keras.layers.RNN(self.cell, return_sequences=True, return_state=True)(
            inputs_reshaped, initial_state=states_reshaped, training=training
        )

        # Reshape outputs and state back to [n, b, t, hidden] and [n, b, hidden]
        outputs_reshaped = tf.reshape(outputs, [n, b, t, self.hidden_size])
        state_reshaped = tf.reshape(tf.stack(state), [n, b, self.hidden_size])

        return outputs_reshaped, state_reshaped

class MetaGRUCell(tf.keras.layers.Layer):
    """
    input [n, b, t, pre_hidden] pre_hidden size is equal to MyGRUCell (after GAT)
    """
    def __init__(self, pre_hidden_size, hidden_size, meta_hiddens):
        super(MetaGRUCell, self).__init__()
        self.pre_hidden_size = pre_hidden_size
        self.hidden_size = hidden_size

        # Define the MetaDense layers
        self.dense_z = MetaDense(pre_hidden_size + hidden_size, hidden_size, meta_hiddens)
        self.dense_r = MetaDense(pre_hidden_size + hidden_size, hidden_size, meta_hiddens)
        self.dense_i2h = MetaDense(pre_hidden_size, hidden_size, meta_hiddens)
        self.dense_h2h = MetaDense(hidden_size, hidden_size, meta_hiddens)

    def call(self, feature, inputs, states, training=None):
        # inputs is [n, b, t, pre_hidden] and states is [n, b, hidden]
        n, b, length, _ = inputs.shape

        if states is None:
            states = tf.zeros((n, b, self.hidden_size))

        outputs = []
        for t in range(length):
            z = tf.sigmoid(self.dense_z(feature, tf.concat([inputs[:, :, t, :], states], axis=-1)))
            r = tf.sigmoid(self.dense_r(feature, tf.concat([inputs[:, :, t, :], states], axis=-1)))
            state = z * states + (1 - z) * tf.tanh(self.dense_i2h(feature, inputs[:, :, t, :]) + self.dense_h2h(feature, r*states))
            outputs.append(state)
            states = state

        outputs = tf.stack(outputs, axis=2)
        return outputs, states


class RNNCell:
    @staticmethod
    def create(rnn_type, pre_hidden_size, hidden_size, meta_hiddens=None):
        """
        Create a RNN cell.

        Parameters
        ----------
        rnn_type: str
            Type of RNN cell ('MyGRUCell' or 'MetaGRUCell').
        pre_hidden_size: int
            The hidden size of the previous layer.
        hidden_size: int
            The hidden size for the RNN cell.
        meta_hiddens: list of int, optional
            The list of hidden units for MetaGRUCell's MetaDense layers.

        Returns
        -------
        An instance of MyGRUCell or MetaGRUCell.
        """
        if rnn_type == 'MyGRUCell':
            return MyGRUCell(hidden_size)
        elif rnn_type == 'MetaGRUCell':
            return MetaGRUCell(pre_hidden_size, hidden_size, meta_hiddens=MODEL['meta_hiddens'])
        else:
            raise ValueError('Unknown rnn type: %s' % rnn_type)
