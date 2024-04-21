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

# #####################################################################
# import numpy as np
# import mxnet as mx
# from mxnet import nd
# from mxnet.gluon import Block, rnn, nn

# from model.basic_structure import MetaDense
# from config import MODEL

# class RNNCell(Block):
#     def __init__(self, prefix):
#         super(RNNCell, self).__init__(prefix=prefix)

#     @staticmethod
#     def create(rnn_type, pre_hidden_size, hidden_size, prefix):
#         if rnn_type == 'MyGRUCell': return MyGRUCell(hidden_size, prefix)
#         elif rnn_type == 'MetaGRUCell': return MetaGRUCell(pre_hidden_size, hidden_size, meta_hiddens=MODEL['meta_hiddens'], prefix=prefix)
#         else: raise Exception('Unknown rnn type: %s' % rnn_type)
    
#     def forward_single(self, feature, data, begin_state):
#         """ Unroll the recurrent cell with one step

#         Parameters
#         ----------
#         data: a NDArray with shape [n, b, d].
#         feature: a NDArray with shape [n, d].
#         begin_state: a NDArray with shape [n, b, d]

#         Returns
#         -------
#         output: ouptut of the cell, which is a NDArray with shape [n, b, d]
#         states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.

#         """
#         raise NotImplementedError("To be implemented")

#     def forward(self, feature, data, begin_state):
#         """ Unroll the temporal sequence sequence.

#         Parameters
#         ----------
#         data: a NDArray with shape [n, b, t, d].
#         feature: a NDArray with shape [n, d].
#         begin_state: a NDArray with shape [n, b, d]

#         Returns
#         -------
#         output: ouptut of the cell, which is a NDArray with shape [n, b, t, d]
#         states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.

#         """
#         raise NotImplementedError("To be implemented")


# class MyGRUCell(RNNCell):
#     """ A common GRU Cell. """
#     def __init__(self, hidden_size, prefix=None):
#         super(MyGRUCell, self).__init__(prefix=prefix)
#         self.hidden_size = hidden_size
#         with self.name_scope():
#             self.cell = rnn.GRUCell(self.hidden_size)

#     def forward_single(self, feature, data, begin_state):
#         # add a temporal axis
#         data = nd.expand_dims(data, axis=2)

#         # unroll
#         data, state = self(feature, data, begin_state)

#         # remove the temporal axis
#         data = nd.mean(data, axis=2)

#         return data, state

#     def forward(self, feature, data, begin_state):
#         n, b, length, _ = data.shape

#         # reshape the data and states for rnn unroll
#         data = nd.reshape(data, shape=(n * b, length, -1)) # [n * b, t, d]
#         if begin_state is not None:
#             begin_state = [
#                 nd.reshape(state, shape=(n * b, -1)) for state in begin_state
#             ] # [n * b, d]
        
#         # unroll the rnn
#         data, state = self.cell.unroll(length, data, begin_state, merge_outputs=True)

#         # reshape the data & states back
#         data = nd.reshape(data, shape=(n, b, length, -1))
#         state = [nd.reshape(s, shape=(n, b, -1)) for s in state]

#         return data, state

# class MetaGRUCell(RNNCell):
#     """ Meta GRU Cell. """

#     def __init__(self, pre_hidden_size, hidden_size, meta_hiddens, prefix=None):
#         super(MetaGRUCell, self).__init__(prefix=prefix)
#         self.pre_hidden_size = pre_hidden_size
#         self.hidden_size = hidden_size
#         with self.name_scope():
#             self.dense_z = MetaDense(pre_hidden_size + hidden_size, hidden_size, meta_hiddens=meta_hiddens, prefix='dense_z_')
#             self.dense_r = MetaDense(pre_hidden_size + hidden_size, hidden_size, meta_hiddens=meta_hiddens, prefix='dense_r_')

#             self.dense_i2h = MetaDense(pre_hidden_size, hidden_size, meta_hiddens=meta_hiddens, prefix='dense_i2h_')
#             self.dense_h2h = MetaDense(hidden_size, hidden_size, meta_hiddens=meta_hiddens, prefix='dense_h2h_')

#     def forward_single(self, feature, data, begin_state):
#         """ unroll one step

#         Parameters
#         ----------
#         feature: a NDArray with shape [n, d].
#         data: a NDArray with shape [n, b, d].        
#         begin_state: a NDArray with shape [n, b, d].
        
#         Returns
#         -------
#         output: ouptut of the cell, which is a NDArray with shape [n, b, d]
#         states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.
        
#         """
#         if begin_state is None:
#             num_nodes, batch_size, _ = data.shape
#             begin_state = [nd.zeros((num_nodes, batch_size, self.hidden_size), ctx=feature.context)]

#         prev_state = begin_state[0]
#         data_and_state = nd.concat(data, prev_state, dim=-1)
#         z = nd.sigmoid(self.dense_z(feature, data_and_state))
#         r = nd.sigmoid(self.dense_r(feature, data_and_state))

#         state = z * prev_state + (1 - z) * nd.tanh(self.dense_i2h(feature, data) + self.dense_h2h(feature, r * prev_state))
#         return state, [state]

#     def forward(self, feature, data, begin_state):
#         num_nodes, batch_size, length, _ = data.shape

#         data = nd.split(data, axis=2, num_outputs=length, squeeze_axis=1)

#         outputs, state = [], begin_state
#         for input in data:
#             output, state = self.forward_single(feature, input, state)
#             outputs.append(output)

#         outputs = nd.stack(*outputs, axis=2)
#         return outputs, state