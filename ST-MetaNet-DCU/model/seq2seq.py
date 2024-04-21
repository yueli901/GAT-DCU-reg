import tensorflow as tf
import numpy as np

from model.cell import RNNCell
from model.graph import Graph
from model.basic_structure import MLP

class Encoder(tf.keras.Model):
    """ Seq2Seq encoder. """
    def __init__(self, cells, graphs, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.cells = cells
        self.graphs = graphs

    def call(self, feature, data):
        """
        Encode the temporal sequence sequence.

        Parameters
        ----------
        feature: a Tensor with shape [n, d].
        data: a Tensor with shape [n, b, t, d].

        Returns
        -------
        states: a list of hidden states (list of hidden units with shape [n, b, hiddens]) of RNNs.
        """

        states = []
        for depth, cell in enumerate(self.cells):
            # rnn unroll
            data, state = cell(feature, data, None, training=None) 
            states.append(state)
            # graph attention
            # if self.graphs[depth] is not None:
            #     data = tf.reduce_mean([g(data, feature) for g in self.graphs[depth]], axis=0)

        return states


class Decoder(tf.keras.Model):
    """ Seq2Seq decoder. """
    def __init__(self, cells, graphs, input_dim, output_dim, use_sampling, cl_decay_steps, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.cells = cells
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_sampling = use_sampling
        self.global_steps = tf.Variable(0.0, trainable=False)
        self.cl_decay_steps = float(cl_decay_steps)

        self.graphs = graphs
        
        # Initialize projection layer for the output
        self.proj = tf.keras.layers.Dense(output_dim)

    def sampling(self):
        """ Schedule sampling: sampling the ground truth. """
        threshold = self.cl_decay_steps / (self.cl_decay_steps + tf.exp(self.global_steps / self.cl_decay_steps))
        return float(np.random.random() < threshold)

    def call(self, feature, label, begin_states, is_training):
        """
        Decode the hidden states to a temporal sequence.

        Parameters
        ----------
        feature: a Tensor with shape [n, d].
        label: a Tensor with shape [n, b, t, d].
        begin_states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.
        is_training: bool

        Returns
        -------
        outputs: the prediction, which is a Tensor with shape [n, b, t, d]
        """
        num_nodes, batch_size, seq_len, _ = label.shape 
        aux = label[:, :, :, self.output_dim:]  # [n, b, t, d]
        label = label[:, :, :, :self.output_dim]  # [n, b, t, d]

        go = tf.zeros(shape=(num_nodes, batch_size, self.input_dim))
        output, states = [], begin_states

        for i in range(seq_len):
            # get next input
            if i == 0:
                data = go
            else:
                prev = tf.concat([output[i - 1], aux[:, :, i - 1]], axis=-1)
                truth = tf.concat([label[:, :, i - 1], aux[:, :, i - 1]], axis=-1)
                value = self.sampling() if is_training and self.use_sampling else 0
                data = value * truth + (1 - value) * prev

            # unroll 1 step
            for depth, cell in enumerate(self.cells):
                data = tf.expand_dims(data, axis=2) # add t axis
                data, states[depth] = cell(feature, data, states[depth], training=is_training)
                data = tf.squeeze(data, axis=2) # delete t axis
                if self.graphs[depth] is not None:
                    data = tf.reduce_mean([g(data, feature) for g in self.graphs[depth]], axis=0)

            # append feature to output
            _feature = tf.expand_dims(feature, axis=1)  # [n, 1, d]
            _feature = tf.broadcast_to(_feature, (num_nodes, batch_size, feature.shape[1]))  # [n, b, d]
            data = tf.concat([data, _feature], axis=-1)  # [n, b, d]

            # project output to prediction
            data = tf.reshape(data, (num_nodes * batch_size, -1))
            data = self.proj(data)
            data = tf.reshape(data, (num_nodes, batch_size, -1))

            output.append(data)

        output = tf.stack(output, axis=2)
        return output
        

class Seq2Seq(tf.keras.Model):
    def __init__(self, 
                 geo_hiddens, 
                 rnn_type, rnn_hiddens,
                 graph_type, graph,
                 input_dim, output_dim,
                 use_sampling,
                 cl_decay_steps,
                 **kwargs):
        super(Seq2Seq, self).__init__(**kwargs)

        # Initialize encoder
        encoder_cells = []
        encoder_graphs = []
        for i, hidden_size in enumerate(rnn_hiddens):
            pre_hidden_size = input_dim if i == 0 else rnn_hiddens[i - 1]
            c = RNNCell.create(rnn_type[i], pre_hidden_size, hidden_size)
            g = Graph.create_graphs('None' if i == len(rnn_hiddens) - 1 else graph_type[i], graph, hidden_size) 
            encoder_cells.append(c)
            encoder_graphs.append(g)
        self.encoder = Encoder(encoder_cells, encoder_graphs)

        # Initialize decoder
        decoder_cells = []
        decoder_graphs = []
        for i, hidden_size in enumerate(rnn_hiddens):
            pre_hidden_size = input_dim if i == 0 else rnn_hiddens[i - 1]
            c = RNNCell.create(rnn_type[i], pre_hidden_size, hidden_size)
            g = Graph.create_graphs(graph_type[i], graph, hidden_size)
            decoder_cells.append(c)
            decoder_graphs.append(g)
        self.decoder = Decoder(decoder_cells, decoder_graphs, input_dim, output_dim, use_sampling, cl_decay_steps)

        # Initialize geo encoder network (node meta knowledge learner)
        self.geo_encoder = MLP(geo_hiddens, act_type='relu', out_act=True)

    def call(self, feature, data, label, is_training):
        # Geo-feature embedding (NMK Learner)
        feature = self.geo_encoder(feature)  # [n, d] # reduce_mean deleted

        # Seq2Seq encoding process
        states = self.encoder(feature, data)
        
        # Seq2Seq decoding process
        output = self.decoder(feature, label, states, is_training)  # [n, b, t, d]

        return output


def net(settings):
    from data.dataloader import get_geo_feature
    _, graph = get_geo_feature()

    net = Seq2Seq(
        geo_hiddens = settings['model']['geo_hiddens'],
        rnn_type    = settings['model']['rnn_type'],
        rnn_hiddens = settings['model']['rnn_hiddens'],
        graph_type  = settings['model']['graph_type'],
        graph       = graph,
        input_dim   = settings['dataset']['input_dim'],
        output_dim  = settings['dataset']['output_dim'],
        use_sampling    = settings['training']['use_sampling'],
        cl_decay_steps  = settings['training']['cl_decay_steps']
    )
    return net
