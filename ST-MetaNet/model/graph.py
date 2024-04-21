import tensorflow as tf
import dgl
from dgl import DGLGraph
from model.basic_structure import MLP
from config import MODEL
import h5py
import datetime
import time

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y%m%d-%H%M%S')

class Graph(tf.keras.Model):
    @staticmethod
    def create(graph_type, in_out, edge_feature, edge, hidden_size):
        if graph_type == 'None': return None
        elif graph_type == 'GAT': return GAT(in_out, edge_feature, edge, hidden_size)
        elif graph_type == 'MetaGAT': return MetaGAT(in_out, edge_feature, edge, hidden_size)
        else: raise ValueError('Unknown graph: %s' % graph_type)

    @staticmethod
    def create_graphs(graph_type, graph, hidden_size):
        """
        dist: normalized distance matrix [n,n]
        e_in/e_out: list of lists, neighbor node index for each node
        hidden_size: RNN hidden size
        """
        if graph_type == 'None': return None
        edge_feature_in, edge_feature_out, e_in, e_out = graph
        return [
            Graph.create(graph_type, 'in', edge_feature_in, e_in, hidden_size),
            Graph.create(graph_type, 'out', edge_feature_out, e_out, hidden_size)
        ]

    def __init__(self, in_out, edge_feature, edge, hidden_size, **kwargs):
        super(Graph, self).__init__(**kwargs)
        self.in_out = in_out
        self.edge_feature = tf.convert_to_tensor(edge_feature, dtype=tf.float32)
        self.edge = edge
        self.hidden_size = hidden_size

        # create graph, make sure the information pass is in the correct direction
        if self.in_out == 'in':
            self.num_nodes = n = len(self.edge)
            src, dst = [], []
            for i in range(n):
                for j in edge[i]:
                    src.append(j)
                    dst.append(i)
        else:
            self.num_nodes = n = len(self.edge)
            src, dst = [], []
            for i in range(n):
                for j in edge[i]:
                    src.append(i)
                    dst.append(j)
        
        self.src = src # paired index for each edge [e,]
        self.dst = dst # paired index for each edge [e,]
        self.build_graph()

    def build_graph(self):
        self.g = dgl.graph((self.src, self.dst), num_nodes=self.num_nodes)
        # self.g = self.g.to("/cpu:0")
        # self.g = dgl.DGLGraph()
        # self.g.add_nodes(self.num_nodes) # based on index
        # self.g.add_edges(self.src, self.dst) # form edge based on index pairs
        # self.edge_feature = self.edge_feature.cpu()
        self.g.edata['feature'] = self.edge_feature # add edge feature [e, d]

    def call(self, state, feature):
        self.g.ndata['state'] = state # add node data [n, b, t, hidden]
        self.g.ndata['feature'] = feature # add node data [n, d]
        self.g.update_all(self.msg_edge, self.msg_reduce) # info pass
        state = self.g.ndata.pop('new_state') # [n, b, t, hidden]
        
        # lambda_ = self.g.ndata.pop('lambda_')
        # current_time = datetime.datetime.now()
        # formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
        # time.sleep(1)
        # with h5py.File(f'lambda-{formatted_time}.h5', 'w') as f:
        #     data = f.create_dataset('lambda', shape=lambda_.shape) 
        #     data[:] = lambda_
        return state

    def msg_edge(self, edge):
        raise NotImplementedError("To be implemented")

    def msg_reduce(self, node):
        raise NotImplementedError("To be implemented")


class GAT(Graph):
    def __init__(self, in_out, edge_feature, edge, hidden_size, **kwargs):
        super(GAT, self).__init__(in_out, edge_feature, edge, hidden_size, **kwargs)
        self.weight = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size),
                                      initializer='random_normal',
                                      trainable=True)

    def msg_edge(self, edge):
        state = tf.concat([edge.src['state'], edge.dst['state']], axis=-1) # [e, b, t, 2*hidden]
        alpha = tf.nn.leaky_relu(tf.matmul(state, self.weight)) # [e, b, t, hidden]

        edge_feature = edge.data['feature']
        while len(dist.shape) < len(alpha.shape):
            edge_feature = tf.expand_dims(edge_feature, axis=-1)

        alpha = alpha * edge_feature # [e, b, t, hidden]
        return {'alpha': alpha, 'state': edge.src['state']}

    def msg_reduce(self, node):
        state = node.mailbox['state'] 
        alpha = node.mailbox['alpha']
        alpha = tf.nn.softmax(alpha, axis=1)

        new_state = tf.nn.relu(tf.reduce_sum(alpha * state, axis=1))
        return {'new_state': new_state}


# class MetaGAT(Graph):
#     def __init__(self, in_out, edge_feature, edge, hidden_size):
#         super(MetaGAT, self).__init__(in_out, edge_feature, edge, hidden_size)
#         self.w_mlp = MLP(MODEL['meta_hiddens'] + [hidden_size * hidden_size * 2], 'sigmoid', False)
#         self.b_mlp = MLP(MODEL['meta_hiddens'] + [1], 'sigmoid', False)
#         self.lambda_mlp = MLP(MODEL['meta_hiddens'] + [1], 'sigmoid', False) # map 2*hidden to 1

#     def msg_edge(self, edge):
#         state = tf.concat([edge.src['state'], edge.dst['state']], axis=-1) # [e, b, t, 2*hidden]
#         feature = tf.concat([edge.src['feature'], edge.dst['feature'], edge.data['feature']], axis=-1) # [e, 2*d_node+d_edge]
#         weight = self.w_mlp(feature)
#         bias = self.b_mlp(feature) # [e, 1]
#         weight = tf.reshape(weight, shape=(-1, self.hidden_size * 2, self.hidden_size)) # [e, 2*hidden, hidden]
#         bias = tf.reshape(bias, shape=(-1, 1, 1)) # [e, 1, 1]
        
#         state_shape = state.shape
#         state = tf.reshape(state, shape=(state_shape[0], -1, state_shape[-1])) # [e, b*t, 2*hidden]

#         alpha = tf.nn.leaky_relu(tf.linalg.matmul(state, weight) + bias) # [e, b*t, hidden]
#         alpha = tf.reshape(alpha, shape=state_shape[:-1] + (self.hidden_size,)) # [e, b, t, hidden]
        
#         return {'alpha': alpha, 'state': edge.src['state']}

#     def msg_reduce(self, node):
#         state = node.mailbox['state'] # 5 dim: [n, neighbors_of_n, b, t, hidden]
#         alpha = node.mailbox['alpha']
#         alpha = tf.nn.softmax(alpha, axis=1)

#         original_state = node.data['state']
#         aggregated_neighbour_state = tf.nn.relu(tf.reduce_sum(alpha * state, axis=1)) # [n, b, t, hidden]
#         concatenation = tf.concat([aggregated_neighbour_state, original_state], axis=-1) # [n, b, t, 2*hidden]
#         lambda_ = tf.sigmoid(self.lambda_mlp(concatenation)) # [n, b, t, 1]
#         new_state = lambda_*aggregated_neighbour_state + (1-lambda_)*original_state

#         return {'new_state': new_state, 'lambda_': lambda_}

class MetaGAT(Graph):
    def __init__(self, in_out, edge_feature, edge, hidden_size):
        super(MetaGAT, self).__init__(in_out, edge_feature, edge, hidden_size)
        with tf.name_scope(self.name):  # Use the Layer's built-in name for unique scoping
            self.w_mlp = MLP(MODEL['meta_hiddens'] + [hidden_size * hidden_size * 2], 'sigmoid', False)
            self.b_mlp = MLP(MODEL['meta_hiddens'] + [1], 'sigmoid', False)
            self.weight = self.add_weight(shape=(1, 1), initializer='random_normal', trainable=True, name='fixed_lambda')

    def msg_edge(self, edge):
        state = tf.concat([edge.src['state'], edge.dst['state']], axis=-1) # [e, b, t, 2*hidden]
        feature = tf.concat([edge.src['feature'], edge.dst['feature'], edge.data['feature']], axis=-1) # [e, 2*d_node+d_edge]
        weight = self.w_mlp(feature)
        bias = self.b_mlp(feature) # [e, 1]
        weight = tf.reshape(weight, shape=(-1, self.hidden_size * 2, self.hidden_size)) # [e, 2*hidden, hidden]
        bias = tf.reshape(bias, shape=(-1, 1, 1)) # [e, 1, 1]
        
        state_shape = state.shape
        state = tf.reshape(state, shape=(state_shape[0], -1, state_shape[-1])) # [e, b*t, 2*hidden]

        alpha = tf.nn.leaky_relu(tf.linalg.matmul(state, weight) + bias) # [e, b*t, hidden]
        alpha = tf.reshape(alpha, shape=state_shape[:-1] + (self.hidden_size,)) # [e, b, t, hidden]
        
        return {'alpha': alpha, 'state': edge.src['state']}

    def msg_reduce(self, node):
        state = node.mailbox['state'] # 5 dim: [n, neighbors_of_n, b, t, hidden]
        alpha = node.mailbox['alpha']
        alpha = tf.nn.softmax(alpha, axis=1)

        weight = tf.sigmoid(self.weight)
        original_state = node.data['state']
        aggregated_neighbour_state = tf.nn.relu(tf.reduce_sum(alpha * state, axis=1))
        new_state = weight*aggregated_neighbour_state + (1-weight)*original_state

        return {'new_state': new_state}


# ###########################################################################
# import mxnet as mx
# from mxnet import nd
# from mxnet.gluon import nn, Block

# import dgl
# from dgl import DGLGraph
# from functools import partial

# from config import MODEL

# class Graph(Block):
#     """
#     The base class of GAT and MetaGAT. We implement the methods based on DGL library.
#     """

#     @staticmethod
#     def create(graph_type, dist, edge, hidden_size, prefix):
#         """ create a graph. """
#         if graph_type == 'None': return None
#         elif graph_type == 'GAT': return GAT(dist, edge, hidden_size, prefix=prefix)
#         elif graph_type == 'MetaGAT': return MetaGAT(dist, edge, hidden_size, prefix=prefix)
#         else: raise Exception('Unknow graph: %s' % graph_type)

#     @staticmethod
#     def create_graphs(graph_type, graph, hidden_size, prefix):
#         """ Create a list of graphs according to graph_type & graph. """
#         if graph_type == 'None': return None
#         dist, e_in, e_out = graph
#         return [
#             Graph.create(graph_type, dist.T, e_in, hidden_size, prefix + 'in_'),
#             Graph.create(graph_type, dist, e_out, hidden_size, prefix + 'out_')
#         ]

#     def __init__(self, dist, edge, hidden_size, prefix=None):
#         super(Graph, self).__init__(prefix=prefix)
#         self.dist = dist
#         self.edge = edge
#         self.hidden_size = hidden_size

#         # create graph
#         self.num_nodes = n = self.dist.shape[0]
#         src, dst, dist = [], [], []
#         for i in range(n):
#             for j in edge[i]:
#                 src.append(j)
#                 dst.append(i)
#                 dist.append(self.dist[j, i])

#         self.src = src
#         self.dst = dst
#         self.dist = mx.nd.expand_dims(mx.nd.array(dist), axis=1)
#         self.ctx = []
#         self.graph_on_ctx = []

#         self.init_model()    

#     def build_graph_on_ctx(self, ctx):
#         g = DGLGraph()
#         g.set_n_initializer(dgl.init.zero_initializer)
#         g.add_nodes(self.num_nodes)
#         g.add_edges(self.src, self.dst)
#         g.edata['dist'] = self.dist.as_in_context(ctx)
#         self.graph_on_ctx.append(g)
#         self.ctx.append(ctx)
    
#     def get_graph_on_ctx(self, ctx):
#         if ctx not in self.ctx:
#             self.build_graph_on_ctx(ctx)
#         return self.graph_on_ctx[self.ctx.index(ctx)]

#     def forward(self, state, feature): # first dimension of state & feature should be num_nodes
#         g = self.get_graph_on_ctx(state.context)
#         g.ndata['state'] = state
#         g.ndata['feature'] = feature        
#         g.update_all(self.msg_edge, self.msg_reduce)
#         state = g.ndata.pop('new_state')
#         return state

#     def init_model(self):
#         raise NotImplementedError("To be implemented")

#     def msg_edge(self, edge):
#         """ Messege passing across edge.
#         More detail usage please refers to the manual of DGL library.

#         Parameters
#         ----------
#         edge: a dictionary of edge data.
#             edge.src['state'] and edge.dst['state']: hidden states of the nodes, which is NDArrays with shape [e, b, t, d] or [e, b, d]
#             edge.src['feature'] and  edge.dst['state']: features of the nodes, which is NDArrays with shape [e, d]
#             edge.data['dist']: distance matrix of the edges, which is a NDArray with shape [e, d]

#         Returns
#         -------
#             A dictionray of messages
#         """
#         raise NotImplementedError("To be implemented")

#     def msg_reduce(self, node):
#         raise NotImplementedError("To be implemented")
        
# class GAT(Graph):
#     def __init__(self, dist, edge, hidden_size, prefix=None):
#         super(GAT, self).__init__(dist, edge, hidden_size, prefix)

#     def init_model(self):
#         self.weight = self.params.get('weight', shape=(self.hidden_size * 2, self.hidden_size))
    
#     def msg_edge(self, edge):
#         state = nd.concat(edge.src['state'], edge.dst['state'], dim=-1)
#         ctx = state.context

#         alpha = nd.LeakyReLU(nd.dot(state, self.weight.data(ctx)))

#         dist = edge.data['dist']
#         while len(dist.shape) < len(alpha.shape):
#             dist = nd.expand_dims(dist, axis=-1)

#         alpha = alpha * dist 
#         return { 'alpha': alpha, 'state': edge.src['state'] }

#     def msg_reduce(self, node):
#         state = node.mailbox['state']
#         alpha = node.mailbox['alpha']
#         alpha = nd.softmax(alpha, axis=1)

#         new_state = nd.relu(nd.sum(alpha * state, axis=1))
#         return { 'new_state': new_state }

# class MetaGAT(Graph):
#     """ Meta Graph Attention. """
#     def __init__(self, dist, edge, hidden_size, prefix=None):
#         super(MetaGAT, self).__init__(dist, edge, hidden_size, prefix)

#     def init_model(self):
#         from model.basic_structure import MLP
#         with self.name_scope():
#             self.w_mlp = MLP(MODEL['meta_hiddens'] + [self.hidden_size * self.hidden_size * 2,], 'sigmoid', False)
#             self.weight = self.params.get('weight', shape=(1,1))
    
#     def msg_edge(self, edge):
#         state = nd.concat(edge.src['state'], edge.dst['state'], dim=-1)
#         feature = nd.concat(edge.src['feature'], edge.dst['feature'], edge.data['dist'], dim=-1)

#         # generate weight by meta-learner
#         weight = self.w_mlp(feature)
#         weight = nd.reshape(weight, shape=(-1, self.hidden_size * 2, self.hidden_size))

#         # reshape state to [n, b * t, d] for batch_dot (currently mxnet only support batch_dot for 3D tensor)
#         shape = state.shape
#         state = nd.reshape(state, shape=(shape[0], -1, shape[-1]))

#         alpha = nd.LeakyReLU(nd.batch_dot(state, weight))

#         # reshape alpha to [n, b, t, d]
#         alpha = nd.reshape(alpha, shape=shape[:-1] + (self.hidden_size,))
#         return { 'alpha': alpha, 'state': edge.src['state'] }

#     def msg_reduce(self, node):
#         state = node.mailbox['state']
#         alpha = node.mailbox['alpha']
#         alpha = nd.softmax(alpha, axis=1)

#         new_state = nd.relu(nd.sum(alpha * state, axis=1)) * nd.sigmoid(self.weight.data(state.context))
#         return { 'new_state': new_state }