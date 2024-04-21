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
        rho = self.g.ndata.pop('rho')
        # current_time = datetime.datetime.now()
        # formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
        # time.sleep(1)
        # with h5py.File(f'rho-{formatted_time}.h5', 'w') as f:
        #     data = f.create_dataset('rho', shape=rho.shape) 
        #     data[:] = rho
        return state, rho

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


class MetaGAT(Graph):
    def __init__(self, in_out, edge_feature, edge, hidden_size):
        super(MetaGAT, self).__init__(in_out, edge_feature, edge, hidden_size)
        self.w_mlp = MLP(MODEL['meta_hiddens'] + [hidden_size * hidden_size * 2], 'sigmoid', False)
        self.b_mlp = MLP(MODEL['meta_hiddens'] + [1], 'sigmoid', False)
        self.rho_mlp = MLP(MODEL['meta_hiddens'] + [1], 'sigmoid', False) # map 2*hidden to 1

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

        original_state = node.data['state']
        aggregated_neighbour_state = tf.nn.relu(tf.reduce_sum(alpha * state, axis=1)) # [n, b, t, hidden]
        concatenation = tf.concat([aggregated_neighbour_state, original_state], axis=-1) # [n, b, t, 2*hidden]
        rho = tf.sigmoid(self.rho_mlp(concatenation)) # [n, b, t, 1]
        new_state = rho*aggregated_neighbour_state + (1-rho)*original_state

        return {'new_state': new_state, 'rho': rho}
