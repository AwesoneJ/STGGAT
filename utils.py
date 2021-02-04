import numpy as np
import mxnet as mx
import dgl
from dgl import DGLGraph
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
from scipy import io as sio
import networkx as nx


class GraphTraffic(object):
    def __init__(self, num_graphs, num_nodes, matrix):
        super(GraphTraffic, self).__init__()
        self.num_graphs = num_graphs
        self.graphs = []
        self.labels = []
        self.matrix = matrix
        self.num_nodes = num_nodes
        self._generate()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.
        Paramters
        ---------
        idx : int
            The sample index.
        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]

    @property
    def num_classes(self):
        """Number of classes."""
        return 1

    def _generate(self):
        self._gen_graph(self.num_graphs)
        # preprocess
        for i in range(self.num_graphs):
            self.graphs[i] = DGLGraph(self.graphs[i])

    def _gen_graph(self, n):
        for _ in range(n):
            num_v = self.num_nodes
            g = build_karate_club_graph(self.matrix)
            g = g.to_networkx()
            self.graphs.append(g)
            self.labels.append(0.0)


def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, label = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    ret = mx.nd.zeros(shape=(int(len(label)), 64), ctx=mx.gpu())
    for i in range(len(label)):
        ret[i] = label[i]
    return batched_graph, ret


def masked_mape_np(y_true, y_pred, null_val=0):
    '''
    MAPE
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def rmse_trainset(model, train_iter, batch_size, num):
    test = mx.nd.zeros(shape=(len(train_iter) * batch_size, num), ctx=mx.gpu())
    pred = mx.nd.zeros(shape=(len(train_iter) * batch_size, num), ctx=mx.gpu())
    count = 0
    for iter, (bg, label) in enumerate(train_iter):
        y = label.astype('float32')
        _ = model(bg.ndata['volumes'])
        test[count: count + len(label), :] = y
        pred[count: count + len(label), :] = _.reshape(-1, num)
        count += len(label)
    return np.sqrt(mean_squared_error(test.asnumpy(), pred.asnumpy()))


def rmse_testset(model, _iter, batch_size, num, mae=False, idx=None, plot=False):
    test = mx.nd.zeros(shape=(len(_iter) * batch_size, num), ctx=mx.gpu())
    pred = mx.nd.zeros(shape=(len(_iter) * batch_size, num), ctx=mx.gpu())
    count = 0
    for iter, (bg, label) in enumerate(_iter):
        y = label.astype('float32')
        _ = model(bg.ndata['volumes'])
        test[count: count + len(label), :] = y
        pred[count: count + len(label), :] = _.reshape(-1, num)
        count += len(label)
    RMSE = np.sqrt(mean_squared_error(test.asnumpy(), pred.asnumpy()))
    if plot:
        tmp = plt.figure(dpi=600)
        plt.plot(test[:, idx].asnumpy(), label='true')
        plt.plot(pred[:, idx].asnumpy(), label='pred')
        plt.legend()
        plt.title('RMSE: %.4f' % (RMSE))
        plt.show()
    if mae:
        MAE = mean_absolute_error(test.asnumpy(), pred.asnumpy())
        MAPE = masked_mape_np(test.asnumpy(), pred.asnumpy())
        return RMSE, MAE, MAPE
    else:
        return RMSE


def load_adjmatrix(filename):
    data = sio.loadmat(filename)
    tratim_matrix = np.array(data['adjacency_matrix'])

    # 构建网络图
    g = build_karate_club_graph(tratim_matrix)
    nx_G = g.to_networkx()
    pos = nx.kamada_kawai_layout(nx_G)
    return g, tratim_matrix


def build_karate_club_graph(matrix):
    g = dgl.DGLGraph()
    # add 32 nodes into the graph; nodes are labeled from 0~31
    g.add_nodes(matrix.shape[0])

    # add edges two lists of nodes: src and dst
    # 起点和终点
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 1:
                g.add_edge(i, j)
    return g


def elu(data):
    return mx.nd.LeakyReLU(data, act_type='elu')


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score > self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        model.save_parameters('model.param')


