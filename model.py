from GATConv import *
from mxnet.gluon.contrib.nn import Identity
import mxnet.gluon.nn as nn
import mxnet as mx


class STGGAT(nn.Block):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 alpha,
                 residual):
        super(STGGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.gat_layers = []
        self.num_classes = num_classes
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(T_GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, alpha, residual))
        # hidden layers
        # for l in range(1, num_layers):
        # due to multi-head, the in_dim = num_hidden*num_heads
        # self.gat_layers.append(GATConv(
        #     num_hidden*self.in_dim*heads[l-1], num_hidden, heads[l],
        #     feat_drop, attn_drop, alpha, True))
        if residual:
            if in_dim != num_classes:
                self.res_fc = nn.Dense(num_classes, use_bias=False,
                                       weight_initializer=mx.init.Xavier(
                                           magnitude=math.sqrt(2.0)))
            else:
                self.res_fc = Identity()
        else:
            self.res_fc = None
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden*self.in_dim*heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, alpha, residual))
        for i, layer in enumerate(self.gat_layers):
            self.register_child(layer, "gat_layer_{}".format(i))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten()
            h = self.activation(h)
        # output projection
        rst = self.gat_layers[-1](self.g, h).mean(1)
        if self.res_fc is not None:
            resval = self.res_fc(inputs).reshape(inputs.shape[0], self.num_classes)
            rst = rst + resval
        return rst
