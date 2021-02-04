import math
import mxnet as mx
from mxnet.gluon import nn, rnn
from mxnet.gluon.contrib.nn import Identity
from dgl import function as fn
from dgl.nn.mxnet.softmax import edge_softmax
from dgl.utils import expand_as_pair


class T_GATConv(nn.Block):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(T_GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        with self.name_scope():
            if self._num_heads != 1:
                self.fc = rnn.GRU(out_feats*num_heads, layout='NTC')
                self.attn_l = self.params.get('attn_l',
                                              shape=(1, num_heads, out_feats*self._in_feats),
                                              init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
                self.attn_r = self.params.get('attn_r',
                                              shape=(1, num_heads, out_feats*self._in_feats),
                                              init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
                self.gated = rnn.LSTM(out_feats*5, layout='NTC', bidirectional=True)
            else:
                self.fc = nn.Dense(out_feats*num_heads, use_bias=False,
                                   weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                                   in_units=in_feats)
                self.attn_l = self.params.get('attn_l',
                                              shape=(1, num_heads, out_feats),
                                              init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
                self.attn_r = self.params.get('attn_r',
                                              shape=(1, num_heads, out_feats),
                                              init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            self.feat_drop = nn.Dropout(feat_drop)
            self.attn_drop = nn.Dropout(attn_drop)
            self.leaky_relu = nn.LeakyReLU(negative_slope)
            if residual:
                if in_feats != out_feats:
                    self.res_fc = nn.Dense(out_feats*num_heads*in_feats, use_bias=False,
                                           weight_initializer=mx.init.Xavier(
                                               magnitude=math.sqrt(2.0)),
                                           in_units=in_feats)
                else:
                    self.res_fc = Identity()
            else:
                self.res_fc = None
            self.activation = activation

    def forward(self, graph, feat):
        graph = graph.local_var()
        h = self.feat_drop(feat)
        if self._num_heads != 1:
            _ = h.expand_dims(-1)
            feat = self.fc(_).swapaxes(1, 2).reshape(-1, self._num_heads, self._out_feats*feat.shape[1])
            # feat = self.fc(_).reshape(-1, self._num_heads, self._out_feats*feat.shape[1])
        else:
            feat = self.fc(h).reshape(
                -1, self._num_heads, self._out_feats)
        el = (feat * self.attn_l.data(feat.context)).sum(axis=-1).expand_dims(-1)
        er = (feat * self.attn_r.data(feat.context)).sum(axis=-1).expand_dims(-1)
        graph.ndata.update({'ft': feat, 'el': el, 'er': er})
        # compute edge attention
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        w = graph.edata['weight'].expand_dims(axis=1).repeat(self._num_heads, axis=1)
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e*w))
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.ndata['ft']
        # gated
        if self._num_heads != 1:
            rst = self.gated(rst)
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).reshape(h.shape[0], -1, self._out_feats*self._in_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst


class GATConv(nn.Block):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._in_feats = in_feats
        self._out_feats = out_feats
        with self.name_scope():
            if isinstance(in_feats, tuple):
                self.fc_src = nn.Dense(out_feats * num_heads, use_bias=False,
                                       weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                                       in_units=self._in_src_feats)
                self.fc_dst = nn.Dense(out_feats * num_heads, use_bias=False,
                                       weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                                       in_units=self._in_dst_feats)
            else:
                self.fc = nn.Dense(out_feats * num_heads, use_bias=False,
                                   weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                                   in_units=in_feats)
            self.attn_l = self.params.get('attn_l',
                                          shape=(1, num_heads, out_feats),
                                          init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            self.attn_r = self.params.get('attn_r',
                                          shape=(1, num_heads, out_feats),
                                          init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            self.feat_drop = nn.Dropout(feat_drop)
            self.attn_drop = nn.Dropout(attn_drop)
            self.leaky_relu = nn.LeakyReLU(negative_slope)
            if self._num_heads != 1:
                self.gated = rnn.LSTM(out_feats*5, layout='NTC', bidirectional=True)
            if residual:
                if in_feats != out_feats:
                    self.res_fc = nn.Dense(out_feats * num_heads, use_bias=False,
                                           weight_initializer=mx.init.Xavier(
                                               magnitude=math.sqrt(2.0)),
                                           in_units=in_feats)
                else:
                    self.res_fc = Identity()
            else:
                self.res_fc = None
            self.activation = activation

    def forward(self, graph, feat):
        graph = graph.local_var()
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).reshape(
                -1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).reshape(
                -1, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).reshape(
                -1, self._num_heads, self._out_feats)
        el = (feat_src * self.attn_l.data(feat_src.context)).sum(axis=-1).expand_dims(-1)
        er = (feat_dst * self.attn_r.data(feat_src.context)).sum(axis=-1).expand_dims(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        w = graph.edata['weight'].expand_dims(axis=1).repeat(self._num_heads, axis=1)
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e*w))
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        # gated
        if self._num_heads != 1:
            rst = self.gated(rst)
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).reshape(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst

