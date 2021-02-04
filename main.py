import time
from utils import *
from model import *
from mxnet import gluon


pre = 1
k = 3
num_hidden = 28
num_heads = 8
seq_len = 10
# load data
area_volumes = np.load(
    'F:/STGGAT/volumes.npy')
area_volumes = np.transpose(area_volumes)
num_entra = area_volumes.shape[0]
features = mx.nd.zeros(shape=(area_volumes.shape[1] - seq_len - pre + 1, num_entra, seq_len), ctx=mx.gpu())
labels = mx.nd.zeros(shape=(area_volumes.shape[1] - seq_len - pre + 1, num_entra, 1), ctx=mx.gpu())
for i in range(area_volumes.shape[1] - seq_len - pre + 1):
    features[i] = area_volumes[:, i: i + seq_len]
    labels[i] = area_volumes[:, i + seq_len + pre - 1].reshape(num_entra, 1)
in_feats = features.shape[1]
# 读取权重矩阵
avg_time = np.load(
    'F:/STGGAT/avg_time.npy')
OD_matrix = np.load(
    'F:/STGGAT/OD_direction.npy')
# Create training and test sets.
num_train = int(features.shape[0] * 0.6)
num_valid = int(features.shape[0] * 0.2)
num_test = int(features.shape[0] * 0.2)
# 随机划分测试集合与训练集
train_idx = np.load('F:/STGGAT/train_idx.npy')
valid_idx = np.load('F:/STGGAT/valid_idx.npy')
matfn = 'F:/STGGAT/adjacency_matrix_'+str(k)+'.mat'
g, tratim_matrix = load_adjmatrix(matfn)
# 随机划分训练集与测试集
trainset = GraphTraffic(num_train, num_entra, tratim_matrix)
validset = GraphTraffic(num_valid, num_entra, tratim_matrix)
testset = GraphTraffic(num_test, num_entra, tratim_matrix)
# 读取邻接矩阵
weighted_matrix = np.zeros(shape=(avg_time.shape))
for i in range(avg_time.shape[0]):
    idx = np.where(tratim_matrix[i, :]!=0)
    idx = idx[0]
    weighted_matrix[i, idx] = OD_matrix[i, idx] / (OD_matrix[i, idx].sum()+1e-4)
for o in range(weighted_matrix.shape[0]):
    weighted_matrix[o, o] = 1
# 构建图数据
start_point = trainset.graphs[0].edges()[0]
end_point = trainset.graphs[0].edges()[1]
weight = mx.nd.zeros(shape=(len(start_point),), ctx=mx.gpu())
for i in range(len(start_point)):
    sta_idx = start_point[i].astype('int32').asscalar()
    end_idx = end_point[i].astype('int32').asscalar()
    weight[i] = weighted_matrix[sta_idx, end_idx]
for i in range(trainset.__len__()):
    trainset.graphs[i].ndata['volumes'] = features[train_idx[i]]
    trainset.graphs[i].edata['weight'] = weight.reshape(-1, 1)
    trainset.labels[i] = labels[train_idx[i]].reshape(-1, num_entra)
for i in range(validset.__len__()):
    validset.graphs[i].ndata['volumes'] = features[valid_idx[i]]
    validset.graphs[i].edata['weight'] = weight.reshape(-1, 1)
    validset.labels[i] = labels[valid_idx[i]].reshape(-1, num_entra)
for i in range(testset.__len__()):
    testset.graphs[i].ndata['volumes'] = features[i+num_train+num_valid]
    testset.graphs[i].edata['weight'] = weight.reshape(-1, 1)
    testset.labels[i] = labels[i+num_train+num_valid].reshape(-1, num_entra)
# 定义模型参数
num_layers = 1
num_out_heads = 1
n_classes = 1
in_drop = 0.0
attn_drop = 0.0
alpha = 0.2
residual = True
lr = 1e-3
epochs = 250
batch_size = 128
early_stop = 1
# create model
heads = ([num_heads] * num_layers) + [num_out_heads]
model = STGGAT(dgl.batch(trainset.graphs[: batch_size]),
            num_layers,
            seq_len,
            num_hidden,
            n_classes,
            heads,
            elu,
            in_drop,
            attn_drop,
            alpha,
            residual)
train_iter = gluon.data.DataLoader(trainset, batch_size, shuffle=True,
                                   batchify_fn=collate, last_batch='discard', num_workers=512, thread_pool=True)
valid_iter = gluon.data.DataLoader(validset, batch_size, shuffle=True,
                                   batchify_fn=collate, last_batch='discard', num_workers=512, thread_pool=True)
test_iter = gluon.data.DataLoader(testset, batch_size, shuffle=False,
                                  batchify_fn=collate, last_batch='discard', num_workers=512, thread_pool=True)
if early_stop:
    stopper = EarlyStopping(patience=15)
model.initialize(mx.init.Xavier(magnitude=math.sqrt(2.0)), ctx=mx.gpu())
trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.L2Loss()
t0 = time.time()
for epoch in range(epochs):
    train_l_sum, n = 0.0, 0
    start_time = time.time()
    for iter, (bg, label) in enumerate(train_iter):
        model.g = bg
        with mx.autograd.record():
            prediction = model(bg.ndata['volumes'])
            l = loss(prediction, label)
        l.backward()
        trainer.step(batch_size)
        train_l_sum += l
        n += bg.__len__()
    train_l = train_l_sum / n
    train_rmse = rmse_trainset(model, train_iter, batch_size, num_entra)
    valid_rmse = rmse_testset(model, valid_iter, batch_size, num_entra)
    test_rmse = rmse_testset(model, test_iter, batch_size, num_entra)
    print('epoch %d, loss %.3f, running time %.2f s, train rmse %.3f, valid rmse %.3f, test rmse %.3f' % (
        epoch + 1, train_l.mean().asnumpy(), time.time() - start_time, train_rmse, valid_rmse, test_rmse))
    if stopper.step(valid_rmse, model):
        break
if early_stop:
    model.load_parameters('model.param')
test_rmse, test_mae, test_mape = rmse_testset(model, test_iter, batch_size, num_entra, mae=True, idx=29, plot=False)
model.save_parameters('STGGAT_'+str(pre)+'hidden_'+str(num_hidden)+'k_'+str(k)+'rmse_'+str(test_rmse)+'_time_'+str(time.time()-t0)+'.param')
print("K: %d, Hidden: %d, Head: %d, Test RMSE: %.4f, Test MAE: %.4f, Test MAPE: %.4f"
      % (k, num_hidden, num_heads, test_rmse, test_mae, test_mape))
