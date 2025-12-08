import time
import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import psutil
from sklearn.metrics import f1_score
from tqdm import tqdm 

# -------------------- 配置 --------------------
BACKEND = 'DGL'
# DATASET = 'ogbn-products'
DEVICE = torch.device('cpu')
SEED = 42
EPOCHS = 500
HIDDEN_DIM = 8


DATASETS = [
    ('Coauthor', 'CS'),
    ('Coauthor', 'Physics'),
    ('Planetoid', 'Cora'),
    ('Planetoid', 'PubMed'),
    # ('ogb', 'ogbn-arxiv'),
    # ('ogb', 'ogbn-products')
]


# -------------------- 固定随机种子 --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(SEED)

# -------------------- 包和数据导入 --------------------

from torch_geometric.datasets import Coauthor, Planetoid
import torch_geometric.transforms as T
import dgl
import dgl.nn.pytorch as dglnn
from ogb.nodeproppred import PygNodePropPredDataset

# 解决 torch.load 报错
_load = torch.load
def load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _load(*args, **kwargs)
torch.load = load

transform = T.NormalizeFeatures()
# 数据集选择（自行切换）
# dataset = Coauthor(root='data/Coauthor', name='CS')
# dataset = Coauthor(root='data/Coauthor', name='Physics')
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# dataset = Planetoid(root='/tmp/PubMed', name='PubMed', transform=transform)
# dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/OGB')
# dataset = PygNodePropPredDataset(name='ogbn-products', root='data/OGB')
# data = dataset[0]

# 构建 DGL 图（用 PyG 的 edge_index 转换）



# 模型定义
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, out_dim)

    def forward(self, graph, features):
        x = F.relu(self.conv1(graph, features))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(graph, x)
        return x

for backend_type, dataset_name in DATASETS:
    print(f"\n================= 数据集: {dataset_name} =================")

    # -------------------- 加载数据集 --------------------
    if backend_type == 'Coauthor':
        dataset = Coauthor(root=f'data/Coauthor', name=dataset_name)
    elif backend_type == 'Planetoid':
        dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name, transform=transform)
    elif backend_type == 'ogb':
        dataset = PygNodePropPredDataset(name=dataset_name, root='data/OGB')
    else:
        raise ValueError(f"Unknown dataset type {backend_type}")

    data = dataset[0]

    # 原始边
    src, dst = data.edge_index
    g = dgl.graph((src, dst), num_nodes=data.num_nodes)
    g = dgl.add_self_loop(g)
    g = g.to(DEVICE)
    g.ndata['feat'] = data.x.to(DEVICE)

    # -------------------- 数据预处理（统一） --------------------
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_idx = indices[:int(0.6 * num_nodes)]
    val_idx = indices[int(0.6 * num_nodes):int(0.8 * num_nodes)]
    test_idx = indices[int(0.8 * num_nodes):]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True


    g.ndata['label'] = data.y.to(DEVICE)
    g.ndata['train_mask'] = train_mask.to(DEVICE)
    g.ndata['val_mask'] = val_mask.to(DEVICE)
    g.ndata['test_mask'] = test_mask.to(DEVICE)


    All_forward_times = []
    All_backward_times = []
    All_epoch_times = []
    All_total_train_time = []
    All_test_acc = []
    ALL_f1 = []
    RR = 50

    for R in tqdm(range(RR)):
        # -------------------- 设备转移 --------------------
        model = GCN(g.ndata['feat'].shape[1], HIDDEN_DIM, dataset.num_classes).to(DEVICE)

        # -------------------- 训练配置 --------------------
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        process = psutil.Process(os.getpid())
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
        peak_memory_mb = memory_usage_mb

        forward_times = []
        backward_times = []
        epoch_times = []

        # -------------------- 训练循环 --------------------
        LOSS_LIST = []
        LOSS_LIST_TEST = []
        LOSS_LIST_VALID = []

        train_start = time.time()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            epoch_start = time.time()

            # 前向传播
            start_fwd = time.time()
            out = model(g, g.ndata['feat'])
            end_fwd = time.time()

            # 计算loss
            loss = criterion(out[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']].squeeze())
            loss_test = criterion(out[g.ndata['test_mask']], g.ndata['label'][g.ndata['test_mask']].squeeze())
            loss_valid = criterion(out[g.ndata['val_mask']], g.ndata['label'][g.ndata['val_mask']].squeeze())
        
            # 反向传播
            start_bwd = time.time()
            loss.backward()
            optimizer.step()
            end_bwd = time.time()

            epoch_end = time.time()

            forward_times.append(end_fwd - start_fwd)
            backward_times.append(end_bwd - start_bwd)
            epoch_times.append(epoch_end - epoch_start)

            LOSS_LIST.append(round(loss.item(),3))
            LOSS_LIST_TEST.append(round(loss_test.item(),3))
            LOSS_LIST_VALID.append(round(loss_valid.item(),3))

            current_memory_mb = process.memory_info().rss / 1024 / 1024
            peak_memory_mb = max(peak_memory_mb, current_memory_mb)

        train_end = time.time()
        total_train_time = train_end - train_start

        # -------------------- 测试函数 --------------------
        @torch.no_grad()
        def evaluate(model, data, mask_key='test_mask'):
            model.eval()
            out = model(g, g.ndata['feat'])
            mask = g.ndata[mask_key]
            pred = out.argmax(dim=1)
            correct = (pred[mask] == g.ndata['label'][mask].squeeze()).sum()
            acc = int(correct) / int(mask.sum())
            macro_f1 = f1_score(g.ndata['label'][mask].squeeze(), pred[mask], average='macro')
            return acc, macro_f1

        test_acc, f1 = evaluate(model, data)

        # -------------------- 结果输出 --------------------
        avg_fwd = sum(forward_times) / len(forward_times)
        avg_bwd = sum(backward_times) / len(backward_times)
        avg_epoch = sum(epoch_times) / len(epoch_times)

        All_forward_times.append(avg_fwd)
        All_backward_times.append(avg_bwd)
        All_epoch_times.append(avg_epoch)
        All_total_train_time.append(total_train_time)
        All_test_acc.append(test_acc)
        ALL_f1.append(f1)

    # print(f'Train_LOSS_{BACKEND} = {LOSS_LIST}')
    # print(f'Valid_LOSS_{BACKEND} = {LOSS_LIST_VALID}')
    # print(f'Test_LOSS_{BACKEND} = {LOSS_LIST_TEST}')


    print("\n======= 统一测试结果汇总 =======")
    print(f"🔹 后端框架: {BACKEND}")
    print(f"🔹 数据集: {dataset_name}")
    print(f"🔥 总训练时间: {sum(All_total_train_time)/RR:.3f} 秒")
    print(f"⏩ 单轮训练平均时间: {sum(All_epoch_times)/RR*1000:.3f} ms")
    print(f"🔁 平均前向传播时间: {sum(All_forward_times)/RR*1000:.3f} ms")
    print(f"↩️ 平均反向传播时间: {sum(All_backward_times)/RR*1000:.3f} ms")
    print(f"📦 内存占用（初始）: {memory_usage_mb:.2f} MB")
    print(f"📈 运行时峰值内存: {peak_memory_mb:.2f} MB")
    print(f"🎯 测试集准确率: {sum(All_test_acc)/RR:.4f}")
    print(f"🎯 测试集F1-score: {sum(ALL_f1)/RR:.4f}")