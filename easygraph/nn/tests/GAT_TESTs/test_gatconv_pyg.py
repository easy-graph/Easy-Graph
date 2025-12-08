import time
import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import psutil
from sklearn.metrics import f1_score
from tqdm import tqdm   
import torch.nn as nn
import statistics
torch.set_num_threads(30)

# -------------------- 配置 --------------------
BACKEND = 'PyG'
DEVICE = torch.device('cpu')  # 如有 GPU 改成 'cuda'
SEED = 42
EPOCHS = 500
HIDDEN_DIM = 256
DROPOUT = 0.6
Heads = 8
EARLY_STOP_WINDOW = 100
RR = 10

DATASETS = [
    # ('Coauthor', 'CS'),
    # ('Coauthor', 'Physics'),
    # ('Planetoid', 'Cora'),
    # ('Planetoid', 'Citeseer'),
    # ('Planetoid', 'PubMed'),
    ('ogb', 'ogbn-arxiv'),
    # ('ogb', 'ogbn-products')
]

# -------------------- 固定随机种子 --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)

set_seed(SEED)

# -------------------- 包和数据导入 --------------------
from torch_geometric.datasets import Coauthor, Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
# from torch_geometric.nn.models import GAT
import easygraph as eg

transform = T.NormalizeFeatures()


_load = torch.load
def load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _load(*args, **kwargs)
torch.load = load

class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.6):
        super(GATNet, self).__init__()
        self.dropout = dropout
        # 第一层: 8个head, 每个head输出8维
        self.conv1 = GATConv(
            in_channels, hidden_channels,
            heads = 8,
            dropout=dropout,  # attention dropout
        )
        # 第二层: 单head输出分类结果
        self.conv2 = GATConv(
            hidden_channels * 8,  # 因为 concat
            out_channels,
            heads=1,
            concat=False,  # 原文最后一层不concat
            dropout=dropout,
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)  # 输入特征 dropout
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # 原文用 ELU

        x = F.dropout(x, p=self.dropout, training=self.training)  # 输入特征 dropout
        x = self.conv2(x, edge_index)

        return x  # CrossEntropyLoss 里自带 softmax

# -------------------- 主循环 --------------------
for backend_type, dataset_name in DATASETS:
    print(f"\n================= 数据集: {dataset_name} =================")

    # -------------------- 加载数据集 --------------------
    if backend_type == 'Coauthor':
        dataset = Coauthor(root=f'/root/Easy-Graph/easygraph/nn/tests/data/Coauthor', name=dataset_name)
    elif backend_type == 'Planetoid':
        dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name, transform=transform)
    elif backend_type == 'ogb':
        dataset = PygNodePropPredDataset(name=dataset_name, root='/root/Easy-Graph/easygraph/nn/tests/data/OGB')
    else:
        raise ValueError(f"Unknown dataset type {backend_type}")

    data = dataset[0]
    num_nodes = data.num_nodes

    # -------------------- 数据类型修正 --------------------
    data.x = data.x.float()
    data.y = data.y.long()

    # -------------------- 数据集划分 --------------------
    if backend_type == 'ogb':  # OGB 用 get_idx_split()
        split_idx = dataset.get_idx_split()
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[split_idx['train']] = True
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[split_idx['valid']] = True
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[split_idx['test']] = True
    elif backend_type in 'Planetoid':
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

    elif backend_type == 'Coauthor': 
        torch.manual_seed(42) 
        indices = torch.randperm(num_nodes)

        train_end = int(0.6 * num_nodes)
        val_end = int(0.8 * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True

    else:
        raise ValueError(f"Unknown dataset type {backend_type}")

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"Train nodes: {train_mask.sum().item()} | Val nodes: {val_mask.sum().item()} | Test nodes: {test_mask.sum().item()}")

    # -------------------- 构建 Easy-Graph 图对象 --------------------
    g = eg.Graph()
    edge_index = data.edge_index
    edge_list = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    g.add_nodes_from(range(num_nodes))
    g.add_edges(edge_list)
    print(len(g.edges))

    # -------------------- 移动数据到设备 --------------------
    data = data.to(DEVICE)

    # -------------------- 结果存储 --------------------
    All_forward_times, All_backward_times, All_epoch_times = [], [], []
    All_total_train_time, All_test_acc, ALL_f1 = [], [], []

    for R in tqdm(range(RR)):
        # -------------------- 初始化模型 --------------------
        # model = GAT(
        #     in_channels=dataset.num_node_features,
        #     hidden_channels=HIDDEN_DIM,
        #     out_channels=dataset.num_classes,
        #     num_layers=2,
        #     heads=Heads,
        #     dropout=DROPOUT
        # ).to(DEVICE)

        model = GATNet(
            in_channels=dataset.num_node_features,
            hidden_channels=HIDDEN_DIM,
            out_channels=dataset.num_classes,
            dropout=DROPOUT
        ).to(DEVICE)
        
        # model = torch.compile(model, backend="inductor")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.005,            # 学习率 0.005
            weight_decay=5e-4    # L2 正则
        )
        criterion = torch.nn.CrossEntropyLoss()

        # 每次重复实验初始化 early stopping
        best_val_loss = float('inf')
        early_stop_counter = 0

        LOSS_LIST, LOSS_LIST_VALID, LOSS_LIST_TEST = [], [], []
        forward_times, backward_times, epoch_times = [], [], []

        process = psutil.Process(os.getpid())
        peak_memory_mb = process.memory_info().rss / 1024 / 1024

        train_start = time.perf_counter()

        for epoch in range(1, EPOCHS+1):
            model.train()
            optimizer.zero_grad()

            start_fwd = time.perf_counter()
            out = model(data.x, data.edge_index)
            end_fwd = time.perf_counter()

            # loss 计算
            loss = criterion(out[data.train_mask], data.y[data.train_mask].squeeze())
            loss_val = criterion(out[data.val_mask], data.y[data.val_mask].squeeze())
            loss_test = criterion(out[data.test_mask], data.y[data.test_mask].squeeze())

            # 反向传播
            start_bwd = time.perf_counter()
            loss.backward()
            optimizer.step()
            end_bwd = time.perf_counter()

            # 记录
            forward_times.append(end_fwd - start_fwd)
            backward_times.append(end_bwd - start_bwd)
            epoch_times.append(end_fwd - start_fwd + end_bwd - start_bwd)
            LOSS_LIST.append(loss.item())
            LOSS_LIST_VALID.append(loss_val.item())
            LOSS_LIST_TEST.append(loss_test.item())

            # early stopping
            # if loss_val.item() < best_val_loss:
            #     best_val_loss = loss_val.item()
            #     early_stop_counter = 0
            # else:
            #     early_stop_counter += 1

            # if early_stop_counter >= EARLY_STOP_WINDOW:
            #     break

            current_memory_mb = process.memory_info().rss / 1024 / 1024
            peak_memory_mb = max(peak_memory_mb, current_memory_mb)

        train_end = time.perf_counter()
        total_train_time = train_end - train_start

        # -------------------- 测试 --------------------
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            test_acc = (pred[data.test_mask] == data.y[data.test_mask].squeeze()).sum().item() / data.test_mask.sum().item()
            macro_f1 = f1_score(data.y[data.test_mask].squeeze(), pred[data.test_mask], average='macro')

        # -------------------- 保存统计结果 --------------------
        All_forward_times.append(sum(forward_times)/len(forward_times))
        All_backward_times.append(sum(backward_times)/len(backward_times))
        All_epoch_times.append(sum(epoch_times)/len(epoch_times))
        All_total_train_time.append(total_train_time)
        All_test_acc.append(test_acc)
        ALL_f1.append(macro_f1)


    print("\n======= 统一测试结果汇总 =======")
    print(f"🔹 后端框架: {BACKEND}")
    print(f"🔹 数据集: {dataset_name}")
    print(f"🔥 总训练时间: {sum(All_total_train_time)/RR:.3f} 秒")
    print(f"⏩ 单轮训练平均时间: {sum(All_epoch_times)/RR*1000:.3f} ms")
    print(f"🔁 平均前向传播时间: {sum(All_forward_times)/RR*1000:.3f} ms")
    print(f"↩️ 平均反向传播时间: {sum(All_backward_times)/RR*1000:.3f} ms")
    print(f"📦 运行时峰值内存: {peak_memory_mb:.2f} MB")
    print(f"🎯 测试集平均准确率: {sum(All_test_acc)/RR:.4f}")
    print(f"🎯 测试集标准差: {statistics.stdev(All_test_acc):.4f}")
    print(f"🎯 测试集平均F1-score: {sum(ALL_f1)/RR:.4f}")
