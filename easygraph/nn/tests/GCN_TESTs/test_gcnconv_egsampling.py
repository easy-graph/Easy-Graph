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
from easygraph.utils.Effective_R import EffectiveResistance
from easygraph.utils.GraphSampler import FERGraphSampler
# from easygraph.utils.GraphSampler import FullGraphHybridSampler
torch.set_num_threads(30)

# -------------------- 配置 --------------------
BACKEND = 'EasyGraph'
DEVICE = torch.device('cpu')  # 如有 GPU 改成 'cuda'
SEED = 42
EPOCHS = 200
HIDDEN_DIM = 256
DROPOUT = 0.5
EARLY_STOP_WINDOW = 10
RR = 50

DATASETS = [
    # ('Coauthor', 'CS'),
    # ('Coauthor', 'Physics'),
    # ('Planetoid', 'Cora'),
    # ('Planetoid', 'Citeseer'),
    # ('Planetoid', 'PubMed'),
    # ('ogb', 'ogbn-arxiv'),
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

def feature_similarity(x, edge_index):
    src, dst = edge_index
    sim = F.cosine_similarity(x[src], x[dst])
    sim = (sim + 1.0) / 2.0  # 归一化到 [0,1]
    return sim

def FER_score(r_ij, s_ij, alpha=1.0, beta=1.0):
    score = (r_ij ** alpha) * (s_ij ** beta)
    return score / (score.max() + 1e-12)

set_seed(SEED)

# -------------------- 包和数据导入 --------------------
from torch_geometric.datasets import Coauthor, Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import easygraph as eg

transform = T.NormalizeFeatures()

# -------------------- 定义 GCN --------------------
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GCN, self).__init__()
        self.gcn1 = eg.GCNConv(in_channels, hidden_channels)
        self.gcn2 = eg.GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, g):
        x = F.relu(self.gcn1(x, g))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, g)
        return x

_load = torch.load
def load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _load(*args, **kwargs)
torch.load = load

# -------------------- 主循环 --------------------
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

    er = EffectiveResistance(edge_index, num_nodes)
    resistance_dict = er.compute_resistance_dict()
    r_ij = torch.tensor([1 / (resistance_dict[(u,v)] + 1e-12) for u, v in edge_index.T.tolist()])
    r_ij = r_ij / r_ij.max()

    s_ij = feature_similarity(data.x, edge_index)
    FER = FER_score(r_ij, s_ij, alpha=1.0, beta=1.0)
    # sample_g = FERGraphSampler(edge_index, num_nodes, FER)
    # sg = FullGraphHybridSampler(edge_index, num_nodes, 2)
    # sample_g = sg(data.x)
    # print(len(sample_g.edges))

    # -------------------- 移动数据到设备 --------------------
    data = data.to(DEVICE)

    # -------------------- 结果存储 --------------------
    All_forward_times, All_backward_times, All_epoch_times = [], [], []
    All_total_train_time, All_test_acc, ALL_f1 = [], [], []

    for R in tqdm(range(RR)):
        
        sample_g = FERGraphSampler(edge_index, num_nodes, FER)
        # -------------------- 初始化模型 --------------------
        model = GCN(dataset.num_node_features, HIDDEN_DIM, dataset.num_classes, dropout=DROPOUT).to(DEVICE)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        optimizer = torch.optim.Adam([
            {'params': model.gcn1.parameters(), 'weight_decay': 5e-4},  # 第一层 GCN
            {'params': model.gcn2.parameters(), 'weight_decay': 0.0}    # 第二层 GCN，不做正则
        ], lr=0.01)
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
            out = model(data.x, sample_g)
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
            if loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= EARLY_STOP_WINDOW:
                break

            current_memory_mb = process.memory_info().rss / 1024 / 1024
            peak_memory_mb = max(peak_memory_mb, current_memory_mb)

        train_end = time.perf_counter()
        total_train_time = train_end - train_start

        # -------------------- 测试 --------------------
        model.eval()
        with torch.no_grad():
            out = model(data.x, g)
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

        # print(f"Run {R+1}/{RR} finished | Test Accuracy: {test_acc:.4f} | Macro-F1: {macro_f1:.4f}")

    # -------------------- 输出最终结果 --------------------
    # print(f'\nTrain Loss = {LOSS_LIST}')
    # print(f'Valid Loss = {LOSS_LIST_VALID}')
    # print(f'Test Loss = {LOSS_LIST_TEST}')

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
