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
from easygraph.utils.GraphSampler import degree_community_sampling
torch.set_num_threads(30)

# -------------------- 配置 --------------------
BACKEND = 'EasyGraph'
DEVICE = torch.device('cpu')  # 如有 GPU 改成 'cuda'
SEED = 42
EPOCHS = 200
HIDDEN_DIM = 512
DROPOUT = 0.5
EARLY_STOP_WINDOW = 10
RR = 50

DATASETS = [
    ('Yelp', 'Yelp'),
]

# -------------------- 固定随机种子 --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

# -------------------- 包和数据导入 --------------------
from torch_geometric.datasets import Yelp, Reddit, Flickr
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

# -------------------- 主循环 --------------------
for backend_type, dataset_name in DATASETS:
    print(f"\n================= 数据集: {dataset_name} =================")
    
    # -------------------- 加载数据集 --------------------
    if backend_type == 'Yelp': 
        dataset = Yelp(root=f'/root/autodl-tmp/data/Yelp')
    elif backend_type == 'Reddit':
        dataset = Reddit(root=f'/root/autodl-tmp/data/Reddit')
    elif backend_type == 'Flickr':
        dataset = Flickr(root=f'/root/autodl-tmp/data/Flickr')
    else:
        raise ValueError(f"Unknown dataset type {backend_type}")

    data = dataset[0]
    num_nodes = data.num_nodes

    # -------------------- 数据类型修正 --------------------
    data.x = data.x.float()
    data.y = data.y.float()  # 多标签任务

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"Train nodes: {train_mask.sum().item()} | Val nodes: {val_mask.sum().item()} | Test nodes: {test_mask.sum().item()}")

    # -------------------- 构建 Easy-Graph 图对象 --------------------
    g = eg.Graph()
    edge_index = data.edge_index
    edge_list = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    print('开始采样')
    g_s, x_s, sampled_nodes_tensor = degree_community_sampling(
        data.edge_index, data.x, data.y, num_nodes, sample_ratio=0.4, 
        min_nodes=150, alpha=0.65, random_ratio=0.08, bridge_ratio=0.1, k=20, nodes_per_class=17)
    
    print('采样完成')
    print(f'节点数量: {len(g_s.nodes)}, 边数:{len(g_s.edges)}')

    g.add_nodes_from(range(num_nodes))
    g.add_edges(edge_list)

    # -------------------- 移动数据到设备 --------------------
    data = data.to(DEVICE)
    x_s = x_s.to(DEVICE)

    # -------------------- 构建训练 mask 子集 --------------------
    train_mask_sub = data.train_mask[sampled_nodes_tensor]
    val_mask_sub = data.val_mask[sampled_nodes_tensor]

    # -------------------- 结果存储 --------------------
    All_forward_times, All_backward_times, All_epoch_times = [], [], []
    All_total_train_time, All_test_acc, ALL_f1 = [], [], []

    for R in tqdm(range(RR)):
        
        # -------------------- 初始化模型 --------------------
        model = GCN(dataset.num_node_features, HIDDEN_DIM, dataset.num_classes, dropout=DROPOUT).to(DEVICE)
        optimizer = torch.optim.Adam([
            {'params': model.gcn1.parameters(), 'weight_decay': 5e-4},  # 第一层 GCN
            {'params': model.gcn2.parameters(), 'weight_decay': 0.0}    # 第二层 GCN，不做正则
        ], lr=0.01)
        criterion = nn.BCEWithLogitsLoss()  # 多标签损失

        # 每次重复实验初始化 early stopping
        best_val_loss = float('inf')
        early_stop_counter = 0

        forward_times, backward_times, epoch_times = [], [], []

        process = psutil.Process(os.getpid())
        peak_memory_mb = process.memory_info().rss / 1024 / 1024
        train_start = time.perf_counter()

        for epoch in range(1, EPOCHS+1):
            model.train()
            optimizer.zero_grad()

            start_fwd = time.perf_counter()
            out = model(x_s, g_s)
            end_fwd = time.perf_counter()

            # -------------------- 子图 loss --------------------
            loss = criterion(out[train_mask_sub], data.y[sampled_nodes_tensor][train_mask_sub])

            # -------------------- 全图验证 --------------------
            model.eval()
            with torch.no_grad():
                out_full = model(data.x, g)
                loss_val = criterion(out_full[data.val_mask], data.y[data.val_mask])

            # 反向传播
            start_bwd = time.perf_counter()
            loss.backward()
            optimizer.step()
            end_bwd = time.perf_counter()

            # 记录
            forward_times.append(end_fwd - start_fwd)
            backward_times.append(end_bwd - start_bwd)
            epoch_times.append(end_fwd - start_fwd + end_bwd - start_bwd)

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
            out = model(data.x, g)
            prob = torch.sigmoid(out)  # logits -> 概率
            pred = (prob > 0.5).int()

            test_y = data.y[data.test_mask].int()
            test_pred = pred[data.test_mask]

            # Accuracy (每个标签独立平均)
            test_acc = (test_pred == test_y).float().mean().item()
            macro_f1 = f1_score(test_y.cpu().numpy(), test_pred.cpu().numpy(), average='macro')

        # -------------------- 保存统计结果 --------------------
        All_forward_times.append(sum(forward_times)/len(forward_times))
        All_backward_times.append(sum(backward_times)/len(backward_times))
        All_epoch_times.append(sum(epoch_times)/len(epoch_times))
        All_total_train_time.append(total_train_time)
        All_test_acc.append(test_acc)
        ALL_f1.append(macro_f1)

    # -------------------- 输出最终结果 --------------------
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
