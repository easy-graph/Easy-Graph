import time
import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import psutil
from sklearn.metrics import f1_score
from tqdm import tqdm
import statistics
import torch.nn as nn

# -------------------- 配置 --------------------
BACKEND = 'PyG'
DEVICE = torch.device('cpu')
SEED = 42
EPOCHS = 200
HIDDEN_DIM = 16
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4
EARLY_STOP_WINDOW = 10
RR = 5

DATASETS = [
    ('Yelp', 'Yelp'),
]

# -------------------- 固定随机种子 --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)

set_seed(SEED)

# -------------------- 包和数据导入 --------------------
from torch_geometric.datasets import Coauthor, Planetoid, Yelp, Reddit, Flickr
from torch_geometric.nn.models import GCN  # PyG 封装好的 GCN
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

transform = T.NormalizeFeatures()

_load = torch.load
def load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _load(*args, **kwargs)
torch.load = load

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

    # -------------------- 结果存储 --------------------
    All_forward_times = []
    All_backward_times = []
    All_epoch_times = []
    All_total_train_time = []
    All_test_acc = []
    ALL_f1 = []

    # -------------------- 重复实验 --------------------
    for R in tqdm(range(RR)):
        # -------------------- 设备转移 --------------------
        model = GCN(
            in_channels=dataset.num_node_features,
            hidden_channels=HIDDEN_DIM,
            out_channels=dataset.num_classes,
            num_layers=2,
            dropout=DROPOUT
        ).to(DEVICE)

        # optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer = torch.optim.Adam([
            {'params': model.convs[0].parameters(), 'weight_decay': 5e-4},  # 第一层 GCN
            {'params': model.convs[1].parameters(), 'weight_decay': 0.0}    # 第二层 GCN，不做正则
        ], lr=0.01)
        criterion = nn.BCEWithLogitsLoss()  # 多标签损失

        process = psutil.Process(os.getpid())
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
        peak_memory_mb = memory_usage_mb

        forward_times = []
        backward_times = []
        epoch_times = []

        LOSS_LIST = []
        LOSS_LIST_TEST = []
        LOSS_LIST_VALID = []

        best_val_loss = float('inf')
        early_stop_counter = 0

        train_start = time.time()

        for epoch in tqdm(range(1, EPOCHS + 1)):
            model.train()
            optimizer.zero_grad()
            epoch_start = time.time()

            # 前向传播
            start_fwd = time.time()
            out = model(data.x, data.edge_index)
            end_fwd = time.time()

            # 计算 loss
            loss = criterion(out[data.train_mask], data.y[data.train_mask].squeeze())
            loss_valid = criterion(out[data.val_mask], data.y[data.val_mask].squeeze())
            loss_test = criterion(out[data.test_mask], data.y[data.test_mask].squeeze())

            # 反向传播
            start_bwd = time.time()
            loss.backward()
            optimizer.step()
            end_bwd = time.time()

            epoch_end = time.time()

            forward_times.append(end_fwd - start_fwd)
            backward_times.append(end_bwd - start_bwd)
            epoch_times.append(epoch_end - epoch_start)

            LOSS_LIST.append(round(loss.item(), 3))
            LOSS_LIST_VALID.append(round(loss_valid.item(), 3))
            LOSS_LIST_TEST.append(round(loss_test.item(), 3))

            # early stopping
            if loss_valid.item() < best_val_loss:
                best_val_loss = loss_valid.item()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= EARLY_STOP_WINDOW:
                # print(f"Early stopping at epoch {epoch}. Validation loss has not decreased for {EARLY_STOP_WINDOW} epochs.")
                break

            current_memory_mb = process.memory_info().rss / 1024 / 1024
            peak_memory_mb = max(peak_memory_mb, current_memory_mb)

        train_end = time.time()
        total_train_time = train_end - train_start

        # -------------------- 测试函数 --------------------
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            prob = torch.sigmoid(out)  # logits -> 概率
            pred = (prob > 0.5).int()

            test_y = data.y[data.test_mask].int()
            test_pred = pred[data.test_mask]

            # Accuracy (每个标签独立平均)
            test_acc = (test_pred == test_y).float().mean().item()
            macro_f1 = f1_score(test_y.cpu().numpy(), test_pred.cpu().numpy(), average='macro')

        avg_fwd = sum(forward_times) / len(forward_times)
        avg_bwd = sum(backward_times) / len(backward_times)
        avg_epoch = sum(epoch_times) / len(epoch_times)

        All_forward_times.append(avg_fwd)
        All_backward_times.append(avg_bwd)
        All_epoch_times.append(avg_epoch)
        All_total_train_time.append(total_train_time)
        All_test_acc.append(test_acc)
        ALL_f1.append(macro_f1)

    # -------------------- 输出结果 --------------------
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
    print(f"🎯 测试集标准差: {statistics.stdev(All_test_acc):.4f}")
    print(f"🎯 测试集F1-score: {sum(ALL_f1)/RR:.4f}")
