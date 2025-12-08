import time
import torch
import torch.nn.functional as F
import random
import numpy as np
import os
from sklearn.metrics import f1_score
from tqdm import tqdm

# -------------------- 配置 --------------------
BACKEND = 'PyG'
DEVICE = torch.device('cpu')  # 如果有GPU改成 'cuda'
SEED = 42
EPOCHS = 200
HIDDEN_DIM = 16
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4
EARLY_STOP_WINDOW = 10
RR = 1  # 测试先只跑1次重复实验

DATASETS = [
    ('Planetoid', 'Cora'),
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
from torch_geometric.nn.models import GCN  # PyG封装的GCN
import torch_geometric.transforms as T

transform = T.NormalizeFeatures()

# -------------------- 主循环 --------------------
for backend_type, dataset_name in DATASETS:
    print(f"\n================= 数据集: {dataset_name} =================")

    # 加载数据集
    if backend_type == 'Coauthor':
        dataset = Coauthor(root=f'data/Coauthor', name=dataset_name)
    elif backend_type == 'Planetoid':
        dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name, transform=transform)
    else:
        raise ValueError(f"Unknown dataset type {backend_type}")

    data = dataset[0]

    # -------------------- 数据类型修正 --------------------
    data.x = data.x.float()
    data.y = data.y.long()

    # -------------------- 数据集划分 --------------------
    if backend_type in ['Coauthor', 'Planetoid']:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        print(f"Train nodes: {train_mask.sum().item()}")
        print(f"Valid nodes: {val_mask.sum().item()}")
        print(f"Test nodes: {test_mask.sum().item()}")
    else:
        raise ValueError(f"Unknown dataset type {backend_type}")

    data = data.to(DEVICE)

    # -------------------- 训练循环 --------------------
    for R in range(RR):
        model = GCN(
            in_channels=dataset.num_node_features,
            hidden_channels=HIDDEN_DIM,
            out_channels=dataset.num_classes,
            num_layers=2,
            dropout=DROPOUT
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        early_stop_counter = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            optimizer.zero_grad()

            out = model(data.x, data.edge_index)
            loss_train = criterion(out[train_mask], data.y[train_mask].squeeze())
            loss_val = criterion(out[val_mask], data.y[val_mask].squeeze())

            loss_train.backward()
            optimizer.step()

            # early stopping
            if loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= EARLY_STOP_WINDOW:
                print(f"Early stopping at epoch {epoch}")
                break

            # 打印每轮信息
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                acc_train = (pred[train_mask] == data.y[train_mask].squeeze()).sum().item() / train_mask.sum().item()
                acc_val = (pred[val_mask] == data.y[val_mask].squeeze()).sum().item() / val_mask.sum().item()
                print(f"Epoch {epoch:03d} | Train Loss: {loss_train.item():.4f} | Val Loss: {loss_val.item():.4f} | Train Acc: {acc_train:.4f} | Val Acc: {acc_val:.4f}")

        # -------------------- 测试 --------------------
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            test_acc = (pred[test_mask] == data.y[test_mask].squeeze()).sum().item() / test_mask.sum().item()
            macro_f1 = f1_score(data.y[test_mask].squeeze(), pred[test_mask], average='macro')
            print(f"Test Accuracy: {test_acc:.4f} | Test Macro-F1: {macro_f1:.4f}")
