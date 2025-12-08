import os
import time
import torch
import torch.nn.functional as F
import random
import numpy as np
import psutil
from sklearn.metrics import f1_score
from tqdm import tqdm   
import torch.nn as nn
import statistics

# ------------------- 引入必要的库 ---------------------
import gensim.models.word2vec # 必须显式引入
from nodevectors import Node2Vec
from scipy.sparse import csr_matrix
import easygraph as eg
from torch_geometric.datasets import Coauthor, Planetoid, Reddit
from torch_geometric.utils import add_self_loops 
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from txtReader import TxtGraphReader

# ==============================================================================
# 🩹【Gensim 4.0+ 兼容性补丁】(防止 size 参数报错)
# ==============================================================================
def patch_gensim_4_compatibility():
    original_init = gensim.models.Word2Vec.__init__
    def patched_init(self, *args, **kwargs):
        if 'size' in kwargs: kwargs['vector_size'] = kwargs.pop('size')
        if 'iter' in kwargs: kwargs['epochs'] = kwargs.pop('iter')
        original_init(self, *args, **kwargs)
    gensim.models.Word2Vec.__init__ = patched_init

patch_gensim_4_compatibility()
# ==============================================================================

# -------------------- 配置 --------------------
BACKEND = 'EasyGraph'
DEVICE = torch.device('cpu') 
SEED = 42
EPOCHS = 200
DROPOUT = 0.5
RR = 1

DATASETS = [
    ('Coauthor', 'CS'),
    ('Coauthor', 'Physics'),
    ('Planetoid', 'Cora'),
    ('Planetoid', 'Citeseer'),
    ('Planetoid', 'PubMed'),
]

# 【设置】输入特征维度
NODE2VEC_DIM = 128  
FORCE_NODE2VEC = True 

transform = T.NormalizeFeatures()

# -------------------- 固定随机种子 --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

# -------------------- 定义 GCN --------------------
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, nparts: int=32):
        super(GCN, self).__init__()
        self.gcn1 = eg.GCNConv(in_channels, hidden_channels)
        self.gcn2 = eg.GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        
    def forward(self, x, g):
        x = self.gcn1(x, g)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, g)
        return x

_load = torch.load
def load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _load(*args, **kwargs)
torch.load = load

# -------------------- 结果展示 --------------------
Result_Chart = {}
def Show_Result(dataset_name, RR, All_total_train_time, All_epoch_times, All_forward_times, All_backward_times, peak_memory_mb, All_test_acc, ALL_f1):
    print("\n======= 统一测试结果汇总 =======")
    print(f"🔹 数据集: {dataset_name}")
    print(f"🔹 输入维度: {NODE2VEC_DIM}")
    print(f"🎯 测试集准确率: {sum(All_test_acc)/RR:.4f}")
    print(f"🎯 F1-score: {sum(ALL_f1)/RR:.4f}")
    
    Result_Chart[dataset_name] = {
        "Input_Dim": NODE2VEC_DIM,
        "Total_Train_Time": sum(All_total_train_time)/RR,
        "Accuracy": sum(All_test_acc)/RR,
        "F1-Score": sum(ALL_f1)/RR
    }

# -------------------- 主函数 --------------------
def main():
    root = "./dataset/" 
    
    for backend_type, dataset_name in DATASETS:
        print(f"\n================= 数据集: {dataset_name} =================")

        # 1. 加载
        if backend_type == 'Coauthor':
            dataset = Coauthor(root=root, name=dataset_name)
        elif backend_type == 'Planetoid':
            dataset = Planetoid(root=root, name=dataset_name, transform=transform)
        elif backend_type == 'txt':
            dataset = TxtGraphReader(root=root, name=dataset_name)
        else:
            continue

        data = dataset[0]
        num_nodes = data.num_nodes

        # -------------------- 【核心修改】Node2Vec --------------------
        if FORCE_NODE2VEC:
            print(f"🔄 [Node2Vec] 正在生成 {NODE2VEC_DIM} 维特征...")
            t0 = time.time()

            # A. CSR 矩阵
            row = data.edge_index[0].cpu().numpy()
            col = data.edge_index[1].cpu().numpy()
            data_ones = np.ones(len(row))
            adj_matrix = csr_matrix((data_ones, (row, col)), shape=(num_nodes, num_nodes))

            # B. 训练
            n2v_model = Node2Vec(n_components=NODE2VEC_DIM, walklen=20, epochs=1, threads=4)
            n2v_model.fit(adj_matrix) 

            # C. 提取特征 (修复 KeyError 问题)
            embeddings = []
            for i in range(num_nodes):
                try:
                    # 优先尝试 String Key (因为 nodevectors 内部常转为 str)
                    vec = n2v_model.predict(str(i))
                except KeyError:
                    try:
                        # 如果失败，尝试 Int Key
                        vec = n2v_model.predict(i)
                    except KeyError:
                        # 如果还失败 (孤立节点)，用 0 填充
                        vec = np.zeros(NODE2VEC_DIM)
                embeddings.append(vec)
            
            data.x = torch.tensor(np.array(embeddings), dtype=torch.float)
            print(f"✅ 特征生成完毕! 耗时: {time.time()-t0:.2f}s | 维度: {data.x.shape}")

        # -------------------- 后续处理 --------------------
        data.x = data.x.float()
        data.y = data.y.long()

        if backend_type == 'Coauthor' or backend_type == 'txt': 
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
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        if NODE2VEC_DIM >= 128:
            HIDDEN_DIM = 64
        else:
            HIDDEN_DIM = 16
            
        # 构建 EasyGraph
        g = eg.Graph()
        edge_list = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
        g.add_nodes_from(range(num_nodes))
        g.add_edges(edge_list)

        try:
            data = data.to(DEVICE)
        except:
            pass
        
        num_node_features = data.x.shape[1] 
        num_classes = int(data.y.max().item()) + 1

        # -------------------- 训练循环 --------------------
        All_forward_times, All_backward_times, All_epoch_times = [], [], []
        All_total_train_time, All_test_acc, ALL_f1 = [], [], []

        x_orig = data.x.clone()
        y_orig = data.y.clone()
        train_mask_orig = train_mask.clone()
        val_mask_orig = val_mask.clone()
        test_mask_orig = test_mask.clone()

        for R in tqdm(range(RR + 1)):
            if 'adj_gp' in g.cache: del g.cache['adj_gp']
            g.build_adj_gp(nparts=32)
            perm = g.cache['gp_perm']
            
            data.x = x_orig[perm]
            data.y = y_orig[perm]
            data.train_mask = train_mask_orig[perm]
            data.val_mask = val_mask_orig[perm]
            data.test_mask = test_mask_orig[perm]

            model = GCN(num_node_features, HIDDEN_DIM, num_classes, dropout=DROPOUT).to(DEVICE)
            optimizer = torch.optim.Adam([
                {'params': model.gcn1.parameters(), 'weight_decay': 5e-4},
                {'params': model.gcn2.parameters(), 'weight_decay': 0.0}
            ], lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()

            train_start = time.perf_counter()
            process = psutil.Process(os.getpid())
            peak_memory_mb = 0

            for epoch in range(1, EPOCHS+1):
                model.train()
                optimizer.zero_grad()
                
                t_s = time.perf_counter()
                out = model(data.x, g)
                t_f = time.perf_counter()
                
                loss = criterion(out[data.train_mask], data.y[data.train_mask].squeeze())
                
                t_b_s = time.perf_counter()
                loss.backward()
                optimizer.step()
                t_b_e = time.perf_counter()

                if R > 0: 
                    All_forward_times.append(t_f - t_s)
                    All_backward_times.append(t_b_e - t_b_s)
                    All_epoch_times.append(t_b_e - t_s)
                
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                peak_memory_mb = max(peak_memory_mb, current_memory_mb)

            total_train_time = time.perf_counter() - train_start
            
            model.eval()
            with torch.no_grad():
                out = model(data.x, g)
                pred = out.argmax(dim=1)
                test_acc = (pred[data.test_mask] == data.y[data.test_mask].squeeze()).sum().item() / data.test_mask.sum().item()
                macro_f1 = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='macro')
            
            if R > 0:
                All_total_train_time.append(total_train_time)
                All_test_acc.append(test_acc)
                ALL_f1.append(macro_f1)

        Show_Result(dataset_name, RR, All_total_train_time, All_epoch_times, All_forward_times, All_backward_times, peak_memory_mb, All_test_acc, ALL_f1)

if __name__ == "__main__":
    main()
    import pandas as pd
    df = pd.DataFrame(Result_Chart).T
    df.to_csv("gcn_node2vec_results.csv")
    print("\n所有结果已保存至 gcn_node2vec_results.csv")