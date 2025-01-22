import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, precision_score, recall_score
from model_torch import myModel_concat_new as myModel
from torch.utils.data import DataLoader, TensorDataset

# 确保使用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
max_len_en = 3000
max_len_pr = 2000
nwords = 4097
emb_dim = 100


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_en_batch, X_pr_batch, y_batch in data_loader:
            X_en_batch, X_pr_batch, y_batch = X_en_batch.to(device), X_pr_batch.to(device), y_batch.to(device)
            y_pred = model(X_en_batch, X_pr_batch).cpu().numpy().flatten()
            all_preds.append(y_pred)
            all_labels.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    aupr = average_precision_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, np.round(all_preds))
    f1 = f1_score(all_labels, np.round(all_preds))
    precision = precision_score(all_labels, np.round(all_preds))
    recall = recall_score(all_labels, np.round(all_preds))

    return auc, aupr, mcc, f1, precision, recall

# 数据集名称
# names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
names = ['HMEC']



# 初始化模型并加载权重
model = myModel(max_len_en, max_len_pr, nwords, emb_dim).to(device)
model.load_state_dict(torch.load("model_DynFusionEPI/HMEC_epoch_16.pth"))
model.eval()
# 评估
with torch.no_grad():
    for name in names:
        data_dir = '../data1/%s/' % name
        test_data = np.load(data_dir + f'{name}_test.npz')
        # test_data = np.load("/root/lanyun-fs/IMR90_test_new.npz")
        X_en_tes, X_pr_tes, y_tes = test_data['X_en_tes'], test_data['X_pr_tes'], test_data['y_tes']

        # 将数据转换为PyTorch张量
        X_en_tes = torch.tensor(X_en_tes, dtype=torch.long)
        X_pr_tes = torch.tensor(X_pr_tes, dtype=torch.long)
        y_tes = torch.tensor(y_tes, dtype=torch.float32)

        # 创建数据加载器
        test_dataset = TensorDataset(X_en_tes, X_pr_tes, y_tes)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        print(f"****************在{name}细胞系上测试模型****************")

        auc, aupr, mcc, f1, precision, recall = evaluate_model(model, test_loader, device)

        print("AUC : ", auc)
        print("AUPR : ", aupr)
        print("MCC : ", mcc)
        print("F1 Score : ", f1)
        print("Precision : ", precision)
        print("Recall : ", recall)