import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, precision_score, recall_score
from model_torch import myModel_concat_new as myModel
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
names = ['HMEC']




model = myModel(max_len_en, max_len_pr, nwords, emb_dim).to(device)
model.load_state_dict(torch.load("model_DynFusionEPI/HMEC_epoch_16.pth"))
model.eval()

with torch.no_grad():
    for name in names:
        data_dir = '../data1/%s/' % name
        test_data = np.load(data_dir + f'{name}_test.npz')
        # test_data = np.load("/root/lanyun-fs/IMR90_test_new.npz")
        X_en_tes, X_pr_tes, y_tes = test_data['X_en_tes'], test_data['X_pr_tes'], test_data['y_tes']

        
        X_en_tes = torch.tensor(X_en_tes, dtype=torch.long)
        X_pr_tes = torch.tensor(X_pr_tes, dtype=torch.long)
        y_tes = torch.tensor(y_tes, dtype=torch.float32)

        
        test_dataset = TensorDataset(X_en_tes, X_pr_tes, y_tes)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



        auc, aupr, mcc, f1, precision, recall = evaluate_model(model, test_loader, device)

        print("AUC : ", auc)
        print("AUPR : ", aupr)
        print("MCC : ", mcc)
        print("F1 Score : ", f1)
        print("Precision : ", precision)
        print("Recall : ", recall)