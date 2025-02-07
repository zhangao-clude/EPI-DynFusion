import numpy as np
import torch.nn.functional as F
from CBAM import *


num_filters = 72
max_len_en = 3000
max_len_pr = 2000
nwords = 4097
emb_dim = 100
pretrain_embeddings = torch.tensor(np.load("embedding_matrix.npy"))


class EPI_DynFusion(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(EPI_DynFusion, self).__init__()
        self.embedding_en = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=False)
        self.embedding_pr = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=False)

        self.enhancer_conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=80),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=15, stride=15)
        )

        self.promoter_conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=61),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=10)
        )
        self.weight_transformer = nn.Parameter(torch.randn(1))
        self.weight_gru = nn.Parameter(torch.randn(1))
        self.cbam_layer = CBAMLayer(channel=emb_dim)

        self.gru = nn.GRU(input_size=emb_dim, hidden_size=50, num_layers=2,
                          batch_first=True, bidirectional=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=10, batch_first=True, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.merger_dense = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.dropout = nn.Dropout(p=0.5)  
        self.bn = nn.BatchNorm1d(num_features=emb_dim)
        self.bn1 = nn.BatchNorm1d(num_features=emb_dim)
        self.bn2 = nn.BatchNorm1d(num_features=emb_dim)
        self.output_layer = nn.Linear(in_features=emb_dim, out_features=1)

    def forward(self, enhancers, promoters):
        enhancers = enhancers.long()
        promoters = promoters.long()

        emb_en = self.embedding_en(enhancers)  # 64，3000，100
        emb_pr = self.embedding_pr(promoters)  # 64，2000，100

        emb_en = emb_en.float()
        emb_pr = emb_pr.float()

        enhancer_conv = self.enhancer_conv_layer(emb_en.transpose(1, 2))  # 64 100 194
        promoter_conv = self.promoter_conv_layer(emb_pr.transpose(1, 2))  # 64 100 194

        ep = torch.cat([enhancer_conv, promoter_conv], dim=-1)  # 64 100 388
        ep = self.bn2(ep)
        ep = self.dropout(ep)

        ep = ep.transpose(1, 2)  # 64 388 100

        ep_attn_output = self.encoder(ep)  # 64 388 100
        ep_gru, _ = self.gru(ep_attn_output)  # 64 388 100


        weight_transformer = torch.sigmoid(self.weight_transformer)
        weight_gru = torch.sigmoid(self.weight_gru)
        ep_combined = weight_transformer * ep_attn_output + weight_gru * ep_gru

        ep_cbam = self.cbam_layer(ep_combined.transpose(1, 2)).transpose(1, 2)
       

        ep_maxpool = ep_cbam.max(dim=1)[0]  # 64 200

        merge = self.bn1(ep_maxpool)
        merge_dense = F.relu(self.merger_dense(merge))
        merge_dense = self.bn(merge_dense)
        merge_dense = self.dropout(merge_dense)  
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output

