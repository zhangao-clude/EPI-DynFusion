import numpy as np
import torch.nn.functional as F
from ResNeXt import *
from CBAM import *


num_filters = 72
max_len_en = 3000
max_len_pr = 2000
nwords = 4097
emb_dim = 100
pretrain_embeddings = torch.tensor(np.load("embedding_matrix.npy"))


class myModel_concat_new(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(myModel_concat_new, self).__init__()
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
        # self.cbam_layer = CBAM_my(channel=emb_dim, in_features=488, convker=3)
        self.cbam_layer = CBAMLayer(channel=emb_dim)

        self.gru = nn.GRU(input_size=emb_dim, hidden_size=50, num_layers=2,
                          batch_first=True, bidirectional=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=10, batch_first=True, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.merger_dense = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
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
        # GRU处理
        ep_gru, _ = self.gru(ep_attn_output)  # 64 388 100

        # 加权融合
        weight_transformer = torch.sigmoid(self.weight_transformer)
        weight_gru = torch.sigmoid(self.weight_gru)
        ep_combined = weight_transformer * ep_attn_output + weight_gru * ep_gru

        ep_cbam = self.cbam_layer(ep_combined.transpose(1, 2)).transpose(1, 2)
        # ep_gru, _ = self.gru(ep_cbam.transpose(1, 2))  # 64 388 200

        # ep_attn_output, _ = self.attention_layer(ep_gru, ep_gru, ep_gru)  # 64 388 200

        ep_maxpool = ep_cbam.max(dim=1)[0]  # 64 200

        merge = self.bn1(ep_maxpool)
        merge_dense = F.relu(self.merger_dense(merge))
        merge_dense = self.bn(merge_dense)
        merge_dense = self.dropout(merge_dense)  # 在全连接层之后添加Dropout
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class EPI_DyFusion_noCBAM(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(EPI_DyFusion_noCBAM, self).__init__()
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
        # self.cbam_layer = CBAM_my(channel=emb_dim, in_features=488, convker=3)
        self.cbam_layer = CBAMLayer(channel=emb_dim)

        self.gru = nn.GRU(input_size=emb_dim, hidden_size=50, num_layers=2,
                          batch_first=True, bidirectional=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=10, batch_first=True, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.merger_dense = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
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
        # GRU处理
        ep_gru, _ = self.gru(ep_attn_output)  # 64 388 100

        # 加权融合
        weight_transformer = torch.sigmoid(self.weight_transformer)
        weight_gru = torch.sigmoid(self.weight_gru)
        ep_combined = weight_transformer * ep_attn_output + weight_gru * ep_gru

        ep_maxpool = ep_combined.max(dim=1)[0]  # 64 200

        merge = self.bn1(ep_maxpool)
        merge_dense = F.relu(self.merger_dense(merge))
        merge_dense = self.bn(merge_dense)
        merge_dense = self.dropout(merge_dense)  # 在全连接层之后添加Dropout
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class EPI_DyFusion_noFusion(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(EPI_DyFusion_noFusion, self).__init__()
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
        # self.cbam_layer = CBAM_my(channel=emb_dim, in_features=488, convker=3)
        self.cbam_layer = CBAMLayer(channel=emb_dim)

        self.gru = nn.GRU(input_size=emb_dim, hidden_size=50, num_layers=2,
                          batch_first=True, bidirectional=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=10, batch_first=True, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.merger_dense = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
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
        # GRU处理
        ep_gru, _ = self.gru(ep_attn_output)  # 64 388 100

        # 加权融合
        # weight_transformer = torch.sigmoid(self.weight_transformer)
        # weight_gru = torch.sigmoid(self.weight_gru)
        # ep_combined = weight_transformer * ep_attn_output + weight_gru * ep_gru

        ep_cbam = self.cbam_layer(ep_gru.transpose(1, 2)).transpose(1, 2)
        ep_maxpool = ep_cbam.max(dim=1)[0]  # 64 200

        merge = self.bn1(ep_maxpool)
        merge_dense = F.relu(self.merger_dense(merge))
        merge_dense = self.bn(merge_dense)
        merge_dense = self.dropout(merge_dense)  # 在全连接层之后添加Dropout
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class EPI_CNN(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(EPI_CNN, self).__init__()
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

        self.merger_dense = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
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

        ep_maxpool = ep.max(dim=1)[0]  # 64 100

        merge = self.bn1(ep_maxpool)
        merge_dense = F.relu(self.merger_dense(merge))
        merge_dense = self.bn(merge_dense)
        merge_dense = self.dropout(merge_dense)  # 在全连接层之后添加Dropout
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class EPI_CNN_transformer(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(EPI_CNN_transformer, self).__init__()
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

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=10, batch_first=True, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.merger_dense = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
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

        ep_maxpool = ep_attn_output.max(dim=1)[0]  # 64 200

        merge = self.bn1(ep_maxpool)
        merge_dense = F.relu(self.merger_dense(merge))
        merge_dense = self.bn(merge_dense)
        merge_dense = self.dropout(merge_dense)  # 在全连接层之后添加Dropout
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class EPI_CNN_BiGRU(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(EPI_CNN_BiGRU, self).__init__()
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
        # self.cbam_layer = CBAM_my(channel=emb_dim, in_features=488, convker=3)
        self.cbam_layer = CBAMLayer(channel=emb_dim)

        self.gru = nn.GRU(input_size=emb_dim, hidden_size=50, num_layers=2,
                          batch_first=True, bidirectional=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=10, batch_first=True, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.merger_dense = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
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

        # ep_attn_output = self.encoder(ep)  # 64 388 100
        # GRU处理
        ep_gru, _ = self.gru(ep)  # 64 388 100

        ep_maxpool = ep_gru.max(dim=1)[0]  # 64 200

        merge = self.bn1(ep_maxpool)
        merge_dense = F.relu(self.merger_dense(merge))
        merge_dense = self.bn(merge_dense)
        merge_dense = self.dropout(merge_dense)  # 在全连接层之后添加Dropout
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class EPI_CBAM(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(EPI_CBAM, self).__init__()
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
        self.cbam_layer = CBAMLayer(channel=emb_dim)

        self.merger_dense = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
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

        ep_cbam = self.cbam_layer(ep.transpose(1, 2)).transpose(1, 2)
        # ep_gru, _ = self.gru(ep_cbam.transpose(1, 2))  # 64 388 200

        # ep_attn_output, _ = self.attention_layer(ep_gru, ep_gru, ep_gru)  # 64 388 200

        ep_maxpool = ep_cbam.max(dim=1)[0]  # 64 200

        merge = self.bn1(ep_maxpool)
        merge_dense = F.relu(self.merger_dense(merge))
        merge_dense = self.bn(merge_dense)
        merge_dense = self.dropout(merge_dense)  # 在全连接层之后添加Dropout
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class EPI_Trans_BiGRU(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(EPI_Trans_BiGRU, self).__init__()
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
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=50, num_layers=2,
                          batch_first=True, bidirectional=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=10, batch_first=True, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.merger_dense = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
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
        # GRU处理
        ep_gru, _ = self.gru(ep_attn_output)  # 64 388 100

        ep_maxpool = ep_gru.max(dim=1)[0]  # 64 200

        merge = self.bn1(ep_maxpool)
        merge_dense = F.relu(self.merger_dense(merge))
        merge_dense = self.bn(merge_dense)
        merge_dense = self.dropout(merge_dense)  # 在全连接层之后添加Dropout
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class myModel_demo1(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(myModel_demo1, self).__init__()
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
        self.fixed_weights = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=False)
        self.cbam_layer = CBAMLayer(channel=emb_dim)

        self.gru = nn.GRU(input_size=emb_dim, hidden_size=50, num_layers=2,
                          batch_first=True, bidirectional=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=10, batch_first=True, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.merger_dense = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
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
        # GRU处理
        ep_gru, _ = self.gru(ep_attn_output)  # 64 388 100

        # 加权融合
        weight_transformer = torch.sigmoid(self.weight_transformer)
        weight_gru = torch.sigmoid(self.weight_gru)
        ep_combined = self.fixed_weights[0] * ep_attn_output + self.fixed_weights[1] * ep_gru

        ep_cbam = self.cbam_layer(ep_combined.transpose(1, 2)).transpose(1, 2)
        # ep_gru, _ = self.gru(ep_cbam.transpose(1, 2))  # 64 388 200

        # ep_attn_output, _ = self.attention_layer(ep_gru, ep_gru, ep_gru)  # 64 388 200

        ep_maxpool = ep_cbam.max(dim=1)[0]  # 64 200

        merge = self.bn1(ep_maxpool)
        merge_dense = F.relu(self.merger_dense(merge))
        merge_dense = self.bn(merge_dense)
        merge_dense = self.dropout(merge_dense)  # 在全连接层之后添加Dropout
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class myModel_demo2(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(myModel_demo2, self).__init__()
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
        self.linear_after_concat = nn.Linear(200, 100)
        self.cbam_layer = CBAMLayer(channel=emb_dim)

        self.gru = nn.GRU(input_size=emb_dim, hidden_size=50, num_layers=2,
                          batch_first=True, bidirectional=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=10, batch_first=True, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.merger_dense = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
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
        # GRU处理
        ep_gru, _ = self.gru(ep_attn_output)  # 64 388 100

        # 加权融合
        ep_combined = torch.cat([ep_attn_output, ep_gru], dim=-1)
        ep_combined = self.linear_after_concat(ep_combined)

        ep_cbam = self.cbam_layer(ep_combined.transpose(1, 2)).transpose(1, 2)
        # ep_gru, _ = self.gru(ep_cbam.transpose(1, 2))  # 64 388 200

        # ep_attn_output, _ = self.attention_layer(ep_gru, ep_gru, ep_gru)  # 64 388 200

        ep_maxpool = ep_cbam.max(dim=1)[0]  # 64 200

        merge = self.bn1(ep_maxpool)
        merge_dense = F.relu(self.merger_dense(merge))
        merge_dense = self.bn(merge_dense)
        merge_dense = self.dropout(merge_dense)  # 在全连接层之后添加Dropout
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class myModel_demo3(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(myModel_demo3, self).__init__()
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

        self.attention_query = nn.Linear(emb_dim, emb_dim)
        self.attention_key = nn.Linear(emb_dim, emb_dim)
        self.attention_value = nn.Linear(emb_dim, emb_dim)
        self.cbam_layer = CBAMLayer(channel=emb_dim)

        self.gru = nn.GRU(input_size=emb_dim, hidden_size=50, num_layers=2,
                          batch_first=True, bidirectional=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=10, batch_first=True, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.merger_dense = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
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
        # GRU处理
        ep_gru, _ = self.gru(ep_attn_output)  # 64 388 100

        # 加权融合
        stacked_features = torch.stack([ep_attn_output, ep_gru], dim=1)  # 64 2 388 100
        # 计算注意力权重
        query = self.attention_query(ep_attn_output).unsqueeze(1)  # 64 1 388 100
        key = self.attention_key(stacked_features)  # 64 2 388 100
        value = self.attention_value(stacked_features)  # 64 2 388 100

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(emb_dim))
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 加权求和得到融合特征
        fused_feature = torch.matmul(attention_weights, value).squeeze(1)
        ep_combined = fused_feature.reshape(64, -1, 100)

        ep_cbam = self.cbam_layer(ep_combined.transpose(1, 2)).transpose(1, 2)
        # ep_gru, _ = self.gru(ep_cbam.transpose(1, 2))  # 64 388 200

        # ep_attn_output, _ = self.attention_layer(ep_gru, ep_gru, ep_gru)  # 64 388 200

        ep_maxpool = ep_cbam.max(dim=1)[0]  # 64 200

        merge = self.bn1(ep_maxpool)
        merge_dense = F.relu(self.merger_dense(merge))
        merge_dense = self.bn(merge_dense)
        merge_dense = self.dropout(merge_dense)  # 在全连接层之后添加Dropout
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class EPI_Trans(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(EPI_Trans, self).__init__()
        self.embedding_en = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=False)
        self.embedding_pr = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=False)

        self.enhancer_conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=80, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=15, stride=15)
        )

        self.promoter_conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=61, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=10)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=num_filters * 2, nhead=9, batch_first=True,
                                                   dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.merger_dense = nn.Linear(in_features=num_filters * 2, out_features=50)
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
        self.bn = nn.BatchNorm1d(num_features=num_filters * 2)
        self.bn1 = nn.BatchNorm1d(num_features=50)
        self.bn2 = nn.BatchNorm1d(num_features=num_filters * 2)
        self.output_layer = nn.Linear(in_features=50, out_features=1)

    def forward(self, enhancers, promoters):
        enhancers = enhancers.long()
        promoters = promoters.long()

        emb_en = self.embedding_en(enhancers)  # 64，3000，100
        emb_pr = self.embedding_pr(promoters)  # 64，2000，100

        emb_en = emb_en.float()
        emb_pr = emb_pr.float()

        enhancer_conv = self.enhancer_conv_layer(emb_en.transpose(1, 2))  # 64 72 199
        promoter_conv = self.promoter_conv_layer(emb_pr.transpose(1, 2))  # 64 72 199

        ep = torch.cat([enhancer_conv, promoter_conv], dim=1)  # 64 144 194
        ep = self.bn2(ep)
        ep = self.dropout(ep)

        ep = ep.transpose(1, 2)  # 64 194 144

        ep_attn_output = self.encoder(ep)  # 64 388 100

        ep_combined = ep_attn_output

        # ep_gru, _ = self.gru(ep_cbam.transpose(1, 2))  # 64 388 200

        # ep_attn_output, _ = self.attention_layer(ep_gru, ep_gru, ep_gru)  # 64 388 200

        ep_maxpool = ep_combined.max(dim=1)[0]  # 64 144

        # merge = self.bn1(ep_maxpool)
        merge_dense = F.relu(self.bn1(self.merger_dense(ep_maxpool)))
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class EPI_Mind(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(EPI_Mind, self).__init__()
        self.embedding_en = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=False)
        self.embedding_pr = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=False)

        self.enhancer_conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=36),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=20, stride=20)
        )

        self.promoter_conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=36),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=20, stride=20)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=num_filters, nhead=8, batch_first=True,
                                                   dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.merger_dense = nn.Linear(in_features=num_filters * 2, out_features=50)
        self.output_layer = nn.Linear(in_features=50, out_features=1)

    def forward(self, enhancers, promoters):
        enhancers = enhancers.long()
        promoters = promoters.long()

        emb_en = self.embedding_en(enhancers)  # 64，3000，100
        emb_pr = self.embedding_pr(promoters)  # 64，2000，100

        emb_en = emb_en.float()
        emb_pr = emb_pr.float()

        enhancer_conv = self.enhancer_conv_layer(emb_en.transpose(1, 2))  # 64 72 148
        promoter_conv = self.promoter_conv_layer(emb_pr.transpose(1, 2))  # 64 72 98

        enhancer_trans = self.encoder(enhancer_conv.transpose(1, 2))  # 64 148 72
        promoter_trans = self.encoder(promoter_conv.transpose(1, 2))  # 64 98 72

        enhancer_maxpool = enhancer_trans.max(dim=1)[0]  # 64 72
        promoter_maxpool = promoter_trans.max(dim=1)[0]  # 64 72

        merge = torch.cat([enhancer_maxpool * promoter_maxpool, torch.abs(enhancer_maxpool - promoter_maxpool)],
                          dim=-1)  # 64 144

        merge_dense = F.relu(self.merger_dense(merge))
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class AttLayer(nn.Module):
    def __init__(self, attention_dim):
        super(AttLayer, self).__init__()
        self.attention_dim = attention_dim

        # 定义权重参数
        self.W = nn.Parameter(torch.randn(attention_dim, attention_dim) * 0.1)  # 初始化权重矩阵 W
        self.b = nn.Parameter(torch.zeros(attention_dim))  # 偏置 b
        self.u = nn.Parameter(torch.randn(attention_dim, 1) * 0.1)  # 权重向量 u

    def forward(self, x, mask=None):
        """
        前向传播计算注意力
        :param x: 输入张量，形状 (batch_size, seq_len, input_dim)
        :param mask: 掩码张量，形状 (batch_size, seq_len)
        :return: 注意力输出，形状 (batch_size, input_dim)
        """
        # 计算 uit = tanh(xW + b)
        uit = torch.tanh(torch.matmul(x, self.W) + self.b)  # (batch_size, seq_len, attention_dim)

        # 计算 ait = uit·u
        ait = torch.matmul(uit, self.u).squeeze(-1)  # (batch_size, seq_len)

        # 对 ait 应用 softmax（加掩码机制）
        if mask is not None:
            ait = ait.masked_fill(mask == 0, float('-inf'))  # 使用 mask 屏蔽无效位置
        ait = F.softmax(ait, dim=1)  # (batch_size, seq_len)

        # 计算权重 ait 乘以输入 x
        ait = ait.unsqueeze(-1)  # (batch_size, seq_len, 1)
        weighted_input = x * ait  # (batch_size, seq_len, input_dim)

        # 求和以获得注意力加权的输出
        output = torch.sum(weighted_input, dim=1)  # (batch_size, input_dim)

        return output


class EPI_DLMH(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(EPI_DLMH, self).__init__()
        self.embedding_en = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=False)
        self.embedding_pr = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=False)

        self.enhancer_conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=64, kernel_size=60, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=30, stride=30)
        )

        self.promoter_conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=64, kernel_size=40, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=20, stride=20)
        )

        self.gru = nn.GRU(input_size=99, hidden_size=25, num_layers=1, batch_first=True, bidirectional=True)

        self.attention_layer = AttLayer(50)

        self.merger_dense = nn.Linear(in_features=emb_dim * 2, out_features=64)
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
        self.bn = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=emb_dim * 2)
        self.output_layer = nn.Linear(in_features=64, out_features=1)

    def forward(self, enhancers, promoters):
        enhancers = enhancers.long()
        promoters = promoters.long()

        emb_en = self.embedding_en(enhancers)  # 64，3000，100
        emb_pr = self.embedding_pr(promoters)  # 64，2000，100

        emb_en = emb_en.float()
        emb_pr = emb_pr.float()

        enhancer_conv = self.enhancer_conv_layer(emb_en.transpose(1, 2))  # 64 64 99
        promoter_conv = self.promoter_conv_layer(emb_pr.transpose(1, 2))  # 64 64 99

        enhancer_gru = self.gru(enhancer_conv)[0]
        promoter_gru = self.gru(promoter_conv)[0]

        enhancer_att = self.attention_layer(enhancer_gru)
        promoter_att = self.attention_layer(promoter_gru)

        l1 = enhancer_att * promoter_att
        l2 = torch.abs(enhancer_att - promoter_att)

        ep = torch.cat([enhancer_att, promoter_att, l1, l2], dim=1)  # 64 200
        ep = self.bn2(ep)
        ep = self.dropout(ep)

        merge_dense = self.merger_dense(ep)
        merge_dense = F.relu(self.bn(merge_dense))
        merge_dense = self.dropout(merge_dense)  # 在全连接层之后添加Dropout
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output


class EPI_VAN(nn.Module):
    def __init__(self, max_len_en, max_len_pr, nwords, emb_dim):
        super(EPI_VAN, self).__init__()
        self.embedding_en = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=False)
        self.embedding_pr = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=False)

        self.enhancer_conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=64, kernel_size=40, padding='valid'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=20, stride=20)
        )

        self.promoter_conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=64, kernel_size=40, padding='valid'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=20, stride=20)
        )

        self.gru = nn.GRU(input_size=64, hidden_size=25, num_layers=1, batch_first=True, bidirectional=True)

        self.attention_layer = AttLayer(50)

        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
        self.bn = nn.BatchNorm1d(num_features=244)
        self.output_layer = nn.Linear(in_features=50, out_features=1)

    def forward(self, enhancers, promoters):
        enhancers = enhancers.long()
        promoters = promoters.long()

        emb_en = self.embedding_en(enhancers)  # 64，3000，100
        emb_pr = self.embedding_pr(promoters)  # 64，2000，100

        emb_en = emb_en.float()
        emb_pr = emb_pr.float()

        enhancer_conv = self.enhancer_conv_layer(emb_en.transpose(1, 2)).transpose(1, 2)  # 64 147 64
        promoter_conv = self.promoter_conv_layer(emb_pr.transpose(1, 2)).transpose(1, 2)  # 64 97 64

        merge = torch.cat([enhancer_conv, promoter_conv], dim=1)  # 64 246 64

        merge = self.bn(merge)
        merge = self.dropout(merge)

        l_gru = self.gru(merge)[0]
        l_att = self.attention_layer(l_gru)

        output = torch.sigmoid(self.output_layer(l_att))

        return output
