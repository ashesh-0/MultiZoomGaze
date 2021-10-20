import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.resnet import ViewDense


class ChannelAttention(nn.Module):
    def __init__(self, input_count, feature_dim, latent_dim):
        super().__init__()
        self._lD = latent_dim
        self._fD = feature_dim
        self._N = input_count
        self._W = nn.ModuleList([nn.Linear(self._fD, self._lD) for _ in range(self._N)])
        self._V = nn.Linear(self._lD * self._N, self._N)

    def forward(self, *features):
        ml = [nn.Tanh()(self._W[i](features[i])) for i in range(self._N)]
        ml = torch.cat(ml, dim=1)
        out = self._V(ml)
        out = nn.Softmax(dim=-1)(out)
        return out


class SpatialAttention(nn.Module):
    # Adapted from https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size % 2 == 1, 'kernel size must be odd'
        padding = kernel_size // 2

        self._conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        """
        Args:
            x: Tensor with shape (batch, 512,sz,sz)
        """
        # import pdb
        # pdb.set_trace()
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # (batch, 2,sz,sz)
        x = torch.cat([avg_out, max_out], dim=1)
        # (batch, 1,sz,sz)
        x = self._conv1(x)
        return x


class SpatialAttentionMultiScale(SpatialAttention):
    def forward(self, embeddings):
        """
        Args:
            embeddings: Its shape should be (batch, seq_len, 512, sz, sz). where sz is 7 in default settings with
                Resnet18
        """

        attention_outputs = []
        N, seq_len, f_len, sz, _ = embeddings.shape
        for i in range(seq_len):
            ao = super().forward(embeddings[:, i, ...])

            # (batch, 1,1,sz,sz)
            ao = ao[:, None, ...]

            attention_outputs.append(ao)

        # (batch, seq_len,1,sz,sz)
        attention = torch.cat(attention_outputs, dim=1)
        # At every pixel, ith final feature now becomes weighted average of ith features of different scales. Also,
        # weights are same for all features. Weights are different for every pixel
        attention = nn.Softmax(dim=1)(attention)

        # (batch, 512,sz,sz)
        output = torch.sum(embeddings * attention, dim=1)
        return output


class SpatialAttentionMultiScaleAligned(SpatialAttentionMultiScale):
    def __init__(self, cropsize_list, kernel_size=3):
        super().__init__(kernel_size=kernel_size)
        assert isinstance(cropsize_list, list)
        self._cropsizes = cropsize_list
        self._img_size = 224

        # corresponding to 224, we have 7*7 feature map at the end.
        self._default_sz = 7

    def aligned_embedding(self, embeddings):
        """
        Args:
            embeddings: Its shape should be (batch, seq_len, 512, sz, sz). where sz is 7 in default settings with
                Resnet18

        """
        min_crop_sz = min(self._cropsizes)
        interp_sz = [int(np.ceil(c * self._default_sz / min_crop_sz)) for c in self._cropsizes]
        max_sz = max(interp_sz)

        batch, seq_len, f_len, sz, sz = embeddings.shape
        aligned_emb_list = []
        for i in range(seq_len):
            emb = embeddings[:, i, ...]
            # upsample to align pixels
            aligned_emb = nn.Upsample(interp_sz[i], mode='bilinear', align_corners=False)(emb)
            # pad for uniformity in spatial dimensions
            lstep = (max_sz - interp_sz[i]) // 2
            rstep = max_sz - (lstep + interp_sz[i])
            aligned_emb = F.pad(aligned_emb, (lstep, rstep, lstep, rstep), mode='constant', value=0)
            aligned_emb_list.append(aligned_emb[:, None, ...])

        output = torch.cat(aligned_emb_list, dim=1)
        return output

    def forward(self, embeddings):
        embeddings = self.aligned_embedding(embeddings)
        return super().forward(embeddings)


class SpatialAttentionBig(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(512, 16, 1),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, input):
        """
        Args:
            input: Tensor with shape (batch, 512,sz,sz)
        """
        return self.attention(input)


class SpatialAttentionMultiScaleV2(SpatialAttentionBig):
    def forward(self, embeddings):
        """
        Args:
            embeddings: Its shape should be (batch, seq_len, 512, sz, sz). where sz is 7 in default settings with
                Resnet18
        """

        attention_outputs = []
        N, seq_len, f_len, sz, _ = embeddings.shape
        # import pdb
        # pdb.set_trace()
        for i in range(seq_len):
            ao = super().forward(embeddings[:, i, ...])
            # import pdb
            # pdb.set_trace()
            # (batch, 1,1,sz,sz)
            ao = ao[:, None, ...]

            attention_outputs.append(ao)

        # (batch, seq_len,1,sz,sz)
        attention = torch.cat(attention_outputs, dim=1)
        # At every pixel, ith final feature now becomes weighted average of ith features of different scales. Also,
        # weights are same for all features. Weights are different for every pixel
        attention = nn.Softmax(dim=1)(attention)

        # (batch, 512,sz,sz)
        output = torch.sum(embeddings * attention, dim=1)
        return output


class FeatureAttention(nn.Module):
    def __init__(self, fc=256):
        super().__init__()

        self.attention = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(3, 16, 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
            ViewDense(),
            nn.Linear(32, fc),
        )

    def forward(self, input):
        """
        Args:
            input: Tensor with shape (batch, 512,sz,sz)
        """
        return self.attention(input)


class AttentionMultiScale(FeatureAttention):
    def forward(self, input, embeddings):
        """
        Args:
            input: (batch,seq,3,224,224)
            embeddings: Its shape should be (batch, seq_len, 512)
        """
        N, seqLen, feLen = embeddings.size()
        static_shape = (-1, 3) + input.size()[-2:]
        inp = input.view(static_shape)

        attention = super().forward(inp)
        attention = nn.Softmax(dim=1)(attention.view(N, seqLen, feLen))
        return torch.sum(embeddings * attention, dim=1)


# class SelfAttention(nn.Module):
#     """
#     Self attention Layer
#     Adapted from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
#     """
#     def __init__(self, in_dim, activation):
#         super(SelfAttention, self).__init__()
#         self.chanel_in = in_dim
#         self.activation = activation

#         self.query_fc = nn.Dense(in_channels=in_dim, out_channels=in_dim // 8)
#         self.key_fc = nn.Dense(in_channels=in_dim, out_channels=in_dim // 8)
#         self.value_fc = nn.Dense(in_channels=in_dim, out_channels=in_dim)
#         self.gamma = nn.Parameter(torch.zeros(1))

#         self.softmax = nn.Softmax(dim=-1)  #

#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#         # We have already applied spatial avg-pooling
#         # input.size(0), self._seq_len, f_len
#         batchLen, sqLen, fLen = x.size()

#         x_flat = x.view(batchLen * sqLen, fLen)
#         # b,c/8,w,h => b,c/8,wh=> b,sQ,c/8 => b,c/8,sQ
#         proj_query = self.query_fc(x_flat).view(batchLen, sqLen, -1).permute(0, 2, 1)

#         # bsQ,c => b,sQ,c/8
#         proj_key = self.key_fc(x_flat).view(batchLen, sqLen, -1)
#         # b,c/8,c/8
#         energy = torch.bmm(proj_query, proj_key)  # transpose check
#         # b,wh,wh <===> b, c, seq
#         attention = self.softmax(energy)  # BX (N) X (N)
#         # b,c,w,h=>b,c,wh <=====> (b,seq,c)-> (b,c,seq)
#         proj_value = self.value_fc(x).view(batchLen, -1, width * height)  # B X C X N
#         # b,c,wh * b,wh,wh=> b,c,wh <===>  (b,c,seq) *(b,seq,c) => (b,c,c)
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         # b,c,wh=> b,c,w,h
#         out = out.view(batchLen, C, width, height)

#         out = self.gamma * out + x
#         return out, attention
