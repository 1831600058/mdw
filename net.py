import math
from torch import nn
import torch
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],
]


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=30.0, m=0.7, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output


class MobileFaceNet(nn.Module):
    def __init__(self,
                 # num_class,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        # 20(10), 4(2), 8(4)
        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)

        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        # self.fc_out = nn.Linear(128, num_class)
        self.fc_out = nn.Linear(128, 41)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        feature = x.view(x.size(0), -1)
        out = self.fc_out(feature)
        return out, feature


class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder_s = nn.Sequential(
            *[nn.Sequential(
                # 313(10) , 63(2), 126(4)
                nn.LayerNorm(313),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for _ in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = self.conv_encoder_s(out)
        return out


class MixData_WeightProcess(nn.Module):
    def __init__(self, num_classes, mode,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 use_arcface=True, m=0.7, s=30, sub=1
                 ):
        super().__init__()

        self.arcface = ArcMarginProduct(in_features=c_dim, out_features=num_classes,
                                        m=m, s=s, sub=sub) if use_arcface else use_arcface
        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        self.mobilefacenet = MobileFaceNet(num_class=num_classes,
                                           bottleneck_setting=bottleneck_setting)
        self.mode = mode

        if mode not in ['arcface', 'noisy_arcface']:
            raise ValueError('Choose one of [arcface, noisy_arcface]')

        self.weight_process = Weight_Process(feature_dim=c_dim)
        self.self_attention = SelfAttention(feature_dim=c_dim, num_heads=8)

    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)

    # mel: B,C,M,T
    def forward(self, x_wav, x_mel, label, train=True):
        x_t = self.tgramnet(x_wav)
        # x_t_self_att = self.self_attention(x_t).unsqueeze(1)
        x_t = x_t.unsqueeze(1)
        x_mel_dw = self.weight_process(x_mel).unsqueeze(1)
        # x_mel = x_mel.squeeze(1)
        # x_mel_self_att= self.self_attention(x_mel).unsqueeze(1)
        # x_mel = x_mel.unsqueeze(1)
        x = torch.cat((x_t, x_mel, x_mel_dw), dim=1)
        # x = torch.cat((x_t, x_mel, x_mel_self_att), dim=1)

        out, feature = self.mobilefacenet(x)

        # if self.mode == 'arcmix':
        #     if train:
        #         out = self.arcface(feature, label[0])
        #         out_shuffled = self.arcface(feature, label[1])
        #         return out, out_shuffled, feature
        #     else:
        #         out = self.arcface(feature, label)
        #         return out, feature
        #
        # else:
        out = self.arcface(feature, label)
        return out, feature


class SelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, mask=None):
        B, M, T = x.shape
        x = x.permute(2, 0, 1)  # T,B,M
        if mask is not None:
            assert mask.shape == (T, B), "Mask should have shape (time_steps, batch_size)"
        attention_output, attention_weights = self.attention(x, x, x, key_padding_mask=mask)
        projected_output = self.proj(attention_output)
        output = self.norm(projected_output + x)
        output = output.permute(1, 2, 0)
        # return output, attention_weights
        return output


# Weight_Process

class Weight_Process(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.squeeze(1)
        dual = x + x
        result_dual = self.sigmoid(dual) * x
        # z = (1-self.sigmoid(dual)) * x
        return result_dual