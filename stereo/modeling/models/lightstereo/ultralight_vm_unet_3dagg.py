import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from mamba_ssm import Mamba


class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list)
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3)), dim=1)
        att = self.get_all_att(att.squeeze(-1).squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))

        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t3)

        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)

        return att1, att2, att3


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(
            nn.Conv3d(2, 1, 7, stride=1, padding=9, dilation=3),
            nn.Sigmoid()
        )

    def forward(self, t1, t2, t3):
        t_list = [t1, t2, t3]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2]


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3):
        r1, r2, r3 = t1, t2, t3

        satt1, satt2, satt3 = self.satt(t1, t2, t3)
        t1, t2, t3 = satt1 * t1, satt2 * t2, satt3 * t3

        r1_, r2_, r3_, = t1, t2, t3
        t1, t2, t3 = t1 + r1, t2 + r2, t3 + r3

        catt1, catt2, catt3 = self.catt(t1, t2, t3)
        t1, t2, t3 = catt1 * t1, catt2 * t2, catt3 * t3

        return t1 + r1_, t2 + r2_, t3 + r3_,


class UltraLight_VM_UNet(nn.Module):

    def __init__(self, input_channels=3, c_list=None,
                 split_att='fc', bridge=True):
        super().__init__()

        if c_list is None:
            c_list = [16, 32, 64]

        self.bridge = bridge

        self.encoder1 = nn.Sequential(PVMLayer(input_dim=input_channels, output_dim=c_list[0]))
        self.encoder2 = nn.Sequential(PVMLayer(input_dim=c_list[0], output_dim=c_list[1]))
        self.encoder3 = nn.Sequential(PVMLayer(input_dim=c_list[1], output_dim=c_list[2]))
        self.encoder4 = nn.Sequential(PVMLayer(input_dim=c_list[2], output_dim=c_list[2]))

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])

        self.encoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[1], output_dim=c_list[2])
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(PVMLayer(input_dim=c_list[2], output_dim=c_list[2]))
        self.decoder2 = nn.Sequential(PVMLayer(input_dim=c_list[2], output_dim=c_list[1]))
        self.decoder3 = nn.Sequential(PVMLayer(input_dim=c_list[1], output_dim=c_list[0]))

        self.dbn1 = nn.GroupNorm(4, c_list[2])
        self.dbn2 = nn.GroupNorm(4, c_list[1])
        self.dbn3 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv3d(c_list[0], 1, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):  # [bz, 48, H/4, W/4]
        out = F.gelu(F.max_pool3d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # [bz, 64, H/8, W/8]

        out = F.gelu(F.max_pool3d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # [bz, 96, H/16, W/16]

        out = F.gelu(F.max_pool3d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # [bz, 128, H/32, W/32]

        if self.bridge:
            t1, t2, t3 = self.scab(t1, t2, t3)

        out = F.gelu(self.encoder4(out))  # [bz, 128, H/32, W/32]

        out3 = F.gelu(self.dbn1(self.decoder1(out)))
        out3 = torch.add(out3, t3)  # # [bz, 128, H/32, W/32]

        out2 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out3)),
                                    scale_factor=(2, 2, 2),
                                    mode='trilinear',
                                    align_corners=True))
        out2 = torch.add(out2, t2)  # [bz, 96, H/16, W/16]

        out1 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out2)),
                                    scale_factor=(2, 2, 2),
                                    mode='trilinear',
                                    align_corners=True))
        out1 = torch.add(out1, t1)   # [bz, 64, H/8, W/8]

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2, 2), mode='trilinear',
                             align_corners=True)  # [bz, 48, H/4, W/4]

        return out0


if __name__ == '__main__':
    model = UltraLight_VM_UNet(input_channels=8).cuda()
    model(torch.randn(4, 8, 48, 128, 128).cuda())
