import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import device
from torch.nn import Softmax


def INF(self, B, H, W, device):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1).half().to(device)
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Moudle"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()

        proj_query = self.query_conv(x)
        # b, c', h, w ===> b, w, c', h ===> b*w, c', h ===> b*w, h, c'
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        # b, c', h, w ===> b, h, c', w ===> b*h, c', w ===> b*h, w, c'
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)

        proj_key = self.key_conv(x)
        # b, c', h, w ===> b, w, c', h ===> b*w, c', h
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # b, c', h, w ===> b, h, c', w ===> b*h, c', w
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(x)
        # b, c', h, w ===> b, w, c', h ===> b*w, c', h
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # b, c', h, w ===> b, h, c', w ===> b*h, c', w
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        # torch.bmm((b*w,h,c')x(b*w,c',h))===>(b*w,h,h)+(b*w,h,h)===>(b*w,h,h)===>(b,w,h,h)===>(b, h, w, h)
        energy_H = (torch.bmm(proj_query_H, proj_key_H).to(device) + self.INF(m_batchsize, height, width).to(
            device)).view(m_batchsize, width,
                          height,
                          height).permute(0,
                                          2,
                                          1,
                                          3)
        # energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
        #                                                                                              height,
        #                                                                                              height).permute(0,
        #                                                                                                              2,
        #                                                                                                              1,
        #                                                                                                              3)
        # # torch.bmm((b*h,w,c')x(b*h,c',w))===>(b*h,w,w)===>(b, h, w, w)
        energy_W = (torch.bmm(proj_query_W, proj_key_W)).view(m_batchsize, height, width, width)
        # torch.cat([(b,h,w,h),(b,h,w,w)], 3)===>(b,h,w,h+w)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        # (b,h,w,h+w)===>(b,h,w,h)===>(b,w,h,h)===>(b*w,h,h)
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # (b,h,w,h+w)===>(b,h,w,w)===>(b*h,w,w)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        # torch.bmm((b*w,c',h)x(b*w,h,h))===>(b*w,c',h)===>(b,w,c',h)===>(b,c',h,w)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        # torch.bmm((b*h,c',w)x(b*h,w,w))===>(b*h,c',w)===>(b,h,c',w)===>(b,c',h,w)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x

