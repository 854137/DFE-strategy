# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math
import torch_dct as DCT
import numpy as np
import torch
import torch.nn as nn
# for DCNv3
import torch.nn.functional as F
from typing import Optional

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
    "Custom_CBAM",
    "SEAttention",
    "SC",
    "FEM",
    "CPFM",
    "BasicRFB",

    "ConvUtr",
    "BasicConv",
    "EMA",
    "RGA",
)




def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        # 无法设置bias,可改为用BasicConv
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            # act参数既可以是bool型，也可以是nn.ReLU()之类的激活函数
            # act=True/nn.ReLU()
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)
        # 这里直接可以写
        # self.conv2 = Conv(c2, c2, k, g=math.gcd(c2, c2), act=act)

    def forward(self, x):
        """
        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        # 复用Conv的初始化，其他方法一致
        # 初始化添加属性才是对父类初始化的覆盖


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)
    # conv1和conv2得到的空间尺寸似乎不一样？
    # 这里意思应该是两个卷积分支表示两个感受野的特征，
    # 扩展感受野，有效建立了不同局部特征之间的关联

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
    # [torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]]
    # 对通道维度进行降维，降成一维并且保留此维度
    # 从[B, C, H, W]-->[B, 1, H, W]


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)
        # 这里的空间注意力用的是大核，
        # 可以使用全局平均和全局最大值对每个位置的所有通道进行操作

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """
        Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
class Custom_CBAM(nn.Module):
    # Real CBAM
    # 严格按照定义设计的CBAM模块
    def __init__(self, c1, ratio=16, kernel_size=7):
        super(Custom_CBAM, self).__init__()
        # CAM initialize
        self.ave_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(c1, c1 // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(c1 // ratio, c1, bias=False),
            # nn.Sigmoid(),
        )

        # CBAM out
        self.sigmoid = nn.Sigmoid()

        # SAM initialize
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=autopad(kernel_size))

    def forward(self, x):
        # CAM
        B, C, H, W = x.shape
        x_max = self.mlp(self.max_pool(x).view(B, C))
        x_ave = self.mlp(self.ave_pool(x).view(B, C))
        # 自带的CBAM没有max_pool
        # 全局平均、最大池化后展平成B*C，进行MLP，
        # 由于，Linear只对最后一个维度进行操作，所以展平成B*C以对通道进行全连接，
        x_camout = self.sigmoid((x_max + x_ave).unsqueeze(-1).unsqueeze(-1)) * x
        # 这里是并行的模块融合
        # 扩展出H,W
        # 乘法过程中， [16, 1024, 1, 1] 会被广播成 [16, 1024, 20, 20]

        # SAM
        x_max = torch.max(x_camout, dim=1, keepdim=True)
        x_ave = torch.mean(x_camout, dim=1, keepdim=True)
        x_samout = self.sigmoid(self.conv(torch.cat([x_max.values, x_ave], dim=1))) * x
        return x_samout
# class Custom_CBAM(nn.Module):
#     # Real CBAM
#     # 严格按照定义设计的CBAM模块
#     def __init__(self, c1, ratio=16,  kernel_size=7):
#         super(Custom_CBAM, self).__init__()
#         # CAM initialize
#         self.ave_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         #shared MLP
#         self.mlp = nn.Sequential(
#             nn.Linear(c1, c1//ratio, bias=False),
#             nn.ReLU(),
#             nn.Linear(c1//ratio, c1 , bias=False),
#             # nn.Sigmoid(),
#         )
#
#         # CBAM out
#         self.sigmoid = nn.Sigmoid()
#
#         # SAM initialize
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=autopad(kernel_size))
#
#     def forward(self, x):
#         # CAM
#         B, C, H, W = x.shape
#         x = DCT.dct_2d(x.float(), norm='ortho')
#         # 上一行是新增修改
#         x_max = self.mlp(self.max_pool(x).view(B, C))
#         x_ave = self.mlp(self.ave_pool(x).view(B, C))
#         # 自带的CBAM没有max_pool
#         # 全局平均、最大池化后展平成B*C，进行MLP，
#         # 由于，Linear只对最后一个维度进行操作，所以展平成B*C以对通道进行全连接，
#         x_camout = self.sigmoid((x_max + x_ave).unsqueeze(-1).unsqueeze(-1))*x
#         # 这里是并行的模块融合
#         # 扩展出H,W
#         # 乘法过程中， [16, 1024, 1, 1] 会被广播成 [16, 1024, 20, 20]
#
#         # SAM
#         x_max = torch.max(x_camout, dim=1, keepdim=True)
#         x_ave = torch.mean(x_camout,dim=1, keepdim=True)
#         x_samout = self.sigmoid(self.conv(torch.cat([x_max.values ,x_ave], dim=1)))*x
#         # return x_samout
#         return DCT.idct_2d(x_samout, norm='ortho')
#         # 上一行是新增修改

class SEAttention(nn.Module):
     def __init__(self, c1, ratio=16):
         super().__init__()
         self.avepool = nn.AdaptiveAvgPool1d(1)
         self.mlp = nn.Sequential(
             nn.Linear(c1, c1 // ratio, bias=False),
             nn.ReLU(),
             nn.Linear(c1 // ratio, c1, bias=False),
             nn.Sigmoid(),
         )

     def forward(self, x):
         B, C ,H , W = x.shape
         x_temp = self.mlp(self.avepool(x).view(B,C)).unsqueeze(-1).unsqueeze(-1)
         # 取B, C以进行mlp， 然后恢复张量形状（即恢复.shape张量为四个维度）

         return x*x_temp
         # 或者自动广播以进行相乘
         # 或者利用.expand_as(x)
         # return x*x_temp.expand_as(x)

# AMSP-VConv
class BrokenBlock(nn.Module):
    # 随即通道重排，增加扰动性
    def __init__(self, dim_len, dim=1, dim_shape=4, group=1, *args, **kwargs):
        super(BrokenBlock, self).__init__(*args, **kwargs)
        self.dim = dim
        self.group = group
        self.dim_len = dim_len
        self.view = list(np.ones(dim_shape, int))

    def getShuffle(self, shape):
        # 生成随机索引，作为位置扰动
        # shape = [b,c,h,w]
        # shape的内容为这个，且shape的shape为1*4
        self.view[self.dim] = shape[self.dim]
        # view有四个维度，通道维保持一致
        perm = torch.randperm(self.dim_len // self.group).unsqueeze(1)
        # 随机生成dim_len/group个索引，
        # 索引形状从unsqueeze[dim_len/group]为[dim_len/group, 1]
        indices = torch.cat([perm * self.group + _ for _ in range(self.group)], dim=1)
        # 对索引值（索引向量列维度）放大到group倍，然后加上相对位置偏移生成一个组，生成group个索引向量
        # 按照维度1进行拼接，形状为[dim_len/group,group]
        indices = indices.view(self.view).expand(shape)
        # 将索引的形状变为1*4，通道维保持为c，
        # 索引形状为[1,c,1,1],再进行expand，则索引的形状为[b,c,h,w]
        return indices

    def forward(self, x):
        if self.training:
            return torch.gather(x, self.dim, self.getShuffle(x.shape).to(x.device))
        else:
            return x



class SC(nn.Module):
    # Split Concat
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, length=4, *args, **kwargs):
        super(SC, self).__init__(*args, **kwargs)
        self.length = length
        c_ = c2 // 2  # hidden channels
        # 对所有SC，c2 = 2c1,则这里的c_ = c1

        # 确保分组后每组至少1通道
        if c_ < length:
            length = c_ if c_ > 0 else 1
        self.length = length
        c_ = (c_ // self.length) * self.length  # 对齐到 length 的整数倍
        if c_ == 0:
            raise ValueError(f"SC: c2={c2} too small for length={length}, c_ became 0!")
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        # AMSP_VC的CBS
        self.cv2 = nn.Conv2d(c_ // length, c_ // length, 5, 1, 2, 1, bias=False)
        # VC，conv2d分组并行处理：每个分割块独立进行卷积,这个作为涡积
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()
        self.brk = BrokenBlock(c_, group=c_ // length)
        # SP，BrokenBlock作为重洗扰动

    def forward(self, x):
        x = self.cv1(x)
        # x经过一层CBS，输出通道数为SC输出的一半，输出图像大小不变（经过计算）
        # AMSP
        if self.training:
            y = self.brk(x)
            #  如果为训练，x经过BrokenBlock层，进行幅度调制AM和洗牌扰动SP
        else:
            y = x
            # 否则不经过BrokenBlock层
        y = y.chunk(self.length, 1)  # 沿通道维度分割成length份
        out = [self.cv2(_) for _ in y]  # 对每个分割块并行进行5x5卷积,卷积前后图像维度不变
        out = self.act(self.bn(torch.cat(out, 1)))  # 合并各个length，维度为c_*Hin*Wout

        result = torch.cat([x, out], 1)
        return result
        # 合并AM输出（c_*Hin*Hout）和out，得到c2*Hout*Wout

# FEM、RFB
class BasicConv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = c2
        self.conv = nn.Conv2d(c1, c2, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(c2, momentum=0.01) if bn else None
        # 如果用做Backbone，则eps默认1e-5,momentum=0.1,affine=True
        # self.bn = nn.BatchNorm2d(c2) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FEM(nn.Module):
    def __init__(self, c1, c2, stride=1, scale=0.1, map_reduce=8):
        # scale为缩放因子，在最后与shortcut进行加权融合
        # 这里shortcut为下采样
        # map_reduce应该不能修改，否则要同时修改各个分支
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = c2
        inter_planes = c1 // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(c1, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(c1, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(c1, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, c2, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(c1, c2, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        # 输出通道数为c2
        short = self.shortcut(x)
        # 输出通道数为c2
        out = out * self.scale + short
        out = self.relu(out)

        return out
# class FEM(nn.Module):
#     def __init__(self, c1, c2, stride=1, scale=0.1, map_reduce=8):
#         # scale为缩放因子，在最后与shortcut进行加权融合
#         # 这里shortcut为下采样
#         # map_reduce应该不能修改，否则要同时修改各个分支
#         # 官方代码
#         super(FEM, self).__init__()
#         self.scale = scale
#         self.out_channels = c2
#         inter_planes = c1 // map_reduce
#         self.branch0 = nn.Sequential(
#             # BasicConv(c1, 2 * inter_planes, kernel_size=1, stride=stride),
#             # BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
#             nn.Conv2d(c1, 2 * inter_planes, kernel_size=1, stride=stride, groups=c1),
#             nn.Conv2d(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, groups=2 * inter_planes),
#         )
#         self.branch1 = nn.Sequential(
#             # BasicConv(c1, inter_planes, kernel_size=1, stride=1),
#             # BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             # BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             # BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#             nn.Conv2d(c1, inter_planes, kernel_size=1, stride=1, groups=c1),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 5), stride=stride, padding=(0, 2)),
#             nn.SiLU(inplace=True),
#             nn.Conv2d((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(5, 1), stride=stride, padding=(2, 0)),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, groups=2 * inter_planes),
#
#
#         )
#         self.branch2 = nn.Sequential(
#             # BasicConv(c1, inter_planes, kernel_size=1, stride=1),
#             # BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             # BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             # BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#             nn.Conv2d(c1, inter_planes, kernel_size=1, stride=1, groups=c1),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(inter_planes, (inter_planes // 2) * 3, kernel_size=(5, 1), stride=stride, padding=(2, 0)),
#             nn.SiLU(inplace=True),
#             nn.Conv2d((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 5), stride=stride, padding=(0, 2)),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, groups=2 * inter_planes)
#
#         )
#
#         self.ConvLinear = BasicConv(6 * inter_planes, c2, k=1, s=1, act=False)
#         self.shortcut = Conv(c1, c2, k=1, s=stride, act=False)
#         self.relu = nn.ReLU(inplace=False)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#
#         out = torch.cat((x0, x1, x2), 1)
#         out = self.ConvLinear(out)
#         # 输出通道数为c2
#         short = self.shortcut(x)
#         # 输出通道数为c2
#         out = out * self.scale + short
#         out = self.relu(out)
#
#         return out

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        # 定义分支0，包含两个卷积层，第一个卷积层的卷积核大小为1，第二个卷积层的卷积核大小为3，步长为1，膨胀系数为visual
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
        )
        # 定义分支1，包含三个卷积层，第一个卷积层的卷积核大小为1，第二个卷积层的卷积核大小为3，步长为stride，第三个卷积层的卷积核大小为3，膨胀系数为visual+1
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2*inter_planes, kernel_size=(3, 3), stride=stride, padding=1),
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
        )
        # 定义分支2，包含四个卷积层，第一个卷积层的卷积核大小为1，第二个卷积层的卷积核大小为3，步长为1，第三个卷积层的卷积核大小为3，步长为stride，第四个卷积层的卷积核大小为3，膨胀系数为2*visual+1
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
        )
        # 定义一个线性卷积层
        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        # 定义一个shortcut路径的卷积层
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        # 定义一个ReLU激活层
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # 前向传播：分别通过三个分支
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # 将三个分支的输出沿通道维度拼接
        out = torch.cat((x0, x1, x2), 1)
        # 通过线性卷积层
        out = self.ConvLinear(out)
        # 通过shortcut路径的卷积层
        short = self.shortcut(x)
        # 进行加权求和
        out = out * self.scale + short
        # 通过ReLU激活层
        out = self.relu(out)
        return out

# 专利：CPFM:Custom-CBAM+FEM
class CPFM(nn.Module):
    def __init__(self, c1, c2=None, stride=1, scale=0.1, map_reduce=8, cbam_ratio=16):
        """
        CPFM模块 - 结合FEM和CBAM
        Args:
            c1: 输入通道数
            c2: 输出通道数（如果为None，则与输入相同）
            stride: 步长
            scale: FEM中的缩放因子
            map_reduce: FEM中的降维比率
            cbam_ratio: CBAM中的通道降维比率
        """
        super(CPFM, self).__init__()
        # 如果未指定输出通道，则与输入相同
        if c2 is None:
            c2 = c1
        self.fem = FEM(c1, c2, stride=stride, scale=scale, map_reduce=map_reduce)
        self.cbam = Custom_CBAM(c1, ratio=cbam_ratio)
        # 如果CBAM的输出通道数与FEM不同，添加1x1卷积调整通道数
        if c1 != c2:
            self.cbam_adjust = BasicConv(c1, c2, kernel_size=1, stride=stride, relu=False)
        else:
            self.cbam_adjust = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            y: 输出特征图 [B, C, H, W]，与输入保持相同尺寸（或根据stride调整）
        """
        # 并行处理两个分支
        y1 = self.fem(x)  # FEM分支
        y2 = self.cbam(x)  # CBAM分支
        return y1+self.cbam_adjust(y2)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvUtr(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, depth:int=1, kernel:int=9):
        # 这个depth作为parse_model的深度进行自适应缩放了，不进行另外设定
        # - [-1, 3, ConvUtr, [ch_out, 9]]
        super(ConvUtr, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                # 大核卷积 + GeLU + 残差(稳定训练)
                # 大卷积核全局建模
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in, kernel_size=(kernel, kernel), groups=ch_in, padding=(kernel // 2, kernel // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                # PointWise-Conv + GeLU + PointWise-Conv + GeLU + 残差(稳定训练)
                # 通道之间进行交互
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
            ) for i in range(depth)]
        )
        """
        depth为1简化后：
        self.block = nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=(kernel, kernel), groups=ch_in, padding=(kernel // 2, kernel // 2)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            )),
            Residual(nn.Sequential(
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ))
        )
        """
        self.up = nn.Sequential(
            # kernel_size应该设为1，将通道数变为设定输出通道数
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            # 这里能改为SiLU
        )

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        # 用于上采样的下采样，因此叫做self.up
        return x

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x) # b*g,c//g,h,1
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2) # # b*g,c//g,w,1
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)) # b*g,c//g,w+h,1
        x_h, x_w = torch.split(hw, [h, w], dim=2) # b*g,c//g,h,1;b*g,c//g,w,1
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        # 自动广播->b*g,c//g,h,w
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # b*g,1,c//g
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # b*g,1,c//g
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # b*g,1,hw->b*g,1,h,w
        return (group_x * weights.sigmoid()).reshape(b, c, h, w) # b*g,1,h,w * b*g,1,h,w->reshape->b,c,h,w


class AgentEMA(nn.Module):
    """严格保留EMA原结构，仅轻量化矩阵乘法环节"""

    def __init__(self, channels, factor=8, agent_num=49):
        super(AgentEMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.agent_num = agent_num
        c_g = channels // self.groups

        # 保留EMA所有原始组件
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(c_g, c_g)
        self.conv1x1 = nn.Conv2d(c_g, c_g, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(c_g, c_g, kernel_size=3, stride=1, padding=1)

        # 新增：代理池化层（仅此一处新增参数）
        agent_size = int(agent_num ** 0.5)
        self.agent_pool = nn.AdaptiveAvgPool2d((agent_size, agent_size))

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.groups
        c_g = c // g

        # 1. 分组（与原EMA完全一致）
        group_x = x.reshape(b * g, c_g, h, w)

        # 2. 生成x1和x2（与原EMA完全一致）
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        # 3. 生成代理令牌（仅此新增逻辑）
        agents_from_x1 = self.agent_pool(x1)  # 从x1生成代理，保留原始特征语义
        agents_flat = agents_from_x1.reshape(b * g, c_g, self.agent_num)

        # 4. 将原全连接矩阵乘法改造为Agent路由
        # 原逻辑: weights = matmul(x11, x12) + matmul(x21, x22)
        # 新逻辑: 通过代理实现通道到空间的映射

        # 分支1：x1 -> 代理 -> x2路径
        x11 = self.softmax(self.agp(x1).reshape(b * g, -1, 1).permute(0, 2, 1))  # [B*g, 1, C/g]
        # 代理聚合：代理从x2收集信息
        x12_proxy = torch.matmul(x11, agents_flat)  # [B*g, 1, n]  x11作为查询权重
        # 代理广播：将聚合后的代理信息映射回空间
        x12_flat = x2.reshape(b * g, c_g, h * w)  # [B*g, C/g, N]
        branch1 = torch.matmul(x12_proxy.transpose(-2, -1), x12_flat)  # [B*g, n, N]

        # 分支2：x2 -> 代理 -> x1路径（对称结构）
        x21 = self.softmax(self.agp(x2).reshape(b * g, -1, 1).permute(0, 2, 1))  # [B*g, 1, C/g]
        x21_proxy = torch.matmul(x21, agents_flat)  # [B*g, 1, n]
        x22_flat = x1.reshape(b * g, c_g, h * w)  # [B*g, C/g, N]
        branch2 = torch.matmul(x21_proxy.transpose(-2, -1), x22_flat)  # [B*g, n, N]

        # 合并双分支并映射回空间维度
        weights = (branch1 + branch2).sum(dim=1, keepdim=True)  # [B*g, 1, N]
        weights = weights.reshape(b * g, 1, h, w)

        # 5. 最终输出（与原EMA完全一致）
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

# class RGA(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True,
#                  cha_ratio=8, spa_ratio=8, down_ratio=8):
#         super(RGA, self).__init__()
#
#         self.in_channel = in_channel # 输入通道数
#         self.in_spatial = in_spatial # 输入空间大小（H*W）
#
#         self.use_spatial = use_spatial # 是否使用空间注意力
#         self.use_channel = use_channel # 是否使用通道注意力
#
#         self.inter_channel = max(in_channel // cha_ratio, 1) # 中间通道数
#         self.inter_spatial = max(in_spatial // spa_ratio, 1) # 中间空间大小
#
#         # 原始特征的嵌入函数
#         if self.use_spatial:
#             self.gx_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
#                           kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(self.inter_channel),
#                 nn.ReLU()
#             )
#         if self.use_channel:
#             self.gx_channel = nn.Sequential(
#                 nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
#                           kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(self.inter_spatial),
#                 nn.ReLU()
#             )
#
#         # 关系特征的嵌入函数
#         if self.use_spatial:
#             self.gg_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
#                           kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(self.inter_spatial),
#                 nn.ReLU()
#             )
#         if self.use_channel:
#             self.gg_channel = nn.Sequential(
#                 nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel,
#                           kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(self.inter_channel),
#                 nn.ReLU()
#             )
#
#         # 学习注意力权重的网络
#         if self.use_spatial:
#             num_channel_s = 1 + self.inter_spatial
#             self.W_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // down_ratio,
#                           kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(num_channel_s // down_ratio),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=num_channel_s // down_ratio, out_channels=1,
#                           kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(1)
#             )
#         if self.use_channel:
#             num_channel_c = max(1 + self.inter_channel, 1) # 确保至少为1
#             self.W_channel = nn.Sequential(
#                 nn.Conv2d(in_channels=num_channel_c, out_channels=max(num_channel_c // down_ratio, 1),
#                           kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(max(num_channel_c // down_ratio, 1)),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=max(num_channel_c // down_ratio, 1), out_channels=1,
#                           kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(1)
#             )
#
#         # 用于建模关系的嵌入函数
#         if self.use_spatial:
#             self.theta_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
#                           kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(self.inter_channel),
#                 nn.ReLU()
#             )
#             self.phi_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
#                           kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(self.inter_channel),
#                 nn.ReLU()
#             )
#         if self.use_channel:
#             self.theta_channel = nn.Sequential(
#                 nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
#                           kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(self.inter_spatial),
#                 nn.ReLU()
#             )
#             self.phi_channel = nn.Sequential(
#                 nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
#                           kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(self.inter_spatial),
#                 nn.ReLU()
#             )
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         if self.use_spatial:
#             # 空间注意力
#             # Q
#             theta_xs = self.theta_spatial(x)
#             theta_xs = theta_xs.view(b, self.inter_channel, -1)
#             theta_xs = theta_xs.permute(0, 2, 1)
#             # K
#             phi_xs = self.phi_spatial(x)
#             phi_xs = phi_xs.view(b, self.inter_channel, -1)
#             Gs = torch.matmul(theta_xs, phi_xs)
#
#             # 以光栅扫描顺序堆叠关系得到关系向量
#             # 第一部分 cat
#             Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)
#             Gs_out = Gs.view(b, h * w, h, w)
#             Gs_joint = torch.cat((Gs_in, Gs_out), 1)
#             Gs_joint = self.gg_spatial(Gs_joint)
#             # 第二部分 cat
#             g_xs = self.gx_spatial(x)
#             g_xs = torch.mean(g_xs, dim=1, keepdim=True)
#             ys = torch.cat((g_xs, Gs_joint), 1)
#
#             W_ys = self.W_spatial(ys)
#
#             if not self.use_channel:
#                 out = torch.sigmoid(W_ys.expand_as(x)) * x
#                 return out
#             else:
#                 x = torch.sigmoid(W_ys.expand_as(x)) * x
#
#         if self.use_channel:
#             # 通道注意力
#             xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
#             # Q
#             theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
#             # K
#             phi_xc = self.phi_channel(xc).squeeze(-1)
#             Gc = torch.matmul(theta_xc, phi_xc)
#
#             # 以光栅扫描顺序堆叠关系得到关系向量
#             # 第一部分 cat
#             Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
#             Gc_out = Gc.unsqueeze(-1)
#             Gc_joint = torch.cat((Gc_in, Gc_out), 1)
#             Gc_joint = self.gg_channel(Gc_joint)
#             # 第二部分 cat
#             g_xc = self.gx_channel(xc)
#             g_xc = torch.mean(g_xc, dim=1, keepdim=True)
#             yc = torch.cat((g_xc, Gc_joint), 1)
#
#             W_yc = self.W_channel(yc).transpose(1, 2)
#             out = torch.sigmoid(W_yc) * x
#             return out
class RGA(nn.Module):
    """Relation-Aware Global Attention with Dynamic Spatial Dimension"""

    def __init__(self, in_channel, in_spatial=None, use_spatial=True, use_channel=True,
                 cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(RGA, self).__init__()

        self.in_channel = in_channel
        self.use_spatial = use_spatial
        self.use_channel = use_channel
        self.cha_ratio = cha_ratio
        self.spa_ratio = spa_ratio
        self.down_ratio = down_ratio

        self.inter_channel = max(in_channel // cha_ratio, 1)

        # 缓存动态模块的字典
        self._modules_cache = {}

        # 空间注意力的固定层（Query/Key/Value网络）
        if self.use_spatial:
            self.theta_spatial = nn.Conv2d(in_channel, self.inter_channel, 1, bias=False)
            self.phi_spatial = nn.Conv2d(in_channel, self.inter_channel, 1, bias=False)
            self.gx_spatial = nn.Conv2d(in_channel, self.inter_channel, 1, bias=False)
            self.bn_theta_sp = nn.BatchNorm2d(self.inter_channel)
            self.bn_phi_sp = nn.BatchNorm2d(self.inter_channel)
            self.bn_gx_sp = nn.BatchNorm2d(self.inter_channel)

    def _get_spatial_mods(self, spatial_size, device):
        """获取空间注意力模块"""
        key = f"sp_{spatial_size}"

        if key not in self._modules_cache:
            inter_spatial = max(spatial_size // self.spa_ratio, 1)

            mods = nn.ModuleDict({
                'gg_spatial': nn.Sequential(
                    nn.Conv2d(spatial_size * 2, inter_spatial, 1, bias=False),
                    nn.BatchNorm2d(inter_spatial),
                    nn.ReLU(inplace=True)
                ),
                'W_spatial': nn.Sequential(
                    nn.Conv2d(1 + inter_spatial, (1 + inter_spatial) // self.down_ratio, 1, bias=False),
                    nn.BatchNorm2d((1 + inter_spatial) // self.down_ratio),
                    nn.ReLU(inplace=True),
                    nn.Conv2d((1 + inter_spatial) // self.down_ratio, 1, 1, bias=False),
                    nn.BatchNorm2d(1)
                )
            })

            self._modules_cache[key] = mods.to(device)

        return self._modules_cache[key]

    def _get_channel_mods(self, spatial_size, channel_size, device):
        """获取通道注意力模块"""
        key = f"ch_{spatial_size}_{channel_size}"

        if key not in self._modules_cache:
            inter_spatial = max(spatial_size // self.spa_ratio, 1)

            # ⚠️ 核心修正：所有卷积的in_channels都是spatial_size（空间尺寸）
            # 因为xc的形状是[b, spatial_size, channel_size, 1]
            mods = nn.ModuleDict({
                'theta_channel': nn.Conv2d(spatial_size, inter_spatial, 1, bias=False),
                'phi_channel': nn.Conv2d(spatial_size, inter_spatial, 1, bias=False),
                'gx_channel': nn.Conv2d(spatial_size, inter_spatial, 1, bias=False),
                'bn_theta_channel': nn.BatchNorm2d(inter_spatial),
                'bn_phi_channel': nn.BatchNorm2d(inter_spatial),
                'bn_gx_channel': nn.BatchNorm2d(inter_spatial),
                'gg_channel': nn.Sequential(
                    nn.Conv2d(channel_size * 2, inter_spatial, 1, bias=False),
                    nn.BatchNorm2d(inter_spatial),
                    nn.ReLU(inplace=True)
                ),
                'W_channel': nn.Sequential(
                    nn.Conv2d(1 + inter_spatial, (1 + inter_spatial) // self.down_ratio, 1, bias=False),
                    nn.BatchNorm2d((1 + inter_spatial) // self.down_ratio),
                    nn.ReLU(inplace=True),
                    nn.Conv2d((1 + inter_spatial) // self.down_ratio, 1, 1, bias=False),
                    nn.BatchNorm2d(1)
                )
            })

            self._modules_cache[key] = mods.to(device)

        return self._modules_cache[key]

    def forward(self, x):
        b, c, h, w = x.size()
        spatial_size = h * w

        # ==================== 空间注意力 ====================
        if self.use_spatial:
            sp_mods = self._get_spatial_mods(spatial_size, x.device)

            # Query & Key
            theta_xs = self.bn_theta_sp(self.theta_spatial(x)).view(b, self.inter_channel, -1)
            phi_xs = self.bn_phi_sp(self.phi_spatial(x)).view(b, self.inter_channel, -1)

            # 关系矩阵: [b, spatial_size, spatial_size]
            Gs = torch.matmul(theta_xs.permute(0, 2, 1), phi_xs)

            # 重塑并合并
            Gs_in = Gs.permute(0, 2, 1).view(b, spatial_size, h, w)
            Gs_out = Gs.view(b, spatial_size, h, w)
            Gs_joint = torch.cat([Gs_in, Gs_out], dim=1)
            Gs_joint = sp_mods['gg_spatial'](Gs_joint)

            # 全局特征
            g_xs = self.bn_gx_sp(self.gx_spatial(x))
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)  # [b, 1, h, w]
            ys = torch.cat([g_xs, Gs_joint], dim=1)
            W_ys = sp_mods['W_spatial'](ys)

            if not self.use_channel:
                return torch.sigmoid(W_ys.expand_as(x)) * x
            x = torch.sigmoid(W_ys.expand_as(x)) * x

        # ==================== 通道注意力 ====================
        if self.use_channel:
            ch_mods = self._get_channel_mods(spatial_size, c, x.device)

            # 重塑: [b, spatial_size, c, 1]
            xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)

            # Query & Key
            theta_xc = ch_mods['bn_theta_channel'](ch_mods['theta_channel'](xc)).squeeze(-1).permute(0, 2, 1)
            phi_xc = ch_mods['bn_phi_channel'](ch_mods['phi_channel'](xc)).squeeze(-1)

            # 关系矩阵: [b, c, c]
            Gc = torch.matmul(theta_xc, phi_xc)

            # 重塑并合并
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
            Gc_out = Gc.unsqueeze(-1)
            Gc_joint = torch.cat([Gc_in, Gc_out], dim=1)
            Gc_joint = ch_mods['gg_channel'](Gc_joint)

            # 全局特征
            g_xc = ch_mods['bn_gx_channel'](ch_mods['gx_channel'](xc))
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)
            yc = torch.cat([g_xc, Gc_joint], dim=1)
            W_yc = ch_mods['W_channel'](yc).transpose(1, 2)

            out = torch.sigmoid(W_yc) * x
            return out

        return x

class OREPAConv(nn.Module):
    """
    OREPA（Online RE-Parameterization Architecture）卷积
    训练阶段：主分支 + 1×1 逐点增强分支 + 3×3 逐深度增强分支
    推理阶段：将 BN、残差缩放、增强分支全部融合成主卷积的 weight & bias
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1):
        super().__init__()

        # ----------- 主分支 -----------
        self.stride = stride
        self.padding = padding
        # 主卷积：负责基础特征提取
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=padding, bias=False)
        # 主 BN：训练时参与融合，推理时消失
        self.bn = nn.BatchNorm2d(out_ch)

        # ----------- 增强分支 -----------
        # 3×3 逐深度卷积（groups = out_ch）
        self.dw_conv = nn.Conv2d(out_ch, out_ch, kernel_size,
                                 stride=1, padding=padding, groups=out_ch, bias=False)
        # 1×1 逐点卷积
        self.pw_conv = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        # 激活
        self.act = nn.ReLU(inplace=True)

        # ----------- 可学习残差缩放 -----------
        # 初始值 1，训练时随梯度更新
        self.alpha = nn.Parameter(torch.ones(1))   # 增强分支整体缩放
        self.beta = nn.Parameter(torch.ones(1))    # 预留，当前代码未使用
        self.gamma = nn.Parameter(torch.ones(1))   # 预留，当前代码未使用

    # --------------------------------------------------
    # 前向：训练 / 推理 分支自动切换
    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # ===== 训练模式：多分支 =====
            main_out = self.bn(self.conv(x))          # 主分支

            enhance = self.dw_conv(main_out)          # 3×3 DW
            enhance = self.bn(enhance)                # 共享主 BN（论文做法）
            enhance = self.pw_conv(enhance)           # 1×1 PW
            enhance = self.act(enhance)               # 激活

            return main_out + self.alpha * enhance    # 残差 + 可学习缩放
        else:
            # ===== 推理模式：融合为单卷积 =====
            fused_weight = self._fuse_conv()          # 融合后的卷积核
            fused_bias = self._fuse_bn()              # 融合后的偏置
            return F.conv2d(x, fused_weight, fused_bias,
                            stride=self.stride, padding=self.padding)

    # --------------------------------------------------
    # 融合增强分支 → 主卷积核
    # --------------------------------------------------
    def _fuse_conv(self) -> torch.Tensor:
        """
        将 3×3 DW + 1×1 PW 融合进主卷积，返回形状 [out_ch, in_ch, k, k]
        """
        # 1. 给 DW 卷积核做 padding，使其形状与主卷积核一致
        dw_weight = self._pad_kernel(self.dw_conv.weight)   # [out_ch, 1, k, k]

        # 2. PW 卷积核形状 [out_ch, out_ch, 1, 1]
        pw_weight = self.pw_conv.weight

        # 3. 计算增强分支等效权重：PW ⊗ DW
        #    einsum 含义：对每一组 (o,i) 做 2D 卷积，输出 [out_ch, 1, k, k]
        enhance_weight = torch.einsum('oi,ijkl->ojkl',
                                      pw_weight.squeeze(-1).squeeze(-1),
                                      dw_weight)            # [out_ch, 1, k, k]

        # 4. 叠加到主卷积，并乘以可学习缩放
        fused_weight = self.conv.weight + self.alpha * enhance_weight
        return fused_weight                                   # [out_ch, in_ch, k, k]

    # --------------------------------------------------
    # 融合 BN → 主卷积偏置
    # --------------------------------------------------
    def _fuse_bn(self) -> torch.Tensor:
        """
        将 BN 的 γ/β 融合进卷积偏置，返回形状 [out_ch]
        """
        # 读取 BN 运行统计量
        mean = self.bn.running_mean   # [out_ch]
        var = self.bn.running_var     # [out_ch]
        gamma = self.bn.weight        # [out_ch]
        beta = self.bn.bias           # [out_ch]
        eps = 1e-5

        # 计算融合后的 scale & bias
        std = torch.sqrt(var + eps)
        scale = gamma / std                                         # [out_ch]

        # 若主卷积本身有 bias 则先取出，否则为 0
        conv_bias = self.conv.bias if self.conv.bias is not None else torch.zeros_like(mean)
        fused_bias = scale * (conv_bias - mean) + beta             # [out_ch]
        return fused_bias

    # --------------------------------------------------
    # 辅助：给 DW 卷积核做 zero-padding，使其与主卷积核尺寸相同
    # --------------------------------------------------
    def _pad_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        """
        kernel: [out_ch, 1, k, k]
        return: [out_ch, 1, K, K]  （K 为主卷积核大小）
        """
        k_main = self.conv.kernel_size[0]          # 主卷积核大小
        k_enh = kernel.shape[-1]                   # 增强分支核大小
        pad = (k_main - k_enh) // 2
        if pad > 0:
            kernel = F.pad(kernel, [pad, pad, pad, pad], mode='constant', value=0.)
        return kernel

class CoordAttMeanMax(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAttMeanMax, self).__init__()
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1_mean = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1_mean = nn.BatchNorm2d(mip)
        self.conv2_mean = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.conv1_max = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1_max = nn.BatchNorm2d(mip)
        self.conv2_max = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Mean pooling branch
        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)
        y_mean = self.conv1_mean(y_mean)
        y_mean = self.bn1_mean(y_mean)
        y_mean = self.relu(y_mean)
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)

        # Max pooling branch
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)
        y_max = self.conv1_max(y_max)
        y_max = self.bn1_max(y_max)
        y_max = self.relu(y_max)
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)

        # Apply attention
        x_h_mean = self.conv2_mean(x_h_mean).sigmoid()
        x_w_mean = self.conv2_mean(x_w_mean).sigmoid()
        x_h_max = self.conv2_max(x_h_max).sigmoid()
        x_w_max = self.conv2_max(x_w_max).sigmoid()

        # Expand to original shape
        x_h_mean = x_h_mean.expand(-1, -1, h, w)
        x_w_mean = x_w_mean.expand(-1, -1, h, w)
        x_h_max = x_h_max.expand(-1, -1, h, w)
        x_w_max = x_w_max.expand(-1, -1, h, w)

        # Combine outputs
        attention_mean = identity * x_w_mean * x_h_mean
        attention_max = identity * x_w_max * x_h_max

        # Sum the attention outputs
        return attention_mean + attention_max

class DCAFE(nn.Module):
    def __init__(self, in_channels):
        super(DCAFE, self).__init__()
        self.coord_att = CoordAttMeanMax(in_channels, oup=1260)

    def forward(self, x):
        return self.coord_att(x)

class DSConv(nn.Module):
    """The Basic Depthwise Separable Convolution."""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(
            c_in, c_in, kernel_size=k, stride=s,
            padding=p, dilation=d, groups=c_in, bias=bias
        )
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))