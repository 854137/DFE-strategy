# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics neural network modules.

This module provides access to various neural network components used in Ultralytics models, including convolution
blocks, attention mechanisms, transformer components, and detection/segmentation heads.

Examples:
    Visualize a module with Netron
    >>> from ultralytics.nn.modules import Conv
    >>> import torch
    >>> import subprocess
    >>> x = torch.ones(1, 128, 40, 40)
    >>> m = Conv(128, 128)
    >>> f = f"{m._get_name()}.onnx"
    >>> torch.onnx.export(m, x, f)
    >>> subprocess.run(f"onnxslim {f} {f} && open {f}", shell=True, check=True)  # pip install onnxslim
"""

from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    RTMBlock,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    MaxSigmoidAttnBlock,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    ResNetLayer_1834,
    SCDown,
    TorchVision,
    pC2f,
    LCPP,
    LCVIT,
    PST,
    HyperACE,
    FullPAD_Tunnel,
    DownsampleConv,
    DSConv,
    DSC3k2,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
    Custom_CBAM,
    SEAttention,
    SC,
    FEM,
    BasicRFB,
    CPFM,
    ConvUtr,
    BasicConv,
    EMA,
    RGA,
)

from .head import (
    OBB,
    Classify,
    Detect,
    LRPCHead,
    Pose,
    RTDETRDecoder,
    Segment,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)

from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
    IntermediateViT,
)
# 自定义模块文件
from .fads import (
    PWConv,
    CUCSP,
    SACSP,
    FAD_C2f,
    FAD_C2fv3,
    EMACSP,
    EMAC2f,
    OREPAConvMSNBv1,
    OREPAConvMSNBv2,
    OREPAConv,

    ENBC2f,
)
from .orepa import OREPA
from .dcnv4 import (
    DCNv4Block,
    LightweightDCNv4Block
)
from .afpn import (
    Detect_AFPN4,
    AFPN,
)
from .dyyolo import (
    StemLayer,
    DownsampleLayer,
    DyReLU,
    MultiScaleAttentionFusion_SA,
    MultiScaleAttentionFusion_TB,
    MultiScaleAttentionFusion_SpA,
    Detect_MultiScaleFusion,
)
from .retinanet import (
    RetinaNetDetect,
    PyramidFeatures,
)
from .longcustom import (
    SDP_Local,
    LGLBlock,
    LGLBottleneck,
)
from .dafpn import (
    DetectDA,
    DAFPN,
    UDM,
    UDM_dct,
)

from .hsfpn import (
    HFP,
    GatedHFP,
    SDP,
    HSFPN,
    Detect_HSFPN,
)

from .mobilenetv4 import (
    MobileNetV4ConvSmall,
    MobileNetV4ConvMedium,
    MobileNetV4ConvLarge,

    MobileNetV4HybridMedium,
    MobileNetV4HybridLarge,
)

from .patconv import (
    PAT_ch_Block,
    PAT_sp_Block,
    PAT_sf_Block,
    PartialNetv1,
    PartialNetv2,
)



__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3k2",
    "RTMBlock",
    "SCDown",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "ResNetLayer_1834",
    "OBB",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
    "LRPCHead",
    "ImagePoolingAttn",
    "MaxSigmoidAttnBlock",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "AConv",
    "ELAN1",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "TorchVision",
    "Index",
    "A2C2f",
    "Custom_CBAM",
    "SEAttention",
    'SC',
    'FEM',
    'BasicRFB',
    'CPFM',
    'EMA',
    'RGA',

    'PWConv',
    'CUCSP',
    'SACSP',
    'pC2f',
    'FAD_C2f',
    'FAD_C2fv3',
    'EMACSP',
    'EMAC2f',
    'ENBC2f',
    'OREPAConvMSNBv1',
    'OREPAConvMSNBv2',
    'OREPAConv',
    'OREPA',


    'LCPP',
    'LCVIT',
    # 'GroupDCNv3Layer',
    # 'InternImageBlock',
    'ConvUtr',
    'BasicConv',
    'LGLBottleneck',
    'LGLBlock',

    'StemLayer',
    'DownsampleLayer',
    'DCNv4Block',
    'DyReLU',
    'MultiScaleAttentionFusion_SA',
    'MultiScaleAttentionFusion_TB',
    'MultiScaleAttentionFusion_SpA',
    'Detect_MultiScaleFusion',

    'IntermediateViT',
    'SDP_Local',
    'HFP',
    'GatedHFP',
    'SDP',

    'RetinaNetDetect',
    'DetectDA',

    'MobileNetV4ConvSmall',
    'MobileNetV4ConvMedium',
    'MobileNetV4ConvLarge',
    'MobileNetV4HybridMedium',
    'MobileNetV4HybridLarge',
)
