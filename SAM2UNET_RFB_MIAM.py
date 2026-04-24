import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2

NUM_CLASSES = 8


# ============================================================
# Basic Blocks
# ============================================================

class DoubleConv(nn.Module):
    """Two sequential Conv-BN-ReLU blocks."""

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
#  Adapter 
# ============================================================    
class Adapter(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features

        self.prompt = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim)
        )

    def forward(self, x):
        return self.block(x + self.prompt(x))


# ============================================================
# Multi-scale CNN Stem 
# ============================================================

class Stem(nn.Module):

    def __init__(self, out_ch=144):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(x)


# ============================================================
# ResBlock (used for deeper CNN stages)
# ============================================================

class ResBlock(nn.Module):
    """Standard pre-activation residual block."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.down = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else None
        )

    def forward(self, x):
        identity = self.down(x) if self.down else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


# ============================================================
# Stacked ResBlocks per stage (2* depth per stage)
# ============================================================

class ResStage(nn.Module):
    """
    Two stacked ResBlocks per encoder stage.
    The first block handles the stride / channel change;
    the second adds representational depth at the same resolution.
    """

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.stage = nn.Sequential(
            ResBlock(in_ch, out_ch, stride=stride),
            ResBlock(out_ch, out_ch, stride=1),
        )

    def forward(self, x):
        return self.stage(x)


# ============================================================
# RFB Module (multi-scale dilated receptive field)
# ============================================================

class BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k, p=0, d=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p, dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RFB_modified(nn.Module):
    """
    Receptive Field Block with four parallel dilated branches.
    Captures objects at scales 1*, 3*, 5*, 7*  well suited for
    UAV imagery where flood regions vary enormously in apparent size.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch0 = BasicConv2d(in_ch, out_ch, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_ch, out_ch, 1),
            BasicConv2d(out_ch, out_ch, 3, p=3, d=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_ch, out_ch, 1),
            BasicConv2d(out_ch, out_ch, 3, p=5, d=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_ch, out_ch, 1),
            BasicConv2d(out_ch, out_ch, 3, p=7, d=7),
        )
        self.conv_cat = BasicConv2d(4 * out_ch, out_ch, 3, p=1)
        self.conv_res = BasicConv2d(in_ch, out_ch, 1)

    def forward(self, x):
        feats = torch.cat([
            self.branch0(x),
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
        ], dim=1)
        return self.conv_cat(feats) + self.conv_res(x)


# ============================================================
# MIAM  Multi-modal Interaction Attention Module
# ============================================================

class MIAM(nn.Module):
    """
    Fuses one SAM2 stage feature with the matching CNN stage feature.
    1. Concatenate  project back to `ch` channels (fuse).
    2. Channel attention re-weights the fused map.

    The bottleneck uses max(ch // 8, 16) to prevent collapsing to
    too few neurons at the smallest stage (ch=144  18 is fine; this
    guards against future configs with smaller ch values).
    """

    def __init__(self, ch):
        super().__init__()
        bottleneck = max(ch // 8, 16)

        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, bottleneck, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, sam_feat, cnn_feat):
        # Align spatial sizes in case of minor rounding differences
        if sam_feat.shape[-2:] != cnn_feat.shape[-2:]:
            sam_feat = F.interpolate(
                sam_feat, size=cnn_feat.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        x   = self.fuse(torch.cat([sam_feat, cnn_feat], dim=1))
        att = self.ca(x)
        return x * att


# ============================================================
# Weight Fusion Block (decoder skip connection)
# ============================================================

class WF(nn.Module):
    """
    Weighted skip-connection fusion used in the decoder.
    Both the upsampled deep feature and the skip feature are
    independently projected to `out_ch` before the learnable
    weighted sum, avoiding potential channel mismatches.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj_skip = nn.Conv2d(in_ch, out_ch, 1)   # project skip feature
        self.proj_up   = nn.Conv2d(in_ch, out_ch, 1)   # project upsampled feature
        self.weights   = nn.Parameter(torch.ones(2))
        self.eps       = 1e-8
        self.post      = DoubleConv(out_ch, out_ch)

    def forward(self, x, res):
        x = F.interpolate(x, size=res.shape[-2:], mode='bilinear', align_corners=False)

        w = torch.relu(self.weights)
        w = w / (w.sum() + self.eps)

        fused = w[0] * self.proj_skip(res) + w[1] * self.proj_up(x)
        return self.post(fused)


# ============================================================
# Feature Refinement Head
# ============================================================

class FeatureRefinementHead(nn.Module):
    """
    Applies both spatial and channel attention to refine the
    pre-final decoder feature map.

    Spatial gate: 7*7 conv  -1 channel-  sigmoid broadcast.
      Learns *where* to look.
    Channel gate: global avg-pool - SE-style squeeze-excite.
      Learns *what* to look at.
    """

    def __init__(self, ch):
        super().__init__()
        self.conv = DoubleConv(ch, ch)

        # Spatial attention - output is a single-channel mask
        self.pa = nn.Sequential(
            nn.Conv2d(ch, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        # Channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(ch // 16, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch // 16, 8), ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x   = self.conv(x)
        x   = self.pa(x) * x   # spatial gate broadcasts over channels
        x   = self.ca(x) * x   # channel gate broadcasts over spatial dims
        return x


# ============================================================
# Separate Auxiliary Heads (independent supervision)
# ============================================================

class AuxHead(nn.Module):
    """
    Independent auxiliary segmentation head for deep supervision.
    Two instances are created (one per decoder stage) so that
    each provides truly independent gradient signals.
    """

    def __init__(self, ch, num_classes):
        super().__init__()
        self.conv = DoubleConv(ch, ch)
        self.drop = nn.Dropout2d(0.1)
        self.out  = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x, size):
        x = self.conv(x)
        x = self.drop(x)
        x = self.out(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


# ============================================================
# SAM2 Intermediate Feature Extractor
# ============================================================

class SAM2FeatureExtractor(nn.Module):
    """
    Wraps the SAM2 Hiera trunk and extracts intermediate features
    from each of the 4 hierarchical stages via forward hooks.

    SAM2's Hiera trunk operates in BHWC layout; we permute to BCHW
    before returning so the rest of the network is layout-agnostic.

    Stage output strides (for a 1024*1024 input):
        s1 - 1/4   (256*256),  C=144
        s2 - 1/8   (128*128),  C=288
        s3 - 1/16  ( 64*64),   C=576
        s4 - 1/32  ( 32*32),   C=1152
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self._feats  = {}
        self._hooks  = []
        self._register_hooks()

    def _register_hooks(self):
        # Hiera groups blocks into q_stride stages; the stage boundaries
        # are stored in encoder.stage_ends (list of block indices).
        stage_ends = self.encoder.stage_ends   # e.g. [1, 3, 26, 31] for Hiera-L

        def make_hook(stage_idx):
            def hook(module, _inp, output):
                # output is (B, H, W, C)  permute to BCHW
                self._feats[stage_idx] = output.permute(0, 3, 1, 2).contiguous()
            return hook

        for i, blk_idx in enumerate(stage_ends):
            h = self.encoder.blocks[blk_idx].register_forward_hook(make_hook(i))
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, x):
        self._feats.clear()
        self.encoder(x)   # triggers all hooks
        s1 = self._feats[0]
        s2 = self._feats[1]
        s3 = self._feats[2]
        s4 = self._feats[3]
        return s1, s2, s3, s4


# ============================================================
# MAIN MODEL
# ============================================================

class SAM2UNet(nn.Module):


    def __init__(self, checkpoint, num_classes=NUM_CLASSES):
        super().__init__()

        # SAM2 Encoder 
        sam     = build_sam2(
            config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
            ckpt_path=checkpoint,
            device="cuda",
            apply_postprocessing=False,
        )
        encoder = sam.image_encoder.trunk

        # Freeze SAM2 backbone weights
        for p in encoder.parameters():
            p.requires_grad = False

        # Inject adapters (trainable, ~tiny param count)
        encoder.blocks = nn.ModuleList(
            [Adapter(b) for b in encoder.blocks]
        )

        # FIX 8: hook-based intermediate feature extractor
        self.sam_extractor = SAM2FeatureExtractor(encoder)

        #  CNN Encoder
        # FIX 2: gradual stem instead of stride-4 single conv
        self.stem = Stem(out_ch=144)            # → 1/4,  C=144

        # FIX 3: 2 ResBlocks per stage
        self.res2 = ResStage(144, 288,  stride=2)   # → 1/8,  C=288
        self.res3 = ResStage(288, 576,  stride=2)   # → 1/16, C=576
        self.res4 = ResStage(576, 1152, stride=2)   # → 1/32, C=1152

        # MIAM Fusion
        self.miam1 = MIAM(144)
        self.miam2 = MIAM(288)
        self.miam3 = MIAM(576)
        self.miam4 = MIAM(1152)

        # RFB Blocks
        self.rfb1 = RFB_modified(144,  64)
        self.rfb2 = RFB_modified(288,  64)
        self.rfb3 = RFB_modified(576,  64)
        self.rfb4 = RFB_modified(1152, 64)

        # Decoder
        # FIX 5: WF now projects both branches before weighted sum
        self.wf1 = WF(64, 64)
        self.wf2 = WF(64, 64)
        self.wf3 = WF(64, 64)

        # Feature Refinement
        # FIX 6: true CBAM-style spatial gate
        self.frh = FeatureRefinementHead(64)

        # Heads
        # FIX 7: two separate aux heads for independent deep supervision
        self.aux_head1 = AuxHead(64, num_classes)
        self.aux_head2 = AuxHead(64, num_classes)

        # FIX 10: dropout before final prediction head
        self.drop = nn.Dropout2d(0.1)
        self.head = nn.Conv2d(64, num_classes, 1)

    # Forward

    def forward(self, x):
        h, w = x.shape[-2:]

        # SAM2 Encoder (hook-based intermediate features)
        s1, s2, s3, s4 = self.sam_extractor(x)
        # s1: (B, 144, H/4,  W/4)
        # s2: (B, 288, H/8,  W/8)
        # s3: (B, 576, H/16, W/16)
        # s4: (B,1152, H/32, W/32)

        # CNN Encoder
        c1 = self.stem(x)    # (B, 144, H/4,  W/4)
        c2 = self.res2(c1)   # (B, 288, H/8,  W/8)
        c3 = self.res3(c2)   # (B, 576, H/16, W/16)
        c4 = self.res4(c3)   # (B,1152, H/32, W/32)

        # MIAM Fusion
        x1 = self.miam1(s1, c1)
        x2 = self.miam2(s2, c2)
        x3 = self.miam3(s3, c3)
        x4 = self.miam4(s4, c4)

        # RFB Processing
        x1 = self.rfb1(x1)   # (B, 64, H/4,  W/4)
        x2 = self.rfb2(x2)   # (B, 64, H/8,  W/8)
        x3 = self.rfb3(x3)   # (B, 64, H/16, W/16)
        x4 = self.rfb4(x4)   # (B, 64, H/32, W/32)

        # Top-down Decoder
        d = self.wf1(x4, x3)                    # H/16
        aux1 = self.aux_head1(d, (h, w))        # auxiliary prediction 1

        d = self.wf2(d, x2)                     # H/8
        aux2 = self.aux_head2(d, (h, w))        # auxiliary prediction 2

        d = self.wf3(d, x1)                     # H/4

        # Refinement + Final Prediction
        d   = self.frh(d)
        d   = self.drop(d)
        out = self.head(d)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return out, aux1, aux2