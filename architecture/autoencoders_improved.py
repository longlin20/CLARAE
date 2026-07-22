import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from architecture.autoencoders import Encoder_CLARAE_SC
except ImportError:
    from autoencoders import Encoder_CLARAE_SC

# =============================================================================
# Section 1 — Building Blocks
# =============================================================================

class MultiScaleConvBlock(nn.Module):
    """
    Three parallel convolutions with different kernel sizes, merged via 1x1 projection.

    Default kernels at fs=500 Hz:
        kernel=3  (~6 ms)  : fine details, signal edges
        kernel=9  (~18 ms) : local EGM morphology
        kernel=21 (~42 ms) : full activation context

    Noise is broadband -> multi-scale captures signal structure at specific scales
    while noise contributions average out across scales in the 1x1 projection.
    InstanceNorm normalizes each sample independently.
    """
    def __init__(self, in_ch, out_ch, kernels=(3, 9, 21)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, k, padding='same') for k in kernels
        ])
        self.proj = nn.Conv1d(out_ch * len(kernels), out_ch, kernel_size=1)
        self.norm = nn.InstanceNorm1d(out_ch, affine=True)

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return F.leaky_relu(self.norm(self.proj(out)), negative_slope=0.2)


class GLUConvBlock(nn.Module):
    """
    Convolutional block with Gated Linear Unit (GLU) activation.

    Produces 2*out_ch channels, splits into (signal, gate):
        output = signal * sigmoid(gate)

    The gate learns to suppress channels activated by noise.
    InstanceNorm applied after gating.
    """
    def __init__(self, in_ch, out_ch, kernel_size=7, dropout=0.0):
        super().__init__()
        self.conv    = nn.Conv1d(in_ch, out_ch * 2, kernel_size, padding='same')
        self.norm    = nn.InstanceNorm1d(out_ch, affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x, gate = x.chunk(2, dim=1)
        return self.norm(self.dropout(x) * torch.sigmoid(gate))


# =============================================================================
# Section 2 — Encoders
# =============================================================================

class Encoder_CLARAE_SCM(nn.Module):
    """
    Multi-scale encoder with MaxPool1d.

    Architecture (fi=filters_initial):
        Layer 1: MultiScale(1    → fi)     + MaxPool -> skip1
        Layer 2: MultiScale(fi   → fi//2)  + MaxPool -> skip2
        Layer 3: MultiScale(fi//2 → fi//4) + MaxPool -> skip3
        Layer 4: MultiScale(fi//4 → fi//8) + MaxPool -> skip4
        FC: flatten → dense_dim → latent
    """
    def __init__(self, input_channels, input_length, latent_dim,
                 filters_initial, dropout_rate=0.2, dense_dim=256,
                 ms_kernels=(3, 9, 21), latent_act=None):
        super().__init__()
        self.filters_initial = filters_initial
        self.latent_act = latent_act if latent_act is not None else torch.tanh
        fi = filters_initial

        self.block1 = MultiScaleConvBlock(input_channels, fi,      ms_kernels)
        self.pool1  = nn.MaxPool1d(2)

        self.block2 = MultiScaleConvBlock(fi,      fi // 2, ms_kernels)
        self.pool2  = nn.MaxPool1d(2)

        self.block3 = MultiScaleConvBlock(fi // 2, fi // 4, ms_kernels)
        self.pool3  = nn.MaxPool1d(2)

        self.block4 = MultiScaleConvBlock(fi // 4, fi // 8, ms_kernels)
        self.pool4  = nn.MaxPool1d(2)

        flatten_size = (input_length // 16) * (fi // 8)
        self.fc1     = nn.Linear(flatten_size, dense_dim)
        self.fc2     = nn.Linear(dense_dim, latent_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        skips = []
        x = self.block1(x); skips.append(x); x = self.pool1(x)
        x = self.block2(x); skips.append(x); x = self.pool2(x)
        x = self.block3(x); skips.append(x); x = self.pool3(x)
        x = self.block4(x); skips.append(x); x = self.pool4(x)

        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.dropout(self.fc1(x)), 0.3)
        latent = self.latent_act(self.fc2(x))
        return latent, skips


class Encoder_CLARAE_SCM_AP(Encoder_CLARAE_SCM):
    """
    Encoder_CLARAE_SCM with AvgPool1d instead of MaxPool1d.

    AvgPool reduces Gaussian noise variance by 1/2 at each pooling step
    (x1/16 total after 4 layers), whereas MaxPool amplifies noise peaks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool1 = nn.AvgPool1d(2)
        self.pool2 = nn.AvgPool1d(2)
        self.pool3 = nn.AvgPool1d(2)
        self.pool4 = nn.AvgPool1d(2)


class Encoder_CLARAE_AP(nn.Module):
    """
    Single-scale AvgPool encoder with skip connections.

    Plain Conv1d + InstanceNorm + LeakyReLU at each layer (no multi-scale branching).
    Returns (latent, skips) like Encoder_CLARAE_SCM_AP.
    Ablation: removes the SCM block from CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST.

    Architecture (fi=filters_initial, kernel_size=7):
        Layer 1: Conv(1 → fi)      + IN + LeakyReLU + AvgPool → skip1
        Layer 2: Conv(fi → fi//2)  + IN + LeakyReLU + AvgPool → skip2
        Layer 3: Conv(fi//2→fi//4) + IN + LeakyReLU + AvgPool → skip3
        Layer 4: Conv(fi//4→fi//8) + IN + LeakyReLU + AvgPool → skip4
        FC: flatten → dense_dim → latent_act(latent)
    """
    def __init__(self, input_channels, input_length, latent_dim,
                 filters_initial, dropout_rate=0.2, dense_dim=256,
                 kernel_size=7, latent_act=None):
        super().__init__()
        self.latent_act = latent_act if latent_act is not None else torch.tanh
        fi = filters_initial

        self.block1 = nn.Sequential(nn.Conv1d(input_channels, fi,    kernel_size, padding='same'),
                                    nn.InstanceNorm1d(fi,    affine=True))
        self.block2 = nn.Sequential(nn.Conv1d(fi,    fi // 2, kernel_size, padding='same'),
                                    nn.InstanceNorm1d(fi // 2, affine=True))
        self.block3 = nn.Sequential(nn.Conv1d(fi // 2, fi // 4, kernel_size, padding='same'),
                                    nn.InstanceNorm1d(fi // 4, affine=True))
        self.block4 = nn.Sequential(nn.Conv1d(fi // 4, fi // 8, kernel_size, padding='same'),
                                    nn.InstanceNorm1d(fi // 8, affine=True))
        self.pool = nn.AvgPool1d(2)

        flatten_size = (input_length // 16) * (fi // 8)
        self.fc1     = nn.Linear(flatten_size, dense_dim)
        self.fc2     = nn.Linear(dense_dim, latent_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        skips = []
        for block in [self.block1, self.block2, self.block3, self.block4]:
            x = F.leaky_relu(block(x), 0.2)
            skips.append(x)
            x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.dropout(self.fc1(x)), 0.3)
        latent = self.latent_act(self.fc2(x))
        return latent, skips


# =============================================================================
# Section 3 — Decoders
# =============================================================================

class Decoder_CLARAE_SCM_GLU(nn.Module):
    """
    GLU decoder with additive skip connections (optionally InstanceNorm-normalized).

    Architecture (fi=filters_initial):
        Layer 1: upsample → x + [IN](skip4) → GLU(fi//8 → fi//4)
        Layer 2: upsample → x + [IN](skip3) → GLU(fi//4 → fi//2)
        Layer 3: upsample → GLU(fi//2 → fi)       [no skip]
        Layer 4: upsample → Conv(fi → 1) + out_act [no skip]

    use_in=True  (default): normalises skip features with InstanceNorm before adding.
    use_in=False (sin IN):  adds skip directly, like classic U-Net.
    """
    def __init__(self, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256, glu_kernel_size=7, out_act=None,
                 use_in=True, skip_dropout=0.0):
        super().__init__()
        self.filters_initial = filters_initial
        self.input_length    = input_length
        self.out_act         = out_act if out_act is not None else torch.tanh
        self.use_in          = use_in
        fi  = filters_initial

        self.fc1 = nn.Linear(latent_dim, dense_dim)
        self.fc2 = nn.Linear(dense_dim, (input_length // 16) * (fi // 8))

        self.glu1 = GLUConvBlock(fi // 8, fi // 4, glu_kernel_size, dropout=dropout_rate)
        self.glu2 = GLUConvBlock(fi // 4, fi // 2, glu_kernel_size, dropout=dropout_rate)
        self.glu3 = GLUConvBlock(fi // 2, fi,      glu_kernel_size, dropout=dropout_rate)

        self.conv_out = nn.Conv1d(fi, 1, kernel_size=7, padding='same')
        self.dropout  = nn.Dropout(dropout_rate)

        self.norm_s4  = nn.InstanceNorm1d(fi // 8, affine=True) if use_in else nn.Identity()
        self.norm_s3  = nn.InstanceNorm1d(fi // 4, affine=True) if use_in else nn.Identity()
        self.skip_drop = nn.Dropout(skip_dropout)

    def forward(self, latent, skips):
        nch = self.filters_initial // 8
        x = F.leaky_relu(self.dropout(self.fc1(latent)), 0.3)
        x = F.leaky_relu(self.fc2(x), 0.3)
        x = x.view(x.shape[0], nch, x.shape[1] // nch)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        s4 = skips[3]
        if x.shape[-1] != s4.shape[-1]:
            s4 = F.interpolate(s4, size=x.shape[-1], mode='linear', align_corners=True)
        x = x + self.skip_drop(self.norm_s4(s4))
        x = self.glu1(x)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        s3 = skips[2]
        if x.shape[-1] != s3.shape[-1]:
            s3 = F.interpolate(s3, size=x.shape[-1], mode='linear', align_corners=True)
        x = x + self.skip_drop(self.norm_s3(s3))
        x = self.glu2(x)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        x = self.glu3(x)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if x.shape[-1] != self.input_length:
            x = F.interpolate(x, self.input_length, mode='linear', align_corners=True)
        return self.out_act(self.conv_out(x))


class Decoder_CLARAE_SCM_GLU_SKN(nn.Module):
    """
    GLU decoder with plain additive skip connections at configurable encoder layers.

    No learned gate — skips are added directly (optionally IN-normalized).
    skip_layers: set of encoder layer indices (1=shallowest, 4=deepest).
    use_in: apply InstanceNorm to each active skip before adding (default False).

    Architecture (fi=filters_initial, default skip_layers=(4,)):
        Layer 1: upsample → [+ [IN](skip4) if 4∈skip_layers] → GLU(fi//8 → fi//4)
        Layer 2: upsample → [+ [IN](skip3) if 3∈skip_layers] → GLU(fi//4 → fi//2)
        Layer 3: upsample → [+ [IN](skip2) if 2∈skip_layers] → GLU(fi//2 → fi)
        Layer 4: upsample → [+ [IN](skip1) if 1∈skip_layers] → Conv(fi → 1) + out_act
    """
    def __init__(self, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256, glu_kernel_size=7,
                 skip_dropout=0.0, out_act=None, skip_layers=(4,), use_in=False):
        super().__init__()
        self.filters_initial = filters_initial
        self.input_length    = input_length
        self.out_act         = out_act if out_act is not None else torch.tanh
        self.skip_layers     = set(skip_layers)
        fi = filters_initial

        self.fc1 = nn.Linear(latent_dim, dense_dim)
        self.fc2 = nn.Linear(dense_dim, (input_length // 16) * (fi // 8))

        self.glu1 = GLUConvBlock(fi // 8, fi // 4, glu_kernel_size, dropout=dropout_rate)
        self.glu2 = GLUConvBlock(fi // 4, fi // 2, glu_kernel_size, dropout=dropout_rate)
        self.glu3 = GLUConvBlock(fi // 2, fi,      glu_kernel_size, dropout=dropout_rate)

        self.conv_out  = nn.Conv1d(fi, 1, kernel_size=7, padding='same')
        self.dropout   = nn.Dropout(dropout_rate)
        self.skip_drop = nn.Dropout(skip_dropout)

        _ch = {4: fi // 8, 3: fi // 4, 2: fi // 2, 1: fi}
        self.norms = nn.ModuleDict({
            f's{l}': (nn.InstanceNorm1d(_ch[l], affine=True) if use_in else nn.Identity())
            for l in self.skip_layers
        })

    def _add_skip(self, x, s, key):
        if s.shape[-1] != x.shape[-1]:
            s = F.interpolate(s, size=x.shape[-1], mode='linear', align_corners=True)
        return x + self.skip_drop(self.norms[key](s))

    def forward(self, latent, skips):
        nch = self.filters_initial // 8
        x = F.leaky_relu(self.dropout(self.fc1(latent)), 0.3)
        x = F.leaky_relu(self.fc2(x), 0.3)
        x = x.view(x.shape[0], nch, x.shape[1] // nch)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if 4 in self.skip_layers:
            x = self._add_skip(x, skips[3], 's4')
        x = self.glu1(x)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if 3 in self.skip_layers:
            x = self._add_skip(x, skips[2], 's3')
        x = self.glu2(x)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if 2 in self.skip_layers:
            x = self._add_skip(x, skips[1], 's2')
        x = self.glu3(x)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if x.shape[-1] != self.input_length:
            x = F.interpolate(x, self.input_length, mode='linear', align_corners=True)
        if 1 in self.skip_layers:
            x = self._add_skip(x, skips[0], 's1')
        return self.out_act(self.conv_out(x))


class Decoder_CLARAE_SCM_ELU(nn.Module):
    """
    Plain Conv decoder with additive skip connections (no GLU), ELU output.

    Replaces GLUConvBlock with standard Conv1d + InstanceNorm + LeakyReLU blocks.
    Same skip fusion scheme as Decoder_CLARAE_SCM_GLU (use_in controls IN on skips).
    Ablation: tests whether the GLU gating mechanism in the decoder adds value.
    """
    def __init__(self, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256, kernel_size=7, out_act=None,
                 use_in=True, skip_dropout=0.0):
        super().__init__()
        self.filters_initial = filters_initial
        self.input_length    = input_length
        self.out_act         = out_act if out_act is not None else torch.tanh
        fi = filters_initial

        self.fc1 = nn.Linear(latent_dim, dense_dim)
        self.fc2 = nn.Linear(dense_dim, (input_length // 16) * (fi // 8))

        self.conv1 = nn.Sequential(nn.Conv1d(fi // 8, fi // 4, kernel_size, padding='same'),
                                   nn.InstanceNorm1d(fi // 4, affine=True))
        self.conv2 = nn.Sequential(nn.Conv1d(fi // 4, fi // 2, kernel_size, padding='same'),
                                   nn.InstanceNorm1d(fi // 2, affine=True))
        self.conv3 = nn.Sequential(nn.Conv1d(fi // 2, fi,      kernel_size, padding='same'),
                                   nn.InstanceNorm1d(fi,       affine=True))

        self.conv_out  = nn.Conv1d(fi, 1, kernel_size=7, padding='same')
        self.dropout   = nn.Dropout(dropout_rate)
        self.norm_s4   = nn.InstanceNorm1d(fi // 8, affine=True) if use_in else nn.Identity()
        self.norm_s3   = nn.InstanceNorm1d(fi // 4, affine=True) if use_in else nn.Identity()
        self.skip_drop = nn.Dropout(skip_dropout)

    def forward(self, latent, skips):
        nch = self.filters_initial // 8
        x = F.leaky_relu(self.dropout(self.fc1(latent)), 0.3)
        x = F.leaky_relu(self.fc2(x), 0.3)
        x = x.view(x.shape[0], nch, x.shape[1] // nch)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        s4 = skips[3]
        if x.shape[-1] != s4.shape[-1]:
            s4 = F.interpolate(s4, size=x.shape[-1], mode='linear', align_corners=True)
        x = x + self.skip_drop(self.norm_s4(s4))
        x = F.leaky_relu(self.conv1(x), 0.2)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        s3 = skips[2]
        if x.shape[-1] != s3.shape[-1]:
            s3 = F.interpolate(s3, size=x.shape[-1], mode='linear', align_corners=True)
        x = x + self.skip_drop(self.norm_s3(s3))
        x = F.leaky_relu(self.conv2(x), 0.2)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        x = F.leaky_relu(self.conv3(x), 0.2)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if x.shape[-1] != self.input_length:
            x = F.interpolate(x, self.input_length, mode='linear', align_corners=True)
        return self.out_act(self.conv_out(x))


class Decoder_CLARAE_SCM_GLU_GATE(nn.Module):
    """
    GLU decoder with LEARNED GATED skip connections on skip3 and skip4.

    Per-channel sigmoid gates (nn.Parameter, init=gate_init≈-5 → sigmoid≈0.007)
    force the model to rely on the bottleneck latent early in training.

    use_in=True  (default): InstanceNorm normalises each skip before gating.
    use_in=False (sin IN):  skips are gated without normalisation.

    Architecture (fi=filters_initial):
        Layer 1: upsample → x + sigmoid(gate4)*skip_drop([IN](skip4)) → GLU
        Layer 2: upsample → x + sigmoid(gate3)*skip_drop([IN](skip3)) → GLU
        Layer 3: upsample → GLU(fi//2 → fi)          [no skip]
        Layer 4: upsample → Conv(fi → 1) + out_act    [no skip]
    """
    def __init__(self, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256, glu_kernel_size=7,
                 skip_dropout=0.0, gate_init=-5.0, out_act=None, use_in=True):
        super().__init__()
        self.filters_initial = filters_initial
        self.input_length    = input_length
        self.out_act         = out_act if out_act is not None else torch.tanh
        fi = filters_initial

        self.fc1 = nn.Linear(latent_dim, dense_dim)
        self.fc2 = nn.Linear(dense_dim, (input_length // 16) * (fi // 8))

        self.glu1 = GLUConvBlock(fi // 8, fi // 4, glu_kernel_size, dropout=dropout_rate)
        self.glu2 = GLUConvBlock(fi // 4, fi // 2, glu_kernel_size, dropout=dropout_rate)
        self.glu3 = GLUConvBlock(fi // 2, fi,      glu_kernel_size, dropout=dropout_rate)

        self.conv_out  = nn.Conv1d(fi, 1, kernel_size=7, padding='same')
        self.dropout   = nn.Dropout(dropout_rate)
        self.skip_drop = nn.Dropout(skip_dropout)

        self.norm_s4 = nn.InstanceNorm1d(fi // 8, affine=True) if use_in else nn.Identity()
        self.norm_s3 = nn.InstanceNorm1d(fi // 4, affine=True) if use_in else nn.Identity()

        self.gate4 = nn.Parameter(torch.full((fi // 8, 1), gate_init))
        self.gate3 = nn.Parameter(torch.full((fi // 4, 1), gate_init))

    def forward(self, latent, skips):
        nch = self.filters_initial // 8
        x = F.leaky_relu(self.dropout(self.fc1(latent)), 0.3)
        x = F.leaky_relu(self.fc2(x), 0.3)
        x = x.view(x.shape[0], nch, x.shape[1] // nch)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        s4 = skips[3]
        if x.shape[-1] != s4.shape[-1]:
            s4 = F.interpolate(s4, size=x.shape[-1], mode='linear', align_corners=True)
        x = x + torch.sigmoid(self.gate4) * self.skip_drop(self.norm_s4(s4))
        x = self.glu1(x)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        s3 = skips[2]
        if x.shape[-1] != s3.shape[-1]:
            s3 = F.interpolate(s3, size=x.shape[-1], mode='linear', align_corners=True)
        x = x + torch.sigmoid(self.gate3) * self.skip_drop(self.norm_s3(s3))
        x = self.glu2(x)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        x = self.glu3(x)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if x.shape[-1] != self.input_length:
            x = F.interpolate(x, self.input_length, mode='linear', align_corners=True)
        return self.out_act(self.conv_out(x))


class Decoder_CLARAE_SCM_GLU_GATE_SKN(nn.Module):
    """
    GLU decoder with LEARNED GATED skip connections at configurable encoder layers.

    skip_layers controls which encoder layers contribute (1=shallowest, 4=deepest).
    Each active skip is scaled by sigmoid(gate), initialized near 0 (gate_init=-5).

    Architecture (fi=filters_initial):
        Layer 1: upsample → [+ sigmoid(gate4)*skip_drop(IN(skip4)) if 4∈skip_layers]
                          → GLU(fi//8 → fi//4)
        Layer 2: upsample → [+ sigmoid(gate3)*skip_drop(IN(skip3)) if 3∈skip_layers]
                          → GLU(fi//4 → fi//2)
        Layer 3: upsample → [+ sigmoid(gate2)*skip_drop(IN(skip2)) if 2∈skip_layers]
                          → GLU(fi//2 → fi)
        Layer 4: upsample → [+ sigmoid(gate1)*skip_drop(IN(skip1)) if 1∈skip_layers]
                          → Conv(fi → 1) + out_act
    """
    def __init__(self, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256, glu_kernel_size=7,
                 skip_dropout=0.0, gate_init=-5.0, out_act=None,
                 skip_layers=(3, 4)):
        super().__init__()
        self.filters_initial = filters_initial
        self.input_length    = input_length
        self.out_act         = out_act if out_act is not None else torch.tanh
        self.skip_layers     = set(skip_layers)
        fi = filters_initial

        self.fc1 = nn.Linear(latent_dim, dense_dim)
        self.fc2 = nn.Linear(dense_dim, (input_length // 16) * (fi // 8))

        self.glu1 = GLUConvBlock(fi // 8, fi // 4, glu_kernel_size, dropout=dropout_rate)
        self.glu2 = GLUConvBlock(fi // 4, fi // 2, glu_kernel_size, dropout=dropout_rate)
        self.glu3 = GLUConvBlock(fi // 2, fi,      glu_kernel_size, dropout=dropout_rate)

        self.conv_out  = nn.Conv1d(fi, 1, kernel_size=7, padding='same')
        self.dropout   = nn.Dropout(dropout_rate)
        self.skip_drop = nn.Dropout(skip_dropout)

        self.norms = nn.ModuleDict()
        if 4 in self.skip_layers:
            self.norms['s4'] = nn.InstanceNorm1d(fi // 8, affine=True)
            self.gate4 = nn.Parameter(torch.full((fi // 8, 1), gate_init))
        if 3 in self.skip_layers:
            self.norms['s3'] = nn.InstanceNorm1d(fi // 4, affine=True)
            self.gate3 = nn.Parameter(torch.full((fi // 4, 1), gate_init))
        if 2 in self.skip_layers:
            self.norms['s2'] = nn.InstanceNorm1d(fi // 2, affine=True)
            self.gate2 = nn.Parameter(torch.full((fi // 2, 1), gate_init))
        if 1 in self.skip_layers:
            self.norms['s1'] = nn.InstanceNorm1d(fi, affine=True)
            self.gate1 = nn.Parameter(torch.full((fi, 1), gate_init))

    def _add_gated_skip(self, x, s, key, gate):
        if s.shape[-1] != x.shape[-1]:
            s = F.interpolate(s, size=x.shape[-1], mode='linear', align_corners=True)
        return x + torch.sigmoid(gate) * self.skip_drop(self.norms[key](s))

    def forward(self, latent, skips):
        nch = self.filters_initial // 8
        x = F.leaky_relu(self.dropout(self.fc1(latent)), 0.3)
        x = F.leaky_relu(self.fc2(x), 0.3)
        x = x.view(x.shape[0], nch, x.shape[1] // nch)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if 4 in self.skip_layers:
            x = self._add_gated_skip(x, skips[3], 's4', self.gate4)
        x = self.glu1(x)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if 3 in self.skip_layers:
            x = self._add_gated_skip(x, skips[2], 's3', self.gate3)
        x = self.glu2(x)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if 2 in self.skip_layers:
            x = self._add_gated_skip(x, skips[1], 's2', self.gate2)
        x = self.glu3(x)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if x.shape[-1] != self.input_length:
            x = F.interpolate(x, self.input_length, mode='linear', align_corners=True)
        if 1 in self.skip_layers:
            x = self._add_gated_skip(x, skips[0], 's1', self.gate1)
        return self.out_act(self.conv_out(x))


class Decoder_CLARAE_SCM_ELU_GATE(nn.Module):
    """
    Plain Conv decoder (no GLU) with LEARNED GATED skip connections on skip3 and skip4.

    Same as Decoder_CLARAE_SCM_GLU_GATE but replaces GLUConvBlock with
    Conv1d + InstanceNorm + LeakyReLU blocks. Ablation: isolates the contribution
    of GLU gating in the decoder when learned skip gates are present.

    Architecture (fi=filters_initial):
        Layer 1: upsample → x + sigmoid(gate4)*skip_drop([IN](skip4)) → Conv+IN+LReLU
        Layer 2: upsample → x + sigmoid(gate3)*skip_drop([IN](skip3)) → Conv+IN+LReLU
        Layer 3: upsample → Conv+IN+LReLU(fi//2 → fi)                 [no skip]
        Layer 4: upsample → Conv(fi → 1) + out_act                    [no skip]
    """
    def __init__(self, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256, kernel_size=7,
                 skip_dropout=0.0, gate_init=-5.0, out_act=None, use_in=True):
        super().__init__()
        self.filters_initial = filters_initial
        self.input_length    = input_length
        self.out_act         = out_act if out_act is not None else torch.tanh
        fi = filters_initial

        self.fc1 = nn.Linear(latent_dim, dense_dim)
        self.fc2 = nn.Linear(dense_dim, (input_length // 16) * (fi // 8))

        self.conv1 = nn.Sequential(nn.Conv1d(fi // 8, fi // 4, kernel_size, padding='same'),
                                   nn.InstanceNorm1d(fi // 4, affine=True))
        self.conv2 = nn.Sequential(nn.Conv1d(fi // 4, fi // 2, kernel_size, padding='same'),
                                   nn.InstanceNorm1d(fi // 2, affine=True))
        self.conv3 = nn.Sequential(nn.Conv1d(fi // 2, fi,      kernel_size, padding='same'),
                                   nn.InstanceNorm1d(fi,       affine=True))

        self.conv_out  = nn.Conv1d(fi, 1, kernel_size=7, padding='same')
        self.dropout   = nn.Dropout(dropout_rate)
        self.skip_drop = nn.Dropout(skip_dropout)

        self.norm_s4 = nn.InstanceNorm1d(fi // 8, affine=True) if use_in else nn.Identity()
        self.norm_s3 = nn.InstanceNorm1d(fi // 4, affine=True) if use_in else nn.Identity()

        self.gate4 = nn.Parameter(torch.full((fi // 8, 1), gate_init))
        self.gate3 = nn.Parameter(torch.full((fi // 4, 1), gate_init))

    def forward(self, latent, skips):
        nch = self.filters_initial // 8
        x = F.leaky_relu(self.dropout(self.fc1(latent)), 0.3)
        x = F.leaky_relu(self.fc2(x), 0.3)
        x = x.view(x.shape[0], nch, x.shape[1] // nch)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        s4 = skips[3]
        if x.shape[-1] != s4.shape[-1]:
            s4 = F.interpolate(s4, size=x.shape[-1], mode='linear', align_corners=True)
        x = x + torch.sigmoid(self.gate4) * self.skip_drop(self.norm_s4(s4))
        x = F.leaky_relu(self.conv1(x), 0.2)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        s3 = skips[2]
        if x.shape[-1] != s3.shape[-1]:
            s3 = F.interpolate(s3, size=x.shape[-1], mode='linear', align_corners=True)
        x = x + torch.sigmoid(self.gate3) * self.skip_drop(self.norm_s3(s3))
        x = F.leaky_relu(self.conv2(x), 0.2)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        x = F.leaky_relu(self.conv3(x), 0.2)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if x.shape[-1] != self.input_length:
            x = F.interpolate(x, self.input_length, mode='linear', align_corners=True)
        return self.out_act(self.conv_out(x))


class Decoder_CLARAE_SCM_ELU_GATE_SKN(nn.Module):
    """
    Plain Conv decoder (no GLU) with LEARNED GATED skip connections at configurable layers.

    Same as Decoder_CLARAE_SCM_GLU_GATE_SKN but replaces GLUConvBlock with
    Conv1d + InstanceNorm + LeakyReLU blocks.

    skip_layers controls which encoder layers contribute (1=shallowest, 4=deepest).
    Each active skip is scaled by sigmoid(gate), initialized near 0 (gate_init=-5).
    """
    def __init__(self, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256, kernel_size=7,
                 skip_dropout=0.0, gate_init=-5.0, out_act=None,
                 skip_layers=(3, 4)):
        super().__init__()
        self.filters_initial = filters_initial
        self.input_length    = input_length
        self.out_act         = out_act if out_act is not None else torch.tanh
        self.skip_layers     = set(skip_layers)
        fi = filters_initial

        self.fc1 = nn.Linear(latent_dim, dense_dim)
        self.fc2 = nn.Linear(dense_dim, (input_length // 16) * (fi // 8))

        self.conv1 = nn.Sequential(nn.Conv1d(fi // 8, fi // 4, kernel_size, padding='same'),
                                   nn.InstanceNorm1d(fi // 4, affine=True))
        self.conv2 = nn.Sequential(nn.Conv1d(fi // 4, fi // 2, kernel_size, padding='same'),
                                   nn.InstanceNorm1d(fi // 2, affine=True))
        self.conv3 = nn.Sequential(nn.Conv1d(fi // 2, fi,      kernel_size, padding='same'),
                                   nn.InstanceNorm1d(fi,       affine=True))

        self.conv_out  = nn.Conv1d(fi, 1, kernel_size=7, padding='same')
        self.dropout   = nn.Dropout(dropout_rate)
        self.skip_drop = nn.Dropout(skip_dropout)

        self.norms = nn.ModuleDict()
        if 4 in self.skip_layers:
            self.norms['s4'] = nn.InstanceNorm1d(fi // 8, affine=True)
            self.gate4 = nn.Parameter(torch.full((fi // 8, 1), gate_init))
        if 3 in self.skip_layers:
            self.norms['s3'] = nn.InstanceNorm1d(fi // 4, affine=True)
            self.gate3 = nn.Parameter(torch.full((fi // 4, 1), gate_init))
        if 2 in self.skip_layers:
            self.norms['s2'] = nn.InstanceNorm1d(fi // 2, affine=True)
            self.gate2 = nn.Parameter(torch.full((fi // 2, 1), gate_init))
        if 1 in self.skip_layers:
            self.norms['s1'] = nn.InstanceNorm1d(fi, affine=True)
            self.gate1 = nn.Parameter(torch.full((fi, 1), gate_init))

    def _add_gated_skip(self, x, s, key, gate):
        if s.shape[-1] != x.shape[-1]:
            s = F.interpolate(s, size=x.shape[-1], mode='linear', align_corners=True)
        return x + torch.sigmoid(gate) * self.skip_drop(self.norms[key](s))

    def forward(self, latent, skips):
        nch = self.filters_initial // 8
        x = F.leaky_relu(self.dropout(self.fc1(latent)), 0.3)
        x = F.leaky_relu(self.fc2(x), 0.3)
        x = x.view(x.shape[0], nch, x.shape[1] // nch)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if 4 in self.skip_layers:
            x = self._add_gated_skip(x, skips[3], 's4', self.gate4)
        x = F.leaky_relu(self.conv1(x), 0.2)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if 3 in self.skip_layers:
            x = self._add_gated_skip(x, skips[2], 's3', self.gate3)
        x = F.leaky_relu(self.conv2(x), 0.2)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if 2 in self.skip_layers:
            x = self._add_gated_skip(x, skips[1], 's2', self.gate2)
        x = F.leaky_relu(self.conv3(x), 0.2)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if x.shape[-1] != self.input_length:
            x = F.interpolate(x, self.input_length, mode='linear', align_corners=True)
        if 1 in self.skip_layers:
            x = self._add_gated_skip(x, skips[0], 's1', self.gate1)
        return self.out_act(self.conv_out(x))


class Decoder_CLARAE_SCM_ELU_SKN(nn.Module):
    """
    Plain Conv decoder (no GLU, no gate) with additive skip connections at configurable layers.

    Analogous to Decoder_CLARAE_SCM_GLU_SKN but with Conv1d + InstanceNorm + LeakyReLU
    instead of GLUConvBlock. No learned gates — skips are added directly (with IN).
    """
    def __init__(self, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256, kernel_size=7,
                 skip_dropout=0.0, out_act=None, skip_layers=(4,)):
        super().__init__()
        self.filters_initial = filters_initial
        self.input_length    = input_length
        self.out_act         = out_act if out_act is not None else torch.tanh
        self.skip_layers     = set(skip_layers)
        fi = filters_initial

        self.fc1 = nn.Linear(latent_dim, dense_dim)
        self.fc2 = nn.Linear(dense_dim, (input_length // 16) * (fi // 8))

        self.conv1 = nn.Sequential(nn.Conv1d(fi // 8, fi // 4, kernel_size, padding='same'),
                                   nn.InstanceNorm1d(fi // 4, affine=True))
        self.conv2 = nn.Sequential(nn.Conv1d(fi // 4, fi // 2, kernel_size, padding='same'),
                                   nn.InstanceNorm1d(fi // 2, affine=True))
        self.conv3 = nn.Sequential(nn.Conv1d(fi // 2, fi,      kernel_size, padding='same'),
                                   nn.InstanceNorm1d(fi,       affine=True))

        self.conv_out  = nn.Conv1d(fi, 1, kernel_size=7, padding='same')
        self.dropout   = nn.Dropout(dropout_rate)
        self.skip_drop = nn.Dropout(skip_dropout)

        _ch = {4: fi // 8, 3: fi // 4, 2: fi // 2, 1: fi}
        self.norms = nn.ModuleDict({
            f's{l}': nn.InstanceNorm1d(_ch[l], affine=True)
            for l in self.skip_layers
        })

    def _add_skip(self, x, s, key):
        if s.shape[-1] != x.shape[-1]:
            s = F.interpolate(s, size=x.shape[-1], mode='linear', align_corners=True)
        return x + self.skip_drop(self.norms[key](s))

    def forward(self, latent, skips):
        nch = self.filters_initial // 8
        x = F.leaky_relu(self.dropout(self.fc1(latent)), 0.3)
        x = F.leaky_relu(self.fc2(x), 0.3)
        x = x.view(x.shape[0], nch, x.shape[1] // nch)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if 4 in self.skip_layers:
            x = self._add_skip(x, skips[3], 's4')
        x = F.leaky_relu(self.conv1(x), 0.2)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if 3 in self.skip_layers:
            x = self._add_skip(x, skips[2], 's3')
        x = F.leaky_relu(self.conv2(x), 0.2)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if 2 in self.skip_layers:
            x = self._add_skip(x, skips[1], 's2')
        x = F.leaky_relu(self.conv3(x), 0.2)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if x.shape[-1] != self.input_length:
            x = F.interpolate(x, self.input_length, mode='linear', align_corners=True)
        if 1 in self.skip_layers:
            x = self._add_skip(x, skips[0], 's1')
        return self.out_act(self.conv_out(x))


# =============================================================================
# Section 4 — CLARAE_AP_ELU: single-scale bottleneck baseline
# Single-scale Conv + AvgPool + InstanceNorm + ELU, NO skip connections.
# Reference starting point for the ablation chain.
# =============================================================================

class _Encoder_AP_ELU(nn.Module):
    """Single-scale AvgPool encoder, no skip connections output."""
    def __init__(self, input_channels, input_length, latent_dim,
                 filters_initial, dropout_rate=0.2, dense_dim=256):
        super().__init__()
        fi = filters_initial
        self.block1 = nn.Sequential(nn.Conv1d(input_channels, fi,    7, padding='same'),
                                    nn.InstanceNorm1d(fi,    affine=True))
        self.block2 = nn.Sequential(nn.Conv1d(fi,    fi//2,  7, padding='same'),
                                    nn.InstanceNorm1d(fi//2, affine=True))
        self.block3 = nn.Sequential(nn.Conv1d(fi//2, fi//4,  7, padding='same'),
                                    nn.InstanceNorm1d(fi//4, affine=True))
        self.block4 = nn.Sequential(nn.Conv1d(fi//4, fi//8,  7, padding='same'),
                                    nn.InstanceNorm1d(fi//8, affine=True))
        self.pool    = nn.AvgPool1d(2)
        flatten_size = (input_length // 16) * (fi // 8)
        self.fc1     = nn.Linear(flatten_size, dense_dim)
        self.fc2     = nn.Linear(dense_dim, latent_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.elu     = nn.ELU()

    def forward(self, x):
        for block in [self.block1, self.block2, self.block3, self.block4]:
            x = F.leaky_relu(block(x), 0.2)
            x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.dropout(self.fc1(x)), 0.3)
        return self.elu(self.fc2(x))


class _Decoder_AP_ELU(nn.Module):
    """Simple upsampling decoder without skip connections, ELU output."""
    def __init__(self, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256, kernel_size=7):
        super().__init__()
        fi = filters_initial
        self.fi8 = fi // 8
        self.input_length = input_length

        self.fc1 = nn.Linear(latent_dim, dense_dim)
        self.fc2 = nn.Linear(dense_dim, (input_length // 16) * (fi // 8))

        self.conv1 = nn.Conv1d(fi//8, fi//4, kernel_size, padding='same')
        self.bn1   = nn.InstanceNorm1d(fi//4, affine=True)
        self.conv2 = nn.Conv1d(fi//4, fi//2, kernel_size, padding='same')
        self.bn2   = nn.InstanceNorm1d(fi//2, affine=True)
        self.conv3 = nn.Conv1d(fi//2, fi,    kernel_size, padding='same')
        self.bn3   = nn.InstanceNorm1d(fi,   affine=True)

        self.conv_out = nn.Conv1d(fi, 1, kernel_size=7, padding='same')
        self.dropout  = nn.Dropout(dropout_rate)
        self.elu      = nn.ELU()

    def forward(self, x):
        x = F.leaky_relu(self.dropout(self.fc1(x)), 0.3)
        x = F.leaky_relu(self.fc2(x), 0.3)
        x = x.view(x.shape[0], self.fi8, -1)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.3)
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.3)
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.3)
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if x.shape[-1] != self.input_length:
            x = F.interpolate(x, self.input_length, mode='linear', align_corners=True)
        return self.elu(self.conv_out(x))


class CLARAE_AP_ELU(nn.Module):
    """
    Single-scale bottleneck AE: AvgPool + InstanceNorm + ELU, no skip connections.

    Baseline #1 of the ablation chain. Uses single-scale Conv1d (not multi-scale),
    AvgPool, InstanceNorm per layer, and ELU activations throughout.
    No skip connections in the decoder — pure bottleneck autoencoder.
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256):
        super().__init__()
        self.encoder = _Encoder_AP_ELU(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim,
        )
        self.decoder = _Decoder_AP_ELU(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# =============================================================================
# Section 5 — SCM + GLU + AP + ELU without and with IN on skip connections
# AvgPool encoder + GLU decoder + ELU activations
# sin IN: skips added directly (U-Net style)
# con IN: skips normalised with InstanceNorm before addition
# =============================================================================

class CLARAE_SCM_GLU_AP_ELU_woIN(nn.Module):
    """
    Multi-scale GLU AE with AvgPool + ELU, WITHOUT InstanceNorm on skip connections.

    Skip features are added directly to the decoder stream without normalisation
    (classic U-Net additive skip). Ablation vs. CLARAE_SCM_GLU_AP_ELU (con IN).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256,
                 ms_kernels=(3, 9, 21), glu_kernel_size=7):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim, ms_kernels,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_GLU(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, glu_kernel_size,
            out_act=nn.ELU(), use_in=False,
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_GLU_AP_ELU(nn.Module):
    """
    Multi-scale GLU AE with AvgPool + ELU + InstanceNorm on skip connections.

    ELU replacing tanh in the latent bottleneck and decoder output.
    InstanceNorm on skip4 and skip3 before additive fusion in the decoder.
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256,
                 ms_kernels=(3, 9, 21), glu_kernel_size=7):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim, ms_kernels,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_GLU(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, glu_kernel_size,
            out_act=nn.ELU(), use_in=True,
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_AP_ELU(nn.Module):
    """
    Multi-scale AE with AvgPool + ELU + IN on skip connections, WITHOUT GLU decoder.

    Ablation of CLARAE_SCM_GLU_AP_ELU: replaces GLUConvBlock in the decoder with
    plain Conv1d + InstanceNorm + LeakyReLU blocks. Allows isolating the contribution
    of the gated linear units in the decoder.
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256,
                 ms_kernels=(3, 9, 21), kernel_size=7):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim, ms_kernels,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_ELU(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, kernel_size,
            out_act=nn.ELU(), use_in=True,
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_AP_ELU_woIN(nn.Module):
    """
    CLARAE_SCM_AP_ELU WITHOUT InstanceNorm on skip connections.

    Ablation of CLARAE_SCM_AP_ELU: removes IN normalisation before adding skips.
    noGLU counterpart of CLARAE_SCM_GLU_AP_ELU_woIN.
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256,
                 ms_kernels=(3, 9, 21), kernel_size=7):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim, ms_kernels,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_ELU(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, kernel_size,
            out_act=nn.ELU(), use_in=False,
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_AP_ELU_K2(nn.Module):
    """
    CLARAE_SCM_AP_ELU with 2-scale encoder (kernels 3 and 9 only), no gate.

    Ablation of CLARAE_SCM_AP_ELU: removes the largest scale (kernel=21).
    Uses skip3 + skip4 with InstanceNorm, no learned gate. ms_kernels=(3, 9).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256, kernel_size=7):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim,
            ms_kernels=(3, 9),
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_ELU(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, kernel_size,
            out_act=nn.ELU(), use_in=True,
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_AP_ELU_GATE_K2(nn.Module):
    """
    CLARAE_SCM_AP_ELU with 2-scale encoder (kernels 3 and 9) + learned gates on skip3+skip4.

    noGLU counterpart of CLARAE_SCM_GLU_AP_ELU_GATE with 2-scale encoder.
    Gates on skip3 and skip4 with InstanceNorm. ms_kernels=(3, 9).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 kernel_size=7, gate_init=-5.0):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim,
            ms_kernels=(3, 9),
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_ELU_GATE(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, kernel_size,
            skip_dropout=skip_dropout, gate_init=gate_init,
            out_act=nn.ELU(), use_in=True,
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


# =============================================================================
# Section 6 — SCM + GLU + AP + ELU + learned gates on skip3 and skip4
# Decoder_CLARAE_SCM_GLU_GATE: fixed 2-gate decoder (skip3 + skip4)
# sin IN: gates applied without prior normalisation
# con IN: gates applied after InstanceNorm normalisation (default)
# =============================================================================

class CLARAE_SCM_GLU_AP_ELU_GATE_woIN(nn.Module):
    """
    CLARAE_SCM_GLU_AP_ELU_GATE WITHOUT InstanceNorm on gated skip connections.

    Per-channel sigmoid gates on skip3 and skip4, but without InstanceNorm
    normalisation before gating. Ablation vs. CLARAE_SCM_GLU_AP_ELU_GATE (con IN).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 ms_kernels=(3, 9, 21), glu_kernel_size=7, gate_init=-5.0):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim, ms_kernels,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_GLU_GATE(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, glu_kernel_size,
            skip_dropout=skip_dropout, gate_init=gate_init,
            out_act=nn.ELU(), use_in=False,
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_GLU_AP_ELU_GATE(nn.Module):
    """
    CLARAE_SCM_GLU_AP_ELU with learned gated skip connections on skip3 and skip4.

    Per-channel sigmoid gates (init=-5 → sigmoid≈0.007) force the model to rely
    on the bottleneck latent early in training. InstanceNorm normalises each skip
    before gating.
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 ms_kernels=(3, 9, 21), glu_kernel_size=7, gate_init=-5.0):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim, ms_kernels,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_GLU_GATE(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, glu_kernel_size,
            skip_dropout=skip_dropout, gate_init=gate_init,
            out_act=nn.ELU(), use_in=True,
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_AP_ELU_GATE(nn.Module):
    """
    CLARAE_SCM_AP_ELU with learned gated skip connections on skip3 and skip4 (no GLU).

    Ablation of CLARAE_SCM_GLU_AP_ELU_GATE: replaces GLUConvBlock in the decoder
    with plain Conv1d + InstanceNorm + LeakyReLU. Isolates whether the performance
    gain from learned gates comes from the gating mechanism itself or the GLU blocks.
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 ms_kernels=(3, 9, 21), kernel_size=7, gate_init=-5.0):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim, ms_kernels,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_ELU_GATE(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, kernel_size,
            skip_dropout=skip_dropout, gate_init=gate_init,
            out_act=nn.ELU(), use_in=True,
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


# =============================================================================
# Section 7 — SCM + GLU + AP + ELU + gate on skip4 ONLY (SKLAST)
# Decoder_CLARAE_SCM_GLU_GATE_SKN(skip_layers=(4,))
# =============================================================================

class CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST(nn.Module):
    """
    CLARAE_SCM_GLU_AP_ELU_GATE with gate ONLY on skip4 (deepest encoder layer).

    No shallow skip connections reach the decoder. The single gated skip at the
    deepest level carries local morphology context while protecting the latent
    from carrying excessive detail. ms_kernels=(3,9,21).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 ms_kernels=(3, 9, 21), glu_kernel_size=7, gate_init=-5.0):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim, ms_kernels,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_GLU_GATE_SKN(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, glu_kernel_size,
            skip_dropout=skip_dropout, gate_init=gate_init,
            out_act=nn.ELU(), skip_layers=(4,),
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST_K2(nn.Module):
    """
    CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST with 2-scale encoder (kernels 3 and 9 only).

    Ablation: removes the largest scale (kernel=21, ~42 ms) to study whether
    the coarsest temporal context is necessary. ms_kernels=(3,9).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 glu_kernel_size=7, gate_init=-5.0):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim,
            ms_kernels=(3, 9),              # 2-scale
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_GLU_GATE_SKN(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, glu_kernel_size,
            skip_dropout=skip_dropout, gate_init=gate_init,
            out_act=nn.ELU(), skip_layers=(4,),
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_AP_ELU_GATE_SKLAST(nn.Module):
    """
    CLARAE_SCM_AP_ELU with learned gate on skip4 ONLY, no GLU decoder.

    Ablation of CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST: replaces GLUConvBlock with
    plain Conv1d + InstanceNorm + LeakyReLU. Gate on deepest skip only (skip4).
    ms_kernels=(3, 9, 21).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 ms_kernels=(3, 9, 21), kernel_size=7, gate_init=-5.0):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim, ms_kernels,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_ELU_GATE_SKN(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, kernel_size,
            skip_dropout=skip_dropout, gate_init=gate_init,
            out_act=nn.ELU(), skip_layers=(4,),
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_AP_ELU_GATE_SKLAST_K2(nn.Module):
    """
    CLARAE_SCM_AP_ELU_GATE_SKLAST with 2-scale encoder (kernels 3 and 9 only), no GLU.

    Ablation of CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST_K2: replaces GLUConvBlock.
    Removes the largest scale (kernel=21) to study coarsest temporal context.
    ms_kernels=(3, 9).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 kernel_size=7, gate_init=-5.0):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim,
            ms_kernels=(3, 9),
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_ELU_GATE_SKN(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, kernel_size,
            skip_dropout=skip_dropout, gate_init=gate_init,
            out_act=nn.ELU(), skip_layers=(4,),
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_AP_ELU_SKLAST(nn.Module):
    """
    CLARAE_SCM_AP_ELU with plain additive skip on skip4 ONLY, no GLU, no gate.

    Ablation of CLARAE_SCM_AP_ELU_GATE_SKLAST: removes the learned sigmoid gate,
    using a direct IN-normalized additive skip. Tests whether gating is necessary
    when the GLU decoder is already removed. ms_kernels=(3, 9, 21).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 ms_kernels=(3, 9, 21), kernel_size=7):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim, ms_kernels,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_ELU_SKN(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, kernel_size,
            skip_dropout=skip_dropout,
            out_act=nn.ELU(), skip_layers=(4,),
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_AP_ELU_SKLAST_K2(nn.Module):
    """
    CLARAE_SCM_AP_ELU_SKLAST with 2-scale encoder (kernels 3 and 9 only), no gate.

    Ablation of CLARAE_SCM_AP_ELU_GATE_SKLAST_K2: removes the learned gate.
    ms_kernels=(3, 9).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 kernel_size=7):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim,
            ms_kernels=(3, 9),
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_ELU_SKN(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, kernel_size,
            skip_dropout=skip_dropout,
            out_act=nn.ELU(), skip_layers=(4,),
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_GLU_AP_ELU_GATE_SKALL(nn.Module):
    """
    CLARAE_SCM_GLU_AP_ELU with learned gated skip connections on ALL 4 encoder layers.

    Extension of CLARAE_SCM_GLU_AP_ELU_GATE (which gates skip3+skip4 only) to also
    gate skip1 and skip2. Each skip is scaled by sigmoid(gate), initialized near 0.
    ms_kernels=(3,9,21).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 ms_kernels=(3, 9, 21), glu_kernel_size=7, gate_init=-5.0):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim, ms_kernels,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_GLU_GATE_SKN(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, glu_kernel_size,
            skip_dropout=skip_dropout, gate_init=gate_init,
            out_act=nn.ELU(), skip_layers=(1, 2, 3, 4),
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_GLU_AP_ELU_SKALL(nn.Module):
    """
    AvgPool encoder + GLU decoder + IN-normalized additive skip on ALL 4 encoder layers.
    No learned gate — skips added directly after InstanceNorm.
    Ablation of CLARAE_SCM_GLU_AP_ELU_GATE_SKALL. ms_kernels=(3,9,21).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 ms_kernels=(3, 9, 21), glu_kernel_size=7):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim, ms_kernels,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_GLU_SKN(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, glu_kernel_size,
            skip_dropout=skip_dropout, out_act=nn.ELU(),
            skip_layers=(1, 2, 3, 4), use_in=True,
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_GLU_AP_ELU_SKLAST_K2(nn.Module):
    """
    Ablation of CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST_K2: no gate, WITH IN on skip4.

    SCM encoder with 2-scale kernels (3,9) + GLU decoder + IN-normalized additive skip4.
    Tests whether learned gating is necessary when InstanceNorm on the skip is kept.
    ms_kernels=(3,9).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256, glu_kernel_size=7):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim,
            ms_kernels=(3, 9),
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_GLU_SKN(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, glu_kernel_size,
            skip_dropout=skip_dropout,
            out_act=nn.ELU(), skip_layers=(4,), use_in=True,
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_GLU_AP_ELU_GATE_SKLAST(nn.Module):
    """
    Single-scale encoder (no SCM) + GLU decoder + gated skip4 only.

    Ablation of CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST: replaces MultiScaleConvBlock
    with plain Conv1d + InstanceNorm to isolate the contribution of multi-scale
    feature extraction. Gate on deepest skip only (skip4).
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 glu_kernel_size=7, gate_init=-5.0):
        super().__init__()
        self.encoder = Encoder_CLARAE_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_GLU_GATE_SKN(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, glu_kernel_size,
            skip_dropout=skip_dropout, gate_init=gate_init,
            out_act=nn.ELU(), skip_layers=(4,),
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_AP_ELU_GATE_SKLAST(nn.Module):
    """
    Single-scale encoder (no SCM) + ELU decoder (no GLU) + gated skip4 only.

    noGLU counterpart of CLARAE_GLU_AP_ELU_GATE_SKLAST: replaces GLUConvBlock
    with Conv1d + InstanceNorm + LeakyReLU in the decoder.
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 kernel_size=7, gate_init=-5.0):
        super().__init__()
        self.encoder = Encoder_CLARAE_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_ELU_GATE_SKN(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, kernel_size,
            skip_dropout=skip_dropout, gate_init=gate_init,
            out_act=nn.ELU(), skip_layers=(4,),
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST_TANH(nn.Module):
    """
    CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST with tanh output activation.

    ELU in the encoder latent, tanh in the decoder output.
    Ablation of the output activation on the best SKLAST model.
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, skip_dropout=0.0, dense_dim=256,
                 ms_kernels=(3, 9, 21), glu_kernel_size=7, gate_init=-5.0):
        super().__init__()
        self.encoder = Encoder_CLARAE_SCM_AP(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim, ms_kernels,
            latent_act=nn.ELU(),
        )
        self.decoder = Decoder_CLARAE_SCM_GLU_GATE_SKN(
            input_length, latent_dim, filters_initial,
            dropout_rate, dense_dim, glu_kernel_size,
            skip_dropout=skip_dropout, gate_init=gate_init,
            skip_layers=(4,),               # out_act=tanh (default)
        )

    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)

