
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import numpy as np
from torchvision.ops import roi_align

try:
    from torchvision.ops import DeformConv2d
    DEFORMCONV_AVAILABLE = True
except ImportError:
    DEFORMCONV_AVAILABLE = False

class FeatureExtractor(nn.Module):

    def __init__(self, name='resnet50', pretrained=True):
        super().__init__()
        self.name = name

        actual_backbone = name
        if '_' in name:

            parts = name.split('_')

            if len(parts) >= 2:
                last_part = parts[-1]

                if '+' in last_part or (last_part.isupper() and len(last_part) <= 20):

                    actual_backbone = '_'.join(parts[:-1])
                    print(f"backbone: '{name}' -> backbone: '{actual_backbone}'")

        if actual_backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.layer0 = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            self.out_channels = [256, 512, 1024, 2048]

        elif actual_backbone == 'vit_l_16':
            try:
                from torchvision.models import vit_l_16, ViT_L_16_Weights
                if pretrained:

                    import torch.hub
                    original_download = torch.hub.download_url_to_file

                    def download_with_retry(url, dst, hash_prefix=None, progress=True, max_retries=3):
                        for attempt in range(max_retries):
                            try:

                                return original_download(url, dst, hash_prefix=None, progress=progress)
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    print(f"Download attempt {attempt + 1} failed, retrying...")
                                    continue
                                raise e

                    torch.hub.download_url_to_file = download_with_retry

                    try:
                        vit = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
                    finally:
                        torch.hub.download_url_to_file = original_download
                else:
                    vit = vit_l_16(weights=None)
                self.vit_model = vit
                self.vit_hidden_dim = 1024

                self.vit_proj1 = nn.Conv2d(1024, 256, 1)
                self.vit_proj2 = nn.Conv2d(1024, 512, 1)
                self.vit_proj3 = nn.Conv2d(1024, 1024, 1)
                self.vit_proj4 = nn.Conv2d(1024, 2048, 1)
                self.out_channels = [256, 512, 1024, 2048]
                self.use_vit = True
            except Exception as e:
                print(f"Error loading ViT-L/16: {e}")
                raise ValueError("ViTtorchvision (>=0.13)")

        else:
            raise ValueError(f"Unsupported backbone: {actual_backbone} (from name: {name})")

    def forward(self, x):
        if hasattr(self, 'use_efficientnet'):
            features = []
            for i, module in enumerate(self.features):
                x = module(x)
                if i in self.feature_indices:
                    features.append(x)
            return features

        elif hasattr(self, 'use_regnet'):
            x = self.stem(x)
            features = []
            for i, block in enumerate(self.trunk_output):
                x = block(x)
                features.append(x)
            return features

        elif hasattr(self, 'use_convnext'):
            features = []
            for i in range(len(self.features)):
                x = self.features[i](x)
                if i in self.feature_indices:
                    features.append(x)
            return features

        elif hasattr(self, 'use_swin'):

            features = []
            curr = x
            for i in range(len(self.swin_features)):
                curr = self.swin_features[i](curr)

                if i in [1, 3, 5, 7]:

                    feature = curr.permute(0, 3, 1, 2).contiguous()
                    features.append(feature)
            return features

        elif hasattr(self, 'use_vit'):

            B, C, H, W = x.shape

            x = self.vit_model.conv_proj(x)

            x = x.flatten(2).transpose(1, 2)
            N = x.shape[1]

            cls_token = self.vit_model.class_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

            pos_embed = self.vit_model.encoder.pos_embedding

            if N + 1 != pos_embed.shape[1]:

                cls_pos_embed = pos_embed[:, 0:1, :]
                patch_pos_embed = pos_embed[:, 1:, :]

                orig_size = int(math.sqrt(patch_pos_embed.shape[1]))
                new_size = int(math.sqrt(N))

                patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, -1)
                patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
                patch_pos_embed = F.interpolate(
                    patch_pos_embed,
                    size=(new_size, new_size),
                    mode='bicubic',
                    align_corners=False
                )
                patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
                patch_pos_embed = patch_pos_embed.flatten(1, 2)

                pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)

            x = x + pos_embed

            x = self.vit_model.encoder.dropout(x)
            for layer in self.vit_model.encoder.layers:
                x = layer(x)
            x = self.vit_model.encoder.ln(x)

            x = x[:, 1:, :]
            grid_size = int(math.sqrt(N))
            x = x.transpose(1, 2).reshape(B, self.vit_hidden_dim, grid_size, grid_size)

            f1 = self.vit_proj1(x)
            f2 = self.vit_proj2(x)
            f3 = self.vit_proj3(x)
            f4 = self.vit_proj4(x)
            return [f1, f2, f3, f4]

        else:
            x = self.layer0(x)
            c1 = self.layer1(x)
            c2 = self.layer2(c1)
            c3 = self.layer3(c2)
            c4 = self.layer4(c3)
            return [c1, c2, c3, c4]

class CBAM(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att

        return x

class DeformableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False, groups=1):
        super().__init__()

        if not DEFORMCONV_AVAILABLE:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                 stride, padding, bias=bias, groups=groups)
            self.use_deform = False
        else:
            self.offset_conv = nn.Conv2d(
                in_channels,
                2 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True
            )
            nn.init.constant_(self.offset_conv.weight, 0.)
            nn.init.constant_(self.offset_conv.bias, 0.)

            self.deform_conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )
            self.use_deform = True

    def forward(self, x):
        if not self.use_deform:
            return self.conv(x)
        offset = self.offset_conv(x)
        return self.deform_conv(x, offset)

class DeformableFPN(nn.Module):

    def __init__(self, in_channels_list, out_channels=256, use_dcn=True):
        super().__init__()
        self.use_dcn = use_dcn and DEFORMCONV_AVAILABLE

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)

            if self.use_dcn:
                fpn_conv = nn.Sequential(
                    DeformableConv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
                smooth_conv = DeformableConv2d(out_channels, out_channels, 3, padding=1)
            else:
                fpn_conv = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
                smooth_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

            attention = CBAM(out_channels)

            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
            self.attention.append(attention)
            self.smooth_convs.append(smooth_conv)

    def forward(self, features):
        laterals = [lateral_conv(f) for f, lateral_conv in zip(features, self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            target_size = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=target_size, mode='bilinear', align_corners=False)
            laterals[i - 1] = laterals[i - 1] + upsampled

        outputs = []
        for lateral, fpn_conv, attention, smooth_conv in zip(laterals, self.fpn_convs, self.attention, self.smooth_convs):
            out = fpn_conv(lateral)
            out = attention(out)
            out = smooth_conv(out)
            outputs.append(out)

        return outputs

class ContextModule(nn.Module):
    def __init__(self, in_channels=256, reduction=4):
        super().__init__()
        mid_channels = in_channels // reduction

        self.ring_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.ring_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.ring_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=3, dilation=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.weight_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Softmax(dim=1)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, heatmap):

        B, C, H, W = x.shape

        if heatmap.shape[2:] != (H, W):
            heatmap = F.interpolate(heatmap, size=(H, W), mode='bilinear', align_corners=False)

        ring1 = self.ring_conv1(x)
        ring2 = self.ring_conv2(x)
        ring3 = self.ring_conv3(x)

        weights = self.weight_net(heatmap)
        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]
        w3 = weights[:, 2:3, :, :]

        ring_feat = w1 * ring1 + w2 * ring2 + w3 * ring3
        ring_feat = self.fusion(ring_feat)

        gate = self.gate(torch.cat([x, ring_feat], dim=1))
        x_enhanced = x + gate * ring_feat

        return x_enhanced

class EncodingModule(nn.Module):

    def __init__(self, in_channels=256):
        super().__init__()

        self.dir_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1)
        )

        self.dir_attention = nn.Sequential(
            nn.Conv2d(2, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1),
            nn.Softmax(dim=1)
        )

        self.horizontal_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.vertical_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):

        ori_field = self.dir_conv(x)

        ori_field = F.normalize(ori_field, p=2, dim=1)

        dir_weights = self.dir_attention(ori_field)
        w_h = dir_weights[:, 0:1, :, :]
        w_v = dir_weights[:, 1:2, :, :]

        feat_h = self.horizontal_conv(x)
        feat_v = self.vertical_conv(x)

        feat_dir = w_h * feat_h + w_v * feat_v

        ch_att = self.channel_attention(feat_dir)
        feat_dir = feat_dir * ch_att

        x_enhanced = x + self.fusion(feat_dir)

        return x_enhanced, ori_field

class AdapterModule(nn.Module):

    def __init__(self, in_channels=256, roi_size=7, hidden_dim=256):
        super().__init__()
        self.roi_size = roi_size
        self.in_channels = in_channels

        self.roi_compress = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.delta_angle = nn.Linear(hidden_dim // 2, 2)
        self.delta_radius = nn.Linear(hidden_dim // 2, 1)
        self.delta_width = nn.Linear(hidden_dim // 2, 1)

        nn.init.zeros_(self.delta_angle.weight)
        nn.init.zeros_(self.delta_angle.bias)
        nn.init.zeros_(self.delta_radius.weight)
        nn.init.zeros_(self.delta_radius.bias)
        nn.init.zeros_(self.delta_width.weight)
        nn.init.zeros_(self.delta_width.bias)

    def forward(self, features, vertices, spatial_scale=1.0):

        B, C, H, W = features.shape
        device = features.device

        all_deltas = []

        for b in range(B):
            verts = vertices[b]

            if len(verts) == 0:
                all_deltas.append({
                    'delta_angle': torch.zeros(0, 2, device=device),
                    'delta_radius': torch.zeros(0, 1, device=device),
                    'delta_width': torch.zeros(0, 1, device=device),
                })
                continue

            N = len(verts)

            half_size = self.roi_size / 2
            boxes = torch.zeros(N, 5, device=device)
            boxes[:, 0] = 0
            boxes[:, 1] = verts[:, 0] - half_size
            boxes[:, 2] = verts[:, 1] - half_size
            boxes[:, 3] = verts[:, 0] + half_size
            boxes[:, 4] = verts[:, 1] + half_size

            single_feat = features[b:b+1]
            roi_feats = roi_align(
                single_feat,
                boxes,
                output_size=(self.roi_size, self.roi_size),
                spatial_scale=spatial_scale,
                aligned=True
            )

            compressed = self.roi_compress(roi_feats)
            compressed = compressed.view(N, -1)

            hidden = self.mlp(compressed)

            d_angle = self.delta_angle(hidden)
            d_radius = self.delta_radius(hidden)
            d_width = self.delta_width(hidden)

            all_deltas.append({
                'delta_angle': d_angle,
                'delta_radius': d_radius,
                'delta_width': d_width,
            })

        return all_deltas

class HairTriangleDetectorV3(nn.Module):

    def __init__(
        self,
        backbone='resnet50',
        pretrained=True,
        use_dcn=True,
        use_cm=True,
        use_em=True,
        use_am=True,
        fpn_channels=256,
    ):
        super().__init__()

        self.use_cm = use_cm
        self.use_em = use_em
        self.use_am = use_am

        self.backbone = Backbone(backbone, pretrained)
        self.fpn = DeformableFPN(self.backbone.out_channels, out_channels=fpn_channels, use_dcn=use_dcn)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(fpn_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.heatmap_head[-2].bias, -4.6)

        self.offset_head = nn.Conv2d(fpn_channels, 2, 1)

        if use_cm:
            self.ctx_mod = ContextModule(fpn_channels)
            print(" FCCM () ")

        if use_em:
            self.enc_mod = EncodingModule(fpn_channels)
            print(" OTE () ")

        if use_am:
            self.ada_mod = AdapterModule(fpn_channels)
            print(" TPA () ")

        self.angle_head = nn.Conv2d(fpn_channels, 2, 1)
        self.radius_head = nn.Conv2d(fpn_channels, 1, 1)
        self.width_head = nn.Conv2d(fpn_channels, 1, 1)

        print(f"\n: FCCM={use_cm}, OTE={use_em}, TPA={use_am}")

    def forward(self, x, vertices_gt=None):

        B = x.shape[0]

        features = self.backbone(x)
        fpn_features = self.fpn(features)

        feat1 = fpn_features[0]
        feat2 = F.interpolate(fpn_features[1], size=feat1.shape[2:], mode='bilinear', align_corners=False)
        feat3 = F.interpolate(fpn_features[2], size=feat1.shape[2:], mode='bilinear', align_corners=False)
        F_fused = feat1 + 0.6 * feat2 + 0.3 * feat3

        heatmap_feat = self.heatmap_head(F_fused)
        offset_feat = self.offset_head(F_fused)

        target_size = (x.shape[2] // 2, x.shape[3] // 2)
        heatmap = F.interpolate(heatmap_feat, size=target_size, mode='bilinear', align_corners=False)
        offset = F.interpolate(offset_feat, size=target_size, mode='bilinear', align_corners=False)

        if self.use_cm:
            F_fccm = self.ctx_mod(F_fused, heatmap_feat)
        else:
            F_fccm = F_fused

        ori_field = None
        if self.use_em:
            F_ote, ori_field_feat = self.enc_mod(F_fccm)
            ori_field = F.interpolate(ori_field_feat, size=target_size, mode='bilinear', align_corners=False)
            ori_field = F.normalize(ori_field, p=2, dim=1)
        else:
            F_ote = F_fccm

        angle_feat = self.angle_head(F_ote)
        radius_feat = F.relu(self.radius_head(F_ote))
        width_feat = F.relu(self.width_head(F_ote))

        angle_sincos = F.interpolate(angle_feat, size=target_size, mode='bilinear', align_corners=False)
        angle_sincos = F.normalize(angle_sincos, p=2, dim=1)
        radius = F.interpolate(radius_feat, size=target_size, mode='bilinear', align_corners=False)
        width = F.interpolate(width_feat, size=target_size, mode='bilinear', align_corners=False)

        tpa_deltas = None
        if self.use_am:

            if vertices_gt is not None:

                scale_x = F_ote.shape[3] / x.shape[3]
                scale_y = F_ote.shape[2] / x.shape[2]
                vertices_feat = []
                for verts in vertices_gt:
                    if len(verts) > 0:
                        verts_scaled = torch.tensor(verts, device=x.device, dtype=torch.float32)
                        verts_scaled[:, 0] *= scale_x
                        verts_scaled[:, 1] *= scale_y
                        vertices_feat.append(verts_scaled)
                    else:
                        vertices_feat.append(torch.zeros(0, 2, device=x.device))

                tpa_deltas = self.ada_mod(F_ote, vertices_feat, spatial_scale=1.0)

        outputs = {
            'heatmap': heatmap,
            'offset': offset,
            'angle_sincos': angle_sincos,
            'radius': radius,
            'width': width,
        }

        if ori_field is not None:
            outputs['ori_field'] = ori_field

        if tpa_deltas is not None:
            outputs['tpa_deltas'] = tpa_deltas

        return outputs

    def inference(self, x, threshold=0.3, nms_kernel=5):

        self.eval()
        with torch.no_grad():

            outputs = self.forward(x, vertices_gt=None)

            heatmap = outputs['heatmap']
            B = heatmap.shape[0]

            results = []

            for b in range(B):
                hm = heatmap[b, 0].cpu().numpy()

                from scipy.ndimage import maximum_filter
                local_max = (hm == maximum_filter(hm, size=nms_kernel))
                peaks = (hm > threshold) & local_max

                y_coords, x_coords = peaks.nonzero()
                confidences = hm[peaks]

                if len(x_coords) == 0:
                    results.append({
                        'centers': np.zeros((0, 2)),
                        'angles': np.zeros(0),
                        'radii': np.zeros(0),
                        'widths': np.zeros(0),
                        'confidences': np.zeros(0),
                    })
                    continue

                angle_sin = outputs['angle_sincos'][b, 0].cpu().numpy()
                angle_cos = outputs['angle_sincos'][b, 1].cpu().numpy()
                radius_map = outputs['radius'][b, 0].cpu().numpy()
                width_map = outputs['width'][b, 0].cpu().numpy()

                angles = np.arctan2(angle_sin[y_coords, x_coords], angle_cos[y_coords, x_coords])
                radii = radius_map[y_coords, x_coords]
                widths = width_map[y_coords, x_coords]

                if self.use_am:

                    vertices_pred = [torch.tensor(np.stack([x_coords, y_coords], axis=1),
                                                   device=x.device, dtype=torch.float32)]

                    scale = outputs['heatmap'].shape[2] / x.shape[2] * 2

                results.append({
                    'centers': np.stack([x_coords, y_coords], axis=1),
                    'angles': angles,
                    'radii': radii,
                    'widths': widths,
                    'confidences': confidences,
                })

        return results

if __name__ == '__main__':
    print("Testing HairTriangleDetectorV3...")

    configs = [
        {'use_cm': False, 'use_em': False, 'use_am': False},
        {'use_cm': True, 'use_em': False, 'use_am': False},
        {'use_cm': True, 'use_em': True, 'use_am': False},
        {'use_cm': True, 'use_em': True, 'use_am': True},
    ]

    x = torch.randn(2, 3, 512, 512)
    vertices_gt = [
        np.array([[100, 100], [200, 200], [300, 150]]),
        np.array([[50, 50]]),
    ]

    for config in configs:
        print(f"\n{'='*50}")
        print(f"Config: {config}")
        print('='*50)

        model = HairTriangleDetectorV3(
            backbone='resnet50',
            pretrained=False,
            use_dcn=False,
            **config
        )

        outputs = model(x, vertices_gt=vertices_gt)

        print("\nOutput shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"  {key}: list of {len(value)} items")
                if len(value) > 0 and isinstance(value[0], dict):
                    for k, v in value[0].items():
                        print(f"    - {k}: {v.shape}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
