from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import BackboneEncoder
from .cbam import CBAMBlock
from .local_branch import LocalBranch


class IdentityMarkerMultipliers(nn.Module):
    """论文 SIMBA 中的可学习 identity multipliers。"""

    def __init__(self, *, use_gender: bool, use_chronological: bool) -> None:
        super().__init__()
        self.use_gender = use_gender
        self.use_chronological = use_chronological
        self.gender_multiplier = nn.Parameter(torch.ones(1))
        self.chronological_multiplier = nn.Parameter(torch.ones(1))

    def forward(self, male: torch.Tensor, chronological: torch.Tensor) -> torch.Tensor:
        features = []
        if self.use_gender:
            features.append(male * self.gender_multiplier)
        if self.use_chronological:
            features.append(chronological * self.chronological_multiplier)
        if not features:
            raise ValueError("IdentityMarkerMultipliers 至少需要启用一种元信息输入。")
        return torch.cat(features, dim=-1)


class HeatmapGuidance(nn.Module):
    """用全局 heatmap 对 backbone 特征做论文式 ROI 引导。"""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        hidden = min(128, max(16, out_channels // 8))
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1, bias=False),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, feature_map: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        resized_heatmap = F.interpolate(
            heatmap,
            size=feature_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        attention = torch.sigmoid(self.encoder(resized_heatmap))
        return feature_map * (1.0 + self.scale * attention)


class MetadataEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        meta_cfg = config["model"]["metadata"]
        self.enabled = bool(meta_cfg["enabled"])
        self.mode = meta_cfg.get("mode", "simba_hybrid")
        self.use_mlp = self.mode in {"mlp", "simba_hybrid"}
        self.use_multiplier = self.mode in {"simba_multiplier", "simba_hybrid"}
        self.use_gender = bool(meta_cfg.get("use_gender", True))
        self.use_chronological = bool(meta_cfg.get("use_chronological", True))
        self.output_dim = 0
        if not self.enabled:
            return
        if not (self.use_gender or self.use_chronological):
            raise ValueError("model.metadata.enabled=true 时至少启用一种元信息输入。")

        if self.use_mlp:
            mlp_input_dim = 0
            if self.use_gender:
                gender_dim = int(meta_cfg["gender_embedding_dim"])
                self.gender_embedding = nn.Embedding(2, gender_dim)
                mlp_input_dim += gender_dim
            else:
                self.gender_embedding = None
            if self.use_chronological:
                chrono_dim = int(meta_cfg["chronological_hidden_dim"])
                self.chronological_proj = nn.Sequential(
                    nn.Linear(1, chrono_dim),
                    nn.ReLU(inplace=True),
                )
                mlp_input_dim += chrono_dim
            else:
                self.chronological_proj = None
            self.encoder = nn.Sequential(
                nn.Linear(mlp_input_dim, int(meta_cfg["hidden_dim"])),
                nn.ReLU(inplace=True),
                nn.Dropout(meta_cfg["dropout"]),
            )
            self.output_dim += int(meta_cfg["hidden_dim"])
        else:
            self.gender_embedding = None
            self.chronological_proj = None
            self.encoder = None

        if self.use_multiplier:
            self.multipliers = IdentityMarkerMultipliers(
                use_gender=self.use_gender,
                use_chronological=self.use_chronological,
            )
            self.output_dim += int(self.use_gender) + int(self.use_chronological)
        else:
            self.multipliers = None

        if self.output_dim <= 0:
            raise ValueError(f"不支持的 model.metadata.mode={self.mode}，或当前元信息输入选择无法生成特征。")

    def forward(
        self,
        male: torch.Tensor,
        male_index: torch.Tensor,
        chronological: torch.Tensor,
        chronological_input: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.enabled:
            return None
        features = []
        if self.use_multiplier:
            features.append(self.multipliers(male, chronological))
        if self.use_mlp:
            mlp_features = []
            if self.use_gender:
                mlp_features.append(self.gender_embedding(male_index))
            if self.use_chronological:
                mlp_features.append(self.chronological_proj(chronological_input))
            features.append(self.encoder(torch.cat(mlp_features, dim=-1)))
        return torch.cat(features, dim=-1)


class FusionHead(nn.Module):
    def __init__(self, visual_dim: int, metadata_dim: int, config: dict) -> None:
        super().__init__()
        head_cfg = config["model"]["head"]
        self.metadata_dim = metadata_dim
        if metadata_dim > 0:
            self.gate = nn.Linear(metadata_dim, visual_dim)
            fusion_input_dim = visual_dim * 2 + metadata_dim
        else:
            self.gate = None
            fusion_input_dim = visual_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, int(head_cfg["hidden_dim"])),
            nn.ReLU(inplace=True),
            nn.Dropout(head_cfg["dropout"]),
            nn.Linear(int(head_cfg["hidden_dim"]), int(head_cfg["hidden_dim"] // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(head_cfg["dropout"]),
        )
        self.regressor = nn.Linear(int(head_cfg["hidden_dim"] // 2), 1)

    def forward(self, visual_features: torch.Tensor, metadata: torch.Tensor | None) -> torch.Tensor:
        if metadata is not None:
            gated = torch.sigmoid(self.gate(metadata)) * visual_features
            fusion_input = torch.cat([visual_features, gated, metadata], dim=-1)
        else:
            fusion_input = visual_features
        fused = self.fusion(fusion_input)
        return self.regressor(fused)


class SingleBackboneModel(nn.Module):
    def __init__(self, backbone_name: str, config: dict) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.config = copy.deepcopy(config)
        self.branch_mode = self.config["model"]["branch_mode"]
        pretrained = bool(self.config["model"]["pretrained"])

        self.global_encoder = BackboneEncoder(backbone_name, pretrained=pretrained)
        cbam_cfg = self.config["model"]["cbam"]
        self.global_cbam = None
        if cbam_cfg["enabled"] and cbam_cfg["global_branch"]:
            self.global_cbam = CBAMBlock(self.global_encoder.out_channels)
        heatmap_cfg = self.config["model"]["heatmap_guidance"]
        self.heatmap_guidance = None
        if heatmap_cfg["enabled"]:
            self.heatmap_guidance = HeatmapGuidance(self.global_encoder.out_channels)

        self.global_proj = nn.Sequential(
            nn.Linear(self.global_encoder.out_channels, int(self.config["model"]["global_dim"])),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config["model"]["head"]["dropout"]),
        )

        self.metadata_encoder = MetadataEncoder(self.config)
        self.local_branch = LocalBranch(self.config, metadata_dim=self.metadata_encoder.output_dim)

        visual_dim = 0
        if self.branch_mode in {"global_only", "global_local"}:
            visual_dim += int(self.config["model"]["global_dim"])
        if self.branch_mode in {"local_only", "global_local"}:
            visual_dim += int(self.config["model"]["local_branch"]["feature_dim"])
        self.head = FusionHead(visual_dim=visual_dim, metadata_dim=self.metadata_encoder.output_dim, config=self.config)

    def _encode_global(self, image: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        feature_map = self.global_encoder.forward_features(image)
        if self.heatmap_guidance is not None:
            feature_map = self.heatmap_guidance(feature_map, heatmap)
        if self.global_cbam is not None:
            feature_map = self.global_cbam(feature_map)
        pooled = self.global_encoder.pool(feature_map).flatten(1)
        return self.global_proj(pooled)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        metadata = self.metadata_encoder(
            batch["male"],
            batch["male_index"],
            batch["chronological"],
            batch["chronological_input"],
        )

        visual_parts = []
        if self.branch_mode in {"global_only", "global_local"}:
            visual_parts.append(self._encode_global(batch["global_image"], batch["global_heatmap"]))
        if self.branch_mode in {"local_only", "global_local"}:
            visual_parts.append(
                self.local_branch(
                    local_images=batch["local_images"],
                    local_heatmaps=batch["local_heatmaps"],
                    patch_mask=batch["local_mask"],
                    roi_vector=batch["roi_vector"],
                    metadata_context=metadata,
                )
            )
        visual_features = torch.cat(visual_parts, dim=-1)
        prediction = self.head(visual_features, metadata)
        return {"prediction": prediction}


class EnsembleBoneAgeModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        ensemble_mode = config["model"]["ensemble_mode"]
        self.ensemble_mode = ensemble_mode

        if ensemble_mode == "resnet":
            self.resnet = SingleBackboneModel(config["model"]["resnet_name"], config)
            self.efficientnet = None
        elif ensemble_mode == "efficientnet":
            self.resnet = None
            self.efficientnet = SingleBackboneModel(config["model"]["efficientnet_name"], config)
        else:
            self.resnet = SingleBackboneModel(config["model"]["resnet_name"], config)
            self.efficientnet = SingleBackboneModel(config["model"]["efficientnet_name"], config)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        predictions = []
        outputs = {}
        if self.resnet is not None:
            # Clone tiny prediction tensors so sequential compiled submodel calls
            # do not alias the same CUDA Graph-managed output buffer.
            resnet_output = self.resnet(batch)["prediction"].clone()
            outputs["resnet_prediction"] = resnet_output
            predictions.append(resnet_output)
        if self.efficientnet is not None:
            efficientnet_output = self.efficientnet(batch)["prediction"].clone()
            outputs["efficientnet_prediction"] = efficientnet_output
            predictions.append(efficientnet_output)

        stacked = torch.stack(predictions, dim=0)
        outputs["prediction"] = stacked.mean(dim=0)
        return outputs


def build_model(config: dict) -> nn.Module:
    return EnsembleBoneAgeModel(config)
