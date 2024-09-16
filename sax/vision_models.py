import dataclasses
import logging
import os

import beartype
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

logger = logging.getLogger(__name__)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class EncodedImgBatch:
    """The output of a `VisionBackbone`'s `img_encode()` method."""

    img_features: Float[Tensor, "batch img_dim"]
    """Image-level features. Each image is represented by a single vector."""
    patch_features: Float[Tensor, "batch n_patches patch_dim"] | None
    """Patch-level features. Only ViTs have patch-level features. These features might be a different dimension that the image features because of projection heads or such."""


@jaxtyped(typechecker=beartype.beartype)
class VisionBackbone(torch.nn.Module):
    """ """

    @jaxtyped(typechecker=beartype.beartype)
    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> EncodedImgBatch:
        err_msg = f"{self.__class__.__name__} must implemented img_encode()."
        raise NotImplementedError(err_msg)

    def make_img_transform(self):
        err_msg = f"{self.__class__.__name__} must implemented make_img_transform()."
        raise NotImplementedError(err_msg)


def get_cache_dir() -> str:
    cache_dir = ""
    for var in ("BIOBENCH_CACHE", "HF_HOME", "HF_HUB_CACHE"):
        cache_dir = cache_dir or os.environ.get(var, "")
    return cache_dir or "."


@jaxtyped(typechecker=beartype.beartype)
class OpenClip(VisionBackbone):
    def __init__(self, ckpt: str, **kwargs):
        super().__init__()
        import open_clip

        if ckpt.startswith("hf-hub:"):
            clip, self._img_transform = open_clip.create_model_from_pretrained(ckpt)
        else:
            arch, ckpt = ckpt.split("/")
            clip, self._img_transform = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=get_cache_dir()
            )

        self.model = clip.visual
        self.model.output_tokens = True  # type: ignore

    def make_img_transform(self):
        return self._img_transform

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> EncodedImgBatch:
        result = self.model(batch)
        # Sometimes the model does not return patch features if it has none.
        if isinstance(result, tuple):
            img, patches = result
            return EncodedImgBatch(img, patches)
        else:
            return EncodedImgBatch(result, None)


class TimmVit(VisionBackbone):
    @jaxtyped(typechecker=beartype.beartype)
    def __init__(self, ckpt: str, **kwargs):
        super().__init__()
        import timm

        err_msg = "You are trying to load a non-ViT checkpoint; the `img_encode()` method assumes `model.forward_features()` will return features with shape (batch, n_patches, dim) which is not true for non-ViT checkpoints."
        assert "vit" in ckpt, err_msg
        self.model = timm.create_model(ckpt, pretrained=True)

        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.img_transform = timm.data.create_transform(**data_cfg)

    def make_img_transform(self):
        return self.img_transform

    @jaxtyped(typechecker=beartype.beartype)
    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> EncodedImgBatch:
        patches = self.model.forward_features(batch)
        # Use [CLS] token if it exists, otherwise do a maxpool
        if self.model.num_prefix_tokens > 0:
            img = patches[:, 0, ...]
        else:
            img = patches.max(axis=1).values

        # Remove all non-image patches, like the [CLS] token or registers
        patches = patches[:, self.model.num_prefix_tokens :, ...]

        return EncodedImgBatch(img, patches)


_global_backbone_registry: dict[str, type[VisionBackbone]] = {}


@beartype.beartype
def load_vision_backbone(model_org: str, ckpt: str) -> VisionBackbone:
    """
    Load a pretrained vision backbone.
    """
    if model_org not in _global_backbone_registry:
        raise ValueError(f"Org '{model_org}' not found.")

    cls = _global_backbone_registry[model_org]
    return cls(ckpt)


def register_vision_backbone(model_org: str, cls: type[VisionBackbone]):
    """
    Register a new vision backbone class.
    """
    if model_org in _global_backbone_registry:
        logger.warning("Overwriting key '%s' in registry.", model_org)
    _global_backbone_registry[model_org] = cls


def list_vision_backbones() -> list[str]:
    """
    List all vision backbone model orgs.
    """
    return list(_global_backbone_registry.keys())


register_vision_backbone("timm-vit", TimmVit)
register_vision_backbone("open-clip", OpenClip)
