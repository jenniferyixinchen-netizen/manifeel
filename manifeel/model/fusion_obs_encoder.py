from typing import Sequence
import torch.nn as nn

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


class FusionMultiImageObsEncoder(MultiImageObsEncoder):
    """
    MultiImageObsEncoder (independent ResNet18 per rgb key + concat with low_dim)
    that skips ImageNet normalization for the keys in no_imagenet_norm_keys.

    Used for TacFF, which is already standardized per channel by the dataset
    normalizer and is not an RGB image.
    """
    def __init__(self,
            shape_meta: dict,
            rgb_model,
            no_imagenet_norm_keys: Sequence[str] = (),
            **kwargs
        ):
        super().__init__(shape_meta=shape_meta, rgb_model=rgb_model, **kwargs)
        self.no_imagenet_norm_keys = list(no_imagenet_norm_keys)
        for key in self.no_imagenet_norm_keys:
            assert key in self.rgb_keys, f"{key} is not an rgb key"
            resizer, randomizer, _ = self.key_transform_map[key]
            assert isinstance(randomizer, nn.Identity), \
                f"crop is not supported for {key}"
            self.key_transform_map[key] = nn.Sequential(resizer, randomizer, nn.Identity())
