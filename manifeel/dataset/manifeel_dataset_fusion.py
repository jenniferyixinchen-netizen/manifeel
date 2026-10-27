from typing import Sequence
import numpy as np

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from manifeel.dataset.manifeel_dataset_equidp import ManifeelDataset


class ManifeelFusionDataset(ManifeelDataset):
    """
    ManifeelDataset for TacRGB + TacFF fusion.

    Identical to ManifeelDataset except for the normalizer of the tactile
    force field (TacFF) keys: instead of the image range normalizer (x*2-1),
    which assumes [0, 1] input and almost erases the ~1e-3 scale TacFF signal,
    TacFF is standardized per channel (normal, shear x, shear y) with mean/std
    computed only from the training episodes.

    Image keys (vision RGB, TacRGB) keep the original image range normalizer.

    lowdim_fit_on selects the frames used to fit the state/action normalizer:
    'all' fits on every episode in the zarr, same as ManifeelDataset (baseline);
    'train' fits only on the training episodes, excluding the validation
    episodes and the episodes dropped by max_train_episodes.
    """
    def __init__(self,
            shape_meta: dict,
            zarr_path: str,
            tacff_keys: Sequence[str] = ('tactile_force_field_right',),
            tacff_std_eps: float = 1e-8,
            lowdim_fit_on: str = 'all',
            **kwargs
            ):
        assert lowdim_fit_on in ('all', 'train'), \
            f"lowdim_fit_on must be 'all' or 'train', got {lowdim_fit_on}"
        super().__init__(shape_meta=shape_meta, zarr_path=zarr_path, **kwargs)
        tacff_keys = list(tacff_keys)
        for key in tacff_keys:
            assert key in self.rgb_keys, f"{key} must be an rgb key in shape_meta"
        self.tacff_keys = tacff_keys
        self.tacff_std_eps = tacff_std_eps
        self.lowdim_fit_on = lowdim_fit_on

    def _train_frame_indices(self) -> np.ndarray:
        # frame indices of the episodes used for training
        # (after val split and max_train_episodes downsampling)
        episode_ends = self.replay_buffer.episode_ends[:]
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        idxs = [np.arange(start, end) for start, end, is_train
                in zip(episode_starts, episode_ends, self.train_mask) if is_train]
        return np.concatenate(idxs)

    def get_tacff_stats(self, key: str) -> dict:
        # per-channel stats over train frames and all taxels, data layout (N, H, W, C)
        data = self.replay_buffer[key][:][self._train_frame_indices()].astype(np.float64)
        C = data.shape[-1]
        data = data.reshape(-1, C)
        return {
            'min': data.min(axis=0),
            'max': data.max(axis=0),
            'mean': data.mean(axis=0),
            'std': data.std(axis=0),
        }

    def get_tacff_normalizer(self, key: str) -> SingleFieldLinearNormalizer:
        stat = self.get_tacff_stats(key)
        C, H, W = self.shape_meta['obs'][key]['shape']
        assert len(stat['mean']) == C

        std = stat['std'].copy()
        ignore = std < self.tacff_std_eps
        std[ignore] = 1.0
        scale_c = 1.0 / std
        offset_c = -stat['mean'] * scale_c

        # LinearNormalizer flattens the input to (-1, len(scale)), so per-channel
        # params must be expanded to the full channel-first (C, H, W) obs layout.
        def expand(x):
            return np.broadcast_to(
                x.reshape(C, 1, 1), (C, H, W)).reshape(-1).astype(np.float32)

        return SingleFieldLinearNormalizer.create_manual(
            scale=expand(scale_c),
            offset=expand(offset_c),
            input_stats_dict={k: expand(v) for k, v in stat.items()}
        )

    def get_normalizer(self, mode='limits', **kwargs):
        if self.lowdim_fit_on == 'train':
            # normalizer for action and state, fit on training episodes only
            idx = self._train_frame_indices()
            data = {
                'action': self.replay_buffer['action'][:][idx],
                'state': self.replay_buffer['state'][:][idx]
            }
            normalizer = LinearNormalizer()
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        else:
            normalizer = super().get_normalizer(mode=mode, **kwargs)
        for key in self.rgb_keys:
            if key in self.tacff_keys:
                normalizer[key] = self.get_tacff_normalizer(key)
            else:
                normalizer[key] = get_image_range_normalizer()
        return normalizer
