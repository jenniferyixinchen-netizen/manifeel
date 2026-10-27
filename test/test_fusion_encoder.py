"""
Offline checks for the Right TacRGB + TacFF fusion config.
No Isaac Gym, no simulation, no training: YAML composition, normalizer chain,
encoder / UNet shapes and one forward pass on a few real samples.

Usage (from repo root, diffusion_policy source on PYTHONPATH):
    PYTHONPATH=/path/to/diffusion_policy:. python test/test_fusion_encoder.py --zarr data/usb_mini.zarr
"""
import argparse
import copy
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torchvision
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

OmegaConf.register_new_resolver("eval", eval, replace=True)
CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1].joinpath('manifeel', 'config'))
TACFF = 'tactile_force_field_right'
TACRGB = 'right_tactile_camera_taxim'


def load_cfg(config_name, overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def check(cond, msg):
    print(('[PASS] ' if cond else '[FAIL] ') + msg)
    assert cond, msg


def capture_resnet_inputs(encoder):
    # record the tensor that enters conv1 of each per-key ResNet
    captured = dict()
    for key, model in encoder.key_model_map.items():
        def hook(module, inputs, key=key):
            captured[key] = inputs[0].detach()
        model.conv1.register_forward_pre_hook(hook)
    return captured


def main(args):
    overrides = [f'dataset_path={args.zarr}', 'training.device=cpu']

    # ---------------- 1. YAML ----------------
    cfg = load_cfg('train_diffusion_fusion_workspace', overrides)
    obs_meta = cfg.task.shape_meta.obs
    check(list(obs_meta.keys()) == ['wrist', TACRGB, TACFF, 'state'],
          f'fusion shape_meta obs keys = {list(obs_meta.keys())}')
    check(list(obs_meta[TACFF].shape) == [3, 10, 14] and obs_meta[TACFF].type == 'rgb',
          'TacFF shape [3,10,14], type rgb')
    check(list(obs_meta[TACRGB].shape) == [3, 256, 256], 'TacRGB shape [3,256,256]')
    check(cfg.policy.obs_encoder._target_ == 'manifeel.model.fusion_obs_encoder.FusionMultiImageObsEncoder',
          'policy uses FusionMultiImageObsEncoder')
    check(list(cfg.policy.obs_encoder.no_imagenet_norm_keys) == [TACFF],
          'no_imagenet_norm_keys resolves to [tactile_force_field_right]')
    check(cfg.policy.obs_encoder.share_rgb_model is False and cfg.policy.obs_encoder.rgb_model.name == 'resnet18',
          'share_rgb_model=False, rgb_model=resnet18')
    check(list(cfg.task.env_runner.tactile_size) == [256, 256], 'env_runner tactile_size matches TacRGB shape')

    # ---------------- 2. Dataset ----------------
    dataset = instantiate(cfg.task.dataset)
    n_ep = dataset.replay_buffer.n_episodes
    n_train = int(dataset.train_mask.sum())
    print(f'episodes={n_ep}, train={n_train}, val={n_ep - n_train}, samples={len(dataset)}')
    check(n_train < n_ep, 'at least one episode held out for validation')

    raw = dataset.sampler.sample_sequence(0)
    expected_ff = np.moveaxis(raw[TACFF][:dataset.n_obs_steps], -1, 1)
    item = dataset[0]
    for key, attr in obs_meta.items():
        check(tuple(item['obs'][key].shape) == (dataset.n_obs_steps, *attr.shape),
              f'dataset obs {key} shape {tuple(item["obs"][key].shape)}')
    check(np.array_equal(item['obs'][TACFF].numpy(), expected_ff),
          'dataset TacFF equals raw zarr values (only HWC->CHW, no rescaling)')
    check(float(item['obs'][TACRGB].min()) >= 0 and float(item['obs'][TACRGB].max()) <= 1,
          'dataset TacRGB still in [0,1]')

    # ---------------- 3. Normalizer ----------------
    normalizer = dataset.get_normalizer()
    ff_all = dataset.replay_buffer[TACFF][:]
    train_idx = dataset._train_frame_indices()
    train_ff = ff_all[train_idx].reshape(-1, 3).astype(np.float64)
    all_ff = ff_all.reshape(-1, 3).astype(np.float64)
    stats = normalizer[TACFF].get_input_stats()
    mean_c = stats['mean'].detach().numpy().reshape(3, 10, 14)[:, 0, 0]
    std_c = stats['std'].detach().numpy().reshape(3, 10, 14)[:, 0, 0]
    print('TacFF train mean per channel:', mean_c, '\nTacFF train std  per channel:', std_c)
    print('TacFF all-episode mean      :', all_ff.mean(0), '\nTacFF all-episode std       :', all_ff.std(0))
    check(np.allclose(mean_c, train_ff.mean(0), rtol=1e-5, atol=1e-12)
          and np.allclose(std_c, train_ff.std(0), rtol=1e-5, atol=1e-12),
          'TacFF stats equal stats of train episodes only')
    check(not np.allclose(std_c, all_ff.std(0), rtol=1e-6, atol=0),
          'TacFF stats differ from all-episode stats (val excluded)')
    if (std_c < 1e-4).any():
        print('note: some TacFF channel std < 1e-4; upstream LinearNormalizer.fit(gaussian) '
              'would have treated it as constant (range_eps=1e-4)')

    nff = normalizer[TACFF].normalize(torch.from_numpy(np.moveaxis(ff_all[train_idx], -1, 1))).detach()
    nff_c = nff.permute(1, 0, 2, 3).reshape(3, -1)
    print('normalized train TacFF mean:', nff_c.mean(1).numpy(), 'std:', nff_c.std(1).numpy())
    check(torch.allclose(nff_c.mean(1), torch.zeros(3), atol=1e-3)
          and torch.allclose(nff_c.std(1), torch.ones(3), atol=1e-3),
          'normalized train TacFF has per-channel mean 0 / std 1')

    # layout check: a tensor whose channel c is constant v_c must map to (v_c-mean_c)/std_c
    v = torch.tensor([1e-3, -2e-3, 3e-3])
    x = v.view(1, 1, 3, 1, 1).expand(2, 2, 3, 10, 14).contiguous()
    y = normalizer[TACFF].normalize(x).detach()
    expected = ((v.double() - torch.from_numpy(mean_c)) / torch.from_numpy(std_c)).float()
    check(torch.allclose(y, expected.view(1, 1, 3, 1, 1).expand_as(y), rtol=1e-4),
          'per-channel scale/offset aligned with channel-first (B,T,C,H,W) layout')
    for key in ['wrist', TACRGB]:
        p = normalizer[key].params_dict
        check(p['scale'].tolist() == [2.0] and p['offset'].tolist() == [-1.0],
              f'{key} keeps image range normalizer (x*2-1)')

    # ---------------- 4. Encoder / UNet shapes ----------------
    policy = instantiate(cfg.policy)
    policy.set_normalizer(normalizer)
    policy.eval()
    enc = policy.obs_encoder
    check(enc.rgb_keys == sorted(['wrist', TACRGB, TACFF]) and enc.low_dim_keys == ['state'],
          f'encoder rgb_keys={enc.rgb_keys}, low_dim_keys={enc.low_dim_keys}')
    for key in enc.rgb_keys:
        last = enc.key_transform_map[key][-1]
        if key == TACFF:
            check(isinstance(last, nn.Identity), f'{key}: no ImageNet normalization in encoder')
        else:
            check(isinstance(last, torchvision.transforms.Normalize), f'{key}: ImageNet normalization kept')
    ptrs = {k: m.conv1.weight.data_ptr() for k, m in enc.key_model_map.items()}
    check(len(set(ptrs.values())) == 3, 'TacRGB, TacFF, wrist use 3 independent ResNet18 instances')
    with torch.no_grad():
        for key in enc.rgb_keys:
            out = enc.key_model_map[key](torch.zeros(1, *obs_meta[key].shape))
            check(out.shape == (1, 512), f'{key} ResNet18 feature dim {tuple(out.shape)}')
    obs_dim = enc.output_shape()[0]
    check(obs_dim == 3 * 512 + 7, f'obs feature dim = {obs_dim} (3*512 + 7)')
    cond_linear = policy.model.down_modules[0][0].cond_encoder[1]
    global_cond_dim = cond_linear.in_features - cfg.policy.diffusion_step_embed_dim
    check(global_cond_dim == obs_dim * cfg.n_obs_steps,
          f'UNet global_cond_dim = {global_cond_dim} = {obs_dim} * n_obs_steps({cfg.n_obs_steps})')

    # ---------------- 5. Forward on real samples ----------------
    captured = capture_resnet_inputs(enc)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)))
    with torch.no_grad():
        nobs = policy.normalizer.normalize(batch['obs'])
        flat = {k: v[:, :cfg.n_obs_steps].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
        feat = enc(flat)
        check(feat.shape == (4 * cfg.n_obs_steps, obs_dim) and torch.isfinite(feat).all(),
              f'encoder forward on real batch -> {tuple(feat.shape)}, finite')
        loss = policy.compute_loss(batch)
        check(torch.isfinite(loss), f'policy.compute_loss on real batch = {loss.item():.4f}')
        policy.num_inference_steps = 2
        action = policy.predict_action({k: v[:, :cfg.n_obs_steps] for k, v in batch['obs'].items()})['action']
        check(action.shape == (4, cfg.n_action_steps, 6), f'predict_action shape {tuple(action.shape)}')
    for key in enc.rgb_keys:
        x = captured[key]
        print(f'ResNet input {key:28s} mean={x.mean():+.3f} std={x.std():.3f} '
              f'min={x.min():+.3f} max={x.max():+.3f}')
    # TacFF has many zero (no-contact) taxels, so samples need not be exactly N(0,1)
    check(captured[TACFF].std() > 0.1, 'TacFF reaches ResNet with non-collapsed spread')

    # ---------------- 6. Baselines for comparison (configs untouched) ----------------
    for task in ['visff_wrist', 'vistac_wrist']:
        bcfg = load_cfg('train_diffusion_workspace', overrides + [f'task={task}'])
        bpolicy = instantiate(bcfg.policy)
        bdim = bpolicy.obs_encoder.output_shape()[0]
        print(f'baseline {task}: obs feature dim {bdim}, global_cond_dim {bdim * bcfg.n_obs_steps}')
        if task == 'visff_wrist':
            bds = instantiate(bcfg.task.dataset)
            bpolicy.set_normalizer(bds.get_normalizer())
            bcap = capture_resnet_inputs(bpolicy.obs_encoder)
            with torch.no_grad():
                bpolicy.compute_loss(next(iter(DataLoader(bds, batch_size=4, num_workers=0))))
            x = bcap[TACFF]
            per_c = x.permute(1, 0, 2, 3).reshape(3, -1)
            print(f'baseline visff TacFF ResNet input per-channel std: {per_c.std(1).numpy()} '
                  f'(range {x.min():+.3f}..{x.max():+.3f})')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr', required=True)
    main(parser.parse_args())
