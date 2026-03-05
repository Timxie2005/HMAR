# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import os
import torch
from utils.sampling_arg_util import Args, get_args
from models import HMAR
import torch

device = 'cuda'

def build_everything(args: Args):
    from models import VQVAE, build_vae_hmar

    vae_local, hmar = build_vae_hmar(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=args.patch_nums,
        num_classes=1000,
        depth=args.depth,
        shared_aln=args.saln,
        attn_l2_norm=args.anorm,
        flash_if_available=args.fuse,
        fused_if_available=args.fuse,
    )

    vae_ckpt = os.path.join(".", "vae_ch160v4096z32.pth")
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location="cpu", weights_only=True), strict=True)

    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    hmar: HMAR = args.compile_model(hmar, args.tfast)

    return hmar


def _extract_state_dict(raw_ckpt):
    if isinstance(raw_ckpt, dict) and "trainer" in raw_ckpt and isinstance(raw_ckpt["trainer"], dict):
        trainer_state = raw_ckpt["trainer"]
        if "transformer_wo_ddp" in trainer_state:
            return trainer_state["transformer_wo_ddp"]
        if "state_dict" in trainer_state:
            return trainer_state["state_dict"]
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt and isinstance(raw_ckpt["state_dict"], dict):
        return raw_ckpt["state_dict"]
    return raw_ckpt


def _load_hmar_checkpoint(hmar: HMAR, ckpt_path: str, args: Args):
    state = _extract_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {ckpt_path}")

    keys = list(state.keys())
    has_split_hmar_keys = any(k.startswith(("base_blocks.", "ns_blocks.", "mask_blocks.")) for k in keys)
    has_legacy_hmar_keys = any(k.startswith("blocks.") for k in keys)
    has_mask_weights = any(k.startswith(("mask_blocks.", "mask_head", "mask_embed")) for k in keys)

    if has_split_hmar_keys:
        hmar.load_state_dict(state, strict=True)
        return

    if has_legacy_hmar_keys:
        hmar.load_base_and_ns_state_dict(state)
        if args.mask and not has_mask_weights:
            print("[warn] Checkpoint has no mask branch weights; switching mask=False for sampling.")
            args.mask = False
        return

    hmar.load_state_dict(state, strict=False)


if __name__ == "__main__":
    args: Args = get_args(cfg_folder='sample')
    hmar = build_everything(args)
    torch.set_default_device(device)

    hmar.eval()
    _load_hmar_checkpoint(hmar, f"{args.checkpoint}.pth", args)
    
    class_id = 3
    b = 8
    seed = 13
    
    with torch.inference_mode():
        imgs = hmar.generate(
            b,
            class_id,
            g_seed=seed,
            num_samples=1,
            top_k=args.top_k,
            top_p=args.top_p,
            cfg=args.cfg,
            more_smooth=args.more_smooth,
            mask=args.mask,
            mask_schedule=args.mask_schedule
        )
        from torchvision.utils import save_image

    save_image(imgs, "sample_hmar.png", nrow=4)
