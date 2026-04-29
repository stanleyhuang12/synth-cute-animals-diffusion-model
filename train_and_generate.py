import argparse
import copy
import json
import os
import pathlib
import random
import shutil

import imageio.v2 as imageio
import numpy as np
import skimage
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.utils import make_image_grid
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

IMAGE_SIZE = 64
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH  = pathlib.Path("/projectnb/dl4ds/materials/datasets/synth-cute")


@dataclass
class TrainingConfig:
    image_size:          int   = IMAGE_SIZE
    train_batch_size:    int   = 64
    num_epochs:          int   = 160
    learning_rate:       float = 1e-4
    num_warmup_steps:    int   = 1000
    save_image_epochs:   int   = 10
    num_train_timesteps: int   = 1000
    grad_clip:           float = 1.0
    ema_decay:           float = 0.9999
    val_split:           float = 0.1    # 10% held out for validation
    model_output_dir:    str   = "project4"
    model_output_name:   str   = "best_model.pt"
    meta_name:           str   = "train_meta.json"


config = TrainingConfig()


# Training transform — augmentations on
train_preprocess = transforms.Compose([
    transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    transforms.ColorJitter(brightness=0.25, contrast=0.3, hue=0.15, saturation=0.2),
    transforms.RandomPosterize(bits=6, p=0.05),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAutocontrast(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Validation transform — no augmentations, just resize + normalize
val_preprocess = transforms.Compose([
    transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class CuteDataSet(torch.utils.data.Dataset):
    def __init__(self, root, image_size=IMAGE_SIZE):
        self.paths      = list(pathlib.Path(root).glob("*/*.png"))
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]   # just return path; preprocessing done in subset


class _PreprocessSubset(torch.utils.data.Dataset):
    """
    Wraps CuteDataSet with a specific transform applied to a subset of indices.
    Necessary because train and val need different augmentation pipelines.
    """
    def __init__(self, dataset: CuteDataSet, indices: list, preprocess):
        self.dataset    = dataset
        self.indices    = indices
        self.preprocess = preprocess

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path  = self.dataset.paths[self.indices[idx]]
        image = Image.fromarray(imageio.imread(path))
        return self.preprocess(image).to(DEVICE)


def build_model() -> UNet2DModel:
    return UNet2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


# ── EMA helper ────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model weights.
    Shadow weights are used for val loss and generation — they are smoother
    than raw weights and produce better FID/IS scores.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v

    def apply_shadow(self, model: torch.nn.Module):
        self._backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model: torch.nn.Module):
        model.load_state_dict(self._backup)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = sd


# ── Trainer ───────────────────────────────────────────────────────────────────

class DiffusionTrainer:
    """
    Resumable DDPM trainer with EMA and a proper train/val split.

    Checkpoint format (always written together):
        best_model.pt   — EMA shadow weights (used for generation)
        train_meta.json — {"last_epoch": N, "best_val_loss": X}

    The split uses a fixed seed so the exact same images are always held out,
    even across resumes.
    """

    def __init__(self, seed: int = 42, start_epoch: int = 0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.config = config
        self.device = DEVICE

        # ── train / val split ─────────────────────────────────────────────
        full_dataset = CuteDataSet(DATA_PATH)
        n_total      = len(full_dataset)
        n_val        = int(n_total * config.val_split)
        n_train      = n_total - n_val

        # Fixed generator → same split on every run / resume
        split_gen = torch.Generator().manual_seed(0)
        train_idx, val_idx = random_split(
            range(n_total), [n_train, n_val], generator=split_gen
        )

        train_set = _PreprocessSubset(full_dataset, list(train_idx), train_preprocess)
        val_set   = _PreprocessSubset(full_dataset, list(val_idx),   val_preprocess)

        self.train_loader = DataLoader(
            train_set, batch_size=config.train_batch_size, shuffle=True, drop_last=True
        )
        self.val_loader = DataLoader(
            val_set, batch_size=config.train_batch_size, shuffle=False, drop_last=False
        )
        print(
            f"Dataset: {n_train} train  |  {n_val} val  |  "
            f"{len(self.train_loader)} train batches/epoch"
        )

        # ── model & optimizer ─────────────────────────────────────────────
        self.model           = build_model().to(self.device)
        self.ema             = EMA(self.model, decay=config.ema_decay)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)
        self.optimizer       = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.lr_scheduler    = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=len(self.train_loader) * config.num_epochs,
        )

        self.start_epoch   = start_epoch
        self.best_val_loss = float("inf")
        self._try_resume()

    # ── paths ────────────────────────────────────────────────────────────

    def _ckpt_path(self) -> str:
        return os.path.join(config.model_output_dir, config.model_output_name)

    def _meta_path(self) -> str:
        return os.path.join(config.model_output_dir, config.meta_name)

    def _save(self, epoch: int):
        os.makedirs(config.model_output_dir, exist_ok=True)
        torch.save(self.ema.state_dict(), self._ckpt_path())
        with open(self._meta_path(), "w") as f:
            json.dump({"last_epoch": epoch, "best_val_loss": self.best_val_loss}, f)

    def _try_resume(self):
        ckpt = self._ckpt_path()
        meta = self._meta_path()
        if not os.path.exists(ckpt):
            print("No checkpoint found — starting fresh.")
            return
        print(f"Loading weights from {ckpt}")
        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state)
        self.ema.load_state_dict(state)
        if os.path.exists(meta):
            with open(meta) as f:
                m = json.load(f)
            # backwards-compatible with old checkpoints that used "best_loss"
            self.best_val_loss = m.get("best_val_loss", m.get("best_loss", float("inf")))
            if self.start_epoch == 0:
                self.start_epoch = m.get("last_epoch", 0) + 1
        print(f"Resuming from epoch {self.start_epoch}  |  best val loss so far: {self.best_val_loss:.4f}")

    # ── validation loss ───────────────────────────────────────────────────

    def _compute_val_loss(self) -> float:
        """One pass over the val set using EMA weights — no augmentations."""
        self.ema.apply_shadow(self.model)
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                images    = batch.to(self.device)
                timesteps = torch.randint(
                    0, config.num_train_timesteps,
                    (images.size(0),), device=self.device
                ).long()
                noise        = torch.randn_like(images)
                noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps=timesteps)
                pred         = self.model(noisy_images, timestep=timesteps, return_dict=False)[0]
                total       += F.mse_loss(pred, noise).item()
        self.ema.restore(self.model)
        self.model.train()
        return total / len(self.val_loader)

    # ── sample grid ───────────────────────────────────────────────────────

    def evaluate(self, epoch: int):
        self.ema.apply_shadow(self.model)
        self.model.eval()
        pipeline = DDPMPipeline(unet=self.model, scheduler=self.noise_scheduler).to(self.device)
        with torch.no_grad():
            images = pipeline(
                batch_size=16,
                generator=torch.Generator(device="cpu").manual_seed(17),
            ).images
        grid     = make_image_grid(images, rows=4, cols=4)
        save_dir = os.path.join(config.model_output_dir, "samples")
        os.makedirs(save_dir, exist_ok=True)
        grid.save(os.path.join(save_dir, f"sample_epoch_{epoch:04d}.png"))
        print(f"  sample grid saved → samples/sample_epoch_{epoch:04d}.png")
        self.ema.restore(self.model)
        self.model.train()

    # ── training loop ────────────────────────────────────────────────────

    def fit(self):
        self.model.train()
        remaining = range(self.start_epoch, config.num_epochs)
        if not remaining:
            print("Already at final epoch — nothing to train.")
            return

        for epoch in tqdm(remaining, total=len(remaining), desc="Epochs"):
            epoch_loss = 0.0

            for batch in tqdm(self.train_loader, leave=False, desc=f"E{epoch}"):
                images    = batch.to(self.device)
                timesteps = torch.randint(
                    0, config.num_train_timesteps,
                    (images.size(0),), device=self.device
                ).long()
                noise        = torch.randn_like(images)
                noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps=timesteps)

                pred = self.model(noisy_images, timestep=timesteps, return_dict=False)[0]
                loss = F.mse_loss(pred, noise)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.ema.update(self.model)
                epoch_loss += loss.detach().item()

            avg_train_loss = epoch_loss / len(self.train_loader)
            val_loss       = self._compute_val_loss()
            print(f"Epoch {epoch:03d}  train: {avg_train_loss:.4f}  val: {val_loss:.4f}")

            # Save on first epoch after resume OR whenever val loss improves
            is_first_epoch = (epoch == self.start_epoch)
            if val_loss < self.best_val_loss or is_first_epoch:
                self.best_val_loss = min(val_loss, self.best_val_loss)
                self._save(epoch)
                print("  checkpoint saved.")

            # Periodic sample grid + safety save every 10 epochs
            if epoch % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                self.evaluate(epoch)
                self._save(epoch)

        print("Training complete.")


# ── generation helpers ────────────────────────────────────────────────────────

def _load_best_model() -> UNet2DModel:
    ckpt = os.path.join(config.model_output_dir, config.model_output_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint at {ckpt} — train first.")
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


def generate_fixed(output_dir: str = "eval_images"):
    """Generate fixed-001.png … fixed-100.png using seeds 1–100."""
    os.makedirs(output_dir, exist_ok=True)
    model     = _load_best_model()
    scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)
    pipeline  = DDPMPipeline(unet=model, scheduler=scheduler).to(DEVICE)

    print(f"Generating 100 images into {output_dir}/")
    for seed in tqdm(range(1, 101), desc="Generating"):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            pil_image = pipeline(batch_size=1, generator=generator).images[0]
        pil_image.save(os.path.join(output_dir, f"fixed-{seed:03d}.png"))
    print(f"Done — 100 images saved to {output_dir}/")


def generate_favorites(favorite_seeds: list, eval_dir: str = "eval_images"):
    """Copy chosen seeds to favorite-1.png, favorite-2.png, … (1-indexed rank)."""
    for rank, seed in enumerate(favorite_seeds, start=1):
        src = os.path.join(eval_dir, f"fixed-{seed:03d}.png")
        dst = os.path.join(eval_dir, f"favorite-{rank}.png")
        if not os.path.exists(src):
            print(f"  WARNING: {src} not found — run generate_fixed() first.")
            continue
        shutil.copy(src, dst)
        print(f"  saved favorite-{rank}.png  (seed {seed})")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DS542 Project 4 — train and/or generate")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--start-epoch",    type=int, default=0,
                        help="Force resume from this epoch (0 = auto-detect)")
    parser.add_argument("--resume",         action="store_true",
                        help="Resume from checkpoint (auto-detected)")
    parser.add_argument("--generate-only",  action="store_true",
                        help="Skip training; generate 100 images from saved checkpoint")
    parser.add_argument("--favorites-only", action="store_true",
                        help="Skip training and generation; just copy favorites")
    parser.add_argument("--num-epochs",     type=int, default=None,
                        help="Override total number of training epochs (default: 80)")
    parser.add_argument("--output-dir",     type=str, default=None,
                        help="Where to save checkpoints/samples (default: project4)")
    parser.add_argument("--eval-dir",       type=str, default="eval_images",
                        help="Where to save fixed-*.png and favorite-*.png")
    parser.add_argument("--favorites",      type=int, nargs="+",
                        default=[3, 7, 12, 25, 41, 55, 63, 72, 88, 94],
                        help="Seeds to copy as favorite-1.png, favorite-2.png, …")
    args = parser.parse_args()

    if args.output_dir:
        config.model_output_dir = args.output_dir
    if args.num_epochs:
        config.num_epochs = args.num_epochs

    if args.favorites_only:
        generate_favorites(favorite_seeds=args.favorites, eval_dir=args.eval_dir)
    elif args.generate_only:
        generate_fixed(output_dir=args.eval_dir)
        generate_favorites(favorite_seeds=args.favorites, eval_dir=args.eval_dir)
    else:
        trainer = DiffusionTrainer(seed=args.seed, start_epoch=args.start_epoch)
        trainer.fit()
        generate_fixed(output_dir=args.eval_dir)
        generate_favorites(favorite_seeds=args.favorites, eval_dir=args.eval_dir)
