import argparse
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
from dataclasses import dataclass, field
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.utils import make_image_grid
from PIL import Image
from torch.utils.data import DataLoader
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
    num_epochs:          int   = 80
    learning_rate:       float = 1e-4
    num_warmup_steps:    int   = 1000
    save_image_epochs:   int   = 10
    num_train_timesteps: int   = 1000
    grad_clip:           float = 1.0
    model_output_dir:    str   = "project4"
    model_output_name:   str   = "best_model.pt"
    meta_name:           str   = "train_meta.json"


config = TrainingConfig()


preprocess = transforms.Compose([
    transforms.ColorJitter(brightness=0.25, contrast=0.3, hue=0.15, saturation=0.2),
    transforms.RandomPosterize(bits=6, p=0.05),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAutocontrast(p=0.1),
    transforms.ToTensor(),
    transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class CuteDataSet(torch.utils.data.Dataset):
    def __init__(self, root, image_size=IMAGE_SIZE, preprocess=None):
        self.paths      = list(pathlib.Path(root).glob("*/*.png"))
        self.image_size = image_size
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.fromarray(imageio.imread(self.paths[idx]))
        if self.preprocess is not None:
            return self.preprocess(image).to(DEVICE)
        image = np.array(image).astype(float) / 255.0
        image = skimage.transform.resize(image, (self.image_size, self.image_size))
        return torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).to(DEVICE)


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

class DiffusionTrainer:
    """
    Resumable DDPM trainer.

    Checkpoint format:
        best_model.pt    — raw model.state_dict()   (loads cleanly, no key magic)
        train_meta.json  — {"last_epoch": N, "best_loss": X}

    Both files are always written together so they can't fall out of sync.
    """

    def __init__(self, seed: int = 42, start_epoch: int = 0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.config = config
        self.device = DEVICE

        dataset = CuteDataSet(DATA_PATH, preprocess=preprocess)
        self.train_loader = DataLoader(
            dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True
        )
        print(f"Dataset: {len(dataset)} images  |  {len(self.train_loader)} batches/epoch")

        self.model           = build_model().to(self.device)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)
        self.optimizer       = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.lr_scheduler    = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=len(self.train_loader) * config.num_epochs,
        )

        self.start_epoch = start_epoch
        self.best_loss   = float("inf")
        self._try_resume()

    # ── paths ────────────────────────────────────────────────────────────

    def _ckpt_path(self) -> str:
        return os.path.join(config.model_output_dir, config.model_output_name)

    def _meta_path(self) -> str:
        return os.path.join(config.model_output_dir, config.meta_name)
        
    def _save(self, epoch: int):
        os.makedirs(config.model_output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), self._ckpt_path())
        with open(self._meta_path(), "w") as f:
            json.dump({"last_epoch": epoch, "best_loss": self.best_loss}, f)

    def _try_resume(self):
        ckpt  = self._ckpt_path()
        meta  = self._meta_path()
        if not os.path.exists(ckpt):
            print("No checkpoint found — starting fresh.")
            return
        print(f"Loading weights from {ckpt}")
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        if os.path.exists(meta):
            with open(meta) as f:
                m = json.load(f)
            self.best_loss = m.get("best_loss", float("inf"))
            if self.start_epoch == 0:           # auto-detect unless forced
                self.start_epoch = m.get("last_epoch", 0) + 1
        print(f"Resuming from epoch {self.start_epoch}  |  best loss so far: {self.best_loss:.4f}")

    def evaluate(self, epoch: int):
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
        self.model.train()

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
                epoch_loss += loss.detach().item()

            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch:03d}  avg loss: {avg_loss:.4f}")

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save(epoch)
                print("new best — checkpoint saved.")

            if epoch % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                self.evaluate(epoch)

        print("Training complete.")

def _load_best_model() -> UNet2DModel:
    ckpt = os.path.join(config.model_output_dir, config.model_output_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint at {ckpt} — train first.")
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


def generate_fixed(output_dir: str = "eval_images"):
    """
    Generate fixed-001.png … fixed-100.png using seeds 1-100.

    Post-processing to avoid black edges:
      - Pipeline internally clamps to [0, 1] and converts to uint8.
      - We additionally pad 1px, crop back to 64x64 via a center-crop so any
        residual border artifact from the UNet is trimmed.  The tiny crop has
        no visual impact on 64x64 images.
    """
    os.makedirs(output_dir, exist_ok=True)
    model     = _load_best_model()
    scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)
    pipeline  = DDPMPipeline(unet=model, scheduler=scheduler).to(DEVICE)

    center_crop = transforms.CenterCrop(IMAGE_SIZE - 2)    # trims 1px border all round
    resize_back = transforms.Resize([IMAGE_SIZE, IMAGE_SIZE], interpolation=transforms.InterpolationMode.BILINEAR)

    print(f"Generating 100 images into {output_dir}/")
    for seed in tqdm(range(1, 101), desc="Generating"):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            pil_image = pipeline(batch_size=1, generator=generator).images[0]

        # trim any residual 1-pixel black border then resize back to 64x64
        pil_image = resize_back(center_crop(pil_image))
        pil_image.save(os.path.join(output_dir, f"fixed-{seed:03d}.png"))

    print(f"Done — 100 images saved to {output_dir}/")


def generate_favorites(
    favorite_seeds: list,
    eval_dir: str = "eval_images",
):
    """Copy chosen seeds to favorite-SEED.png (no zero-padding needed)."""
    for seed in favorite_seeds:
        src = os.path.join(eval_dir, f"fixed-{seed:03d}.png")
        dst = os.path.join(eval_dir, f"favorite-{seed}.png")
        if not os.path.exists(src):
            print(f"  WARNING: {src} not found — run generate_fixed() first.")
            continue
        shutil.copy(src, dst)
        print(f"  saved favorite-{seed}.png")


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
    parser.add_argument("--output-dir",     type=str, default=None,
                        help="Where to save checkpoints/samples (default: project4)")
    parser.add_argument("--eval-dir",       type=str, default="eval_images",
                        help="Where to save fixed-*.png and favorite-*.png")
    parser.add_argument("--favorites",      type=int, nargs="+",
                        default=[3, 7, 12, 25, 41, 55, 63, 72, 88, 94],
                        help="Seeds to copy as favorite-SEED.png")
    args = parser.parse_args()

    # override output dir before anything runs
    if args.output_dir:
        config.model_output_dir = args.output_dir

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
