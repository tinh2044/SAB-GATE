import os
import random
from pathlib import Path
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class PairedTransform:
    """Custom transform class to ensure input and target are transformed consistently"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, input_img, target_img):
        # Set seed to ensure same random state
        seed = random.randint(0, 2**32 - 1)

        # Transform input
        random.seed(seed)
        torch.manual_seed(seed)
        input_transformed = self.transform(input_img)

        # Transform target with same seed
        random.seed(seed)
        torch.manual_seed(seed)
        target_transformed = self.transform(target_img)

        return input_transformed, target_transformed


class DebluringDataset(data.Dataset):
    def __init__(self, root, cfg, split="train"):
        self.root = root
        self.cfg = cfg
        self.split = split

        # Setup paths
        if split == "train":
            self.input_dir = os.path.join(root, cfg["train_dir"], cfg["input_dir"])
            self.target_dir = os.path.join(root, cfg["train_dir"], cfg["target_dir"])
        else:
            self.input_dir = os.path.join(root, cfg["test_dir"], cfg["input_dir"])
            self.target_dir = os.path.join(root, cfg["test_dir"], cfg["target_dir"])
        self.extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

        # Get image files
        self.image_files = self._get_image_files()

        # Setup transforms
        self.transform = self._get_transforms()

        print(f"DebluringDataset[{split}] -> {len(self.image_files)} samples")
        print(
            f"  Input: {self.input_dir} ({len(self._list_files(self.input_dir))} files)"
        )
        print(
            f"  Target: {self.target_dir} ({len(self._list_files(self.target_dir))} files)"
        )

    def _list_files(self, folder):
        if not os.path.exists(folder):
            return []
        files = []
        for fp in Path(folder).rglob("*"):
            if fp.is_file() and fp.suffix.lower() in self.extensions:
                files.append(fp)
        return files

    def _get_image_files(self):
        """Get list of image files"""
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_files = []

        if os.path.exists(self.input_dir) and os.path.exists(self.target_dir):
            input_files = sorted(
                [
                    f
                    for f in os.listdir(self.input_dir)
                    if f.lower().endswith(valid_extensions)
                ]
            )

            for file in input_files:
                target_file = os.path.join(self.target_dir, file)
                if os.path.exists(target_file):
                    image_files.append(file)

        return image_files

    def _get_transforms(self):
        """Get image transforms"""
        image_size = self.cfg.get("image_size", 256)
        if self.split == "train":
            base_transform = transforms.Compose(
                [
                    transforms.Resize((image_size + 20, image_size + 20)),
                    transforms.RandomCrop((image_size, image_size)),
                    transforms.RandomRotation(degrees=5, expand=False),
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.05,
                        hue=0.02,
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            base_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            )

        paired_transform = PairedTransform(base_transform)

        return paired_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Get a single sample"""
        filename = self.image_files[idx]

        # Load images
        input_path = os.path.join(self.input_dir, filename)
        target_path = os.path.join(self.target_dir, filename)

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        # Apply transforms - using PairedTransform to ensure consistency
        input_tensor, target_tensor = self.transform(input_image, target_image)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "filename": filename,
            "idx": idx,
        }

    def data_collator(self, batch):
        """Custom collate function for batching"""
        inputs = torch.stack([item["input"] for item in batch])
        targets = torch.stack([item["target"] for item in batch])
        filenames = [item["filename"] for item in batch]
        indices = [item["idx"] for item in batch]

        return {
            "inputs": inputs,
            "targets": targets,
            "filenames": filenames,
            "indices": indices,
        }


def get_training_set(root, cfg):
    """Get training dataset"""
    return DebluringDataset(root, cfg, split="train")


def get_test_set(root, cfg):
    """Get test dataset"""
    return DebluringDataset(root, cfg, split="test")
