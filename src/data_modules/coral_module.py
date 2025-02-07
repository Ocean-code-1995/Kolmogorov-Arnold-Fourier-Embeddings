import torch
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random


def worker_init_fn(worker_id):
    "Set the random seed for each worker"
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# In a module accessible to your data module (e.g., in the same file or in a utils file)
def pi_scale_transform(x):
    return x * 2 * np.pi - np.pi


class CoralDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=64,
        img_size=300,
        scaler_type='min-max',
        test_size=0.2,
        val_size=0.1,
        num_workers=2,
        padding_mode='reflect' # Padding mode can be 'reflect' or 'edge'
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.padding_mode = padding_mode
        self.transform = self.get_transform()



    def custom_pad(self, img):
        left = top = right = bottom = 0
        if img.width < 300 or img.height < 300:
            delta_w = max(300 - img.width, 0)
            delta_h = max(300 - img.height, 0)
            left = delta_w // 2
            right = delta_w - left
            top = delta_h // 2
            bottom = delta_h - top
        return transforms.functional.pad(img, (left, top, right, bottom), padding_mode=self.padding_mode)
    def get_transform(self):
        """
        Selects the appropriate scaling transform based on the `scaler_type`.
        """
        base_transforms = [
            transforms.Lambda(self.custom_pad),
            transforms.Resize((self.img_size, self.img_size)),   #!!!! Forced to resize to 300x300 (no-op for padded & 300x300 images)
            transforms.ToTensor()
        ]
        
        # Apply scaling based on `scaler_type`
        if self.scaler_type == 'min-max':
            # Default: ToTensor scales between 0 and 1, no extra scaling needed
            return transforms.Compose(base_transforms)
        elif self.scaler_type == 'pi-scale':
            # Scale data to [-pi, +pi]
            base_transforms.append(transforms.Lambda(pi_scale_transform))
            return transforms.Compose(base_transforms)
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")


    def setup(self, stage=None):
        def is_image_file(filename):
            return any(filename.endswith(extension) for extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif'])

        def filter_large_images(path):
            if not is_image_file(path):
                return False
            with Image.open(path) as img:
                return img.width <= 300 and img.height <= 300

        full_dataset = datasets.ImageFolder(
            root=self.data_dir, transform=self.transform, is_valid_file=filter_large_images
        )
        # Display class to index mapping
        print("Class to index mapping:", full_dataset.class_to_idx)

        # Splitting the dataset
        train_val_idx, test_idx = train_test_split(
            range(len(full_dataset)),
            test_size=self.test_size,
            shuffle=True,
            stratify=full_dataset.targets,
            random_state=14473
        )

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=self.val_size / (1 - self.test_size),
            shuffle=True,
            stratify=[full_dataset.targets[i] for i in train_val_idx],
            random_state=14473
        )

        self.train_dataset = Subset(full_dataset, train_idx)
        self.val_dataset = Subset(full_dataset, val_idx)
        self.test_dataset = Subset(full_dataset, test_idx)


    def display_sample_images(self):
        sample_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=True)
        data = next(iter(sample_loader))
        images = data[0]
        grid = make_grid(images)
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title('Sample Padded Images')
        plt.axis('off')
        plt.show()
    
    def _calculate_rgb_statistics(self, dataset):
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        mean = torch.tensor([0.0, 0.0, 0.0])
        std = torch.tensor([0.0, 0.0, 0.0])
        for images, _ in loader:
            for i in range(3):  # RGB channels
                mean[i] += images[:, i, :, :].mean()
                std[i] += images[:, i, :, :].std()

        mean /= len(loader)
        std /= len(loader)
        return mean, std

    def _verify_rgb_distribution(self):
        train_mean, train_std = self._calculate_rgb_statistics(self.train_dataset)
        val_mean, val_std = self._calculate_rgb_statistics(self.val_dataset)
        test_mean, test_std = self._calculate_rgb_statistics(self.test_dataset)
        
        print("RGB Distributions:")
        print(f"   - Train Mean: {train_mean}, Std: {train_std}")
        print(f"   - Val Mean: {val_mean}, Std: {val_std}")
        print(f"   - Test Mean: {test_mean}, Std: {test_std}")

        # Here you could implement checks to see if the differences are too large and potentially adjust your splits
    def _verify_class_distribution(self):
        def get_class_counts(dataset):
            targets = [dataset[i][1] for i in range(len(dataset))]
            class_counts = np.bincount(targets, minlength=len(self.train_dataset.dataset.classes))
            return class_counts
        
        train_counts = get_class_counts(self.train_dataset)
        val_counts = get_class_counts(self.val_dataset)
        test_counts = get_class_counts(self.test_dataset)
        
        print("\nLabel Distributions:")
        print("   - Train set:", train_counts)
        print("   - Val set:", val_counts)
        print("   - Test set:", test_counts)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True
        )
