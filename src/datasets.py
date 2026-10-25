from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.transforms.functional import to_tensor


class LeafDataset(Dataset):
    def __init__(self, root: Path, topic: str):
        assert topic in ["disease", "leaf"]
        self.topic = topic
        self.paths = []
        for category in root.iterdir():
            self.paths.extend([
                p for p in Path(category).rglob("*")
                if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
                and "_" not in p.stem
            ])
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        image = Image.open(p).convert("RGB").resize((256, 256))
        mask = Image.open(f"{p.parent}/{p.stem}_{self.topic}_mask.png").convert("L").resize((256, 256), Image.Resampling.NEAREST)
        return to_tensor(image), (to_tensor(mask) > 0).float()


def prepare_dataset(path: str, cat: str):
    dataset = LeafDataset(Path(path), cat)

    train_size = int(0.6 * len(dataset))
    valid_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = random_split(dataset, 
                                                              [train_size, valid_size, test_size],
                                                              generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=24, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

    return train_loader, val_loader, test_loader