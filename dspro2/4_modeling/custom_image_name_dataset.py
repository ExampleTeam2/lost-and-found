from torch.utils.data import Dataset

class CustomImageNameDataset(Dataset):
    def __init__(self, image_paths, json_paths, transform=None):
        self.image_paths = image_paths
        self.json_paths = json_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx], self.json_paths[idx]
