import os
import string
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

class CaptchaDataset(Dataset):
    def __init__(self, dataset_dir, img_size=(64, 128)):
        self.vocab = string.ascii_lowercase + string.digits
        self.vocab_size = len(self.vocab)
        self.char2num = {char: i + 1 for i, char in enumerate(self.vocab)}
        self.num2char = {label: char for char, label in self.char2num.items()}
        self.dataset_dir = dataset_dir
        self.all_imgs = os.listdir(dataset_dir)

        self.transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor()
        ])

    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_dir, self.all_imgs[index])
        pil_image = Image.open(image_path).convert('RGB')
        X = self.transform(pil_image)

        label = self.all_imgs[index].split(".")[0]
        encode_label = [self.char2num[c] for c in label]

        y = torch.LongTensor(encode_label)
        
        return X, y

    def __len__(self):

        return len(self.all_imgs)