import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm
import os

from dataset import CaptchaDataset
from model import CRNN
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="parameters for training")

parser.add_argument('--dataset_dir', default="./dataset", type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-4, type=str)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--save_path', default="./crnn_net.pth", type=str)

args = parser.parse_args()

DATASET_DIR = args.dataset_dir
SAVE_PATH = args.save_path
BATCH_SIZE = args.batch_size
IMG_SIZE = (3, 64, 128)
EPOCHS = args.epochs
LR = float(args.lr)
MOMENTUM = 0.9
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = CaptchaDataset(TRAIN_DIR, img_size=IMG_SIZE[1:])
    test_dataset = CaptchaDataset(TEST_DIR, img_size=IMG_SIZE[1:])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    network = CRNN(img_size=IMG_SIZE, vocab_size=train_dataset.vocab_size)
    if torch.cuda.is_available():
        network.cuda()

    criterion = nn.CTCLoss(blank=train_dataset.vocab_size-1)
    optimizer = optim.SGD(network.parameters(), lr=LR, momentum=MOMENTUM)

    for epoch in tqdm(range(EPOCHS)):
        training_loss = 0.0
        network.train()
        for index, sample in enumerate(train_dataloader, 0):
            X, y = sample

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = network(X)
            outputs_softmax = nn.functional.log_softmax(outputs, dim=2)

            pred_length = torch.LongTensor([outputs_softmax.size(0)] * outputs_softmax.size(1))
            target_length = torch.tensor([len(arr) for arr in y])

            loss = criterion(outputs_softmax, y, pred_length, target_length)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            if index % BATCH_SIZE == 0:
                print()
                print(f'[EPOCH {epoch + 1}, {index + 1}] train loss: {training_loss / BATCH_SIZE}')

        network.eval()
        test_loss = 0.0
        with torch.no_grad():
            for index, sample in enumerate(test_dataloader):
                X, y = sample
                X = X.to(device)
                y = y.to(device)
                
                outputs = network(X)
                outputs_softmax = nn.functional.log_softmax(outputs, dim=2)
                pred_length = torch.LongTensor([outputs_softmax.size(0)] * outputs_softmax.size(1))
                target_length = torch.tensor([len(arr) for arr in y])
                loss = criterion(outputs_softmax, y, pred_length, target_length,)
                test_loss += loss.item()
            
            print(f'\n[EPOCH {epoch + 1}] test loss: {loss / len(test_dataloader)}\n')


    print('Finished Training')
    print('Saving model...')
    torch.save(network.state_dict(), SAVE_PATH)
    print('Done')

if __name__ == '__main__':
    main()



