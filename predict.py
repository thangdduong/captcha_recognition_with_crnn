import torch
import torch.nn as nn
import string
import torchvision.transforms as T

from PIL import Image
from model import CRNN


class GreedyCTCDecoder(nn.Module):
    def __init__(self, vocab, blank=0):
        super(GreedyCTCDecoder, self).__init__()

        self.vocab = string.ascii_lowercase + string.digits
        self.blank = blank
    
    def forward(self, x):
        indices = torch.argmax(x, dim=-1)  
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.vocab[i] for i in indices])

        return joined.replace("|", " ").strip().split()

def main():
    img_path = './dataset/test/2b827.png'
    model_path = './crnn_model.pth'
    decoder = GreedyCTCDecoder()
    network = CRNN()
    network.load_state_dict(torch.load(model_path))
    network.eval()

    img = Image(img_path)

    transform = T.Compose([
        T.Resize(img[1:]),
        T.ToTensor()
    ])

    img = transform(img)
    img = img.unsqueeze(0)
    outputs = network(img)
    transcription = decoder(outputs)

    print(transcription)


if __name__ == '__main__':
    main()