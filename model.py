import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_size, vocab_size, seq_dim=64, hidden_dim=256, n_rnn_layers=24):
        super(CRNN, self).__init__()
        
        self.hidden_dim = hidden_dim

        c, h, w = img_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64), # batch normalization layer
            nn.ReLU(), # relu layers
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64), # batch normalization layer
            nn.ReLU(), # relu layers
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # max pooling layer (64, )
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128), # batch normalization layer
            nn.ReLU(), # relu layers
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128), # batch normalization layer
            nn.ReLU(), # relu layers
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # max pooling layer
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256), # batch normalization layer
            nn.ReLU(), # relu layers
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256), # batch normalization layer
            nn.ReLU(), # relu layers
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256), # batch normalization layer
            nn.ReLU(), # relu layers
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 2)) # max pooling layer
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512), # batch normalization layer
            nn.ReLU(), # relu layers
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512), # batch normalization layer
            nn.ReLU(), # relu layers
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512), # batch normalization layer
            nn.ReLU(), # relu layers
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 2)) # max pooling layer
        )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=512), # batch normalization layer
        #     nn.ReLU(), # relu layers
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=512), # batch normalization layer
        #     nn.ReLU(), # relu layers
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=512), # batch normalization layer
        #     nn.ReLU(), # relu layers
        #     nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 2)) # max pooling layer
        # )
        
        self.sequence_features = nn.Linear(512 * (h // 16), seq_dim)
        self.rnn = nn.GRU(seq_dim, hidden_dim, n_rnn_layers, bidirectional=True) # recurrent layers
        self.fc1 = nn.Linear(hidden_dim * 2, 32) # fully connected layers
        self.fc2 = nn.Linear(32, vocab_size) # fully connected layers


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = self.conv5(x)

        batch_size, c, h, w = x.size()

        x = x.view(batch_size, c * h, w)
        x = x.permute(2, 0, 1) 
        x = self.sequence_features(x)

        x, _ = self.rnn(x)

        x = self.fc1(x)
        x = self.fc2(x)


        return x