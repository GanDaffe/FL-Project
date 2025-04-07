import torch
from torch import nn 
import torchvision.models as models
import torch.nn.functional as F

def calculate(size, kernel, stride, padding):
    return int(((size+(2*padding)-kernel)/stride) + 1)


class CNN(nn.Module):

    def __init__(self, in_feat, im_size, out_feat, hidden): 
        
        super(CNN, self).__init__()
        out = im_size 

        self.conv1 = nn.Conv2d(in_feat, 32, kernel_size=3, stride=1, padding=1)
        out = calculate(out, 3, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        out = calculate(out, kernel=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        out = calculate(out, 3, 1, 1)
        out = calculate(out, kernel=2, stride=2, padding=0)

        self.dropout = nn.Dropout(p=0.5)
        self.after_conv = out * out * 64
        self.fc1 = nn.Linear(in_features=self.after_conv, out_features=hidden) 
        self.fc2 = nn.Linear(in_features=hidden, out_features=out_feat) 
    
    def forward(self, X): 
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))

        X = X.view(-1, self.after_conv)
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))

        return X

class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(2000, 50)
        self.lstm = nn.LSTM(input_size=50, hidden_size=64)

        self.fc1 = nn.Linear(64, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

class fMLP(nn.Module):
    def __init__(self) -> None:
        super(fMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, 10)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout1(x) 

        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.layer3(x)
        return x

class VGG16(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG16, self).__init__()

        self.vgg = models.vgg16(weights=None)

        self.vgg.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg(x)

class ResNet50(nn.Module):
    def __init__(self, num_channel=3, num_classes=100):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=False)

        self.resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        return x

class ResNet101(nn.Module):
    def __init__(self, num_channel=3, num_classes=100):
        super(ResNet101, self).__init__()
        self.resnet = models.resnet101(pretrained=False)

        self.resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        return x


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.classes = torch.unique(torch.tensor(targets)).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
