import torch.nn as nn
import torch.nn.functional as F


class IEGMNet(nn.Module):
    def __init__(self):
        super(IEGMNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(10, 1), stride=(4, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(2, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(4, 1), stride=(3, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 1), stride=(2, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 1), stride=(2, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=60, out_features=10),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=10, out_features=2)
        )

        self.maxpool2d = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 2), padding=(1, 0))

    def forward(self, input):
        # input 1250
        # c1 311 * 3
        # c2 154 * 5
        # c3 51 * 5
        # c4 25 * 5
        # c5 12 * 5
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv2_output = self.maxpool2d(conv2_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv4_output = self.maxpool2d(conv4_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.view(-1, 60)

        fc1_output = F.relu(self.fc1(conv5_output))
        fc2_output = self.fc2(fc1_output)
        return fc2_output
