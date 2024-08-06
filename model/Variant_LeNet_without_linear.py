import torch.nn as nn
import torch
import torch.nn.functional as F


class Variant_LeNet_without_linear(nn.Module):
    def __init__(self, in_channels: int):
        super(Variant_LeNet_without_linear, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=21,
            ),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=11,
            ),
            nn.BatchNorm1d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
            ),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # Bacteria MRSA MSSA
        # self.dense1 = nn.Sequential(
        #     nn.Linear(in_features=7552, out_features=1024),
        #     nn.BatchNorm1d(num_features=1024),
        #     nn.Tanh(),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        # )

        # Bile_acids
        # self.dense1 = nn.Sequential(
        #     nn.Linear(in_features=7744, out_features=2048),
        #     nn.BatchNorm1d(num_features=2048),
        #     nn.Tanh(),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        # )

        # Cancer
        # self.dense1 = nn.Sequential(
        #     nn.Linear(in_features=5952, out_features=1024),
        #     nn.BatchNorm1d(num_features=1024),
        #     nn.LeakyReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        # )

        # Bacteria
        self.dense1 = nn.Sequential(
            nn.Linear(in_features=7552, out_features=2048),
            nn.BatchNorm1d(num_features=2048),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # Covid
        # self.dense1 = nn.Sequential(
        #     nn.Linear(in_features=6720, out_features=128),
        #     nn.BatchNorm1d(num_features=128),
        #     nn.LeakyReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        # )
        
        # Melanoma
        # self.dense1 = nn.Sequential(
        #     nn.Linear(in_features=16256, out_features=2048),
        #     nn.BatchNorm1d(num_features=2048),
        #     nn.Tanh(),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        # )

    def forward(self, x):
        x = self.conv1(x)
        # melanoma : 16256
        # bacteria : 7552
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.dense1(x)
        # print(x.shape)
        # x = self.dense2(x)
        # x = self.dense1(x)
        return x
    
if __name__ == "__main__":
    input_data = torch.randn(32, 1, 1024)
    print('input', input_data.shape)
    model = Variant_LeNet_without_linear(in_channels=1)
    output = model(input_data)