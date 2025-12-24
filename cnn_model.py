import torch
import torch.nn as nn
import torch.nn.functional as F

class GunshotCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GunshotCNN, self).__init__()
        
        # 1D 卷积层: 提取特征间的局部关联
        # 输入形状: (Batch, 1, Num_Features)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)

        # 动态计算全连接层输入大小
        # 经过3次池化(除以2)，长度变为原长的 1/8
        final_dim = num_features // 8
        self.fc_input_dim = 256 * final_dim
        
        # 全连接层 (Classifier)
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.dropout = nn.Dropout(0.5) # 防止过拟合
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x shape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1) 
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
