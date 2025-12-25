import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionCRNN(nn.Module):
    def __init__(self, num_classes):
        super(FusionCRNN, self).__init__()
        
        # === Stream 1: Log-Mel (2D CNN) ===
        # 输入: (B, 1, 128, T)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d((4, 2)) # 128->32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d((4, 2)) # 32->8
        
        # 这里的输出将是 (B, 32, 8, T/4)
        # 展平频域维度: 32*8 = 256
        self.mel_rnn_input = 256
        
        # === Stream 2: Statistics (1D CNN) ===
        # 输入: (B, 4, T) -> 能量、质心、滚降、ZCR
        # 使用 1D 卷积提取数值变化趋势
        self.stat_conv1 = nn.Conv1d(4, 16, kernel_size=5, padding=2)
        self.stat_pool1 = nn.MaxPool1d(2) # T -> T/2
        self.stat_conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.stat_pool2 = nn.MaxPool1d(2) # T/2 -> T/4
        
        self.stat_rnn_input = 32
        
        # === 融合 LSTM ===
        # 我们将两边的特征在“时间维度”对齐后，拼接在一起送入 LSTM
        # 总特征数 = 256 (Mel) + 32 (Stat) = 288
        self.fusion_lstm = nn.LSTM(
            input_size=self.mel_rnn_input + self.stat_rnn_input,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # === Attention & Output ===
        self.attention = nn.Linear(256, 1)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x_mel, x_stat):
        # --- Stream 1 ---
        # x_mel: (B, 1, 128, T)
        m = self.pool1(F.relu(self.conv1(x_mel)))
        m = self.pool2(F.relu(self.conv2(m)))
        # (B, 32, 8, T/4) -> (B, T/4, 256)
        b, c, h, w = m.size()
        m = m.permute(0, 3, 1, 2).reshape(b, w, c * h)
        
        # --- Stream 2 ---
        # x_stat: (B, 4, T)
        s = self.stat_pool1(F.relu(self.stat_conv1(x_stat)))
        s = self.stat_pool2(F.relu(self.stat_conv2(s)))
        # (B, 32, T/4) -> (B, T/4, 32)
        s = s.permute(0, 2, 1)
        
        # --- Fusion ---
        # 拼接: (B, T/4, 288)
        # 注意：由于池化可能有取整误差，强制裁剪到相同长度
        min_t = min(m.shape[1], s.shape[1])
        m = m[:, :min_t, :]
        s = s[:, :min_t, :]
        
        fused = torch.cat([m, s], dim=2)
        
        # LSTM
        out, _ = self.fusion_lstm(fused)
        
        # Attention
        attn_weights = F.softmax(self.attention(out), dim=1)
        out = torch.sum(out * attn_weights, dim=1)
        
        # Classifier
        return self.fc(out)
