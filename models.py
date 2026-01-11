import torch
import torch.nn as nn

class FallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2):
        super(FallLSTM, self).__init__()
        
        # LSTM Layer
        # batch_first=True: 입력 데이터 형태가 (Batch, Sequence, Features)임
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2
        )
        
        # Fully Connected Layer (분류기)
        # LSTM의 마지막 결과값을 받아서 낙상(1)인지 정상(0)인지 판단
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch_Size, Sequence_Length, Input_Size)
        
        # LSTM 통과
        # h0, c0는 자동으로 0으로 초기화됨
        lstm_out, _ = self.lstm(x)
        
        # 마지막 타임스텝의 결과만 사용 (Many-to-One)
        # lstm_out[:, -1, :] -> (Batch_Size, Hidden_Size)
        last_out = lstm_out[:, -1, :]
        
        # 분류
        out = self.fc(last_out)
        return out