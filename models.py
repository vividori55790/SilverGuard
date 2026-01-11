import torch
import torch.nn as nn

class FallLSTM(nn.Module):
    """
    낙상 감지를 위한 LSTM 모델 정의
    입력: (Batch_Size, Sequence_Length, Input_Size)
    출력: (Batch_Size, Num_Classes) -> [Normal_Prob, Fall_Prob]
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2):
        super(FallLSTM, self).__init__()
        
        # LSTM Layer
        # batch_first=True: 입력 텐서의 첫 번째 차원이 Batch 크기임을 명시 (N, L, F)
        # dropout: 과적합 방지를 위해 레이어 사이의 연결을 일부 끊음 (0.2 = 20%)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2
        )
        
        # Fully Connected Layer (분류기)
        # LSTM의 마지막 시점(t=30)의 은닉 상태(Hidden State)를 받아 최종 분류 수행
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch_Size, Sequence_Length, Input_Size)
        
        # 1. LSTM 통과
        # h0, c0는 별도 지정 없으면 자동으로 0으로 초기화됨
        # lstm_out shape: (Batch, Seq, Hidden)
        lstm_out, _ = self.lstm(x) 
        
        # 2. Many-to-One 방식: 시퀀스의 가장 마지막 타임스텝의 결과만 사용
        # lstm_out[:, -1, :] -> 모든 배치의 마지막 시점 데이터 추출
        last_out = lstm_out[:, -1, :]
        
        # 3. 분류 레이어 통과
        out = self.fc(last_out)
        return out