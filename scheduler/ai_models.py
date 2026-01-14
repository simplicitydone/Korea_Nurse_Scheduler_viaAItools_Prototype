# scheduler/ai_models.py

import torch
import torch.nn as nn

class ShiftLSTM(nn.Module):
    """
    간호사 근무 패턴 예측을 위한 LSTM 모델 (Generator용)
    Input: 과거 근무 시퀀스 (Integer encoded shifts)
    Output: 다음 날 근무 확률 (Logits for 4 classes: D, E, N, O)
    """
    def __init__(self, num_shifts=4, embedding_dim=16, hidden_dim=32):
        super(ShiftLSTM, self).__init__()
        self.embedding = nn.Embedding(num_shifts, embedding_dim)
        # batch_first=True: (batch, seq, feature)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_shifts)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embeds = self.embedding(x)
        # lstm_out shape: (batch, seq, hidden)
        lstm_out, _ = self.lstm(embeds)
        # 마지막 시점(last time step)의 출력만 사용
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


class OperatorDQN(nn.Module):
    """
    Local Search용 Contextual DQN 모델
    Input: State Vector (4-dim: progress, violations, score, fail_rate)
    Output: Q-Values for each Operator (N-dim)
    """
    def __init__(self, input_dim=4, output_dim=3):  # operator 개수(3)에 따라 output_dim 설정
        super(OperatorDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)