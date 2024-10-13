import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, config) -> None:
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size + 1, config.embedding_dim, -1)
        self.convs = nn.ModuleList([nn.Conv2d(1, 1, (k, config.embedding_dim))for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(len(config.filter_sizes), config.num_classes)
    def forward(self, x):
        x = self.embedding(x) # batch_size, seq_len, embedding_dim
        x = x.unsqueeze(1) # batch_size, 1, seq_len, embedding_dim
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # [(batch_size, 1, seq_len), ...]*len(filter_sizes)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x] # [(batch_size, 1), ...]*len(filter_sizes)
        x = torch.cat(x, 1) # batch_size, len(filter_sizes)
        x = self.dropout(x) # batch_size, len(filter_sizes)
        logits = self.fc(x) # batch_size, num_classes
        return logits
    def predict(self, x):
        logits = self.forward(x)
        tag = torch.argmax(logits, dim=-1)
        return tag
    

class TextTransformer(nn.Module):
    def __init__(self, config) -> None:
        super(TextTransformer, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size + 1, config.embedding_dim, -1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.embedding_dim, nhead=config.num_heads),
            num_layers=config.num_layers
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.embedding_dim, config.num_classes)
        self.max_len = config.window_size

    def truncate_and_pad(self, x):
        batch_size, seq_len = x.shape
        if seq_len > self.max_len:
            return x[:, :self.max_len]  # 截断至 max_len
        else:
            padding = torch.full((batch_size, self.max_len - seq_len), 8019, dtype=x.dtype)
            return torch.cat((x, padding), dim=1)

    def forward(self, x):
        x = self.truncate_and_pad(x)
        x = self.embedding(x)  # batch_size, seq_len, embedding_dim
        x = x.permute(1, 0, 2)  # seq_len, batch_size, embedding_dim
        x = self.transformer_encoder(x)  # seq_len, batch_size, embedding_dim
        x = x.mean(dim=0)  # batch_size, embedding_dim
        x = self.dropout(x)  # batch_size, embedding_dim
        x = self.fc(x)  # batch_size, num_classes
        return x

    def predict(self, x):
        logits = self.forward(x)
        tag = torch.argmax(logits, dim=-1)
        return tag

class TextLSTM(nn.Module):
    def __init__(self, config):
        super(TextLSTM, self).__init__()
        # 定义嵌入层
        self.embedding = nn.Embedding(config.vocab_size + 1, config.embedding_dim)
        
        # 定义LSTM层，config.hidden_dim表示隐藏层维度，config.num_layers表示LSTM层数
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim, config.num_layers, batch_first=True)
        
        # 定义dropout层
        self.dropout = nn.Dropout(config.dropout)
        
        # 定义全连接层，用于分类
        self.fc = nn.Linear(config.hidden_dim, config.num_classes)
        
        # 设定截断的BPTT步长
        self.bptt_steps = config.bptt_steps
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim

    def forward(self, x):
        # x: batch_size, seq_len
        
        # 嵌入层映射，得到词嵌入
        x = self.embedding(x)  # batch_size, seq_len, embedding_dim
        
        # 初始化LSTM的隐状态和细胞状态
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # 逐步处理输入数据，按照BPTT设定的步长截断反向传播
        outputs = []
        for i in range(0, x.size(1), self.bptt_steps):
            # 获取当前步长内的输入
            x_chunk = x[:, i:i+self.bptt_steps, :]
            
            # 前向传播LSTM
            lstm_out, (h_0, c_0) = self.lstm(x_chunk, (h_0, c_0))  
            
            # 将最后一个时间步的输出保存
            outputs.append(lstm_out[:, -1, :])  # 取出最后时间步的隐藏状态

        # 拼接所有时间步的输出
        x = torch.cat(outputs, dim=1)  # batch_size, hidden_dim * (seq_len / bptt_steps)
        
        # Dropout层
        x = self.dropout(x)  # 防止过拟合
        
        # 全连接层，用于分类
        logits = self.fc(x)  # batch_size, num_classes
        
        return logits

    def predict(self, x):
        # 预测标签
        logits = self.forward(x)
        tag = torch.argmax(logits, dim=-1)
        return tag
