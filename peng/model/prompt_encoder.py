import torch
import torch.nn as nn

class PromptEncoder(nn.Module):
    def __init__(self, config, embedding_layer):
        super().__init__()
        self.embeddings = embedding_layer
        # self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size)
        self.lstm_dropout = config.lstm_dropout if hasattr(config, 'lstm_dropout') else 0.0
        self.lstm_head = torch.nn.LSTM(input_size=config.hidden_size,
                                        hidden_size=config.hidden_size // 2,
                                        num_layers=2,
                                        dropout=self.lstm_dropout,
                                        bidirectional=True,
                                        batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(config.hidden_size, config.hidden_size))

    def forward(self, input_ids):
        # if not next(self.lstm_head[0].parameters()).is_cuda:
        #     for layer in self.lstm_head:
        #         layer.to(input_ids.device)
        #     for layer in self.mlp_head:
        #         layer.to(input_ids.device
        input_embeds = self.embeddings(input_ids)
        temp_embeds = self.lstm_head(input_embeds)
        output_embeds = self.mlp_head(temp_embeds[0]).squeeze()
        return output_embeds