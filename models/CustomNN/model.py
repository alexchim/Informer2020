from math import e
import torch
import torch.nn as nn

class CustomNN(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len, 
                 hidden_dim=256, num_layers=4, kernel_size=3, dropout=0.1, width=3, depth=2,
                 device=torch.device('cuda:0')):
        super(CustomNN, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.width = width
        self.depth = depth

        self.device = device

        # Define multi-level LSTM blocks for encoder and decoder
        self.encoder_lstm_blocks = self.create_multilevel_lstm_blocks(enc_in, hidden_dim, width, depth, dropout)
        self.decoder_lstm_blocks = self.create_multilevel_lstm_blocks(dec_in, hidden_dim, width, depth, dropout)

        # # CNN layer for feature extraction
        # self.enc_cnn = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        # self.dec_cnn = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        # Define CNN layers for encoder and decoder
        self.enc_cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim * self.width, out_channels=self.hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
        )

        self.dec_cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim * self.width, out_channels=self.hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
        )


        # Fully connected layer to predict output
        self.fc_out = nn.Linear(hidden_dim * 2, c_out)
        
        # Optional: Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def create_multilevel_lstm_blocks(self, input_dim, hidden_dim, width, depth, dropout):
        levels = []
        for i in range(depth):
            level = nn.ModuleList([nn.LSTM(input_dim, hidden_dim, num_layers=self.num_layers, dropout=dropout, batch_first=True)
                                   for _ in range(width)])
            levels.append(level)
            input_dim = hidden_dim * width  # Output of one level is the input to the next
        return nn.ModuleList(levels)

    def forward(self, x_enc, x_dec):
        # Apply LSTM blocks
        enc_out = self.apply_lstm_blocks(x_enc, self.encoder_lstm_blocks)
        dec_out = self.apply_lstm_blocks(x_dec, self.decoder_lstm_blocks)

        # Apply CNN to extracted features
        enc_out = enc_out.transpose(1, 2)
        enc_out = self.enc_cnn(enc_out)
        enc_out = enc_out.transpose(1, 2)

        dec_out = dec_out.transpose(1, 2)
        dec_out = self.dec_cnn(dec_out)
        dec_out = dec_out.transpose(1, 2)

        min_length = min(enc_out.size(1), dec_out.size(1))
        enc_out = enc_out[:, :min_length, :]
        dec_out = dec_out[:, :min_length, :]
        # Concatenate and normalize
        combined = torch.cat((enc_out, dec_out), dim=2)
        combined = self.layer_norm(combined)

        # Output layer
        output = self.fc_out(combined)
        return output[:, -self.pred_len:, :]

    def apply_lstm_blocks(self, x, lstm_blocks):
        for level in lstm_blocks:
            level_outputs = []
            for lstm in level:
                out, _ = lstm(x)
                level_outputs.append(out)
            x = torch.cat(level_outputs, dim=2)  # Concatenate along the feature dimension
        return x
