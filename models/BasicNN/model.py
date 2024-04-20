import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

class BasicNN(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len, 
                 hidden_dim=256, num_layers=2, kernel_size=3, dropout=0.1,
                 device=torch.device('cuda:0')):
        super(BasicNN, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        # Encoder using LSTM
        self.enc_lstm = nn.LSTM(input_size=enc_in, hidden_size=hidden_dim, num_layers=num_layers, 
                                batch_first=True, dropout=dropout, bidirectional=True)

        # Decoder using LSTM
        self.dec_lstm = nn.LSTM(input_size=dec_in, hidden_size=hidden_dim, num_layers=num_layers, 
                                batch_first=True, dropout=dropout, bidirectional=True)
        
        # CNN layer for feature extraction
        self.cnn = nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)

        # Fully connected layer to predict output
        self.fc_out = nn.Linear(hidden_dim, c_out)
        
        # Optional: Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_enc, x_dec):
        # Encode
        enc_out, _ = self.enc_lstm(x_enc)
        enc_out = enc_out.contiguous().view(enc_out.shape[0], enc_out.shape[2], enc_out.shape[1])
        enc_out = self.cnn(enc_out)
        enc_out = enc_out.view(enc_out.shape[0], enc_out.shape[2], enc_out.shape[1])
        enc_out = self.layer_norm(enc_out)

        # Decode
        dec_out, _ = self.dec_lstm(x_dec)
        dec_out = dec_out.contiguous().view(dec_out.shape[0], dec_out.shape[2], dec_out.shape[1])
        dec_out = self.cnn(dec_out)
        dec_out = dec_out.view(dec_out.shape[0], dec_out.shape[2], dec_out.shape[1])
        dec_out = self.layer_norm(dec_out)

        # Output
        dec_out = self.fc_out(dec_out)
        
        # Assuming the last outputs are the required prediction outputs
        return dec_out[:, -self.pred_len:, :]