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
        self.seq_len = seq_len  # Assuming this is the common sequence length after processing
        self.label_len = label_len
        self.pred_len = pred_len

        # Encoder using simple RNN
        self.enc_rnn = nn.RNN(input_size=enc_in, hidden_size=hidden_dim, num_layers=num_layers, 
                              batch_first=True, dropout=dropout, nonlinearity='tanh')

        # Decoder using simple RNN
        self.dec_rnn = nn.RNN(input_size=dec_in, hidden_size=hidden_dim, num_layers=num_layers, 
                              batch_first=True, dropout=dropout, nonlinearity='tanh')
        
        # CNN layer for feature extraction
        self.enc_cnn = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.dec_cnn = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)

        # Fully connected layer to predict output
        self.fc_out = nn.Linear(hidden_dim * 2, c_out)
        
        # Optional: Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x_enc, x_dec, *args, **kwargs):
        # Encode
        enc_out, _ = self.enc_rnn(x_enc)
        enc_out = enc_out.transpose(1, 2)
        enc_out = self.enc_cnn(enc_out)
        enc_out = enc_out.transpose(1, 2)

        # Decode
        dec_out, _ = self.dec_rnn(x_dec)
        dec_out = dec_out.transpose(1, 2)
        dec_out = self.dec_cnn(dec_out)
        dec_out = dec_out.transpose(1, 2)

        # Ensure both enc_out and dec_out are the same length in the sequence dimension
        min_length = min(enc_out.size(1), dec_out.size(1))
        enc_out = enc_out[:, :min_length, :]
        dec_out = dec_out[:, :min_length, :]

        combined = torch.cat((enc_out, dec_out), dim=2)
        combined = self.layer_norm(combined)

        # Output
        dec_out = self.fc_out(combined)
        
        # Assuming the last outputs are the required prediction outputs
        return dec_out[:, -self.pred_len:, :]