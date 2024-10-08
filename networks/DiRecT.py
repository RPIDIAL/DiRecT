import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.init import xavier_uniform_

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        # Create a long enough P.E. table
        self.pe = nn.Parameter(torch.zeros(1, max_seq_length, d_model).float(), requires_grad=False)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))        
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class DiRecT(nn.Module):
    def __init__(self, feature_size, embed_size, num_layers, num_heads, dim_feedforward, num_classes):
        super(DiRecT, self).__init__()        
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.lmk_mask_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.embedding = nn.Linear(feature_size, embed_size)  # Embedding layer
        self.pos_encoder = PositionalEncoding(d_model=embed_size, max_seq_length=400)
        encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=dim_feedforward, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers, norm=nn.LayerNorm(embed_size))
        decoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=dim_feedforward, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = TransformerEncoder(encoder_layer=decoder_layers, num_layers=num_layers, norm=nn.LayerNorm(embed_size))
        self.rec_linear = nn.Linear(embed_size, 3)
        self.cls_linear = nn.Linear(embed_size, num_classes)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, data):
        src, batch_size, unmasked_face_lmk_ids = data.x, data.num_graphs, data.unmasked_face_lmk_ids
        graph_node_num = src.shape[0] // batch_size
        unmasked_face_lmk_num = unmasked_face_lmk_ids.shape[0] // batch_size
        unmasked_face_lmk_ids = unmasked_face_lmk_ids[:unmasked_face_lmk_num]

        # src: Batch x Nodes x Features (B x N x F)
        src = src.view(batch_size, graph_node_num, -1)        
        src = self.embedding(src)  # Apply embedding
        
        src_mask_token = self.lmk_mask_token.expand_as(src)
        src_mask = torch.ones_like(src, dtype=bool)
        src_mask[:, unmasked_face_lmk_ids, :] = False
        src = torch.where(src_mask, src_mask_token, src)

        cls_tokens = self.class_token.expand(batch_size, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)  # B x (N+1) x F
        src = self.pos_encoder(src)
        encoder_output = self.encoder(src)
        encoder_cls_token = encoder_output[:, 0:1, :]  # Get the final state of the class token
        y = self.cls_linear(encoder_cls_token.squeeze(1))

        msk_input = encoder_output[:, 1:, :]
        msk_tokens = self.mask_token.expand_as(msk_input)
        z = torch.cat((encoder_cls_token, msk_tokens), dim=1)  # B x (N+1) x F
        z = self.pos_encoder(z)
        lmk_coord = self.decoder(z)
        lmk_coord = lmk_coord[:, 1:, :]
        lmk_coord = self.rec_linear(lmk_coord)
        lmk_coord = self.tanh(lmk_coord)

        if not self.training:
            y = self.softmax(y)
        return y, lmk_coord, encoder_cls_token.squeeze(1)
