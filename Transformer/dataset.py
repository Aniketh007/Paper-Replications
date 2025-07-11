import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(nn.Module):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([self.tokenizer_src.token_to_id('[SOS]')], dtype=torch.long)
        self.eos_token = torch.tensor([self.tokenizer_src.token_to_id('[EOS]')], dtype=torch.long)
        self.pad_token = torch.tensor([self.tokenizer_src.token_to_id('[PAD]')], dtype=torch.long)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        encode_input_tokens = self.tokenizer_src.encode(src_text).ids
        decode_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(encode_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(decode_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")
        
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(encode_input_tokens, dtype = torch.long),
            self.eos_token,
            torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype = torch.long)
        ])    

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(decode_input_tokens, dtype = torch.long),
            torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.long)
        ])        

        label = torch.cat([
            torch.tensor(decode_input_tokens, dtype = torch.long),
            self.eos_token,
            torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.long)
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input' : encoder_input,
            'decoder_input' : decoder_input,
            'encoder_mask' : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask' : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            'label' : label,
            'src_text' : src_text,
            'tgt_text' : tgt_text
        }
    
def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0