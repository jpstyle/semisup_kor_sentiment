import torch
import torch.nn as nn
import torch.nn.functional as F
import jamo
from itertools import accumulate
from collections import defaultdict

CH_SIZE = 200
CH_KERNELS = [(2, 25), (3, 50), (4, 75)]

JM_SIZE = 100
JM_KERNELS = [(3, 25), (5, 50), (7, 75)]

LSTM_DIM = 300


class CBiLSTM(nn.Module):

    def __init__(self, ch_count, jm_count):

        super(CBiLSTM, self).__init__()

        self.ch_count = ch_count
        self.jm_count = jm_count

        # Character/Jamo embeddings
        self.ch_embs = nn.Embedding(self.ch_count, CH_SIZE, padding_idx=0)
        self.jm_embs = nn.Embedding(self.jm_count, JM_SIZE, padding_idx=0)

        # Conv layers
        self.ch_convs = [nn.Conv1d(CH_SIZE, k_out, k_size) for k_size, k_out in CH_KERNELS]
        self.jm_convs = [nn.Conv1d(JM_SIZE, k_out, k_size) for k_size, k_out in JM_KERNELS]

        # 2-layer bidirectional LSTM
        self.lstm = nn.LSTM(input_size=LSTM_DIM, hidden_size=LSTM_DIM, num_layers=2, bidirectional=True, dropout=0.3, batch_first=True)

        # Squasher
        self.u = nn.Linear(LSTM_DIM * 2, 1)

        # Xavier inits
        for name, prm in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(prm)
            elif "bias" in name:
                nn.init.constant(prm, 0.0)


    def forward(self, batch_ch, batch_jm, batch_lens):
        b_size = batch_ch.shape[0]; max_T = batch_ch.shape[1]
        max_ch_len = batch_ch.shape[2]; max_jm_len = batch_jm.shape[2]

        # Embed sequence batch
        ch_seqs = self.ch_embs(batch_ch)
        jm_seqs = self.jm_embs(batch_jm)

        # Prepare conv. with view change
        ch_seqs = ch_seqs.view(-1, max_ch_len, CH_SIZE).transpose(-1, -2)
        jm_seqs = jm_seqs.view(-1, max_jm_len, JM_SIZE).transpose(-1, -2)

        # Apply convolution layer & max pool
        ch_seqs = [ch_cv(ch_seqs).max(dim=-1).values.view(b_size, max_T, -1) for ch_cv in self.ch_convs]
        jm_seqs = [jm_cv(jm_seqs).max(dim=-1).values.view(b_size, max_T, -1) for jm_cv in self.jm_convs]

        # Concat to single vecs
        ch_seqs = torch.cat(ch_seqs, dim=-1); jm_seqs = torch.cat(jm_seqs, dim=-1)
        seqs = torch.cat([ch_seqs, jm_seqs], dim=-1)

        # Feed to LSTM and retrieve summarizing representations
        _, (final_states, _) = self.lstm(seqs)
        seq_embs = final_states.view(2, 2, b_size, -1) # (num_layer, num_direction, b_size, hidden_size)
        seq_embs = seq_embs[1,:,:,:]
        seq_embs = torch.cat([seq_embs[0], seq_embs[1]], dim=-1)

        squashed = self.u(seq_embs)
        squashed = torch.sigmoid(squashed)
 
        return squashed * 4 # Range of 0 ~ 4


def batch_samples(gen, batch_size, c2i, j2i):
    batch = []

    # Helper method to pack samples into a single tensor
    def pack_samples(batch):
        # Return val
        b_as_char_tensor = []; b_as_jamo_tensor = []

        for e in batch:
            e_char_seq = [torch.LongTensor([c2i[c] for c in tok]) for tok in e[0].split()]
            e_jamo_seq = [torch.LongTensor([j2i[j] for j in jamo.j2hcj(jamo.h2j(tok))]) for tok in e[0].split()]

            b_as_char_tensor.append(e_char_seq); b_as_jamo_tensor.append(e_jamo_seq)

        b_lens = [len(t) for t in b_as_char_tensor]

        b_ch_padded = nn.utils.rnn.pad_sequence(sum(b_as_char_tensor, []), batch_first=True)
        b_jm_padded = nn.utils.rnn.pad_sequence(sum(b_as_jamo_tensor, []), batch_first=True)

        b_as_char_tensor = [b_ch_padded[x-y:x] for x, y in zip(accumulate(b_lens), b_lens)]
        b_as_jamo_tensor = [b_jm_padded[x-y:x] for x, y in zip(accumulate(b_lens), b_lens)]

        b_as_char_tensor = nn.utils.rnn.pad_sequence(b_as_char_tensor, batch_first=True)
        b_as_jamo_tensor = nn.utils.rnn.pad_sequence(b_as_jamo_tensor, batch_first=True)

        assert b_as_char_tensor.shape[0] == b_as_char_tensor.shape[0] # Same batch size
        assert b_as_char_tensor.shape[1] == b_as_char_tensor.shape[1] # Same max token count
        assert b_as_jamo_tensor.shape[0] == b_as_jamo_tensor.shape[0]
        assert b_as_jamo_tensor.shape[1] == b_as_jamo_tensor.shape[1]

        if batch[0][1] is not None:
            b_scores = torch.FloatTensor([float(e[1]) for e in batch])
        else:
            b_scores = None

        return b_as_char_tensor, b_as_jamo_tensor, b_lens, b_scores

    for ex in gen:
        # Ignore NA-scored instances
        if ex[1] == "NA":
            continue

        if len(batch) >= batch_size:
            yield pack_samples(batch)
            batch = []
        else:
            batch.append(ex)

    # Final batch
    if len(batch) > 0:
        yield pack_samples(batch)


if __name__ == "__main__":
    # Test CBiLSTM
    import data

    labeled = data.read_csv("data/sample.csv")
    unlabeled = data.read_csv("data/thaad_relevant.csv")

    # First get the char/jamo vocabulary
    c2i = defaultdict(lambda: len(c2i))
    j2i = defaultdict(lambda: len(j2i))
    c2i["[PAD]"]; j2i["[PAD]"]

    for i, ex in enumerate(labeled):
        print(f"Reading char & jamo vocabulary from labeled texts: {i}", end="\r")
        for c in ex[0]: c2i[c]
        for j in jamo.j2hcj(jamo.h2j(ex[0])): j2i[j]
    # for i, ex in enumerate(unlabeled):
    #     print(f"Reading char & jamo vocabulary from unlabeled texts: {i}", end="\r")
    #     for c in ex[0]: c2i[c]
    #     for j in jamo.j2hcj(jamo.h2j(ex[0])): j2i[j]

    c2i = dict(c2i); j2i = dict(j2i)

    labeled = data.read_csv("data/sample.csv")
    b_gen = batch_samples(labeled, 8, c2i, j2i)

    model = CBiLSTM(len(c2i), len(j2i))
    model.train()

    input_b = next(b_gen)
    print(model(input_b[0], input_b[1], input_b[2]))
    print(model(input_b[0], input_b[1], input_b[2]))
