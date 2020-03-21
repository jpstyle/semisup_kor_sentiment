import torch
import torch.nn as nn
import torch.nn.functional as F
from kobert_transformers import get_kobert_model, get_tokenizer

tokenizer = get_tokenizer()


class BertSentimentPredictor(nn.Module):

    def __init__(self, clf_or_reg):

        super(BertSentimentPredictor, self).__init__()

        self.clf_or_reg = clf_or_reg

        self.bert = get_kobert_model()

        if self.clf_or_reg:
            # Classification task
            self.u = nn.Linear(self.bert.pooler.dense.out_features, 3)
        else:
            # Regression task
            self.u = nn.Linear(self.bert.pooler.dense.out_features, 1)

        # Xavier inits
        nn.init.xavier_normal_(self.u.weight)
        nn.init.constant_(self.u.bias, 0.0)


    # Takes in batch of torch LongTensors processed by batch_samples
    def forward(self, batch):

        # Attention mask
        b_mask = (batch != tokenizer.pad_token_id).to(torch.long)
        b_type = torch.zeros_like(batch, dtype=torch.long)

        _, bert_cls = self.bert(batch, b_mask, b_type)

        if self.clf_or_reg:
            squashed = self.u(bert_cls)

            return squashed
        else:
            squashed = self.u(bert_cls).squeeze()
            squashed = torch.sigmoid(squashed)
 
            return squashed * 4 # Range of 0 ~ 4


# Process and batch samples from generator
def batch_samples(examples, batch_size, cuda_device):
    batch = []

    # Helper method to pack samples into a single tensor
    def pack_samples(batch):
        tokenized_b = [tokenizer.tokenize(e[0]) for e in batch]
        max_len = max([len(e) for e in tokenized_b])

        # Return val
        b_as_tensor = []

        # Fill in pad tensors
        for e in tokenized_b:
            padded_with_cls = [tokenizer.cls_token] + e + ([tokenizer.pad_token] * (max_len - len(e)))
            padded_with_cls = tokenizer.convert_tokens_to_ids(padded_with_cls)

            b_as_tensor.append(torch.LongTensor(padded_with_cls))

        # Stack batch as single tensor
        b_as_tensor = torch.stack(b_as_tensor, dim=0)

        if batch[0][1] is not None:
            b_scores = torch.FloatTensor([float(e[1]) for e in batch])
        else:
            b_scores = None

        if cuda_device > -1:
            b_as_tensor = b_as_tensor.to(f"cuda:{cuda_device}")

            if b_scores is not None:
                b_scores = b_scores.to(f"cuda:{cuda_device}")

        return b_as_tensor, b_scores

    for ex in examples:
        # Ignore NA-scored instances
        if ex[1] == "NA":
            continue

        if len(batch) >= batch_size:
            yield pack_samples(batch)
            batch = [ex]
        else:
            batch.append(ex)

    # Final batch
    if len(batch) > 0:
        yield pack_samples(batch)


if __name__ == "__main__":
    # Test KoBERT
    import data

    labeled = data.read_csv("data/sample.csv")
    b_gen = batch_samples(labeled, 8)

    model = BertSentimentPredictor()
    model.train()

    input_b = next(b_gen)[0]
    print(model(input_b))
    print(model(input_b))
