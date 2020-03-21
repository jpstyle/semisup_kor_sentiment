import random
import argparse
import data
import cbilstm
import kobert
import jamo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict


def parse_arguments():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Semisupervised sentiment analysis by co-training")
    parser.add_argument("-lp", "--labeled_path",
        type=str,
        default="",
        help="path to labeled data tsv file")
    parser.add_argument("-up", "--unlabeled_path",
        type=str,
        default="",
        help="path to unlabeled data tsv file")
    parser.add_argument("-ep", "--epoch",
        type=int,
        default=100,
        help="Co-training epoch number")
    parser.add_argument("-ph", "--preheat",
        type=int,
        default=5,
        help="Epochs of 'pre-heat' training")
    parser.add_argument("-b", "--batch_size",
        type=int,
        default=24,
        help="Training batch size")
    parser.add_argument("-c", "--cuda_device",
        type=int,
        default=-1,
        help="CUDA device to use")
    parser.add_argument("-m", "--exp_mode",
        type=str,
        choices = ["clf", "reg"],
        default="clf",
        help="experiment mode; 3-category classification or real-value regression")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Setting shortcuts
    CLF_OR_REG = args.exp_mode == "clf" # True is CLF, False is REG
    CUDA = args.cuda_device
    COTR_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    N = BATCH_SIZE * 3
    U_SUB_SIZE = BATCH_SIZE * 12

    # Char/jamo vocabularies
    c2i = defaultdict(lambda: len(c2i))
    j2i = defaultdict(lambda: len(j2i))
    c2i["[PAD]"]; j2i["[PAD]"]

    # Data generators
    labeled = data.read_csv(args.labeled_path)
    unlabeled = data.read_csv(args.unlabeled_path)

    # Labeled & unlabeled pool of data
    L = []
    U = []
    U_sub = []

    # Fill in L, U and U_sub
    for i, ex in enumerate(labeled):
        print(f"Reading from labeled texts: {i}", end="\r")
        for c in ex[0]: c2i[c]
        for j in jamo.j2hcj(jamo.h2j(ex[0])): j2i[j]
        L.append(ex)

    for i, ex in enumerate(unlabeled):
        print(f"Reading from unlabeled texts: {i}", end="\r")
        for c in ex[0]: c2i[c]
        for j in jamo.j2hcj(jamo.h2j(ex[0])): j2i[j]
        U.append(ex)

    rnd_inds = sorted(random.sample(range(len(U)), U_SUB_SIZE), reverse=True)
    for i in rnd_inds:
        U_sub.append(U[i])
        del U[i]

    c2i = dict(c2i); j2i = dict(j2i)

    # Train/dev/test split of L
    sp1 = int(len(L)*0.8); sp2 = int(len(L)*0.9)
    L_tr = L[:sp1]; L_dev = L[sp1:sp2]; L_test = L[sp2:]

    L_c = L_tr; L_b = L_tr

    # Initialize two training models
    m_cbilstm = cbilstm.CBiLSTM(len(c2i), len(j2i), CLF_OR_REG)
    m_kobert = kobert.BertSentimentPredictor(CLF_OR_REG)

    if CUDA > -1:
        print(f"\nUsing CUDA device {CUDA}")
        m_cbilstm.to(f"cuda:{CUDA}"); m_kobert.to(f"cuda:{CUDA}")

    optim_c = optim.Adam(m_cbilstm.parameters())
    optim_b = optim.Adam(m_kobert.parameters())

    if CLF_OR_REG:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()


    # Dev loss before training
    m_cbilstm.eval(); m_kobert.eval()

    dev_gen_c = cbilstm.batch_samples(L_dev, BATCH_SIZE, c2i, j2i, CUDA)
    dev_gen_b = kobert.batch_samples(L_dev, BATCH_SIZE, CUDA)

    dev_loss_c = 0.0
    for i, batch in enumerate(dev_gen_c):
        print(f"Epoch 0: Computing CBiLSTM dev loss, batch {i+1}", end="\r")

        batch_ch, batch_jm, batch_lens, batch_scores = batch

        if CLF_OR_REG:
            # Cast to 3-category representation
            target = (batch_scores == 2.0) * 1 + (batch_scores >= 3.0) * 2
        else:
            # As-is regression target
            target = batch_scores

        # Compute loss
        out = m_cbilstm(batch_ch, batch_jm, batch_lens)
        loss = criterion(out, target)

        dev_loss_c += loss.item()

    dev_loss_b = 0.0
    for i, batch in enumerate(dev_gen_b):
        print(f"Epoch 0: Computing KoBERT dev loss, batch {i+1}", end="\r")

        batch_bert, batch_scores = batch

        if CLF_OR_REG:
            # Cast to 3-category representation
            target = (batch_scores == 2.0) * 1 + (batch_scores >= 3.0) * 2
        else:
            # As-is regression target
            target = batch_scores

        # Compute loss
        out = m_kobert(batch_bert)
        loss = criterion(out, target)

        dev_loss_b += loss.item()

    print(f"Before training: Dev loss CBiLSTM {dev_loss_c:.4f}, KoBERT {dev_loss_b:.4f}")

    ## Co-training loop
    for t in range(COTR_EPOCH):
        m_kobert.train(); m_cbilstm.train() # Train mode

        # Batch generators
        random.shuffle(L_c); random.shuffle(L_b)
        tr_gen_c = cbilstm.batch_samples(L_c, BATCH_SIZE, c2i, j2i, CUDA)
        tr_gen_b = kobert.batch_samples(L_b, BATCH_SIZE, CUDA)

        # Train both models for an epoch with L_c/L_b
        for i, batch in enumerate(tr_gen_c):
            batch_ch, batch_jm, batch_lens, batch_scores = batch

            if CLF_OR_REG:
                # Cast to 3-category representation
                target = (batch_scores == 2.0) * 1 + (batch_scores >= 3.0) * 2
            else:
                # As-is regression target
                target = batch_scores

            # Clear grads
            optim_c.zero_grad()

            # Compute loss
            out = m_cbilstm(batch_ch, batch_jm, batch_lens)
            loss = criterion(out, target)
            loss.backward()

            print(f"Epoch {t+1}: Training CBiLSTM, batch {i+1} - loss {loss.item():.4f}", end="\r")

            # Update
            optim_c.step()

        for i, batch in enumerate(tr_gen_b):
            batch_bert, batch_scores = batch

            if CLF_OR_REG:
                # Cast to 3-category representation
                target = (batch_scores == 2.0) * 1 + (batch_scores >= 3.0) * 2
            else:
                # As-is regression target
                target = batch_scores

            # Clear grads
            optim_b.zero_grad()

            # Compute loss
            out = m_kobert(batch_bert)
            loss = criterion(out, target)
            loss.backward()

            print(f"Epoch {t+1}: Training KoBERT, batch {i+1} - loss {loss.item():.4f}", end="\r")

            # Update
            optim_b.step()
        

        # Dev loss after each epoch
        m_cbilstm.eval(); m_kobert.eval()

        dev_gen_c = cbilstm.batch_samples(L_dev, BATCH_SIZE, c2i, j2i, CUDA)
        dev_gen_b = kobert.batch_samples(L_dev, BATCH_SIZE, CUDA)

        dev_loss_c = 0.0
        for i, batch in enumerate(dev_gen_c):
            print(f"Epoch {t+1}: Computing CBiLSTM dev loss, batch {i+1}", end="\r")

            batch_ch, batch_jm, batch_lens, batch_scores = batch

            if CLF_OR_REG:
                # Cast to 3-category representation
                target = (batch_scores == 2.0) * 1 + (batch_scores >= 3.0) * 2
            else:
                # As-is regression target
                target = batch_scores

            # Compute loss
            out = m_cbilstm(batch_ch, batch_jm, batch_lens)
            loss = criterion(out, target)

            dev_loss_c += loss.item()

        dev_loss_b = 0.0
        for i, batch in enumerate(dev_gen_b):
            print(f"Epoch {t+1}: Computing KoBERT dev loss, batch {i+1}", end="\r")

            batch_bert, batch_scores = batch

            if CLF_OR_REG:
                # Cast to 3-category representation
                target = (batch_scores == 2.0) * 1 + (batch_scores >= 3.0) * 2
            else:
                # As-is regression target
                target = batch_scores

            # Compute loss
            out = m_kobert(batch_bert)
            loss = criterion(out, target)

            dev_loss_b += loss.item()

        print(f"Epoch {t+1}: Dev loss CBiLSTM {dev_loss_c:.4f}, KoBERT {dev_loss_b:.4f}")


        # 'Pre-heat' for certain number of epochs, not adding instances to L_c/L_b
        if t < args.preheat:
            continue


        # Choose most confident examples from U', for both models
        confs_c = []; confs_b = []

        u_gen_c = cbilstm.batch_samples(U_sub, BATCH_SIZE, c2i, j2i, CUDA)
        u_gen_b = kobert.batch_samples(U_sub, BATCH_SIZE, CUDA)

        for i, batch in enumerate(u_gen_c):
            batch_ch, batch_jm, batch_lens, _ = batch

            if CLF_OR_REG:
                # Straightforward softmax-based confidence metric
                out = m_cbilstm(batch_ch, batch_jm, batch_lens)
                out = F.softmax(out, dim=1).max(dim=1).values
                confs_c.append(out)
            else:
                # Drop-out based confidence metric
                with torch.no_grad():
                    m_cbilstm.train()
                    outs = []
                    for n in range(20):
                        outs.append(m_cbilstm(batch_ch, batch_jm, batch_lens))
                out = -torch.stack(outs, dim=-1).std(dim=-1)
                confs_c.append(out)
                m_cbilstm.eval()

        confs_c = torch.cat(confs_c)
        topN_inds_c = [i.item() for i in confs_c.topk(N).indices]

        for i, batch in enumerate(u_gen_b):
            batch_bert, _ = batch

            if CLF_OR_REG:
                # Straightforward softmax-based confidence metric
                out = m_kobert(batch_bert)
                out = F.softmax(out, dim=1).max(dim=1).values
                confs_b.append(out)
            else:
                # Drop-out based confidence metric
                with torch.no_grad():
                    m_kobert.train()
                    outs = []
                    for n in range(15):
                        outs.append(m_kobert(batch_bert))
                out = -torch.stack(outs, dim=-1).std(dim=-1)
                confs_b.append(out)
                m_kobert.eval()

        confs_b = torch.cat(confs_b)
        topN_inds_b = [i.item() for i in confs_b.topk(N).indices]

        # Add those to L_c/L_b
        for i in topN_inds_c:
            ins = list(cbilstm.batch_samples([U_sub[i]], 1, c2i, j2i, CUDA))[0]

            if CLF_OR_REG:
                label = m_cbilstm(ins[0], ins[1], ins[2]).max(dim=-1).indices.item() + 1
            else:
                label = round(m_cbilstm(ins[0], ins[1], ins[2]).item(), 3)

            L_c.append((U_sub[i][0], str(label)))
        for i in topN_inds_b:
            ins = list(kobert.batch_samples([U_sub[i]], 1, CUDA))[0]

            if CLF_OR_REG:
                label = m_kobert(ins[0]).max(dim=-1).indices.item() + 1
            else:
                label = round(m_kobert(ins[0]).item(), 3)

            L_b.append((U_sub[i][0], str(label)))


        # Remove the added instances from U_sub
        to_remove = sorted(list(set(topN_inds_c) | set(topN_inds_b)), reverse=True)
        for i in to_remove:
            del U_sub[i]

        # Replenish U' from U
        rnd_inds = sorted(random.sample(range(len(U)), len(to_remove)), reverse=True)
        for i in rnd_inds:
            U_sub.append(U[i])
            del U[i]

    # One final training epoch

    # Final evaluation on test data; accuracy for clf, squared err for reg

    # Save trained models
    checkpoint = {
        "c2i": c2i,
        "j2i": j2i,
        "model_c": m_cbilstm.module.state_dict(),
        "model_b": m_kobert.module.state_dict()
    }
    
    if not os.path.isdir("models"):
        os.mkdir("models")
    save_path = os.path.join("models", f"{model_name}.pt")
    torch.save(checkpoint, save_path)

    print("----------------")
    print(f"Saved model checkpoint to {model_save_path}")
