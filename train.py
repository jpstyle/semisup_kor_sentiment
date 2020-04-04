import os
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

import faulthandler; faulthandler.enable()


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
    parser.add_argument("-mp", "--model_path",
        type=str,
        help="path to model, for continuing training")
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
        default=[], nargs="+",
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
    N = BATCH_SIZE * 4
    U_SUB_SIZE = BATCH_SIZE * 12

    # Data generators
    labeled = data.read_csv(args.labeled_path)
    unlabeled = data.read_csv(args.unlabeled_path)

    # Char/jamo vocabularies
    if args.model_path is None:
        c2i = defaultdict(lambda: len(c2i))
        j2i = defaultdict(lambda: len(j2i))
        c2i["[PAD]"]; j2i["[PAD]"]
    else:
        checkpoint = torch.load(args.model_path)
        c2i = checkpoint["c2i"]
        j2i = checkpoint["j2i"]

    # Labeled & unlabeled pool of data
    L = []
    U = []
    U_sub = []

    l = u = 0

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

    lr_c = 0.00001 if CLF_OR_REG else 0.00005
    lr_b = 0.000001 if CLF_OR_REG else 0.000005

    optim_c = optim.AdamW(m_cbilstm.parameters(), lr=lr_c)
    optim_b = optim.AdamW(m_kobert.parameters(), lr=lr_b)

    if args.model_path is not None:
        m_cbilstm.load_state_dict(checkpoint["model_c"])
        m_kobert.load_state_dict(checkpoint["model_b"])

        optim_c.load_state_dict(checkpoint["optim_c"])
        optim_b.load_state_dict(checkpoint["optim_b"])

    if len(CUDA) > 0:
        print(f"\nUsing CUDA device {CUDA}")

        if len(CUDA) > 1:
            m_cbilstm = nn.DataParallel(m_cbilstm, device_ids=CUDA)
            m_kobert = nn.DataParallel(m_kobert, device_ids=CUDA)

        m_cbilstm.to(f"cuda:{CUDA[0]}"); m_kobert.to(f"cuda:{CUDA[0]}")

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

        dev_loss_c += loss.item() * batch_ch.shape[0]

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

        dev_loss_b += loss.item() * batch_bert.shape[0]

    print(f"Before training: Dev loss CBiLSTM {dev_loss_c:.4f}, KoBERT {dev_loss_b:.4f}")

    if args.model_path is None:
        ep_start = 0
        c_devloss_best = dev_loss_c
        b_devloss_best = dev_loss_b
    else:
        ep_start = checkpoint["epoch"]
        c_devloss_best = checkpoint["devloss_c"]
        b_devloss_best = checkpoint["devloss_b"]

    ## Co-training loop
    for t in range(ep_start, COTR_EPOCH):

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

            dev_loss_c += loss.item() * batch_ch.shape[0]

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

            dev_loss_b += loss.item() * batch_bert.shape[0]

        print(f"Epoch {t+1}: Dev loss CBiLSTM {dev_loss_c:.4f}, KoBERT {dev_loss_b:.4f}")


        # 'Pre-heat' for certain number of epochs, not adding instances to L_c/L_b
        if t < args.preheat:
            continue

        # Finish here for the last epoch
        if t == COTR_EPOCH-1:
            break


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
                    for n in range(15):
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

        if dev_loss_c < c_devloss_best and dev_loss_b < b_devloss_best:
            c_devloss_best = dev_loss_c
            b_devloss_best = dev_loss_b

            # Save trained models
            checkpoint = {
                "epoch": t,
                "c2i": c2i,
                "j2i": j2i,
                "model_c": m_cbilstm.module.state_dict() if len(args.cuda_device) > 1 else m_cbilstm.state_dict(),
                "model_b": m_kobert.module.state_dict() if len(args.cuda_device) > 1 else m_kobert.state_dict(),
                "optim_c": optim_c.state_dict(),
                "optim_b": optim_b.state_dict(),
                "devloss_c": c_devloss_best,
                "devloss_b": b_devloss_best
            }
    
            if not os.path.isdir("models"):
                os.mkdir("models")
            save_path = os.path.join("models", f"{args.exp_mode}_{args.preheat}_{args.batch_size}.pt")
            torch.save(checkpoint, save_path)
