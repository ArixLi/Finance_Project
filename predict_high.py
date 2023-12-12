import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from dataloader import MyDataset
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import my_utils as ut
import logging


def main():
    parser = argparse.ArgumentParser(
        description='Predict the High_Open_Ratio with historical data')
    parser.add_argument('--num_epochs', type=int, default=15, help='number of epochs to train the soft-prompt')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--delta', type=float, default=0.08, help='threshold for entering trading (%)')
    parser.add_argument('--threshold', type=float, default=0.55, help='min confidence level')
    parser.add_argument('--len_days', type=int, default=30, help='the number of days for an entity')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load('dataset/pca_data_vix.npy')
    ratio_raw_price = data[:, -3:]
    data = data[:, :-2]
    train_data, val_data = ut.split_data(data, args)
    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)

    train_ds = DataLoader(MyDataset(train_data, args), batch_size=args.bs,
                          shuffle=True, num_workers=0)
    val_ds = DataLoader(MyDataset(val_data, args), batch_size=args.bs,
                        shuffle=True, num_workers=0)

    model = ut.SimpleLinearModel(input_size=train_data.size(-1)-1, days=args.len_days)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    file_name = f"SimpleModel_{args.len_days}"
    writer = SummaryWriter('logs/' + file_name)

    model.to(device)

    for epoch in range(args.num_epochs):
        model.train()
        tot_loss = 0
        loop = tqdm(enumerate(train_ds), total=len(train_ds))
        for step, batch in loop:
            optimizer.zero_grad()

            feature = batch["feature"].to(device)
            label = batch["label"].unsqueeze(1).to(device)
            outputs = model(feature)
            loss = loss_fn(outputs, label)

            optimizer.step()
            loss.backward()
            tot_loss += loss.item()

            loop.set_description(f"Train Epoch: [{epoch + 1}/{args.num_epochs}]")
            loop.set_postfix(loss=loss.item())

        tr_loss = tot_loss / len(train_ds)
        tot_loss = 0
        pred = []
        with torch.inference_mode():
            loop = tqdm(enumerate(val_ds), total=len(val_ds))
            for step, batch in loop:
                feature = batch["feature"].to(device)
                label = batch["label"].unsqueeze(1).to(device)
                outputs = model(feature)
                loss = loss_fn(outputs, label)
                tot_loss += loss.item()
                pred.append(outputs.cpu().numpy())
                loop.set_description(f"Test Epoch: [{epoch + 1}/{args.num_epochs}]")
                loop.set_postfix(loss=loss.item())

            test_loss = tot_loss / len(val_ds)
            pred = np.concatenate(pred, axis=0).reshape(-1)
            PL = ut.profit_and_loss(pred, ratio_raw_price[-len(pred):], args)
            if epoch+1 == args.num_epochs:
                print(pred)
                gt = ratio_raw_price[-len(pred):, 0]
                gt = np.where(gt > args.delta, 1, 0)
                fpr, tpr, thresholds = roc_curve(gt, pred)
                roc_auc = auc(fpr, tpr)
            writer.add_scalar('Train/Loss', tr_loss, epoch)
            writer.add_scalar('Test/Loss', test_loss, epoch)
            writer.add_scalar('Test/PL', PL, epoch)

    writer.flush()
    writer.close()

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve\nAUC:{roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    pred = np.where(pred >= args.threshold, 1, 0)
    print(pred)
    conf_matrix = confusion_matrix(gt, pred)
    precision = precision_score(gt, pred)
    recall = recall_score(gt, pred)
    f1 = f1_score(gt, pred)

    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()