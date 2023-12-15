import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import my_utils as ut


def main():
    parser = argparse.ArgumentParser(
        description='Predict the High_Open_Ratio with historical data with XGBoost')
    parser.add_argument('--delta', type=float, default=0.5, help='threshold for entering trading (%)')
    parser.add_argument('--threshold', type=float, default=0.5, help='min confidence level')
    parser.add_argument('--len_days', type=int, default=4, help='the number of days for an entity')
    args = parser.parse_args()

    # Load the dataset
    data = np.load('dataset/pca_data_vix.npy')
    ratio_raw_price = data[:, -3:]
    data = data[:, :-2]
    # split the data to fixed time-frame and labels
    train_data, val_data = ut.split_data(data, args)
    train_feature, train_label = train_data[:, :-1], train_data[:, -1, -1]
    val_feature, val_label = val_data[:, :-1], val_data[:, -1, -1]

    # flat the time-frame into one dimension
    train_feature = train_feature.reshape(train_feature.shape[0], -1)
    val_feature = val_feature.reshape(val_feature.shape[0], -1)

    # convert the labels to 0/1 labels according to their values w.r.t the setting delta
    train_label = np.where(train_label >= args.delta, 1.0, 0)
    val_label = np.where(val_label >= args.delta, 1.0, 0)

    print(np.sum(train_label) / len(train_label), np.sum(val_label)/len(val_label))

    # Convert the datasets to DMatrix format (XGBoost's internal data structure)
    dtrain = xgb.DMatrix(train_feature, label=train_label)
    dtest = xgb.DMatrix(val_feature, label=val_label)

    # Define XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',  # or 'error' for classification error
        'max_depth': 3,
        'learning_rate': 0.08,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 43
    }

    # Train the XGBoost model
    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'eval')], early_stopping_rounds=10)

    # Make predictions on the test set, the probability for long
    pred = np.array(model.predict(dtest))

    # get roc and auc for the classification
    gt = ratio_raw_price[-len(pred):, 0]
    gt = np.where(gt >= args.delta, 1, 0)
    fpr, tpr, thresholds = roc_curve(gt, pred)
    roc_auc = auc(fpr, tpr)

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

    # convert prediction to binary format and get confusion matrix
    bin_pred = np.where(pred >= args.threshold, 1, 0)

    conf_matrix = confusion_matrix(gt, bin_pred)
    tn, fp, fn, tp = confusion_matrix(gt, bin_pred).ravel()
    print(f"TN: {tn}, FN: {fn}, TP: {tp}, FP: {fp}")
    precision = precision_score(gt, bin_pred)
    recall = recall_score(gt, bin_pred)
    f1 = f1_score(gt, bin_pred)

    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Compute the profit and loss value
    PL = ut.profit_and_loss(pred, ratio_raw_price, args)
    print(PL)


if __name__ == "__main__":
    main()

