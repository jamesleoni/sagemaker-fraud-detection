import argparse
import json
import os
import random
import pandas as pd
import glob
import pickle as pkl

import xgboost

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.05)  # 0.2
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--min_child_weight", type=int, default=6)
    parser.add_argument("--silent", type=int, default=0)
    parser.add_argument("--objective", type=str, default="multi:softmax")
    parser.add_argument("--num_class", type=int, default=15)
    parser.add_argument("--num_round", type=int, default=10)
    
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    args = parser.parse_args()

    return args

def model_fn(model_dir):
    model_file = model_dir + '/model.bin'
    model = pkl.load(open(model_file, 'rb'))
    return model

def main():

    args = parse_args()

    train_file_path='/opt/ml/input/data/train/train.csv'
    val_file_path='/opt/ml/input/data/validation/validation.csv'
    
    columns=['Class', 'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
       'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
       'Amount']
    
    print('Loading training dataframe...')
    df_train = pd.read_csv(train_file_path, names=columns)
    print('Loading validation dataframe...')
    df_val = pd.read_csv(val_file_path, names=columns)
    print('Data loading completed.')
    
    y = df_train.Class.values
    X =  df_train.drop(['Class'], axis=1).values
    val_y = df_val.Class.values
    val_X = df_val.drop(['Class'], axis=1).values

    dtrain = xgboost.DMatrix(X, label=y)
    dval = xgboost.DMatrix(val_X, label=val_y)

    watchlist = [(dtrain, "train"), (dval, "validation")]

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "silent": args.silent,
        "objective": args.objective,
        "num_class": args.num_class
    }

    bst = xgboost.train(
        params=params,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round)
    
    model_dir = os.environ.get('SM_MODEL_DIR')
    pkl.dump(bst, open(model_dir + '/model.bin', 'wb'))

if __name__ == "__main__":
    main()
