import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


def to_device(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))


def train_val_test_split(df,
                         test_size=0.3,
                         ignore_pumas=True,
                         data_folder='/content/gdrive/MyDrive/IFT6756/data/TrainValTest',
                         save=True,
                         random_state=42,
                         shuffle=True,):
    if ignore_pumas:
        df = df.drop(['PUMA', 'ST'], axis=1)

    train, test = train_test_split(df,
                                   test_size=test_size,
                                   random_state=random_state,
                                   shuffle=shuffle)

    train, val = train_test_split(train,
                                  test_size=test_size,
                                  shuffle=False)

    if save:
        train.to_csv(path_or_buf=data_folder+"/train.csv")
        val.to_csv(path_or_buf=data_folder+"/val.csv")
        test.to_csv(path_or_buf=data_folder+"/test.csv")

    return train, val, test


def generate_samples(generator, batch_size, random_dim):
    z = torch.FloatTensor(np.random.normal(
        0, 1, size=(batch_size, random_dim)))
    batch_synthetic = generator(z)
    batch_synthetic = generator.decoder(batch_synthetic)
    return np.round(batch_synthetic.cpu().detach().numpy())


def DWP(train_real: pd.DataFrame, 
        train_synthetic: pd.DataFrame, 
        val: pd.DataFrame, 
        label: str):
    # Data
    label_real = train_real[label].values
    train_real = train_real.drop([label], axis=1).values
    label_synthetic = train_synthetic[label].values
    train_synthetic = train_synthetic.drop([label], axis=1).values
    val_label = val[label].values
    val = val.drop([label], axis=1)

    # Logistic regression
    model_real = LogisticRegression().fit(train_real, label_real)
    model_synthetic = LogisticRegression().fit(train_synthetic, label_synthetic)

    # Evaluate performance
    pred_real = model_real.predict(val)
    pred_synthetic = model_synthetic.predict(val)
    f1_real = f1_score(val_label, pred_real, average="weighted")
    f1_synthetic = f1_score(val_label, pred_synthetic, average="weighted")

    return f1_real, f1_synthetic
