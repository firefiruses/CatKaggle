
from sklearn.metrics import mean_absolute_error

def mae(dataloader_out, model):
    preds = None
    real = None
    for x,y in dataloader_out:
        preds = model.forward(x).tolist()
        real = y.tolist()
    return mean_absolute_error(real, preds)
