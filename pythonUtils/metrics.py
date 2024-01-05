
from sklearn.metrics import mean_absolute_error

def mae(dataloader_out, model):
    preds = None
    real = None
    for x,y in dataloader_out:
        preds = model.forward(x).tolist()
        real = y.tolist()
    return mean_absolute_error(real, preds)

def cmae_many(y_real, y_pred):
    c = len(y_real)
    loss = 0
    for i in range(c):
        loss += cmae_one(y_real[i], y_pred[i])
    return loss

def cmae_one(y_real, y_pred):
    c = len(y_real)
    loss = 0
    for i in range(c):
        loss += 
    return loss/c

def cmae(dataloader_out, model):
    preds = None
    real = None
    for x,y in dataloader_out:
        preds = model.forward(x).tolist()
        real = y.tolist()
    return cmae_many(real, preds)