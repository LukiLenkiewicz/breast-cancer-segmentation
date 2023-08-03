import torch


def get_precision(true: torch.Tensor, pred: torch.Tensor):
    return float(((true*pred).sum()/pred.sum()))

def get_recall(true: torch.Tensor, pred: torch.Tensor):
    return float(((true*pred).sum()/true.sum()))

def get_accuracy(y: torch.Tensor , y_pred: torch.Tensor) -> float:
    return float(torch.sum(y == y_pred)/torch.numel(y))
