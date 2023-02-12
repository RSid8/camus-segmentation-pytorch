import torch


def multiclass_dice(y_true, y_pred, smooth=1e-7, num_classes=4):
    """
    Multiclass Dice score. Ignores bactorchground pixel label 0
    Pass to model as metric during compile statement
    """
    y_true_f = torch.flatten(
        torch.one_hot(y_true.type(torch.int), num_classes=4)[:, 1:, :, :]
    )
    y_pred_f = torch.flatten(
        torch.one_hot(torch.argmax(y_pred, axis=3), num_classes=4)[:, 1:, :, :]
    )
    intersect = torch.sum(y_true_f * y_pred_f, axis=-1)
    denom = torch.sum(y_true_f + y_pred_f, axis=-1)
    return torch.mean((2.0 * intersect / (denom + smooth)))


def dice_lv(y_true, y_pred, smooth=1e-7, num_classes=4):
    """
    Multiclass Dice score. Ignores bactorchground pixel label 0
    Pass to model as metric during compile statement
    """
    y_true_f = torch.flatten(
        torch.one_hot(y_true.type(torch.int), num_classes=4)[:, 1:2, :, :]
    )
    y_pred_f = torch.flatten(
        torch.one_hot(torch.argmax(y_pred, axis=3), num_classes=4)[:, 1:2, :, :]
    )
    intersect = torch.sum(y_true_f * y_pred_f, axis=-1)
    denom = torch.sum(y_true_f + y_pred_f, axis=-1)
    return torch.mean((2.0 * intersect / (denom + smooth)))


def dice_la(y_true, y_pred, smooth=1e-7, num_classes=4):
    """
    Multiclass Dice score. Ignores bactorchground pixel label 0
    Pass to model as metric during compile statement
    """
    y_true_f = torch.flatten(
        torch.one_hot(y_true.type(torch.int), num_classes=4)[:, 3:4, :, :]
    )
    y_pred_f = torch.flatten(
        torch.one_hot(torch.argmax(y_pred, axis=3), num_classes=4)[:, 3:4, :, :]
    )
    intersect = torch.sum(y_true_f * y_pred_f, axis=-1)
    denom = torch.sum(y_true_f + y_pred_f, axis=-1)
    return torch.mean((2.0 * intersect / (denom + smooth)))


def dice_myo(y_true, y_pred, smooth=1e-7, num_classes=4):
    """
    Multiclass Dice score. Ignores bactorchground pixel label 0
    Pass to model as metric during compile statement
    """
    y_true_f = torch.flatten(
        torch.one_hot(y_true.type(torch.int), num_classes=4)[:, 2:3, :, :]
    )
    y_pred_f = torch.flatten(
        torch.one_hot(torch.argmax(y_pred, axis=3), num_classes=4)[:, 2:3, :, :]
    )
    intersect = torch.sum(y_true_f * y_pred_f, axis=-1)
    denom = torch.sum(y_true_f + y_pred_f, axis=-1)
    return torch.mean((2.0 * intersect / (denom + smooth)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
