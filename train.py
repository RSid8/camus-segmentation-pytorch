import torch
import numpy as np
from camus_unet.camus_unet2 import CamusUnet2
from metrics import multiclass_dice, dice_myo, dice_la, dice_lv, count_parameters
from camus_dataset import Camus, ResizeImagesAndLabels
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceCELoss
from torchvision import transforms
from torch.nn.functional import one_hot

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = CamusUnet2()  # initialise the U-NET 2 model
print("Num Parameters: ", count_parameters(model))
# move initialised model to chosen device
model = model.to(device)

# Dataset loading
root = "/home/sidharth/Desktop/CAMUS_public/database/"  # Path to data
# save_path = "./../../data" # Path to save the extracted images
split = "train"
global_transforms = transforms.Resize(size=[224, 224])
augment_transforms = transforms.ToTensor()

train_dataset = Camus(
    root=root,
    split=split,
    global_transforms=global_transforms,
    augment_transforms=augment_transforms,
)
val_dataset = Camus(
    root=root,
    split="val",
    global_transforms=global_transforms,
    augment_transforms=augment_transforms,
)

param_Loader = {"batch_size": 1, "shuffle": True, "num_workers": 4}

train_dataloader = torch.utils.data.DataLoader(train_dataset, **param_Loader)
val_dataloader = torch.utils.data.DataLoader(val_dataset, **param_Loader)

# Put in the required training loop intitialisations
writer = SummaryWriter("runs/2DUnetCamus")
criterion = DiceCELoss(softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
losses = []
val_losses = []
dice_list = []
dice_lv_list = []
dice_la_list = []
dice_myo_list = []
accumulate_batch = 8  # mini-batch size by gradient accumulation
accumulated = 0

# Training Loop
filename = "camus_unet_model.pt"


def run(lr=1e-4, epochs=500):
    accumulated = 0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    for epoch in range(epochs):
        # training loop with gradient accumulation
        running_loss = 0.0
        for data in tqdm(train_dataloader):
            CH2_ED = data["2CH_ED"]
            CH2_ES = data["2CH_ES"]
            CH4_ED = data["4CH_ED"]
            CH4_ES = data["4CH_ES"]
            CH2_ED_gt = data["2CH_ED_gt"]
            CH2_ES_gt = data["2CH_ES_gt"]
            CH4_ED_gt = data["4CH_ED_gt"]
            CH4_ES_gt = data["4CH_ES_gt"]
            inputs = torch.stack((CH2_ED, CH2_ES, CH4_ED, CH4_ES), dim=0)
            labels = torch.stack((CH2_ED_gt, CH2_ES_gt, CH4_ED_gt, CH4_ES_gt), dim=0)
            print(inputs.shape)
            print(labels.shape)
            labels = labels.long()
            labels_one_hot = (
                one_hot(labels, num_classes=4).squeeze(1).permute(0, 3, 1, 2)
            )
            outputs = model(inputs)
            loss = criterion(outputs, labels_one_hot) / accumulate_batch
            loss.backward()
            accumulated += 1
            if accumulated == accumulate_batch:
                optimizer.step()
                optimizer.zero_grad()
                accumulated = 0

            running_loss += loss.item() * accumulate_batch
        writer.add_scalar("Loss/train", running_loss / len(train_dataloader), epoch)
        losses.append(running_loss / len(train_dataloader))
        # validation loop
        with torch.no_grad():
            running_loss = 0.0
            running_dice = 0.0
            running_dice_lv = 0.0
            running_dice_la = 0.0
            running_dice_myo = 0.0
            for data in tqdm(val_dataloader):
                CH2_ED = data["2CH_ED"]
                CH2_ES = data["2CH_ES"]
                CH4_ED = data["4CH_ED"]
                CH4_ES = data["4CH_ES"]
                CH2_ED_gt = data["2CH_ED_gt"]
                CH2_ES_gt = data["2CH_ES_gt"]
                CH4_ED_gt = data["4CH_ED_gt"]
                CH4_ES_gt = data["4CH_ES_gt"]
                inputs = torch.stack((CH2_ED, CH2_ES, CH4_ED, CH4_ES), dim=0)
                labels = torch.stack(
                    (CH2_ED_gt, CH2_ES_gt, CH4_ED_gt, CH4_ES_gt), dim=0
                )
                labels = labels.long()
                labels_one_hot = (
                    one_hot(labels, num_classes=4).squeeze(1).permute(0, 3, 1, 2)
                )
                # labels = torch.argmax(labels, axis=1)
                outputs = model(inputs)
                loss = criterion(outputs, labels_one_hot)
                # loss = criterion(outputs, labels_one_hot)/ accumulate_batch
                running_loss += loss.item()
                outputs = torch.softmax(outputs, 1)
                outputs = torch.argmax(outputs, axis=1)
                running_dice += multiclass_dice(outputs, labels)
                running_dice_la += dice_la(outputs, labels)
                running_dice_lv += dice_lv(outputs, labels)
                running_dice_myo += dice_myo(outputs, labels)
            writer.add_scalar("Loss/valid", running_loss / len(val_dataloader), epoch)
            val_losses.append(running_loss / len(val_dataloader))
            writer.add_scalar(
                "Loss/dice_la", running_dice_la / len(val_dataloader), epoch
            )
            writer.add_scalar(
                "Loss/dice_lv", running_dice_lv / len(val_dataloader), epoch
            )
            writer.add_scalar(
                "Loss/dice_myo", running_dice_myo / len(val_dataloader), epoch
            )
            writer.add_scalar("Loss/dice", running_dice / len(val_dataloader), epoch)
            if np.argmin(val_losses) == len(val_losses) - 1 and loss < 0.4:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": str(running_loss / len(val_dataloader)),
                    },
                    filename,
                )


run()
