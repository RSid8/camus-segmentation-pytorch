import torch
from camus_unet.camus_unet2 import CamusUnet2

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = CamusUnet2()  # initialise the U-NET 1 model
print(model)
# move initialised model to chosen device
model = model.to(device)
