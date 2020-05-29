import torch
from models import MemNet
from LaMemDataset import LaMemEvalDataset
from torchvision import transforms
import numpy as np
import PIL.Image

root_dir = "lamem/images"
csv_file = "lamem/splits/val_1.txt"

#Load pretrained MemNet
model = MemNet()
checkpoint = torch.utils.model_zoo.load_url("https://github.com/andrewrkeyes/Memnet-Pytorch/raw/master/model.ckpt")
model.load_state_dict(checkpoint["state_dict"])


mean = np.load("image_mean.npy")

transform = transforms.Compose([
    transforms.Resize((256,256), PIL.Image.BILINEAR),
    lambda x: np.array(x),
    lambda x: np.subtract(x[:,:,[2, 1, 0]], mean), #Subtract average mean from image (opposite order channels)
    lambda x: x[15:242, 15:242], #Center crop
    transforms.ToTensor()
])

eval_dataset = LaMemEvalDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=False)
model.eval()

with torch.no_grad():
    for idx, (img, target) in enumerate(eval_loader):
        output = model(img)
