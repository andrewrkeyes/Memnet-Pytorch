# Memnet-Pytorch
Pytorch implementation of MemNet (http://memorability.csail.mit.edu/index.html)

Setup:
1) Install pip dependencies
```
python3 -m pip install -r requirements.txt
```
2) Download LaMem dataset from http://memorability.csail.mit.edu/download.html

3) Modify code to point to a csv file and root image directory
```python

import torch
from models import MemNet
from LaMemDataset import LaMemEvalDataset
from torchvision import transforms
import numpy as np
import PIL.Image

#Update to point to intended dataset
root_dir = "lamem/images"
csv_file = "lamem/splits/val_1.txt"

model = MemNet()
checkpoint = torch.load("model.ckpt")
model.load_state_dict(checkpoint["state_dict"])

```
