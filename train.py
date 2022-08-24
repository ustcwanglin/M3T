from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from configs import dataset_PATH, get_aug, BatchSize, model_config
from torchvision.datasets import ImageFolder
from os.path import join
from model import VisionTransformer
import getresults

# loading data set
train_aug = get_aug(True)
test_aug = get_aug(False)
train_DataSet = ImageFolder(join(dataset_PATH, 'train'), transform=train_aug)
test_DataSet = ImageFolder(join(dataset_PATH, 'val'), transform=test_aug)
train_dataloader = DataLoader(train_DataSet, batch_size=BatchSize, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_DataSet, batch_size=BatchSize, shuffle=False, num_workers=4)
print('finished loading dataset')

# train
model = VisionTransformer(**model_config)
print('打印模型--------------------------------------------')
print(model)
from torchsummary import summary
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = model.to(device)
summary(model1, (3, 320, 320))
trainer = Trainer(gpus=1, precision=16)

print('开始训练：')
trainer = trainer.fit(model, train_dataloader, test_dataloader)
print(trainer)
print('训练完成')
acc, f1 = getresults.results()
print('accuracy:', acc)
print('f1 score:', f1)