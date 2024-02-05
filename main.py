import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from vae import VAE
import yaml
import torch
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

with open('config.yaml', 'r') as f:
        file = yaml.safe_load(f)


os.makedirs('data', exist_ok= True)
model = VAE(file= file)

bottleneck_layer_size = '256'
model_path = f'/home/harsh/AI-Projects/VAE/models/cifar/size_variation/{bottleneck_layer_size}/model.pth'
logger = TensorBoardLogger(model_path[:-9] + 'logs/', name= 'modelv1')

if not os.path.exists(model_path):

    trainer = pl.Trainer(max_epochs= file['hyperparameters']['epochs'],  enable_progress_bar= True, logger= logger, 
                         callbacks= EarlyStopping(monitor="validation loss", min_delta=0.5, patience=3, verbose=False, mode="min"))
    trainer.fit(model=model)
    torch.save(model.state_dict(), model_path)
else:
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict=state_dict)
    trainer = pl.Trainer(logger= logger)
    result = trainer.test(model)

#model.sample(12)

model.reconstruct(12)

