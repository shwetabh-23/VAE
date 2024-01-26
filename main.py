import pytorch_lightning as pl
from vae import VAE
import yaml
import torch
import os

with open('config.yaml', 'r') as f:
        file = yaml.safe_load(f)

os.makedirs('data')
model = VAE(file= file)

if not os.path.exists('model.pth'):

    trainer = pl.Trainer(max_epochs= file['hyperparameters']['epochs'],  enable_progress_bar= True)
    trainer.fit(model=model)
    torch.save(model.state_dict(), 'model.pth')
    trainer.save_checkpoint('model.ckpt')
else:
    state_dict = torch.load('model.pth')
    model.load_state_dict(state_dict=state_dict)
    trainer = pl.Trainer()
    result = trainer.test(model)

model.sample(12)

model.reconstruct(12)

