# pytorch-lightning

## LightningDataModule
* Data Preparation
    * prepare_data
        * called only once and on 1 GPU
    * setup
        * called on each GPU separately 
        * whether at fit or test step


## LightningModule
* Functions to Implement
    * train
    * validation
    * test

## Using Wandb Logger
```
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(name='RUN NAME',project='PRJ NAME')
trainer = Trainer(...,logger=wandb_logger,...)
```