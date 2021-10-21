import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningDataModule, LightningModule


from module_template import *


# Parameters Order
# Model Checkpoint File Path ("./.../checkpoint.ckpt")
# Map Location: Map Saved weight device allocated to current machine 
# hparams File Path
# LightningModule Init Params

chk_dir="./checkpoint_dir/epoch0.ckpt"

#Map GPU 0 saved model -> CPU of loading machine
map_location={
    'cuda:0': 'cpu',
}

#Map GPU 0,1 saved model -> GPU 0 of loading machine
map_location={
    'cuda:0': 'cuda:0',
    'cuda:1': 'cuda:1'
}

model_params={
    'l1_size':128,
    'l2_size':64
}
model=LightningModuleTemplate.load_from_checkpoint(chk_dir, 
        map_location=map_location,
        **model_params
    )