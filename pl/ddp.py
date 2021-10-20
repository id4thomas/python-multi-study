from pytorch_lightning import Trainer, LightningDataModule, LightningModule
import argparse


def main(args):
    k=args.num_gpus
    
    # Single Node & k gpus all possible ways
    # First k gpus
    trainer = Trainer(gpus=k, accelerator="ddp")

    # Specify as list [0,1]
    trainer = Trainer(gpus=range(k), accelerator="ddp")

    # Use String "0,1"
    trainer = Trainer(gpus=",".join(range(k)), accelerator="ddp")

    # Use All Available
    trainer = Trainer(gpus=-1, accelerator="ddp")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Distributed Data Parallel (script-based)')
    parser.add_argument('--num_gpus', type=int, default=0)
    args = parser.parse_args()

    main(args)
