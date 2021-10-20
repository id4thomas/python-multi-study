# python-multi-study
Studying python distributed execution

# Concepts
## multiprocessing (./mp/)
* offers both local & remote concurrency
    * sidesteps GIL (Global Intepreter Lock) by using subprocesses instead of threads
    * GIL: used by Cython to assure that only one thread executes Python bytecode at a time

## threading (./th/)
* only one thread can execute Python code at once

## pytorch-lightning (./pl/)
* multi-gpu
    * dp (Data Parallel - multiple gpus, 1 machine)
        + Split batch across k GPUs, replicate same model to all GPUs
        + "works because of linearity of gradient operator. computing gradients individually & averaging is same as calculating at one"
    * ddp (Distributed Data Parallel - multiple gpus, multiple nodes)
        * python script based: calls script multiple times
        ``` 
        MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=0 LOCAL_RANK=0 python my_file.py --gpus 3 --etc
        MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=1 LOCAL_RANK=0 python my_file.py --gpus 3 --etc
        MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=2 LOCAL_RANK=0 python my_file.py --gpus 3 --etc
        ```
        * Each GPU gets own process, each process inits model, gradients are synced & averaged
        * Cases where ddp can't be used (Use ddp_spawn)
            * Jupyter Notebook based
            * have a nested script without a root package
    * ddp_spawn (Distributed Data Parallel Spawn - multiple gpus, multiple nodes)
        * Not Recommended (Use DDP if possible)
            * passed model will not update
        * spawn based: 
        ```
        mp.spawn(self.ddp_train, nprocs=self.num_processes, args=(model,))
        ```
## huggingface-transformers(./ha)
* parallelize

## huggingface-accelerate (./ha/)
* multi-gpu

# Experiments
## multiprocessing (./mp/)
* mp_file_access.ipynb
    * Multiple processes safely writing to file
    * mp.Pool
        * Creates specified number of processes & 

## pytorch-lightning (./pl/)
* ddp.py
    * script based DDP
    * LightningModule, LightningDataModule
        
## References
* https://docs.python.org/3/library/multiprocessing.html
* https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html
* https://devblog.pytorchlightning.ai/distributed-deep-learning-with-pytorch-lightning-part-1-8df1d032e6d3