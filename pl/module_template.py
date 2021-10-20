from pytorch_lightning import Trainer, LightningDataModule, LightningModule

class BaselineDataModule(LightningDataModule):
    def __init__(self, data_dir,tokenizer,max_seq_length, batch_size=256):
        super().__init__()
        self.batch_size = batch_size
        # self.data_dir=data_dir
        self.data_dir="../abductive-commonsense-reasoning/data/anli/"
        self.tokenizer=tokenizer
        self.max_seq_length=max_seq_length

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        pass

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:

            train_data=read_jsonl_inputs(f"{self.data_dir}/train.jsonl")[:200]
            train_labels=read_lines(f"{self.data_dir}/train-labels.lst")[:-1][:200]
            self.train_examples=convert_data_to_features(train_data,train_labels,self.tokenizer,self.max_seq_length)

            dev_data=read_jsonl_inputs(f"{self.data_dir}/dev.jsonl")[:200]
            dev_labels=read_lines(f"{self.data_dir}/dev-labels.lst")[:-1][:200]
            self.dev_examples=convert_data_to_features(dev_data,dev_labels,self.tokenizer,self.max_seq_length)

        if stage == 'test' or stage is None:
            pass

    def train_dataloader(self):
        '''returns training dataloader'''
        return examples_to_dataloader(self.train_examples,self.batch_size,is_train=True,is_predict=False)

    def val_dataloader(self):
        '''returns validation dataloader'''
        return examples_to_dataloader(self.dev_examples,self.batch_size,is_train=False,is_predict=False)

class BaselineBERT(LightningModule):
    def __init__(self,model,t_total):
        super().__init__()
        self.model=model
        self.t_total=t_total

        self.accuracy=pl.metrics.Accuracy()

    def forward(self,batch):
        batch = tuple(t for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        model_output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
        return model_output
    
    def training_step(self,batch,batch_idx):
        input_ids, input_mask, segment_ids, label_ids = batch
        model_output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
        loss=model_output[0]
        logits=model_output[1]

        #Log Train Loss
        self.log('train_loss',loss)

        #Get ACC
        acc=self.accuracy(logits, label_ids)
        self.log('train_acc',acc)

        return {'loss': loss}

    def validation_step(self,batch,batch_idx):
        input_ids, input_mask, segment_ids, label_ids = batch
        model_output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
        loss=model_output[0]
        logits=model_output[1]

        #Log Train Loss
        # self.log('val_loss',loss)

        #Get ACC
        acc=self.accuracy(logits, label_ids)
        # self.log('val_acc',acc)

        return {'val_loss': loss,'val_acc':acc}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        # return avg_loss
        self.log('val_loss',avg_loss)
        self.log('val_acc',avg_acc)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        param_optimizer = list(self.model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)

        warmup_proportion=0.2
        t_total=self.t_total
        scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=math.floor(warmup_proportion * t_total),
                                     t_total=t_total)
        return optimizer, scheduler