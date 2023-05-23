from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging, TQDMProgressBar
from pytorch_lightning.utilities.model_summary import ModelSummary

from torchmetrics import Accuracy, AUROC
import hydra
from omegaconf import OmegaConf
from functools import lru_cache

#hide warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import warnings
warnings.filterwarnings("ignore")
from pkg_resources import PkgResourcesDeprecationWarning
warnings.simplefilter("ignore", category=PkgResourcesDeprecationWarning)

from models.chordmixer import ChordMixer, ChordMixerNoDec

from optim.register import map_scheduler
from optim.custom_metrics import CustomAccuracy
from dataloaders.register import map_dataset

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split


class LitMixer(pl.LightningModule):
    def __init__(self, backbone, cfg, args):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model_config = self.hparams.backbone
        self.training_config = self.hparams.training
        self.args = args
        self.backbone = backbone(**self.model_config)
        
        # Metrics
        if self.args.metric == 'acc':
            self.train_acc = Accuracy(task='multiclass', num_classes=self.model_config.output_size, top_k=1)
            self.val_acc = Accuracy(task='multiclass', num_classes=self.model_config.output_size, top_k=1)
            self.test_acc = Accuracy(task='multiclass', num_classes=self.model_config.output_size, top_k=1)
        elif self.args.metric == 'rocauc':
            self.train_acc = AUROC(task='binary')
            self.val_acc = AUROC(task='binary')
            self.test_acc = AUROC(task='binary')
        elif self.args.metric == 'reg_acc':
            self.train_acc = CustomAccuracy()
            self.val_acc = CustomAccuracy()
            self.test_acc = CustomAccuracy()
            
    def forward(self, x, lengths):
        return self.backbone(x, lengths)

    def _calculate_loss(self, batch, mode):
        sequences, target, lengths = batch
        preds = self(sequences, lengths)
        if self.args.loss == 'crossentropy':
            loss = F.cross_entropy(preds, target)
            preds = preds.argmax(dim=-1)
        elif self.args.loss == 'mse':
            preds = preds.squeeze()
            loss = F.mse_loss(preds, target)
        self.log("%s_loss" % mode, loss)
        return {"loss": loss, 'preds': preds, 'target': target}

    def training_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch, mode='train')
        return outputs
    
    def training_step_end(self, outputs):
        self.train_acc(outputs['preds'], outputs['target'])
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return {'loss': outputs['loss'].sum()}
        
    def validation_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch, mode='val')
        return outputs
    
    def validation_step_end(self, outputs):
        self.val_acc(outputs['preds'], outputs['target'])
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        return {'loss': outputs['loss'].sum()}
        
    def test_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch, mode='test')
        return outputs

    def test_step_end(self, outputs):
        self.test_acc(outputs['preds'], outputs['target'])
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return {'loss': outputs['loss'].sum()}
    @lru_cache
    def total_steps(self):
        l = len(self.trainer.datamodule.train_dataloader())
        # l = 1000
        print('Num devices', self.trainer.num_devices)
        max_epochs = self.trainer.max_epochs
        accum_batches = self.trainer.accumulate_grad_batches
        manual_total_steps = l // accum_batches * max_epochs
        print('MANUAL Total steps', manual_total_steps)
        return manual_total_steps
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay
        )
        lr_scheduler = map_scheduler(
            scheduler_name=self.training_config.scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.training_config.n_warmup_steps,
            num_training_steps=self.total_steps()
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


class LitMixerImage(pl.LightningModule):
    def __init__(self, backbone, cfg, args):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model_config = self.hparams.backbone
        self.training_config = self.hparams.training
        self.args = args
        self.backbone = backbone(**self.model_config)
        
        # Metrics
        if self.args.metric == 'acc':
            self.train_acc = Accuracy(task='multiclass', num_classes=self.model_config.output_size, top_k=1)
            self.val_acc = Accuracy(task='multiclass', num_classes=self.model_config.output_size, top_k=1)
            self.test_acc = Accuracy(task='multiclass', num_classes=self.model_config.output_size, top_k=1)
        elif self.args.metric == 'rocauc':
            self.train_acc = AUROC(task='binary')
            self.val_acc = AUROC(task='binary')
            self.test_acc = AUROC(task='binary')
        elif self.args.metric == 'reg_acc':
            self.train_acc = CustomAccuracy()
            self.val_acc = CustomAccuracy()
            self.test_acc = CustomAccuracy()
            
    def forward(self, x):
        return self.backbone(x)

    def _calculate_loss(self, batch, mode):
        sequences, target = batch
        preds = self(sequences)
        loss = F.cross_entropy(preds, target)
        preds = preds.argmax(dim=-1)
        self.log("%s_loss" % mode, loss)
        return {"loss": loss, 'preds': preds, 'target': target}

    def training_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch, mode='train')
        return outputs
    
    def training_step_end(self, outputs):
        self.train_acc(outputs['preds'], outputs['target'])
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return {'loss': outputs['loss'].sum()}
        
    def validation_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch, mode='val')
        return outputs
    
    def validation_step_end(self, outputs):
        self.val_acc(outputs['preds'], outputs['target'])
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        return {'loss': outputs['loss'].sum()}
        
    def test_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch, mode='test')
        return outputs

    def test_step_end(self, outputs):
        self.test_acc(outputs['preds'], outputs['target'])
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return {'loss': outputs['loss'].sum()}
    
    @lru_cache
    def total_steps(self):
        l = len(self.trainer.datamodule.train_dataloader())
        print('Num devices', self.trainer.num_devices)
        max_epochs = self.trainer.max_epochs
        accum_batches = self.trainer.accumulate_grad_batches
        manual_total_steps = l // accum_batches * max_epochs
        print('MANUAL Total steps', manual_total_steps)
        return manual_total_steps
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay
        )
        lr_scheduler = map_scheduler(
            scheduler_name=self.training_config.scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.training_config.n_warmup_steps,
            num_training_steps=self.total_steps()
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

class NetDual(nn.Module):
    def __init__(self, backbone, dim, linear_decoder=False, n_class=2):
        super(NetDual, self).__init__()
        self.model = backbone
        if linear_decoder:
            self.decoder = nn.Linear(dim*4, n_class)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(dim*4, dim),
                nn.GELU(),
                nn.Linear(dim, n_class)
            )

    def forward(self, x1, x2, lengths1, lengths2):
        y_dim1 = self.model(x1, lengths1)
        y_dim2 = self.model(x2, lengths2)
        y_class = torch.cat([y_dim1, y_dim2, y_dim1 * y_dim2, y_dim1 - y_dim2], dim=1)
        y = self.decoder(y_class)
        return y
    
class LitMixerRetrieval(pl.LightningModule):
    def __init__(self, backbone, cfg, args):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model_config = self.hparams.backbone
        self.training_config = self.hparams.training
        self.args = args
        self.backbone = backbone(**self.model_config)
        dim = int(13 * self.model_config.track_size)
        self.backbone = NetDual(
            backbone=self.backbone,
            dim=dim,
            linear_decoder=False
        )
        # Metrics
        if self.args.metric == 'acc':
            self.train_acc = Accuracy(task='multiclass', num_classes=self.model_config.output_size, top_k=1)
            self.val_acc = Accuracy(task='multiclass', num_classes=self.model_config.output_size, top_k=1)
            self.test_acc = Accuracy(task='multiclass', num_classes=self.model_config.output_size, top_k=1)
        elif self.args.metric == 'rocauc':
            self.train_acc = AUROC(task='binary')
            self.val_acc = AUROC(task='binary')
            self.test_acc = AUROC(task='binary')
        elif self.args.metric == 'reg_acc':
            self.train_acc = CustomAccuracy()
            self.val_acc = CustomAccuracy()
            self.test_acc = CustomAccuracy()
            
    def forward(self, x1, x2, lengths1, lengths2):
        return self.backbone(x1, x2, lengths1, lengths2)

    def _calculate_loss(self, batch, mode):
        sequences1, sequences2, target, lengths1, lengths2 = batch
        preds = self(sequences1, sequences2, lengths1, lengths2)
        loss = F.cross_entropy(preds, target)
        preds = preds.argmax(dim=-1)
        self.log("%s_loss" % mode, loss)
        return {"loss": loss, 'preds': preds, 'target': target}

    def training_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch, mode='train')
        return outputs
    
    def training_step_end(self, outputs):
        self.train_acc(outputs['preds'], outputs['target'])
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return {'loss': outputs['loss'].sum()}
        
    def validation_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch, mode='val')
        return outputs
    
    def validation_step_end(self, outputs):
        self.val_acc(outputs['preds'], outputs['target'])
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        return {'loss': outputs['loss'].sum()}
        
    def test_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch, mode='test')
        return outputs

    def test_step_end(self, outputs):
        self.test_acc(outputs['preds'], outputs['target'])
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return {'loss': outputs['loss'].sum()}

    @lru_cache
    def total_steps(self):
        l = len(self.trainer.datamodule.train_dataloader())
        print('Num devices', self.trainer.num_devices)
        max_epochs = self.trainer.max_epochs
        accum_batches = self.trainer.accumulate_grad_batches
        manual_total_steps = l // accum_batches * max_epochs
        print('MANUAL Total steps', manual_total_steps)
        return manual_total_steps
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay
        )
        lr_scheduler = map_scheduler(
            scheduler_name=self.training_config.scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.training_config.n_warmup_steps,
            num_training_steps=self.total_steps()
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


@hydra.main(config_path="./conf", config_name="config", version_base="1.1")
def cli_main(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    pl.seed_everything(42)

    # Get the original working directory
    original_cwd = hydra.utils.get_original_cwd()

    # Load the cli_args.yaml configuration
    default_cli_args = OmegaConf.load(os.path.join(original_cwd, "conf/cli_args.yaml"))

    # Merge the main configuration with the default CLI arguments
    cfg = OmegaConf.merge(default_cli_args, cfg)

    # Access command line arguments using cfg.<argument_name>
    task_config = cfg[cfg.problem][cfg.model]
    model_config = task_config['backbone']
    training_config = task_config['training']

    print(model_config)
    print(training_config)
    
    # ------------
    # data
    # ------------
    dm = map_dataset(
        data_dir=cfg.data_dir,
        taskname=cfg.problem,
        num_workers=cfg.num_workers,
        batch_size=training_config['batch_size'],
        diff_lengths=cfg.diff_len
    )

    # ------------
    # init model
    # ------------

    if cfg.model == 'chordmixer':
        backbone = ChordMixer
    
    if cfg.problem in ['image', 'pathfinder', 'pathfinderx']:
        model = LitMixerImage(backbone, task_config, cfg)
    elif cfg.problem == 'retrieval':
        backbone = ChordMixerNoDec
        model = LitMixerRetrieval(backbone, task_config, cfg)
    else:
        model = LitMixer(backbone, task_config, cfg)
        
    summary = ModelSummary(model, max_depth=-1)
    

    # ------------
    # init trainer
    # ------------
    
    wandb_logger = WandbLogger(project=f"{cfg.problem}_{cfg.model}", log_model="all", offline=False)
    callbacks_for_trainer = [
        TQDMProgressBar(refresh_rate=10), 
        LearningRateMonitor(logging_interval="step")
    ]
    
    trainer = Trainer(
        fast_dev_run=False,
        gradient_clip_val=1.,
        accelerator='gpu',
        devices=cfg.n_devices,
        strategy='dp',
        enable_progress_bar=True,
        max_epochs=training_config.n_epochs,
        logger=wandb_logger,
        callbacks=callbacks_for_trainer
    )
    
    # ------------
    # training
    # ------------
    print(summary)
    trainer.fit(model, datamodule=dm)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{cfg.ckpt_dir}/{cfg.problem}_{cfg.model}_{cfg.diff_len}.pt')
    # ------------
    # testing
    # ------------
    
    result = trainer.test(datamodule=dm, ckpt_path="last")
    print(result)

if __name__ == '__main__':
    cli_main()
    
