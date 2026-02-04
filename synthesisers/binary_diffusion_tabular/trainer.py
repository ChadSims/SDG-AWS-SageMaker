from abc import ABC, abstractmethod
from typing import Dict, Optional, Literal, Any
from pathlib import Path
from collections import defaultdict

import accelerate
from tqdm.auto import tqdm

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from accelerate import Accelerator
from ema_pytorch import EMA
import wandb

from synthesisers.binary_diffusion_tabular.model import SimpleTableGenerator
from synthesisers.binary_diffusion_tabular.diffusion import BaseDiffusion, BinaryDiffusion1D
from synthesisers.binary_diffusion_tabular.dataset import FixedSizeBinaryTableDataset
from synthesisers.binary_diffusion_tabular.utils import (
    PathOrStr,
    exists,
    cycle,
    zero_out_randomly,
    get_base_model,
    save_config,
    get_random_labels,
)


__all__ = ["BaseTrainer", "FixedSizeTableBinaryDiffusionTrainer"]


OPTIMIZERS = Literal["adam", "adamw"]


class BaseTrainer(ABC):
    """Base class for training."""

    def __init__(
        self,
        *,
        diffusion: BaseDiffusion,
        train_num_steps: int = 200_000,
        log_every: int = 100,
        save_every: int = 10_000,
        save_num_samples: int = 64,
        max_grad_norm: Optional[float] = None,
        gradient_accumulate_every: int = 1,
        ema_decay: float = 0.995,
        ema_update_every: int = 10,
        lr: float = 3e-4,
        opt_type: OPTIMIZERS,
        opt_params: Dict[str, Any] = None,
        batch_size: int = 256,
        dataloader_workers: int = 16,
        logger,
        results_folder: PathOrStr,
    ):
        """
        Args:
            diffusion: diffusion model to train, should be a BaseDiffusion subclass
            train_num_steps: number of training steps. Default is 200000
            log_every: log every n steps. Default is 100
            save_every: saving generated samples frequency. Default is 10000
            save_num_samples: number of samples save. Default is 64
            max_grad_norm: norm to clip gradients. Defaults to None, which means no clipping
            gradient_accumulate_every: gradient accumulation frequency. Defaults to 1
            ema_decay: decay factor for EMA updates. Defaults to 0.995
            ema_update_every: ema update frequency. Defaults to 10
            lr: learning rate. Defaults to 3e-4
            opt_type: optimizer type. Can be "adam" or "adamw"
            opt_params: optimizer parameters. See each optimizer parameters
            batch_size: batch size. Defaults to 256
            dataloader_workers: number of dataloader workers. Defaults to 16
            logger: wandb logger to use
            results_folder: results folder, where to save samples and trained checkpoints
        """

        self.diffusion = diffusion

        self.train_num_steps = train_num_steps
        self.log_every = log_every
        self.save_every = save_every
        self.save_num_samples = save_num_samples
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulate_every = gradient_accumulate_every
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.lr = lr
        self.opt_type = opt_type
        self.opt_params = {} if opt_params is None else opt_params
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulate_every
        )
        self.device = self.accelerator.device
        self.ema = EMA(
            self.diffusion, beta=self.ema_decay, update_every=self.ema_update_every
        ).to(self.device)

        self.opt = self._create_optimizer()

        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True, parents=True)
            self.logger = logger

        self.step = 0

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseTrainer":
        pass

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, checkpoint: PathOrStr) -> "BaseTrainer":
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    def load_checkpoint(self, path_checkpoint: PathOrStr) -> None:
        ckpt = torch.load(path_checkpoint)

        model = self.accelerator.unwrap_model(self.diffusion)
        model.load_state_dict(ckpt["diffusion"])

        self.step = ckpt["step"]
        self.opt.load_state_dict(ckpt["opt"])

        try:
            self.ema.load_state_dict(ckpt["diffusion_ema"])
        except:
            for name, param in ckpt["diffusion_ema"].items():
                if name == "initted" or name == "step":
                    ckpt["diffusion_ema"][name] = param.unsqueeze(
                        0
                    )  # Convert from shape [] to [1]

            # Load the adjusted state dict
            self.ema.load_state_dict(ckpt["diffusion_ema"])

        if exists(self.accelerator.scaler) and exists(ckpt["scaler"]):
            self.accelerator.scaler.load_state_dict(ckpt["scaler"])

        self.diffusion, self.opt = self.accelerator.prepare(model, self.opt)
        print(f"Loaded model from {path_checkpoint}")

    def save_checkpoint(self, milestone) -> None:
        if not self.accelerator.is_local_main_process:
            return

        config = {
            "train_num_steps": self.train_num_steps,
            "log_every": self.log_every,
            "save_every": self.save_every,
            "save_num_samples": self.save_num_samples,
            "max_grad_norm": self.max_grad_norm,
            "gradient_accumulate_every": self.gradient_accumulate_every,
            "ema_decay": self.ema_decay,
            "ema_update_every": self.ema_update_every,
            "lr": self.lr,
            "opt_type": self.opt_type,
            "opt_params": self.opt_params,
        }

        data = {
            "step": self.step,
            "diffusion": self.accelerator.get_state_dict(self.diffusion),
            # save diffusion and model configs for easy loading without dataset preprocessing
            "config_diffusion": self.diffusion.config,
            "config_model": self.diffusion.model.config,
            "opt": self.opt.state_dict(),
            "diffusion_ema": self.ema.state_dict(),
            "scaler": (
                self.accelerator.scaler.state_dict()
                if exists(self.accelerator.scaler)
                else None
            ),
            # save train config as well
            "config_train": config,
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    @abstractmethod
    def sample_save_samples(self, milestone, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def _create_dataloader(self):
        pass

    def _create_optimizer(self):
        if hasattr(self.diffusion, "model"):
            params = self.diffusion.model.parameters()
        else:
            params = self.diffusion.parameters()

        if self.opt_type == "adam":
            opt = optim.Adam(params, lr=self.lr, **self.opt_params)
        elif self.opt_type == "adamw":
            opt = optim.AdamW(params, lr=self.lr, **self.opt_params)
        else:
            raise ValueError(f"Unknown optimizer type: {self.opt_type}")
        return opt


class FixedSizeTableBinaryDiffusionTrainer(BaseTrainer):
    """Trainer for binary diffusion"""

    def __init__(
        self,
        *,
        diffusion: BinaryDiffusion1D,
        dataset: FixedSizeBinaryTableDataset,
        train_num_steps: int = 200_000,
        log_every: int = 100,
        save_every: int = 10_000,
        save_num_samples: int = 64,
        max_grad_norm: Optional[float] = None,
        gradient_accumulate_every: int = 1,
        ema_decay: float = 0.995,
        ema_update_every: int = 10,
        lr: float = 3e-4,
        opt_type: OPTIMIZERS,
        opt_params: Dict[str, Any] = None,
        batch_size: int = 256,
        dataloader_workers: int = 16,
        classifier_free_guidance: bool,
        zero_token_probability: float = 0.0,
        logger,
        results_folder: PathOrStr,
        worker_id: Optional[int] = 0,
    ):
        """
        Args:
            diffusion: diffusion model to train, should be a BinaryDiffusion1D
            dataset: dataset to train on, should be a FixedSizeBinaryTableDataset
            train_num_steps: number of training steps. Default is 200000
            log_every: log every n steps. Default is 100
            save_every: saving generated samples frequency. Default is 10000
            save_num_samples: number of samples save. Default is 64
            max_grad_norm: norm to clip gradients. Defaults to None, which means no clipping
            gradient_accumulate_every: gradient accumulation frequency. Defaults to 1
            ema_decay: decay factor for EMA updates. Defaults to 0.995
            ema_update_every: ema update frequency. Defaults to 10
            lr: learning rate. Defaults to 3e-4
            opt_type: optimizer type. Can be "adam" or "adamw"
            opt_params: optimizer parameters. See each optimizer parameters
            batch_size: batch size. Defaults to 256
            dataloader_workers: number of dataloader workers. Defaults to 16
            classifier_free_guidance: if True classifier free guidance is applied, when training
            zero_token_probability: zero token probability for classifier free guidance training. Defaults to 0.0
            logger: wandb logger to use
            results_folder: results folder, where to save samples and trained checkpoints
            worker_id: process id for distributed training. Defaults to 0
        """

        if not (dataset.split_feature_target == diffusion.conditional):
            raise ValueError(
                "split_feature_target must be same as diffusion.conditional"
            )

        if classifier_free_guidance and zero_token_probability == 0:
            raise ValueError(
                "zero_token_probability must be non-zero when classifier_free_guidance is True"
            )

        if not (diffusion.classifier_free_guidance == classifier_free_guidance):
            raise ValueError(
                "classifier_free_guidance must be same as diffusion.classifier_free_guidance"
            )
        self.conditional = diffusion.conditional
        self.classifier_free_guidance = classifier_free_guidance
        self.n_classes = diffusion.n_classes
        self.task = dataset.task
        self.zero_token_probability = zero_token_probability
        self.worker_id = worker_id

        super().__init__(
            diffusion=diffusion,
            train_num_steps=train_num_steps,
            log_every=log_every,
            save_every=save_every,
            save_num_samples=save_num_samples,
            max_grad_norm=max_grad_norm,
            gradient_accumulate_every=gradient_accumulate_every,
            ema_decay=ema_decay,
            ema_update_every=ema_update_every,
            lr=lr,
            opt_type=opt_type,
            opt_params=opt_params,
            batch_size=batch_size,
            dataloader_workers=dataloader_workers,
            logger=logger,
            results_folder=results_folder,
        )

        self.dataset = dataset
        self.transformation = dataset.transformation
        self.dataloader = self.accelerator.prepare(self._create_dataloader())
        self.dataloader = cycle(self.dataloader)

        if self.task == "classification" and not (
            self.dataset.n_classes == self.diffusion.n_classes
        ):
            raise RuntimeError("dataset.n_classes must equal diffusion.n_classes")

        # save transformation in joblib format
        self.dataset.transformation.save_checkpoint(self.results_folder / "transformation.joblib")

    @classmethod
    def from_checkpoint(
        cls, path_checkpoint: PathOrStr
    ) -> "FixedSizeTableBinaryDiffusionTrainer":
        """Loads trainer from checkpoint.

        Args:
            path_checkpoint: path to the checkpoint

        Returns:
            FixedSizeTableBinaryDiffusionTrainer: trainer
        """

        ckpt = torch.load(path_checkpoint)
        config = ckpt["config_train"]
        logger = wandb.init(
            project="binary-diffusion-tabular", config=config, name=config["comment"]
        )
        trainer = FixedSizeTableBinaryDiffusionTrainer.from_config(config, logger)

        trainer.load_checkpoint(path_checkpoint)
        return trainer

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], logger
    ) -> "FixedSizeTableBinaryDiffusionTrainer":
        """Builds trainer, model, diffusion and dataset from config.

        Config should have the following structure:

        data:
           path_table: path to the csv file with table data
           numerical_columns: list of numerical column names
           categorical_columns: list of categorical column names
           columns_to_drop: list of column names to drop
           dropna: if True, will drop columns with NaNs
           fillna: if True, will fill NaNs. Numerical replaced with mean, categorical replaced with mode
           target_column: optional target column name, should be provided for conditional training
           split_feature_target: if True, will split the feature target into training and test sets, should be True for conditional training
           task: task for which dataset is used. Options: classification, regression

        model:
           dim: internal dimension of model
           n_res_blocks: number of residual blocks to use

           other parameters are filled from dataset

        diffusion:
           schedule: noise schedule for diffusion. Options: linear, quad, sigmoid
           n_timesteps: number of diffusion steps
           target: target for diffusion. Options: mask, target, two_way

        trainer:
           train_num_steps: number of training steps
           log_every: log every n steps
           save_every: saving generated samples frequency
           save_num_samples: number of samples save
           max_grad_norm: norm to clip gradients. If None, no clipping
           gradient_accumulate_every: gradient accumulation frequency
           ema_decay: decay factor for EMA updates
           ema_update_every: ema update frequency
           lr: learning rate
           opt_type: optimizer type. Can be "adam" or "adamw"
           opt_params: optimizer parameters. See each optimizer parameters
           batch_size: batch size
           dataloader_workers: number of dataloader workers
           classifier_free_guidance: if True classifier free guidance is applied, when training
           zero_token_probability: zero token probability for classifier free guidance training

        comment: <comment how to name results folder>

        Args:
            config: config with parameters for dataset, denoising model, diffusion and trainer
            logger: wandb logger. Create by `wandb.init(project=<project name>)`

        Returns:
            FixedSizeTableBinaryDiffusionTrainer: trainer
        """

        config_data = config["data"]
        config_model = config["model"]
        config_diffusion = config["diffusion"]
        config_trainer = config["trainer"]

        task = config_data["task"]
        dataset = FixedSizeBinaryTableDataset.from_config(config_data)

        classifier_free_guidance = config_trainer["classifier_free_guidance"]
        diffusion_target = config_diffusion["target"]

        device = accelerate.Accelerator().device

        # row_size is given from FixedSizeBinaryTableDataset
        # later SimpleTableGenerator can be loaded from config
        model = SimpleTableGenerator(
            data_dim=dataset.row_size,
            out_dim=(
                dataset.row_size * 2
                if diffusion_target == "two_way"
                else dataset.row_size
            ),
            task=dataset.task,
            conditional=dataset.conditional,
            n_classes=0 if task == "regression" else dataset.n_classes,
            classifier_free_guidance=classifier_free_guidance,
            **config_model,
        ).to(device)

        diffusion = BinaryDiffusion1D(
            denoise_model=model,
            **config_diffusion,
        ).to(device)

        comment = config["comment"]
        results_folder = config["result_folder"]
        # results_folder = Path(f"results/{comment}")
        # results_folder = Path(f"{results_folder}/{comment}")
        # results_folder.mkdir(exist_ok=True)

        # save config as yaml file in results_folder
        save_config(config, results_folder / "config.yaml")

        if logger is None:
            logger = wandb.init(
                project="binary-diffusion-tabular", config=config, name=comment
            )

        return cls(
            diffusion=diffusion,
            dataset=dataset,
            results_folder=results_folder,
            logger=logger,
            **config_trainer,
        )

    def train(self, tune=False) -> None | pd.DataFrame:
        self.diffusion.to(self.device)
        self.diffusion.train()

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not self.accelerator.is_main_process,
            position=self.worker_id,
        ) as pbar:
            while self.step < self.train_num_steps:
                total_loss = defaultdict(float)
                total_acc = defaultdict(float)

                with self.accelerator.accumulate(self.diffusion):
                    inp = next(self.dataloader)
                    if self.conditional:
                        data, label = inp
                        label = self._preprocess_labels(label)
                    else:
                        data = inp
                        label = None

                    with self.accelerator.autocast():
                        loss, losses, accs = self.diffusion(x=data, y=label)
                        loss = loss / self.gradient_accumulate_every

                        gathered_losses = {}
                        gathered_accs = {}

                        for key in losses:
                            gathered_losses[key] = self.accelerator.gather(
                                losses[key].detach()
                            )

                        for key in accs:
                            gathered_accs[key] = self.accelerator.gather(
                                accs[key].detach()
                            )

                    self.accelerator.wait_for_everyone()

                    if self.max_grad_norm is not None:
                        self.accelerator.clip_grad_norm_(
                            self.diffusion.parameters(), self.max_grad_norm
                        )

                    self.accelerator.backward(loss)
                    self.opt.step()
                    self.opt.zero_grad()

                if self.accelerator.is_main_process:
                    message = f"Process: {self.worker_id} | Loss: {loss.item():.4f}"
                    for key in gathered_accs:
                        acc_val = torch.mean(gathered_accs[key]).item()
                        total_acc[key] += acc_val
                        message += f" | {key}: {acc_val:.4f}"

                    for key in gathered_losses:
                        loss_val = torch.mean(gathered_losses[key]).item()
                        total_loss[key] += loss_val

                    pbar.set_description(message)

                    self.ema.update()
                    if self.step % self.log_every == 0:
                        log_dict = {}

                        for key in total_loss:
                            log_dict[key] = total_loss[key]

                        for key in total_acc:
                            log_dict[key] = total_acc[key]

                        wandb.log(log_dict)

                    if self.step != 0 and self.step % self.save_every == 0:
                        milestone = self.step // self.save_every
                        # self.sample_save_samples(milestone, tune=tune)
                        self.accelerator.wait_for_everyone()
                        self.save_checkpoint(milestone)

                self.step += 1
                pbar.update(1)

        if self.accelerator.is_main_process:
            milestone = "final-tune" if tune else "final-train"
            sample = self.sample_save_samples(milestone, tune=tune)
            # save final model
            self.accelerator.wait_for_everyone()
            self.save_checkpoint(milestone)


        if tune:
            return sample

    @torch.inference_mode()
    def sample_save_samples(self, milestone, tune=False) -> None | pd.DataFrame:
        base_model = get_base_model(self.diffusion)
        base_model.eval()
        base_model_ema = get_base_model(self.ema.ema_model)
        base_model_ema.eval()

        with self.accelerator.autocast():
            labels_val = get_random_labels(
                conditional=self.conditional,
                task=self.task,
                n_classes=self.n_classes,
                classifier_free_guidance=self.classifier_free_guidance,
                n_labels=self.save_num_samples,
                device=self.device,
            )

            # sampling without
            rows = base_model.sample(n=self.save_num_samples, y=labels_val)
            rows_ema = base_model_ema.sample(n=self.save_num_samples, y=labels_val)

        if self.conditional:
            if self.classifier_free_guidance:
                labels_val = torch.argmax(labels_val, dim=1).detach()

            rows_df, labels_df = self.transformation.inverse_transform(rows, labels_val)
            rows_ema_df, labels_ema_df = self.transformation.inverse_transform(
                rows_ema, labels_val
            )
            rows_df[self.dataset.target_column] = labels_df
            rows_ema_df[self.dataset.target_column] = labels_ema_df
        else:
            rows_df = self.transformation.inverse_transform(rows)
            rows_ema_df = self.transformation.inverse_transform(rows_ema)

        # if tune return else save (if train)
        if tune:
            if milestone == "final-tune":
                rows_df.to_csv(self.results_folder / f"samples-{milestone}.csv", index=False)
                rows_ema_df.to_csv(
                    self.results_folder / f"samples-ema-{milestone}.csv", index=False
                )
            return rows_ema_df
        else:
            rows_df.to_csv(self.results_folder / f"samples_{milestone}.csv", index=False)
            if milestone == 'final-train':
                rows_ema_df.to_csv(
                    self.results_folder / f"sample.csv", index=False
                )

    def _preprocess_labels(self, label: torch.Tensor) -> torch.Tensor:
        if self.task == "regression" and len(label.shape) == 1:
            label = label.unsqueeze(1)

        if self.classifier_free_guidance:
            if self.task == "classification":
                label = F.one_hot(label.long(), num_classes=self.n_classes).to(
                    torch.float
                )
                label = zero_out_randomly(label, self.zero_token_probability)
            else:
                # regression
                # -1 is zero-token for regression
                mask = torch.rand_like(label) < self.zero_token_probability
                label[mask] = -1

        return label.to(self.device)

    def _create_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
