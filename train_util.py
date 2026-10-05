import copy
import functools
import os
import blobfile as bf
import torch as th
import torch.distributed as dist
import wandb
import socket
import random
import glob
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import dist_util, logger
from fp16_util import MixedPrecisionTrainer
from nn import update_ema
from resample import LossAwareSampler, UniformSampler
from tqdm.auto import tqdm
import sys

INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            val_data,  # validation data iterator
            batch_size,
            microbatch,
            ema_rate,
            log_interval,
            save_interval,
            eval_interval,  # validation interval
            patience,  # early stopping patience
            resume_checkpoint,
            lr=0,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            class_cond=False,
            use_db=False,
            num_classes=0,
            use_tqdm=True,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.val_data = val_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr

        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.patience = patience

        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.class_cond = class_cond
        self.num_classes = num_classes
        self.step = 1
        self.resume_step = 0
        self.global_batch = self.batch_size * dist_util.get_world_size()
        self.use_db = use_db
        self.use_tqdm = use_tqdm

        # Early stopping variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        if self.use_db == True and dist_util.get_rank() == 0:
            wandb.init(
                project=f"{logger.get_dir().split('/')[-2]}",
                notes=socket.gethostname(),
                name=f"{logger.get_dir().split('/')[-1]}",
                job_type="training",
                reinit=True
            )

        self._load_and_sync_parameters()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        self.output_model_stastics()

        if th.cuda.is_available() and dist.is_initialized():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
        else:
            if dist_util.get_world_size() > 1:
                logger.warn("Distributed training requires CUDA.")
            self.use_ddp = False
            self.ddp_model = self.model

    def output_model_stastics(self):
        num_params_total = sum(p.numel() for p in self.model.parameters())
        num_params_train = 0
        for param_group in self.opt.param_groups:
            if param_group['lr'] > 0:
                num_params_train += sum(p.numel() for p in param_group['params'] if p.requires_grad)

        if num_params_total > 1e6:
            label = 'M'
            div = 1e6
        else:
            label = 'k'
            div = 1e3

        logger.log(f"Total Parameters: {num_params_total / div:.2f}{label}")
        logger.log(f"Total Training Parameters: {num_params_train / div:.2f}{label}")

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist_util.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = self.resume_checkpoint

        if not main_checkpoint:
            dist_util.sync_params(ema_params)
            return ema_params

        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)

        if ema_checkpoint:
            if dist_util.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        else:
            if dist_util.get_rank() == 0:
                logger.log(f"main checkpoint provided but EMA checkpoint not found. Using master params.")

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        if not main_checkpoint:
            return

        # [Fix]: directly read fixed opt_latest.pt, no longer append numbers
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), "opt_latest.pt"
        )

        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        pbar = None
        if self.use_tqdm:
            from tqdm.auto import tqdm
            total_steps = self.lr_anneal_steps if self.lr_anneal_steps > 0 else None
            pbar = tqdm(total=total_steps, initial=self.step + self.resume_step, dynamic_ncols=True)
            logger.set_progress_bar(pbar)

        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # 1. Training step
            batch = next(self.data)
            cond = {}

            if self.class_cond and 'label' in batch:
                cond['label'] = batch['label']
            if 'cond_vec' in batch:
                cond['cond_vec'] = batch['cond_vec']
            if 'pert_id' in batch:
                cond['pert_id'] = batch['pert_id']
            if 'x_ctrl' in batch:
                cond['x_ctrl'] = batch['x_ctrl']

            loss = self.run_step(batch, cond)

            if dist_util.get_rank() == 0 and self.use_db:
                wandb_log = {'loss': loss["loss"].mean().item()}

            if self.step % self.log_interval == 0:
                logger.dumpkvs()
                if dist_util.get_rank() == 0 and self.use_db:
                    wandb.log(wandb_log)

                # 2. Validation and early stopping logic
                if self.step % self.eval_interval == 0:
                    val_loss = self.run_validation()

                    # --- [Core fix] Resolve deadlock issue ---

                    # 1. Define flag to decide whether to save
                    should_save_best = False

                    # 2. Only Rank 0 determines if best model
                    if dist_util.get_rank() == 0:
                        logger.logkv("val_loss", val_loss)
                        if self.use_db:
                            wandb.log({'val_loss': val_loss})

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            should_save_best = True  # Marked for saving
                        else:
                            self.patience_counter += 1
                            logger.log(
                                f"Validation loss did not improve. Patience: {self.patience_counter}/{self.patience}")

                    # 3. [Key step] Broadcast Rank 0's decision to all GPUs
                    if self.use_ddp:
                        # Create tensor to store decision (1=save, 0=skip)
                        # Note: must be on current device
                        flag_tensor = th.tensor([1 if should_save_best else 0], dtype=th.int,
                                                device=dist_util.dev())
                        # Broadcast: from Rank 0 to all
                        dist.broadcast(flag_tensor, src=0)
                        # Update current process decision
                        should_save_best = (flag_tensor.item() == 1)

                        # Sync early stopping counter (patience_counter)
                        # Rank 1 needs patience counter broadcast too
                        patience_tensor = th.tensor([self.patience_counter], dtype=th.int, device=dist_util.dev())
                        dist.broadcast(patience_tensor, src=0)
                        self.patience_counter = patience_tensor.item()

                    # 4. All GPUs execute save simultaneously (only Rank 0 writes file, all execute barrier)
                    if should_save_best:
                        self.save(tag="best")

                    # 5. Check if early stopping triggered
                    if self.patience_counter >= self.patience:
                        if dist_util.get_rank() == 0:
                            logger.log(f"Early stopping triggered at step {self.step}! (patience={self.patience})")
                        break

            if self.step % self.save_interval == 0:
                self.save()

            self.step += 1
            if pbar is not None:
                pbar.update(1)

        if (self.step - 1) % self.save_interval != 0:
            self.save()

        if pbar is not None:
            pbar.close()

        if dist_util.get_rank() == 0:
            logger.log_important(f"Training finished. Total steps: {self.step - 1} | Best val loss: {self.best_val_loss:.4f}")

    def run_step(self, batch, cond):
        self.mp_trainer.zero_grad()
        loss = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return loss

    # Validation function
    def run_validation(self):
        self.model.eval()  # switch to eval mode
        val_losses = []
        num_val_batches = 300  # validation batch count, adjust as needed

        # 1. initialize iterator
        # If self.val_data is infinite stream, use range(num_val_batches) to control count
        # If self.val_data is standard DataLoader，use iter_wrapper = self.val_data directly
        iter_wrapper = range(num_val_batches)

        # 2. If tqdm enabled, wrap iterator
        if self.use_tqdm:
            from tqdm.auto import tqdm
            # desc="Validating": show prefix
            # leave=False: Clear progress bar after completion
            iter_wrapper = tqdm(iter_wrapper, desc="Validating", leave=False, dynamic_ncols=True)

        with th.no_grad():
            for _ in iter_wrapper:
                # Note: if infinite stream，still use next(self.val_data)
                # If iter_wrapper is DataLoader，use for batch in iter_wrapper
                batch = next(self.val_data)

                cond = {}
                if self.class_cond and 'label' in batch:
                    cond['label'] = batch['label']
                if 'cond_vec' in batch:
                    cond['cond_vec'] = batch['cond_vec']
                if 'pert_id' in batch:
                    cond['pert_id'] = batch['pert_id']
                if 'x_ctrl' in batch:
                    cond['x_ctrl'] = batch['x_ctrl']

                loss_dict = self.forward_backward(batch, cond, train=False)
                val_losses.append(loss_dict["loss"].mean().item())

        self.model.train()  # switch back to training mode
        return np.mean(val_losses)

    def forward_backward(self, batch, cond, train=True):
        if isinstance(batch, dict):
            # When Dataset returns dict, x is the main data
            batch_data = batch['x'].to(dist_util.dev())
        else:
            batch_data = batch.to(dist_util.dev())

        # Move all conditions to GPU
        cond = {k: v.to(dist_util.dev()) for k, v in cond.items()}

        micro_batch_size = self.microbatch
        batch_size = batch_data.shape[0]

        t, weights = self.schedule_sampler.sample(batch_size, dist_util.dev())

        losses = {}
        for i in range(0, batch_size, micro_batch_size):
            micro = batch_data[i: i + micro_batch_size]
            micro_cond = {k: v[i: i + micro_batch_size] for k, v in cond.items()}
            micro_t = t[i: i + micro_batch_size]
            micro_weights = weights[i: i + micro_batch_size]

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                micro_t,
                model_kwargs=micro_cond,
            )

            if not self.use_ddp:
                micro_losses = compute_losses()
            else:
                # No need to sync grad during validation
                if not train:
                    with self.ddp_model.no_sync():
                        micro_losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        micro_losses = compute_losses()

            loss = (micro_losses["loss"] * micro_weights).mean()

            # Backward pass only during training
            if train:
                log_loss_dict(
                    self.diffusion, micro_t, {k: v * micro_weights for k, v in micro_losses.items()}
                )
                self.mp_trainer.backward(loss)

            losses = micro_losses

        return losses

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self, tag=None):

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist_util.get_rank() == 0:
                # Remove dynamic step count, use fixed suffix
                if tag:
                    suffix = tag  # If tag='best', suffix is 'best'
                else:
                    suffix = "latest"  # Regular epoch save, suffix fixed to 'latest'

                logger.log(f"saving model {rate} ({suffix})...")

                # Generate fixed filename，e.g., model_best.pt or ema_0.9999_latest.pt
                if not rate:
                    filename = f"model_{suffix}.pt"
                else:
                    filename = f"ema_{rate}_{suffix}.pt"

                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # Save optimizer only for regular saves，best checkpoint usually doesn't need optimizer resume
        if dist_util.get_rank() == 0 and tag is None:
            # Optimizer name also fixed to opt_latest.pt，continuously overwriting old ones
            with bf.BlobFile(
                    bf.join(get_blob_logdir(), "opt_latest.pt"),
                    "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist_util.barrier()


# ... (Helper functions remain the same)
def parse_resume_step_from_filename(filename):
    split = filename.split("model")
    if len(split) < 2: return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return logger.get_dir()


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None: return None

    # Auto-match EMA filename based on main model filename
    basename = os.path.basename(main_checkpoint)
    if "best" in basename:
        suffix = "best"
    else:
        suffix = "latest"

    filename = f"ema_{rate}_{suffix}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)

    if bf.exists(path): return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)