import os

import torch
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from smalllm.config import Config
from smalllm.model.lm import LanguageModel


class Trainer:
    def __init__(
        self,
        model: LanguageModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-9,
        )

        warmup_steps = config.warmup_steps
        scheduler_warmup = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_steps
        )
        scheduler_decay = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_iters - warmup_steps,
            eta_min=config.min_lr,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[scheduler_warmup, scheduler_decay],
            milestones=[warmup_steps],
        )

        self.scaler = GradScaler(enabled=(config.device == "cuda"))

        self.train_loss_list = []
        self.val_loss_list = []
        self.best_val_loss = float("inf")
        os.makedirs("checkpoints", exist_ok=True)

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        total_loss, count = 0.0, 0
        for xb, yb in self.val_loader:
            xb, yb = xb.to(self.config.device), yb.to(self.config.device)
            _, loss = self.model(xb, yb)
            total_loss += loss.item()
            count += 1
        return total_loss / count

    def train(self):
        self.model.train()
        step = 0
        while step < self.config.max_iters:
            for batch_idx, (xb, yb) in enumerate(self.train_loader):
                if step >= self.config.max_iters:
                    break

                xb, yb = xb.to(self.config.device), yb.to(self.config.device)

                with torch.autocast(
                    device_type=self.config.device,
                    dtype=torch.float16
                    if self.config.device == "cuda"
                    else torch.bfloat16,
                ):
                    logits, loss = self.model(xb, yb)

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

                # Periodic evaluation
                if step % self.config.eval_interval == 0:
                    val_loss = self._evaluate()
                    self._log(step, loss.item(), val_loss)

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        torch.save(self.model.state_dict(), self.config.best_model_path)
                        print(
                            f"🚀 Best model saved at step {step} - val loss: {val_loss:.4f}"
                        )

                step += 1

        return self.train_loss_list, self.val_loss_list

    def _log(self, step, train_loss, val_loss):
        self.train_loss_list.append(train_loss)
        self.val_loss_list.append(val_loss)
        print(f"Step: {step} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
