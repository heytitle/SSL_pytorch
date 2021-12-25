from os import path
import time

import torch
from torch.optim import lr_scheduler

from utils import check_existing_model, Linear_Protocoler


class SSL_Trainer(object):
    def __init__(self, model, ssl_data, device="cuda", use_momentum=False):
        # Define device
        self.device = torch.device(device)

        # Define if use momentum
        self.use_momentum = use_momentum

        # Init
        self.loss_hist = []
        self.eval_acc = {"lin": [], "knn": []}
        self._iter_scheduler = False
        self._hist_lr = []

        # Model
        self.model = model.to(self.device)

        # Define data
        self.data = ssl_data

    def train_epoch(self, epoch_id):
        for i, ((x1, x2), _) in enumerate(self.data.train_dl):
            x1, x2 = x1.to(self.device), x2.to(self.device)

            # Forward pass
            loss = self.model(x1, x2)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update momentum encoder
            if self.use_momentum:
                # get τ
                tau = self.model.get_tau(
                    1 + i + self._train_len * epoch_id, self._total_iters
                )
                self.model.update_moving_average(tau)

            # save learning rate
            self._hist_lr.append(self.scheduler.get_last_lr())

            # Save loss
            self._epoch_loss += loss.item()

    def evaluate(self, num_epochs, lr, milestones=None):
        # Linear protocol
        evaluator = Linear_Protocoler(
            self.model.backbone_net, repre_dim=self.model.repre_dim, device=self.device
        )
        # knn accuracy
        self.eval_acc["knn"].append(
            evaluator.knn_accuracy(self.data.train_eval_dl, self.data.test_dl)
        )
        # linear protocol
        evaluator.train(self.data.train_eval_dl, num_epochs, lr, milestones)
        self.eval_acc["lin"].append(evaluator.linear_accuracy(self.data.test_dl))

    def train(
        self,
        save_root,
        num_epochs,
        optimizer,
        scheduler,
        optim_params,
        scheduler_params,
        eval_params,
        warmup_epochs=10,
        iter_scheduler=True,
        evaluate_at=[100, 200, 400],
        verbose=True,
    ):

        # Check and Load existing model
        epoch_start, optim_state, sched_state = self.load_model(
            save_root, return_vals=True
        )

        # Extract training length
        self._train_len = len(self.data.train_dl)
        self._total_iters = num_epochs * self._train_len

        # Define Optimizer
        self.optimizer = optimizer(self.model.parameters(), **optim_params)
        if optim_state:
            self.optimizer.load_state_dict(optim_state)

        # Define Scheduler
        if warmup_epochs and epoch_start < warmup_epochs:
            self.scheduler = lr_scheduler.LambdaLR(
                self.optimizer, lambda it: (it + 1) / (warmup_epochs * self._train_len)
            )
            self._iter_scheduler = True
        else:
            if scheduler:
                self.scheduler = scheduler(self.optimizer, **scheduler_params)
                self._iter_scheduler = iter_scheduler
            else:  # scheduler = None
                self.scheduler = scheduler
        # Load existing scheduler
        if self.scheduler and sched_state:
            self.scheduler.load_state_dict(sched_state)

        # Run Training
        for epoch in range(epoch_start, num_epochs):
            self._epoch_loss = 0
            start_time = time.time()

            self.train_epoch(epoch)

            if self.scheduler and not self._iter_scheduler:
                # Scheduler only every epoch
                self.scheduler.step()

            # Switch to new schedule after warmup period
            if warmup_epochs and epoch + 1 == warmup_epochs:
                if scheduler:
                    self.scheduler = scheduler(self.optimizer, **scheduler_params)
                    self._iter_scheduler = iter_scheduler
                else:  # scheduler = None
                    self.scheduler = scheduler

            # Log
            self.loss_hist.append(self._epoch_loss / self._train_len)
            if verbose:
                print(
                    f"Epoch: {epoch}, Loss: {self.loss_hist[-1]}, Time epoch: {time.time() - start_time}"
                )

            # Run evaluation
            if (epoch + 1) in evaluate_at:
                self.evaluate(**eval_params)
                # print
                print(
                    f'Accuracy after epoch {epoch}: KNN:{self.eval_acc["knn"][-1]}, Linear: {self.eval_acc["lin"][-1]}'
                )

                # Save model
                self.save_model(save_root, epoch)

        # Evaluate after Training
        self.evaluate(**eval_params)
        # print
        print(
            f'Accuracy after full Training: KNN:{self.eval_acc["knn"][-1]}, Linear: {self.eval_acc["lin"][-1]}'
        )

        # Save final model
        self.save_model(save_root, epoch)

    def save_model(self, save_root, epoch):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "sched": self.scheduler.state_dict() if self.scheduler else None,
                "loss_hist": self.loss_hist,
                "eval_acc": self.eval_acc,
                "lr_hist": self._hist_lr,
            },
            path.join(save_root, f"epoch_{epoch+1:03}.tar"),
        )

    def load_model(self, save_root, return_vals=False):
        # Check for trained model
        epoch_start, saved_data = check_existing_model(save_root, self.device)

        if saved_data is None and return_vals:
            return epoch_start, None, None
        else:
            self.model.load_state_dict(saved_data["model"])
            self.loss_hist = saved_data["loss_hist"]
            self.eval_acc = saved_data["eval_acc"]
            self._hist_lr = saved_data["lr_hist"]
            if return_vals:
                return epoch_start, saved_data["optim"], saved_data["sched"]
