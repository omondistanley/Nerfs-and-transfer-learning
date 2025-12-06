import json
import copy
import torch
import torch.nn as nn
from models.neural_renderer import get_swapped_indices
from pytorch_msssim import SSIM
from torchvision.utils import save_image
from torchvision import models as tv_models
import torch.nn.functional as F


class Trainer():
    """Class used to train neural renderers.

    Args:
        device (torch.device): Device to train model on.
        model (models.neural_renderer.NeuralRenderer): Model to train.
        lr (float): Learning rate.
        rendering_loss_type (string): One of 'l1', 'l2'.
        ssim_loss_weight (float): Weight assigned to SSIM loss.
    """
    def __init__(self, device, model, lr=2e-4, rendering_loss_type='l1',
                 ssim_loss_weight=0.05,
                 perceptual_loss_weight=0.0,
                 perceptual_layers=None,
                 use_ema=False,
                 progressive_freezing=None):
        self.device = device
        self.model = model
        self.lr = lr
        self.rendering_loss_type = rendering_loss_type
        self.ssim_loss_weight = ssim_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        # layer names or indices to extract from VGG; default sensible set
        self.perceptual_layers = perceptual_layers or [2, 7, 12]
        self.use_ema = use_ema
        self.progressive_freezing = progressive_freezing
        self.use_ssim = self.ssim_loss_weight != 0
        # If False doesn't save losses in loss history
        self.register_losses = True
        # Check if model is multi-gpu
        self.multi_gpu = isinstance(self.model, nn.DataParallel)

        # Initialize optimizer (only params that require grad)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        # Setup EMA shadow weights if requested
        if self.use_ema:
            # store a copy of model.state_dict()
            self.ema_state = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}

        # Setup VGG perceptual model if needed
        if self.perceptual_loss_weight and self.perceptual_loss_weight > 0:
            vgg = tv_models.vgg19(pretrained=True).features.eval().to(self.device)
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg
        else:
            self.vgg = None
        
        
        # Initialize loss functions
        # For rendered images
        if self.rendering_loss_type == 'l1':
            self.loss_func = nn.L1Loss()
        elif self.rendering_loss_type == 'l2':
            self.loss_func = nn.MSELoss()

        # For SSIM
        if self.use_ssim:
            self.ssim_loss_func = SSIM(data_range=1.0, size_average=True,
                                       channel=3, nonnegative_ssim=False)

        # Loss histories
        self.recorded_losses = ["total", "regression", "ssim"]
        self.loss_history = {loss_type: [] for loss_type in self.recorded_losses}
        self.epoch_loss_history = {loss_type: [] for loss_type in self.recorded_losses}
        self.val_loss_history = {loss_type: [] for loss_type in self.recorded_losses}

    def train(self, dataloader, epochs, save_dir=None, save_freq=1,
              test_dataloader=None):
        """Trains a neural renderer model on the given dataloader.

        Args:
            dataloader (torch.utils.DataLoader): Dataloader for a
                misc.dataloaders.SceneRenderDataset instance.
            epochs (int): Number of epochs to train for.
            save_dir (string or None): If not None, saves model and generated
                images to directory described by save_dir. Note that this
                directory should already exist.
            save_freq (int): Frequency with which to save model.
            test_dataloader (torch.utils.DataLoader or None): If not None, will
                test model on this dataset after every epoch.
        """
        if save_dir is not None:
            # Try to extract one batch of data for logging; dataloader may be empty
            try:
                batch = next(iter(dataloader))
            except StopIteration:
                print("Warning: dataloader is empty. Skipping saving sample images and fixed batch setup.")
                self.fixed_batch = None
            else:
                # Save original images
                save_image(batch["img"], save_dir + "/imgs_ground_truth.png", nrow=4)
                # Store batch to check how rendered images improve during training
                self.fixed_batch = batch
                # Render images before any training (guard against render errors)
                try:
                    rendered = self._render_fixed_img()
                    save_image(rendered.cpu(),
                               save_dir + "/imgs_gen_{}.png".format(str(0).zfill(3)), nrow=4)
                except Exception as e:
                    print("Warning: could not render fixed batch before training:", e)
                    self.fixed_batch = None

        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            # If progressive freezing schedule provided, apply changes for this epoch
            if self.progressive_freezing:
                self._apply_progressive_freeze(epoch + 1)
            self._train_epoch(dataloader)
            # Update epoch loss history with mean loss over epoch
            for loss_type in self.recorded_losses:
                self.epoch_loss_history[loss_type].append(
                    sum(self.loss_history[loss_type][-len(dataloader):]) / len(dataloader)
                )
            # Print epoch losses
            print("Mean epoch loss:")
            self._print_losses(epoch_loss=True)

            # Optionally save generated images, losses and model
            if save_dir is not None:
                # Save generated images if we have a fixed batch
                if getattr(self, "fixed_batch", None) is not None:
                    try:
                        rendered = self._render_fixed_img()
                        save_image(rendered.cpu(),
                                   save_dir + "/imgs_gen_{}.png".format(str(epoch + 1).zfill(3)), nrow=4)
                    except Exception as e:
                        print("Warning: could not render fixed batch during training:", e)
                else:
                    print("Warning: no fixed batch available; skipping saving generated images for epoch {}".format(epoch + 1))
                # Save losses
                with open(save_dir + '/loss_history.json', 'w') as loss_file:
                    json.dump(self.loss_history, loss_file)
                # Save epoch losses
                with open(save_dir + '/epoch_loss_history.json', 'w') as loss_file:
                    json.dump(self.epoch_loss_history, loss_file)
                # Save model
                if (epoch + 1) % save_freq == 0:
                    if self.multi_gpu:
                        self.model.module.save(save_dir + "/model.pt")
                    else:
                        self.model.save(save_dir + "/model.pt")

            if test_dataloader is not None:
                regression_loss, ssim_loss, total_loss = mean_dataset_loss(self, test_dataloader)
                print("Validation:\nRegression: {:.4f}, SSIM: {:.4f}, Total: {:.4f}".format(regression_loss, ssim_loss, total_loss))
                self.val_loss_history["regression"].append(regression_loss)
                self.val_loss_history["ssim"].append(ssim_loss)
                self.val_loss_history["total"].append(total_loss)
                if save_dir is not None:
                    # Save validation losses
                    with open(save_dir + '/val_loss_history.json', 'w') as loss_file:
                        json.dump(self.val_loss_history, loss_file)
                    # If current validation loss is the lowest, save model as best
                    # model
                    if min(self.val_loss_history["total"]) == total_loss:
                        print("New best model!")
                        if self.multi_gpu:
                            self.model.module.save(save_dir + "/best_model.pt")
                        else:
                            self.model.save(save_dir + "/best_model.pt")

        # Save model after training
        if save_dir is not None:
            if self.multi_gpu:
                self.model.module.save(save_dir + "/model.pt")
            else:
                self.model.save(save_dir + "/model.pt")

    def _train_epoch(self, dataloader):
        """Trains model for a single epoch.

        Args:
            dataloader (torch.utils.DataLoader): Dataloader for a
                misc.dataloaders.SceneRenderDataset instance.
        """
        num_iterations = len(dataloader)
        for i, batch in enumerate(dataloader):
            # Train inverse and forward renderer on batch
            self._train_iteration(batch)

            # Print iteration losses
            print("{}/{}".format(i + 1, num_iterations))
            self._print_losses()

    def _train_iteration(self, batch):
        """Trains model for a single iteration.

        Args:
            batch (dict): Batch of data as returned by a Dataloader for a
                misc.dataloaders.SceneRenderDataset instance.
        """
        imgs, rendered, scenes, scenes_rotated = self.model(batch)
        self._optimizer_step(imgs, rendered)

    def _optimizer_step(self, imgs, rendered):
        """Updates weights of neural renderer.

        Args:
            imgs (torch.Tensor): Ground truth images. Shape
                (batch_size, channels, height, width).
            rendered (torch.Tensor): Rendered images. Shape
                (batch_size, channels, height, width).
        """
        self.optimizer.zero_grad()

        loss_regression = self.loss_func(rendered, imgs)
        loss_total = loss_regression
        if self.use_ssim:
            # We want to maximize SSIM, i.e. minimize -SSIM
            loss_ssim = 1. - self.ssim_loss_func(rendered, imgs)
            loss_total = loss_total + self.ssim_loss_weight * loss_ssim

        # Perceptual (VGG) loss
        if self.vgg is not None and self.perceptual_loss_weight > 0:
            # Expect imgs and rendered in [0,1], convert to 3xHxW tensors
            # VGG expects normalization like ImageNet; use simple scaling to [0,1]
            # Compute features at requested layer indices
            with torch.no_grad():
                pass
            # forward and collect features
            def extract_vgg_feats(x):
                feats = []
                out = x
                for idx, layer in enumerate(self.vgg):
                    out = layer(out)
                    if idx in self.perceptual_layers:
                        feats.append(out)
                return feats

            # Ensure inputs are normalized to VGG expected range: 0-1 -> mean/std
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
            imgs_vgg = (imgs - mean) / std
            rendered_vgg = (rendered - mean) / std
            feats_img = extract_vgg_feats(imgs_vgg)
            feats_render = extract_vgg_feats(rendered_vgg)
            # L2 between feature maps
            loss_perc = 0.0
            for f_r, f_i in zip(feats_render, feats_img):
                loss_perc = loss_perc + F.mse_loss(f_r, f_i)
            loss_total = loss_total + self.perceptual_loss_weight * loss_perc

        loss_total.backward()
        self.optimizer.step()

        # Update EMA shadow weights if requested
        if self.use_ema:
            for k, v in self.model.state_dict().items():
                self.ema_state[k] = 0.999 * self.ema_state[k].to(v.device) + 0.001 * v.detach().cpu()

        # Record total loss
        if self.register_losses:
            self.loss_history["total"].append(loss_total.item())
            self.loss_history["regression"].append(loss_regression.item())
            # If SSIM is not used, register 0 in logs
            if not self.use_ssim:
                self.loss_history["ssim"].append(0.)
            else:
                self.loss_history["ssim"].append(loss_ssim.item())

    def _render_fixed_img(self):
        """Reconstructs fixed batch through neural renderer (by inferring
        scenes, rotating them and rerendering).
        """
        _, rendered, _, _ = self.model(self.fixed_batch)
        return rendered

    def _print_losses(self, epoch_loss=False):
        """Prints most recent losses."""
        loss_info = []
        for loss_type in self.recorded_losses:
            if epoch_loss:
                loss = self.epoch_loss_history[loss_type][-1]
            else:
                loss = self.loss_history[loss_type][-1]
            loss_info += [loss_type, loss]
        print("{}: {:.3f}, {}: {:.3f}, {}: {:.3f}".format(*loss_info))

    def _apply_progressive_freeze(self, epoch):
        """Applies progressive freezing schedule if provided. Expects
        self.progressive_freezing to be a dict of phases with keys 'phase_*'
        each containing: 'epochs': [start, end], 'freeze_parts': [list], 'lr': value
        """
        for phase_name, phase in self.progressive_freezing.items():
            start, end = phase.get("epochs", [0, -1])
            if start <= epoch <= end:
                freeze_parts = phase.get("freeze_parts", [])
                # set requires_grad
                for name, param in self.model.named_parameters():
                    if any(fp in name for fp in freeze_parts):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                # Recreate optimizer to respect new requires_grad settings
                new_lr = phase.get("lr", self.lr)
                self.lr = new_lr
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=new_lr)
                print(f"Applied progressive freezing phase {phase_name}: freeze_parts={freeze_parts}, lr={new_lr}")
                break

    def _save_model_with_ema(self, save_path):
        """If EMA is enabled, temporarily swap in EMA weights and save model.
        Otherwise save current model normally.
        """
        if not self.use_ema:
            if self.multi_gpu:
                self.model.module.save(save_path)
            else:
                self.model.save(save_path)
            return

        # Save current state
        orig_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        # Load EMA (convert to correct device/dtype)
        ema_state_device = {k: v.to(orig_state[k].device) for k, v in self.ema_state.items()}
        self.model.load_state_dict(ema_state_device)
        # Save
        if self.multi_gpu:
            self.model.module.save(save_path)
        else:
            self.model.save(save_path)
        # Restore original
        self.model.load_state_dict(orig_state)


def mean_dataset_loss(trainer, dataloader):
    """Returns the mean loss of a model across a dataloader.

    Args:
        trainer (training.Trainer): Trainer instance containing model to
            evaluate.
        dataloader (torch.utils.DataLoader): Dataloader for a
            misc.dataloaders.SceneRenderDataset instance.
    """
    # No need to calculate gradients during evaluation, so disable gradients to
    # increase performance and reduce memory footprint
    with torch.no_grad():
        # Ensure calculated losses aren't registered as training losses
        trainer.register_losses = False

        regression_loss = 0.
        ssim_loss = 0.
        total_loss = 0.
        for i, batch in enumerate(dataloader):
            imgs, rendered, scenes, scenes_rotated = trainer.model(batch)

            # Update losses
            # Use _loss_func here and not _loss_renderer since we only want regression term
            current_regression_loss = trainer.loss_func(rendered, imgs).item()
            if trainer.use_ssim:
                current_ssim_loss = 1. - trainer.ssim_loss_func(rendered, imgs).item()
            else:
                current_ssim_loss = 0.
            regression_loss += current_regression_loss
            ssim_loss += current_ssim_loss
            total_loss += current_regression_loss + trainer.ssim_loss_weight * current_ssim_loss

        # Average losses over dataset
        regression_loss /= len(dataloader)
        ssim_loss /= len(dataloader)
        total_loss /= len(dataloader)

        # Reset boolean so we register losses if we continue training
        trainer.register_losses = True

    return regression_loss, ssim_loss, total_loss
