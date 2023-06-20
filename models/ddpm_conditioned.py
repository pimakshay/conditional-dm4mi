
from typing import Any

from tqdm.auto import tqdm
from einops import rearrange, repeat
import numpy as np
import torch, torchvision

from models.ddpm_st_diffusion import DDPM
from diffusion_modules.utils import exists, default, noise_like, extract_into_tensor
from diffusion_modules.diffusion_utils import dataloader

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


class ConditionDDPM(DDPM):
    def __init__(self,
                 cond=None,
                 timesteps=1000,
                 beta_schedule="linear",
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 loss_type="l2",
                 epochs=10,
                 max_tsteps=int(5e4),
                 learning_rate=1e-4,
                 warmup_steps=0,
                 lr_scheduler=False,
                 image_size=256,
                 channels=3,
                 num_of_train_samples=10000,
                 num_of_val_samples=2000,
                 train_percent_check=0.5,  # train_percent_check=0.1 -> train only on 50% of data
                 batch_size=128,
                 scheduler_config=None,
                 modified_unet_config=None,  # change to unet_config
                 unet_rosinality_config=None,
                 openai_unet_config=None,
                 unet_config=None,
                 conditioning_key=None,
                 dataset=None,
                 use_ema=False,
                 ema_decay_factor=0.9999,
                 original_elbo_weight=0.,
                 l_simple_weight=1.,
                 learn_logvar=False,
                 logvar_init=0.,
                 use_positional_encodings=False,
                 # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 v_posterior=0.,
                 parameterization="eps",  # "eps" or "x0"-> all assuming fixed variance schedules
                 ckpt_path=None
                 ) -> None:
        super().__init__(timesteps=timesteps,
                         beta_schedule=beta_schedule,
                         linear_start=linear_start,
                         linear_end=linear_end,
                         cosine_s=cosine_s,
                         given_betas=given_betas,
                         loss_type=loss_type,
                         max_tsteps=max_tsteps,
                         learning_rate=learning_rate,
                         warmup_steps=warmup_steps,
                         image_size=image_size,
                         channels=channels,
                         num_of_train_samples=num_of_train_samples,
                         num_of_val_samples=num_of_val_samples,
                         batch_size=batch_size,
                         unet_rosinality_config=unet_rosinality_config,
                         conditioning_key=conditioning_key,
                         dataset=dataset,
                         use_ema=use_ema,
                         ema_decay_factor=ema_decay_factor,
                         original_elbo_weight=original_elbo_weight,
                         l_simple_weight=l_simple_weight,
                         learn_logvar=learn_logvar,
                         logvar_init=logvar_init,
                         use_positional_encodings=use_positional_encodings,
                         # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                         v_posterior=v_posterior,
                         # "eps" or "x0"-> all assuming fixed variance schedules
                         parameterization=parameterization,
                         ckpt_path=ckpt_path)
        self.cond = cond

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        """
        In case of denoising task:
        x_noisy: noisy version of prior (ground truth) 
        cond: low res images (PET images or images to be denoised)
        """
        x_recon = self.model(x_noisy, t, cond)
        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon       


    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        x_cond_batch = batch['x_cond']
        x_prior_batch = batch['x_prior']
        x_cond_batch = x_cond_batch.to(memory_format=torch.contiguous_format).float()
        x_prior_batch = x_prior_batch.to(memory_format=torch.contiguous_format).float()
        return x_prior_batch, x_cond_batch

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, *args, **kwargs)
    

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()    
        loss = self.get_loss(model_output, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})
        
        loss_dict.update({f'trade-off/{log_prefix}': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img


    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)    


    def training_step(self, batch, batch_idx):
        bs_size = list(batch.values())[0].size()
        for key in batch.keys():
            batch[key] = batch[key].reshape(int(bs_size[0]*bs_size[2]),bs_size[1],bs_size[3],bs_size[4])       
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=False, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in self.optimizers().param_groups:
                pg["lr"] = lr_scale * self.learning_rate

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        bs_size = list(batch.values())[0].size()
        for key in batch.keys():
            batch[key] = batch[key].reshape(int(bs_size[0]*bs_size[2]),bs_size[1],bs_size[3],bs_size[4])        
        loss, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return loss    
    
    def train_dataloader(self):
        
        cond_dataset = dataloader.CombinedDataset(self.noisy_train_dataset, self.train_dataset)
        train_dataset = torch.utils.data.Subset (cond_dataset, np.arange(self.num_of_train_samples))

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

        return train_loader
    
    def val_dataloader(self):
        cond_dataset = dataloader.CombinedDataset(self.noisy_val_dataset, self.val_dataset)
        val_dataset = torch.utils.data.Subset (cond_dataset, np.arange(self.num_of_val_samples))
       # Data loader
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )
        return val_loader