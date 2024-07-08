from diffusers import DDPMScheduler, DDIMScheduler
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, randn_tensor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
import numpy as np
import torch

class OurDDPMScheduler(DDPMScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        variance_type: str = "learned_range",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        rescale_timesteps: bool = False, # ours
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            variance_type=variance_type,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
            thresholding=thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            clip_sample_range=clip_sample_range,
            sample_max_value=sample_max_value,
        )
        self.rescale_timesteps = rescale_timesteps


    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
        cond_fn=None,
        classifier=None,
        labels=None,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDPMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        assert not (cond_fn is not None and classifier is None), "cond_fn and classifier must be given together"
        with torch.no_grad():
            t = timestep
            num_inference_steps = self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
            prev_t = timestep - self.config.num_train_timesteps // num_inference_steps

            if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
                model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1) # p_mean, p_var
            else:
                predicted_variance = None

            # 1. compute alphas, betas
            alpha_prod_t = self.alphas_cumprod[t]
            alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta_t = 1 - current_alpha_t

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if self.config.prediction_type == "epsilon":
                pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif self.config.prediction_type == "sample":
                pred_original_sample = model_output
            elif self.config.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                    " `v_prediction`  for the DDPMScheduler."
                )

            # 3. Clip or threshold "predicted x_0"
            if self.config.thresholding:
                pred_original_sample = self._threshold_sample(pred_original_sample)
            elif self.config.clip_sample:
                pred_original_sample = pred_original_sample.clamp(
                    -self.config.clip_sample_range, self.config.clip_sample_range
                )

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
            current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

            # 6. Add noise
            variance = 0
            if t > 0:
                device = model_output.device
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                )
                if self.variance_type == "fixed_small_log":
                    variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
                elif self.variance_type == "learned_range":
                    variance = self._get_variance(t, predicted_variance=predicted_variance)
                    variance = torch.exp(0.5 * variance) * variance_noise
                else:
                    variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        if cond_fn is not None:
            gradient = cond_fn(model_output, self._scale_timesteps(timestep), labels, classifier, sample)
            if predicted_variance is None:
                new_mean = pred_prev_sample + self._get_variance(t,predicted_variance=predicted_variance) * gradient
            else:
                new_mean = pred_prev_sample + predicted_variance * gradient
            pred_prev_sample = new_mean # from guided_diffusion

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
    
    def _scale_timesteps(self, timestep):
        if self.rescale_timesteps:
            return timestep.float() * (1000.0 / self.num_timesteps)
        return timestep
    

class OurDDIMScheduler(DDIMScheduler):
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        rescale_timesteps: bool = False, # ours
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            clip_sample=clip_sample,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
            prediction_type=prediction_type,
            thresholding=thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            clip_sample_range=clip_sample_range,
            sample_max_value=sample_max_value,
        )
        self.rescale_timesteps = rescale_timesteps

    
    def _scale_timesteps(self, timestep):
        if self.rescale_timesteps:
            return timestep.float() * (1000.0 / self.num_timesteps)
        return timestep
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        cond_fn=None,
        classifier=None,
        labels=None,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
                predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
                `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
                coincide with the one provided as input and `use_clipped_model_output` will have not effect.
            generator: random number generator.
            variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
                can directly provide the noise for the variance itself. This is useful for methods such as
                CycleDiffusion. (https://arxiv.org/abs/2210.05559)
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        with torch.no_grad():
            prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

            # 2. compute alphas, betas
            alpha_prod_t = self.alphas_cumprod[timestep]
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

            beta_prod_t = 1 - alpha_prod_t

        if cond_fn is not None:
            gradient, pred_orig_sample_from_cond = cond_fn(model_output, self._scale_timesteps(timestep), labels, classifier, sample)
            # model_output = (model_output - beta_prod_t**(0.5) * gradient).detach().clone()
            classifier.zero_grad()
        else:
            gradient = None
        
        # if gradient is not None:
        #     model_output = (model_output - beta_prod_t**(0.5) * gradient).detach().clone()
        #     sample = sample.detach()


        # with torch.no_grad():

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            if gradient is not None:
                # pred_original_sample = (sample.detach() - beta_prod_t ** (0.5) * (model_output.detach().clone() - beta_prod_t**(0.5)*gradient)) / alpha_prod_t ** (0.5)

                # model_output = (model_output + beta_prod_t**(0.5) * gradient).detach().clone()
                # pred_original_sample = (sample.detach() - beta_prod_t ** (0.5) * (model_output.detach().clone())) / alpha_prod_t ** (0.5)
                # pred_original_sample=pred_original_sample + gradient
                # pred_epsilon = model_output.detach().clone()


                model_output = (model_output - beta_prod_t**(0.5) * gradient).detach().clone()
                # model_output=model_output.detach().clone()
                # pred_original_sample = (sample - beta_prod_t ** (0.5) * (model_output)) / alpha_prod_t ** (0.5)
                pred_original_sample = (pred_orig_sample_from_cond - beta_prod_t ** (0.5) * (model_output)) / alpha_prod_t ** (0.5)
                # pred_original_sample = (pred_orig_sample_from_cond + gradient) / alpha_prod_t ** (0.5)
                pred_original_sample=pred_original_sample
                pred_epsilon = model_output
            else:
                pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        with torch.no_grad():
            # 5. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            variance = self._get_variance(timestep, prev_timestep)
            std_dev_t = eta * variance ** (0.5)
            

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # with torch.no_grad():
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

