"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

class DeYO(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, args, optimizer, scaler, steps=1, episodic=False, deyo_margin=0.5*math.log(1000), margin_e0=0.4*math.log(1000)):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.args = args
        # if args.wandb_log:
        #     import wandb
        self.steps = steps
        self.episodic = episodic
        # args.counts = [1e-6,1e-6,1e-6,1e-6]
        # args.correct_counts = [0,0,0,0]

        self.deyo_margin = deyo_margin
        self.margin_e0 = margin_e0

    def forward(self, x, iter_ = None, targets=None, flag=True, group=None):
        original_features = None
        if self.args.use_duel:
            with torch.no_grad():
                _ = self.model(x)
                original_features = self.model.image_features.detach().clone()
        
        if targets is None:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward = forward_and_adapt_sar(x, iter_, self.model, self.args,
                                                                              self.optimizer, self.scaler, self.deyo_margin,
                                                                              self.margin_e0, targets, flag, group,
                                                                              original_features=original_features)
                else:
                    outputs = forward_and_adapt_sar(x, iter_, self.model, self.args,
                                                    self.optimizer, self.deyo_margin,
                                                    self.margin_e0, targets, flag, group,
                                                    original_features=original_features)
        else:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward, corr_pl_1, corr_pl_2 = forward_and_adapt_sar(x, iter_, self.model, 
                                                                                                    self.args, 
                                                                                                    self.optimizer, 
                                                                                                    self.deyo_margin,
                                                                                                    self.margin_e0,
                                                                                                    targets, flag, group,
                                                                                                    original_features=original_features)
                else:
                    outputs = forward_and_adapt_sar(x, iter_, self.model, 
                                                    self.args, self.optimizer, 
                                                    self.deyo_margin,
                                                    self.margin_e0,
                                                    targets, flag, group,
                                                    original_features=original_features)
        if targets is None:
            if flag:
                return outputs, backward, final_backward
            else:
                return outputs
        else:
            if flag:
                return outputs, backward, final_backward, corr_pl_1, corr_pl_2
            else:
                return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def duel_loss(outputs, current_features, original_features, lambda_ood, lambda_anchor):
    evidence = F.softplus(outputs)
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True) # Total evidence
    K = outputs.size(1)
    u = K / S # Uncertainty mass [N, 1]
    
    # Probability estimate from Dirichlet
    prob = alpha / S 
    prob = torch.clamp(prob, min=1e-8) # Stability

    L_rel = -torch.sum(prob * torch.log(prob), dim=1, keepdim=True) # [N, 1]
    L_ood = torch.sum(evidence, dim=1, keepdim=True) # [N, 1]
    L_adapt_per_sample = (1 - u) * L_rel + lambda_ood * u * L_ood
    L_adapt = L_adapt_per_sample.mean()

    L_anchor_val = 0.0
    if original_features is not None and current_features is not None:
        if current_features.shape == original_features.shape:
             L_anchor_val = F.mse_loss(current_features, original_features)
        else:
             # Broadcasting or shape mismatch fallback
             pass
    total_loss = L_adapt + lambda_anchor * L_anchor_val
    return total_loss

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_sar(x, iter_, model, args, optimizer, scaler, deyo_margin, margin, targets=None, flag=True, group=None, original_features=None):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    outputs = model(x)
    if not flag:
        return outputs
    
    if args.use_duel:
        current_features = None
        if hasattr(model, 'image_features'):
            current_features = model.image_features
            
        loss = duel_loss(
            outputs, 
            current_features, 
            original_features, 
            lambda_ood=args.lambda_ood, 
            lambda_anchor=args.lambda_anchor
        )
        
        backward = outputs.shape[0] 
        final_backward = outputs.shape[0]
        
        optimizer.zero_grad()
        scaler.scale(loss).backward() 
        scaler.step(optimizer) 
        scaler.update()
        
        if targets is not None:
            prob_outputs = outputs.softmax(1)
            corr_pl_1 = (targets == prob_outputs.argmax(dim=1)).sum().item()
            corr_pl_2 = corr_pl_1
            return outputs, backward, final_backward, corr_pl_1, corr_pl_2
            
        return outputs, backward, final_backward

    entropys = softmax_entropy(outputs)
    if args.filter_ent:
        filter_ids_1 = torch.argsort(entropys, descending=False)[:int(entropys.size()[0] * args.selection_p)] 
    else:    
        filter_ids_1 = torch.where((entropys <= math.log(1000)))
    
    entropys = entropys[filter_ids_1]
    backward = len(entropys)
    if backward==0:
        if targets is not None:
            return outputs, 0, 0, 0, 0
        return outputs, 0, 0
    
    if args.filter_plpd:
        x_prime = x[filter_ids_1].detach()
        if args.aug_type=='occ':
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, args.occlusion_size, args.occlusion_size)
            x_prime[:, :, args.row_start:args.row_start+args.occlusion_size,args.column_start:args.column_start+args.occlusion_size] = occlusion_window
        elif args.aug_type=='patch':
            resize_t = torchvision.transforms.Resize(((x.shape[-1]//args.patch_len)*args.patch_len,(x.shape[-1]//args.patch_len)*args.patch_len))
            resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
            x_prime = resize_o(x_prime)
        elif args.aug_type=='pixel':
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
        
        outputs_prime = model(x_prime)
        prob_outputs = outputs[filter_ids_1].softmax(1)
        prob_outputs_prime = outputs_prime.softmax(1)
        cls1 = prob_outputs.argmax(dim=1)
        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
        plpd = plpd.reshape(-1)
    
        filter_ids_2 = torch.where(plpd > args.plpd_threshold) if args.filter_plpd else torch.where(plpd >= -2.0)
        entropys = entropys[filter_ids_2]
        
    final_backward = len(entropys)
        
    if targets is not None:
        corr_pl_1 = 0
        corr_pl_2 = 0
        if len(filter_ids_1[0]) > 0:
             corr_pl_1 = (targets[filter_ids_1] == outputs[filter_ids_1].argmax(dim=1)).sum().item()
             if args.filter_plpd and len(filter_ids_2[0]) > 0:
                 pass 

    if args.reweight_ent or args.reweight_plpd:
        coeff = (args.reweight_ent * (1 / (torch.exp(((entropys.clone().detach()) - margin)))))
        entropys = entropys.mul(coeff) 

    loss = entropys.mean(0)

    if final_backward != 0:
        optimizer.zero_grad()
        scaler.scale(loss).backward() 
        scaler.step(optimizer) 
        scaler.update() 
    
    if targets is not None:
        return outputs, backward, final_backward, corr_pl_1, corr_pl_2
    return outputs, backward, final_backward

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model):
    """Configure model for use with DeYO."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model