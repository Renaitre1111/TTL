import torch
import torch.nn as nn
import torch.jit
import math
import torch.nn.functional as F

class GTF(nn.Module):
    def __init__(self, model, args, optimizer, scaler, steps=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.args = args
        self.steps = steps

        self.beta = args.gtf_beta
        self.lambda_e = args.gtf_lambda
        self.svd_k = args.gtf_k
        self.mu_e = None
        self.momentum = args.gtf_momentum

        self.projectors = {}
        self.setup_geometric_constraints()

    def setup_geometric_constraints(self):
        target_modules = []
        if self.args.lora_encoder == 'image':
            layers = self.model.image_encoder.vision_model.encoder.layers
        elif self.args.lora_encoder == 'text':
            layers = self.model.text_encoder.text_model.encoder.layers
        else:
            return
        
        for i, layer in enumerate(layers):
            if self.args.layer_range[0] <= i <= self.args.layer_range[1]:
                target_modules.append((f"layer_{i}_q", layer.self_attn.q_proj))
                target_modules.append((f"layer_{i}_v", layer.self_attn.v_proj))

        for name, module in target_modules:
            if hasattr(module, 'base_layer'):
                W0 = module.base_layer.weight.detach()
            else:
                W0 = module.weight.detach()
            U, S, Vh = torch.linalg.svd(W0.float(), full_matrices=False)
            k = min(self.svd_k, S.shape[0])
            Uk = U[:, :k] # [d_out, k]
            Vk_T = Vh[:k, :] # [k, d_in]

            identity_L = torch.eye(Uk.shape[0], device=W0.device)
            P_L_perp = identity_L - (Uk @ Uk.T)
            identity_R = torch.eye(Vk_T.shape[1], device=W0.device)

            Vk = Vh.T[:, :k]
            P_R_perp = identity_R - (Vk @ Vk.T)

            self.projectors[name] = (P_L_perp, P_R_perp)
            
            # Clean up SVD results to save memory
            del U, S, Vh, Uk, Vk

    def apply_geometric_constraints(self):
        if self.args.lora_encoder == 'image':
            layers = self.model.image_encoder.vision_model.encoder.layers
        elif self.args.lora_encoder == 'text':
            layers = self.model.text_encoder.text_model.encoder.layers
        else:
            return

        for i, layer in enumerate(layers):
            if self.args.layer_range[0] <= i <= self.args.layer_range[1]:
                q_name = f"layer_{i}_q"
                if q_name in self.projectors:
                    Pl, Pr = self.projectors[q_name]
                    # lora_A shape: [rank, d_in], lora_B shape: [d_out, rank]
                    # Project lora_B (Left side): B = Pl @ B
                    with torch.no_grad():
                        layer.self_attn.q_proj.lora_B.default.weight.data = \
                            Pl @ layer.self_attn.q_proj.lora_B.default.weight.data
                        layer.self_attn.q_proj.lora_A.default.weight.data = \
                            layer.self_attn.q_proj.lora_A.default.weight.data @ Pr

                v_name = f"layer_{i}_v"
                if v_name in self.projectors:
                    Pl, Pr = self.projectors[v_name]
                    with torch.no_grad():
                        layer.self_attn.v_proj.lora_B.default.weight.data = \
                            Pl @ layer.self_attn.v_proj.lora_B.default.weight.data
                        
                        layer.self_attn.v_proj.lora_A.default.weight.data = \
                            layer.self_attn.v_proj.lora_A.default.weight.data @ Pr
                        
    def forward(self, x, iter_=None):
        outputs = self.model(x)
        energy = -1.0 * torch.logsumexp(outputs, dim=1) # [N_views]
        with torch.no_grad():
            mean_energy = energy.mean()
            if self.mu_e is None:
                self.mu_e = mean_energy
            else:
                self.mu_e = self.momentum * self.mu_e + (1 - self.momentum) * mean_energy
        weights = torch.sigmoid(-(energy - self.mu_e) / self.beta) # [N_views]
        if weights.sum() == 0:
            weights_norm = torch.ones_like(weights) / len(weights)
        else:
            weights_norm = weights / weights.sum()
        probs = outputs.softmax(dim=1) # [N_views, N_classes]
        soft_prototype = torch.matmul(weights_norm.unsqueeze(0), probs).squeeze(0) # [N_classes]

        target_p = soft_prototype.unsqueeze(0).expand(outputs.shape[0], -1).detach()
        loss_consistency = F.kl_div(outputs.log_softmax(dim=1), target_p, reduction='none').sum(dim=1) # [N_views]

        loss_energy = energy
        total_loss = (1 - self.lambda_e) * loss_consistency.mean() + self.lambda_e * loss_energy.mean()

        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.apply_geometric_constraints()
        return outputs
