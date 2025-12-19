import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MC_LoRA(nn.Module):
    def __init__(self, model, optimizer, scaler, args):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.args = args

        self.tau = args.mclora_tau
        self.alpha = args.mclora_alpha
        self.n_min = args.mclora_n_min
        self.lambda_tma = args.mclora_lambda

        if not hasattr(self.model, 'manifold_basis'):
            self.manifold_basis = self.model.compute_text_manifold(tau=self.tau)

    def forward(self, images):
        with torch.cuda.amp.autocast():
            outputs = self.model(images) # [N, C]
            probs = torch.softmax(outputs, dim=1)
            avg_probs = probs.mean(dim=0)

            n_classes = outputs.shape[1]
            n_safe = max(self.n_min, int(math.ceil(self.alpha * n_classes)))

            # Sort classes by confidence
            sorted_probs, sorted_indices = torch.sort(avg_probs, descending=True)

            neg_indices = sorted_indices[n_safe:]
            neg_probs = sorted_probs[n_safe:]
            
            loss_neg = -torch.sum(torch.log(neg_probs + 1e-7))

        self.optimizer.zero_grad()
        self.scaler.scale(loss_neg).backward(retain_graph=True)

        g_neg = []
        for param in self.model.parameters():
            if param.grad is not None:
                g_neg.append(param.grad.clone().detach())
            else:
                g_neg.append(None)

        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss_ent = -(probs * torch.log(probs + 1e-7)).sum(dim=1).mean()
            loss_tma = 0.0
            if self.lambda_tma > 0:
                visual_features = self.model.image_features
                B = self.manifold_basis
                f_rec = visual_features @ B @ B.t()
                residual = visual_features - f_rec
                loss_tma = torch.norm(residual, p=2, dim=1).mean()

            loss_total = loss_ent + self.lambda_tma * loss_tma

        self.scaler.scale(loss_total).backward()

        eps = 1e-8
        param_idx = 0
        for param in self.model.parameters():
            if param.grad is not None and g_neg[param_idx] is not None:
                g_pos_vec = param.grad
                g_neg_vec = g_neg[param_idx]

                g_pos_flat = g_pos_vec.view(-1)
                g_neg_flat = g_neg_vec.view(-1)

                dot_prod = torch.dot(g_pos_flat, g_neg_flat)
                norm_sq = torch.dot(g_neg_flat, g_neg_flat)

                scalar = dot_prod / (norm_sq + eps)

                proj_update = scalar * g_neg_vec

                param.grad.data.sub_(proj_update)

            param_idx += 1
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return outputs