import torch
import torch.nn as nn
import torch.nn.functional as F

class DUEL(nn.Module):
    def __init__(self, model, optimizer, scaler, args):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.args = args
        self.steps = args.tta_steps

    def forward(self, x):
        with torch.no_grad():
            _ = self.model(x)
            anchor_features = self.model.image_features.clone().detach()

        for _ in range(self.steps):
            self.forward_and_adapt(x, anchor_features)
        
        return

    def forward_and_adapt(self, x, anchor_features):
        outputs = self.model(x)
        current_features = self.model.image_features

        evidence = F.softplus(outputs)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        K = alpha.shape[1]

        uncertainty = K / S

        probs = alpha / S

        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        evidence_suppression = torch.sum(evidence, dim=1)

        u_score = uncertainty.squeeze().detach()
        w_reliable = 1 - u_score
        w_ood = u_score

        loss_reliable = (w_reliable * entropy).mean()
        loss_ood = (w_ood * evidence_suppression).mean()

        loss_anchor = F.mse_loss(current_features, anchor_features)

        total_loss = loss_reliable + self.args.lambda_ood * loss_ood + self.args.lambda_anchor * loss_anchor

        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
