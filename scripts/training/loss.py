import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha, 1 - alpha]).to(self.device)
            elif isinstance(alpha, list):
                self.alpha = torch.tensor(alpha).to(self.device)
            else:
                self.alpha = alpha.to(device) # Handle Tensor objects  
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            at = self.alpha[targets]
            ce_loss = ce_loss * at
        loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss