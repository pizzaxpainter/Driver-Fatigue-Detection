import torch
import torch.nn as nn

class AdversarialAttack:
    def __init__(self, model, attack_type='fgsm', epsilon=0.01, alpha=0.01, iters=10):
        """
        Initialize the adversarial attack class.

        Args:
            model (nn.Module): The neural network model.
            attack_type (str): Type of attack ('fgsm', 'pgd', 'awp').
            epsilon (float): Perturbation magnitude (input perturbation for FGSM and PGD, weight perturbation for AWP).
            alpha (float): Step size for iterative methods (PGD).
            iters (int): Number of iterations for iterative methods (PGD).
        """
        self.model = model
        self.attack_type = attack_type.lower()
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters

    def generate(self, inputs, labels, masks=None):
        """
        Generate adversarial examples.

        Args:
            inputs (torch.Tensor): Original inputs.
            labels (torch.Tensor): True labels.
            masks (torch.Tensor, optional): Masks for valid inputs.

        Returns:
            torch.Tensor: Adversarial examples.
        """
        if self.attack_type == 'fgsm':
            return self.fgsm_attack(inputs, labels, masks)
        elif self.attack_type == 'pgd':
            return self.pgd_attack(inputs, labels, masks)
        elif self.attack_type == 'awp':
            return self.awp_attack(inputs, labels, masks)
        else:
            raise ValueError(f"Unsupported attack type: {self.attack_type}")

    def fgsm_attack(self, inputs, labels, masks=None):
        """
        Perform FGSM attack.

        Args:
            inputs (torch.Tensor): Original inputs.
            labels (torch.Tensor): True labels.
            masks (torch.Tensor, optional): Masks for valid inputs.

        Returns:
            torch.Tensor: Adversarial examples.
        """
        # Ensure the model is in evaluation mode and gradients are disabled for parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable gradients for inputs
        inputs_adv = inputs.clone().detach().requires_grad_(True)

        # Forward pass
        outputs = self.model(inputs_adv, img_mask=masks, seq_mask=masks)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Backward pass to compute gradients w.r.t. inputs
        loss.backward()

        # Generate adversarial perturbation
        perturbation = self.epsilon * inputs_adv.grad.sign()

        # Create adversarial examples
        inputs_adv = inputs_adv + perturbation

        # Clamp to valid pixel range
        inputs_adv = torch.clamp(inputs_adv, 0, 1)

        # Re-enable gradients for model parameters
        for param in self.model.parameters():
            param.requires_grad = True

        return inputs_adv.detach()

    def pgd_attack(self, inputs, labels, masks=None):
        """
        Perform PGD attack.

        Args:
            inputs (torch.Tensor): Original inputs.
            labels (torch.Tensor): True labels.
            masks (torch.Tensor, optional): Masks for valid inputs.

        Returns:
            torch.Tensor: Adversarial examples.
        """
        # Ensure the model is in evaluation mode and gradients are disabled for parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize adversarial examples with random noise within epsilon ball
        inputs_adv = inputs.clone().detach()
        inputs_adv += torch.empty_like(inputs_adv).uniform_(-self.epsilon, self.epsilon)
        inputs_adv = torch.clamp(inputs_adv, 0, 1)
        inputs_adv.requires_grad = True

        # Perform iterative attacks
        for _ in range(self.iters):
            # Zero gradients for inputs
            if inputs_adv.grad is not None:
                inputs_adv.grad.zero_()

            # Forward pass
            outputs = self.model(inputs_adv, img_mask=masks, seq_mask=masks)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # Backward pass to compute gradients w.r.t. inputs
            loss.backward()

            # Generate adversarial perturbation
            perturbation = self.alpha * inputs_adv.grad.sign()

            # Update adversarial examples with projection
            inputs_adv = inputs_adv.detach() + perturbation
            inputs_adv = torch.max(torch.min(inputs_adv, inputs + self.epsilon), inputs - self.epsilon)
            inputs_adv = torch.clamp(inputs_adv, 0, 1)
            inputs_adv.requires_grad = True

        # Re-enable gradients for model parameters
        for param in self.model.parameters():
            param.requires_grad = True

        return inputs_adv.detach()

    def awp_attack(self, inputs, labels, masks=None):
        """
        Perform AWP attack (Adversarial Weight Perturbation).

        Args:
            inputs (torch.Tensor): Original inputs.
            labels (torch.Tensor): True labels.
            masks (torch.Tensor, optional): Masks for valid inputs.

        Returns:
            torch.Tensor: Original inputs (AWP modifies weights, not inputs).
        """
        # Store original weights and set up perturbation variables
        original_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.data.clone()
                param.requires_grad = False  # Temporarily disable gradient computation

        # Compute adversarial perturbation for weights
        # For demonstration, we use a simple approximation
        # In practice, this should be optimized to find the worst-case perturbation

        # Enable gradient computation for parameters
        for param in self.model.parameters():
            param.requires_grad = True

        # Compute gradients w.r.t. parameters to find worst-case direction
        self.model.zero_grad()
        outputs = self.model(inputs, img_mask=masks, seq_mask=masks)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        # Perturb weights in the direction of the gradient
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    perturbation = self.epsilon * param.grad.sign()
                    param.data += perturbation

        # Compute loss with perturbed weights
        outputs_adv = self.model(inputs, img_mask=masks, seq_mask=masks)
        loss_adv = nn.CrossEntropyLoss()(outputs_adv, labels)

        # Restore original weights
        for name, param in self.model.named_parameters():
            if name in original_params:
                param.data = original_params[name]

        # Return the loss with perturbed weights for adversarial training
        return loss_adv