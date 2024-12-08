import torch
import torch.nn as nn
import logging
from typing import Union, List, Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_eval_for_layers(model):
    """
    Sets Dropout and BatchNorm layers to evaluation mode.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()

def set_train_for_layers(model):
    """
    Sets Dropout and BatchNorm layers back to training mode.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train()

class AdversarialAttack:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        attack_type: str = 'fgsm',
        epsilon: float = 0.01,
        alpha: float = 0.01,
        iters: int = 10,
        emb_names: Union[str, List[str]] = ['embedding'],
        noise_var: float = 1e-5,
        gamma: float = 1e-2
    ):
        """
        Initialize the adversarial attack class.

        Args:
            model (nn.Module): The neural network model.
            loss_fn (nn.Module): The loss function to use (e.g., FocalLoss).
            attack_type (str): Type of attack ('fgsm', 'pgd', 'awp').
            epsilon (float): Perturbation magnitude.
            alpha (float): Step size for iterative methods (PGD).
            iters (int): Number of iterations for iterative methods (PGD).
            emb_names (Union[str, List[str]]): Names of layers to target or 'all' to target all layers.
            noise_var (float): Variance for AWP noise.
            gamma (float): AWP learning rate.

        Raises:
            ValueError: If any numerical parameters are invalid or method is unsupported.
            TypeError: If emb_names is neither string nor list.
        """
        # Validate numerical parameters
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if iters <= 0:
            raise ValueError("iters must be positive")
        if noise_var < 0:
            raise ValueError("noise_var must be non-negative")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        
        self.model = model
        self.loss_fn = loss_fn
        self.attack_type = attack_type.lower()
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.noise_var = noise_var
        self.gamma = gamma

        # Validate and process embedding names
        if isinstance(emb_names, str):
            if emb_names.lower() == 'all':
                # Collect all trainable layers excluding BatchNorm and Dropout
                self.emb_names = [name for name, module in self.model.named_modules() 
                                  if isinstance(module, (nn.Embedding, nn.Conv2d, nn.Linear))]
                if not self.emb_names:
                    logger.warning("No trainable embedding layers found in the model.")
            else:
                self.emb_names = [emb_names]
        elif isinstance(emb_names, (list, tuple)):
            self.emb_names = list(emb_names)
        else:
            raise TypeError("emb_names should be a string or a list of strings")

        # Validate attack type
        valid_methods = {'fgsm', 'pgd', 'awp'}
        if self.attack_type not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

        # Get the device from the model
        self.device = next(model.parameters()).device

        # Initialize backup dictionaries
        self.emb_backup: Dict[str, torch.Tensor] = {}
        self.weight_backup: Dict[str, torch.Tensor] = {}

        # Initialize method-specific components
        self._init_method_specific_components()
    
    def _init_method_specific_components(self) -> None:
        """Initialize components specific to each adversarial method."""
        if self.attack_type == 'awp':
            self._init_awp_noise()
        elif self.attack_type == 'pgd':
            # Pre-compute random signs for PGD initialization if needed
            self.random_signs = torch.randint(0, 2, (1,), device=self.device) * 2 - 1

    def _init_awp_noise(self) -> None:
        """
        Initialize AWP noise for trainable parameters.
        Creates noise tensors for each trainable parameter.
        """
        self.noise_weights = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param, device=param.device) * self.noise_var
                    self.noise_weights[name] = noise

    def _save_params(self, targeted_params: Optional[set] = None) -> None:
        """
        Save original parameters with proper cleanup of previous backups.
        
        Args:
            targeted_params: Optional set of parameter names to save
        """
        # Clear previous backups first to prevent memory leaks
        self.emb_backup.clear()
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if targeted_params is None or name in targeted_params:
                        self.emb_backup[name] = param.data.clone()

    def _save_weight_backup(self) -> None:
        """Save all trainable parameters for AWP with proper cleanup."""
        self.weight_backup.clear()
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.weight_backup[name] = param.data.clone()

    def _restore_params(self) -> None:
        """Restore original parameters with device compatibility check."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.emb_backup:
                    backup_tensor = self.emb_backup[name]
                    if backup_tensor.device != param.device:
                        backup_tensor = backup_tensor.to(param.device)
                    param.data.copy_(backup_tensor)
            self.emb_backup.clear()

    def _restore_weight_backup(self) -> None:
        """Restore all parameters for AWP with device compatibility check."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.weight_backup:
                    backup_tensor = self.weight_backup[name]
                    if backup_tensor.device != param.device:
                        backup_tensor = backup_tensor.to(param.device)
                    param.data.copy_(backup_tensor)
            self.weight_backup.clear()

    def _project(self, param: torch.nn.Parameter, epsilon: float) -> torch.Tensor:
        """
        Project perturbations onto L2-ball with improved error handling.
        
        Args:
            param: Parameter to project
            epsilon: Radius of L2-ball
            
        Returns:
            Projected perturbation
            
        Logs warnings for various edge cases (None gradients, zero norms, NaN values)
        """
        if param.grad is None:
            logger.warning("Encountered None gradient during projection")
            return torch.zeros_like(param.data)
        
        grad = param.grad.data
        grad_norm = torch.norm(grad)
        
        if grad_norm == 0:
            logger.warning("Encountered zero gradient norm during projection")
            return torch.zeros_like(grad)
        
        if torch.isnan(grad_norm):
            logger.warning("Encountered NaN gradient norm during projection")
            return torch.zeros_like(grad)
        
        perturb = epsilon * grad / grad_norm
        return perturb

    def generate(self, inputs: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None, 
                masks: Optional[torch.Tensor] = None, 
                batch_loss: Optional[torch.Tensor] = None) -> Union[torch.Tensor, float, None]:
        """
        Generate adversarial examples or compute adversarial loss based on attack type.
        
        Args:
            inputs (torch.Tensor, optional): Original inputs. Required for 'fgsm' and 'pgd'.
            labels (torch.Tensor, optional): True labels. Required for 'fgsm' and 'pgd'.
            masks (torch.Tensor, optional): Masks for valid inputs. Required for 'fgsm' and 'pgd'.
            batch_loss (torch.Tensor, optional): Current batch loss. Required for 'awp'.
        
        Returns:
            torch.Tensor: Adversarial examples for 'fgsm' and 'pgd'.
            float: Adversarial loss for 'awp'.
            None: If attack_type is unsupported or misused.
        
        Raises:
            ValueError: If required arguments are missing based on attack type.
        """
        if self.attack_type in ['fgsm', 'pgd']:
            if inputs is None or labels is None or masks is None:
                raise ValueError(f"Inputs, labels, and masks are required for {self.attack_type.upper()} attack.")
            return self._input_attack(inputs, labels, masks)
        elif self.attack_type == 'awp':
            if batch_loss is None:
                raise ValueError("batch_loss is required for AWP attack.")
            return self._awp_attack(batch_loss)
        else:
            logger.error(f"Unsupported attack type: {self.attack_type}")
            return None

    def _input_attack(self, inputs: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Perform input-based adversarial attacks ('fgsm' or 'pgd') and return adversarial examples.
        
        Args:
            inputs (torch.Tensor): Original inputs.
            labels (torch.Tensor): True labels.
            masks (torch.Tensor): Masks for valid inputs.
        
        Returns:
            torch.Tensor: Adversarial examples.
        """
        set_eval_for_layers(self.model)

        if self.attack_type == 'fgsm':
            # FGSM Attack
            inputs_adv = inputs.clone().detach().requires_grad_(True)
            outputs = self.model(inputs_adv, img_mask=masks, seq_mask=masks)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            perturbation = self.epsilon * inputs_adv.grad.sign()
            inputs_adv = inputs_adv + perturbation
            inputs_adv = torch.clamp(inputs_adv, 0, 1).detach()
            set_train_for_layers(self.model)
            return inputs_adv

        elif self.attack_type == 'pgd':
            # PGD Attack
            inputs_adv = inputs.clone().detach() + torch.empty_like(inputs).uniform_(-self.epsilon, self.epsilon)
            inputs_adv = torch.clamp(inputs_adv, 0, 1).requires_grad_(True)

            for _ in range(self.iters):
                outputs = self.model(inputs_adv, img_mask=masks, seq_mask=masks)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                perturbation = self.alpha * inputs_adv.grad.sign()
                inputs_adv = inputs_adv + perturbation
                # Project back to epsilon ball
                delta = torch.clamp(inputs_adv - inputs, min=-self.epsilon, max=self.epsilon)
                inputs_adv = torch.clamp(inputs + delta, 0, 1).detach().requires_grad_(True)

            set_train_for_layers(self.model)
            return inputs_adv

    def _awp_attack(self, batch_loss: torch.Tensor) -> float:
        """
        Perform Adversarial Weight Perturbation (AWP) attack and return adversarial loss.
        
        Args:
            batch_loss (torch.Tensor): Current batch loss.
        
        Returns:
            float: Adversarial loss after weight perturbation.
        """
        self._save_weight_backup()
        
        try:
            # Compute gradients w.r.t current loss
            batch_loss.backward(retain_graph=True)
            
            # Perturb weights in the direction of the gradient
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        perturbation = self.epsilon * param.grad.sign()
                        param.data += perturbation
            
            # Compute loss with perturbed weights
            # Ensure that inputs, masks, and labels are accessible here.
            # You may need to modify the interface to pass them if necessary.
            # Placeholder:
            # outputs_adv = self.model(inputs_adv, img_mask=masks, seq_mask=masks)
            # loss_adv = self.loss_fn(outputs_adv, labels)
            # For demonstration, assuming 'batch_loss' represents the forward pass.
            outputs_adv = self.model(inputs=batch_loss.detach(), img_mask=None, seq_mask=None)  # Adjust as per your model's forward signature
            loss_adv = self.loss_fn(outputs_adv, batch_loss.detach())  # Adjust targets accordingly
            
        finally:
            # Ensure gradients are cleared even if an error occurs
            self.model.zero_grad()
        
        # Restore original weights
        self._restore_weight_backup()
        
        # Restore specific layers to training mode
        set_train_for_layers(self.model)
        
        return loss_adv.item()

    def restore(self) -> None:
        """Restore model to natural state based on attack method."""
        if self.attack_type in ['fgsm', 'pgd']:
            self._restore_params()
        elif self.attack_type == 'awp':
            self._restore_weight_backup()
        
        # Restore specific layers to training mode
        set_train_for_layers(self.model)

    def to(self, device: torch.device) -> 'AdversarialAttack':
        """
        Move all internal tensors to the specified device.
        
        Args:
            device: The target device
            
        Returns:
            Self for method chaining
        """
        self.device = device
        
        # Move backup tensors
        self.emb_backup = {
            name: param.to(device) 
            for name, param in self.emb_backup.items()
        }
        self.weight_backup = {
            name: param.to(device) 
            for name, param in self.weight_backup.items()
        }
        
        # Move method-specific tensors
        if hasattr(self, 'noise_weights'):
            self.noise_weights = {
                name: noise.to(device) 
                for name, noise in self.noise_weights.items()
            }
        if hasattr(self, 'random_signs'):
            self.random_signs = self.random_signs.to(device)
        
        return self

    def state_dict(self) -> Dict:
        """
        Get complete state dict for checkpointing.
        
        Returns:
            Dictionary containing all necessary state for model restoration
        """
        state = {
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'iters': self.iters,
            'attack_type': self.attack_type,
            'noise_var': self.noise_var,
            'gamma': self.gamma,
            'emb_names': self.emb_names
        }
        
        if self.attack_type == 'awp' and hasattr(self, 'noise_weights'):
            state['noise_weights'] = {
                name: noise.detach().cpu() 
                for name, noise in self.noise_weights.items()
            }
        
        return state

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Load state dict from checkpoint.
        
        Args:
            state_dict: Dictionary containing model state
        """
        self.epsilon = state_dict['epsilon']
        self.alpha = state_dict['alpha']
        self.iters = state_dict['iters']
        self.attack_type = state_dict['attack_type']
        self.noise_var = state_dict['noise_var']
        self.gamma = state_dict['gamma']
        self.emb_names = state_dict['emb_names']
        
        if self.attack_type == 'awp' and 'noise_weights' in state_dict:
            self.noise_weights = {
                name: noise.to(self.device)
                for name, noise in state_dict['noise_weights'].items()
            }
