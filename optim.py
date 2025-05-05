import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class CustomOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(CustomOptimizer, self).__init__(params, defaults)


    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data += -group['lr']*d_p
    
class CustomZeroOrderOptimizer(CustomOptimizer):

    def __init__(self, params, lr=1e-3, loss_fn=None, epsilon=1e-3, rand_directions=10):
        super(CustomZeroOrderOptimizer, self).__init__(params, lr)
        self.loss_fn = loss_fn
        if self.loss_fn is None:
            raise ValueError("loss_fn must be provided for CustomZeroOrderOptimizer")
        self.epsilon = epsilon # Perturbation size
        self.rand_directions = rand_directions # Number of random directions

    def step(self, model, X, y):
        # Ensure model is in training mode if applicable
        # model.train() # Or handle outside if preferred

        # We will manually update p.data, bypassing autograd for the update itself.
        # However, the forward passes inside need to be runnable.
        # No need for an outer torch.no_grad() in step itself.

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                # Skip parameters that shouldn't be updated (e.g., frozen layers) or are None
                if p is None or not p.requires_grad:
                    continue
                    
                # Zero-order doesn't use p.grad
                # if p.grad is None: continue # This check is not needed for this method

                # Store the original parameter value before perturbation
                original_p_data = p.data.clone()

                # d_p will accumulate the gradient estimate for this parameter
                d_p = torch.zeros_like(p.data)

                # Compute the gradient estimate using finite differences along random directions
                for i in range(self.rand_directions):
                    # Generate a random direction in the *parameter space* of p
                    # Using randn_like gives values around 0, common for directions
                    rand_direction = torch.randn_like(p.data)
                    norm = torch.norm(rand_direction)

                    # Normalize the direction
                    if norm > 1e-6: # Add small epsilon to avoid division by zero
                        rand_direction /= norm
                    else:
                        # If norm is essentially zero, skip this direction
                        continue

                    # --- Perturb the parameter p and evaluate the loss ---

                    # Perturb parameter in the positive direction: p + epsilon * rand_direction
                    # We manually update the .data tensor
                    p.data.copy_(original_p_data + self.epsilon * rand_direction)
                    
                    # Evaluate the loss using the *perturbed parameter* and the *original input*
                    # Need to ensure model(X) runs correctly even with manual p.data changes.
                    # If the outer training loop *was* in no_grad, we'd need torch.enable_grad() here.
                    # But since we removed the outer no_grad, this forward pass works normally.
                    loss_plus = self.loss_fn(model(X), y)


                    # Perturb parameter in the negative direction: p - epsilon * rand_direction
                    p.data.copy_(original_p_data - self.epsilon * rand_direction)

                    # Evaluate the loss using the *perturbed parameter* and the *original input*
                    loss_minus = self.loss_fn(model(X), y)

                    # --- Accumulate the gradient estimate component ---
                    # The formula is (f(theta + epsilon*u) - f(theta - epsilon*u)) / (2*epsilon) * u
                    # This gives the directional derivative estimate multiplied by the direction vector
                    gradient_component_estimate = (loss_plus - loss_minus) / (2.0 * self.epsilon)
                    d_p += gradient_component_estimate * rand_direction

                # --- Restore the parameter value before applying the update ---
                # It's important to set p back to its original value *before* applying the step
                p.data.copy_(original_p_data)

                # --- Apply the parameter update ---
                # Update rule: p = p - lr * grad_estimate
                # Using p.data.add_(-lr, d_p) which is p.data = p.data + (-lr * d_p)
                p.data.add_(-lr, d_p)

class AdamOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(AdamOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        # closure = closure() # Standard optimizers might do this, but keeping it brief like example

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta1, beta2 = betas

                state['step'] += 1
                step = state['step']

                # Adam algorithm
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Denominator term
                denom = v.sqrt().add_(eps)

                # Update parameters
                p.data.addcdiv_(m, denom, value=-lr / bias_correction1) # Apply learning rate and first moment correction
                # Simplified application combining LR and bias correction 1
                # The standard update is -lr * m_hat / (sqrt(v_hat) + eps)
                # = -lr * (m / (1-b1^t)) / (sqrt(v / (1-b2^t)) + eps)
                # The most common implementation uses the corrected moments m_hat, v_hat directly
                # Let's stick to the standard formula: p.data.addcdiv_(-lr, m_hat, v_hat.sqrt().add_(eps))
                # Need m_hat and v_hat explicitly
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                denom_hat = v_hat.sqrt().add_(eps) # Denominator with corrected v

                p.data.addcdiv_(m_hat, denom_hat, value=-lr) 

class AdagradOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8):
        defaults = dict(lr=lr, eps=eps)
        super(AdagradOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['sum'] = torch.zeros_like(p.data)

                sum_sq = state['sum']
                sum_sq.addcmul_(grad, grad, value=1) # sum_sq += grad^2

                # Update parameters
                std = sum_sq.sqrt().add_(eps)
                p.data.addcdiv_(grad, std, value=-lr) # p -= lr * grad / (sqrt(sum_sq) + eps)                   

class AdagradOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8):
        defaults = dict(lr=lr, eps=eps)
        super(AdagradOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['sum'] = torch.zeros_like(p.data)

                sum_sq = state['sum']
                sum_sq.addcmul_(grad, grad, value=1) # sum_sq += grad^2

                # Update parameters
                std = sum_sq.sqrt().add_(eps)
                p.data.addcdiv_(grad, std, value=-lr) # p -= lr * grad / (sqrt(sum_sq) + eps)                   

class PAdagradOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8):
        defaults = dict(lr=lr, eps=eps)
        super(PAdagradOptimizer, self).__init__(params, defaults)

    def step(self, closure=None, ball_size = 2):
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['sum'] = torch.zeros_like(p.data)

                sum_sq = state['sum']
                sum_sq.addcmul_(grad, grad, value=1) # sum_sq += grad^2

                # Update parameters
                std = sum_sq.sqrt().add_(eps)
                p.data.addcdiv_(grad, std, value=-lr) # p -= lr * grad / (sqrt(sum_sq) + eps)   
                if torch.norm(p.data) > ball_size:
                    p.data = p.data / torch.norm(p.data) * ball_size
                
class Averager:
    def __init__(self,model):
        self.model = model
        self.count = 0

    def update(self,model):
        for param, avg_param in zip(model.parameters(), self.model.parameters()):
            if self.count == 0:
                avg_param.data = param.data.clone()
            else:
                avg_param.data = (avg_param.data * self.count + param.data) / (self.count + 1)
        self.count += 1
        return self.model
    def get_model(self):
        return self.model