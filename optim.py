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

class SGDMomentum(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(SGDMomentum, self).__init__(params, defaults)


    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                momentum = state['momentum']
                momentum.mul_(0.9).add_(d_p)
                p.data.add_(-group['lr'], momentum)
    
class CustomZeroOrderOptimizer(CustomOptimizer):

    def __init__(self, params, lr=1e-3, loss_fn=None, epsilon=1e-3, rand_directions=10):
        super(CustomZeroOrderOptimizer, self).__init__(params, lr)
        self.loss_fn = loss_fn
        if self.loss_fn is None:
            raise ValueError("loss_fn must be provided for CustomZeroOrderOptimizer")
        self.epsilon = epsilon
        self.rand_directions = rand_directions

    def step(self, model, X, y):

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p is None or not p.requires_grad:
                    continue
                    
                original_p_data = p.data.clone()

                d_p = torch.zeros_like(p.data)

                for i in range(self.rand_directions):
                    rand_direction = torch.randn_like(p.data)
                    norm = torch.norm(rand_direction)

                    if norm > 1e-6:
                        rand_direction /= norm
                    else:
                        
                        continue

                    p.data.copy_(original_p_data + self.epsilon * rand_direction)
                    
                    loss_plus = self.loss_fn(model(X), y)


                    p.data.copy_(original_p_data - self.epsilon * rand_direction)

                    loss_minus = self.loss_fn(model(X), y)
                    gradient_component_estimate = (loss_plus - loss_minus) / (2.0 * self.epsilon)
                    d_p += gradient_component_estimate * rand_direction
                p.data.copy_(original_p_data)
                p.data.add_(-lr, d_p)

class AdamOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(AdamOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):

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
                p.data.addcdiv_(m, denom, value=-lr / bias_correction1)
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                denom_hat = v_hat.sqrt().add_(eps)

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
                sum_sq.addcmul_(grad, grad, value=1) 

                # Update parameters
                std = sum_sq.sqrt().add_(eps)
                p.data.addcdiv_(grad, std, value=-lr)

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

                if len(state) == 0:
                    state['sum'] = torch.zeros_like(p.data)

                sum_sq = state['sum']
                sum_sq.addcmul_(grad, grad, value=1)

                std = sum_sq.sqrt().add_(eps)
                p.data.addcdiv_(grad, std, value=-lr)

class PAdagradOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8):
        defaults = dict(lr=lr, eps=eps)
        super(PAdagradOptimizer, self).__init__(params, defaults)

    def step(self, closure=None, ball_size = 2):
        avg_grads = []
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['sum'] = torch.zeros_like(p.data)

                sum_sq = state['sum']
                sum_sq.addcmul_(grad, grad, value=1)

                std = sum_sq.sqrt().add_(eps)
                p.data.addcdiv_(grad, std, value=-lr)
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