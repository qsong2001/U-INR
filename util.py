import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad



def get_image_tensor(image_path, sidelength):
    img = Image.open(image_path).convert('RGB')  
    transform = Compose([
        Resize((sidelength, sidelength)),  
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

class ImageFitting(Dataset):
    def __init__(self, sidelength, img):
        super().__init__()
        self.pixels = img.permute(1, 2, 0).reshape(-1, 3)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels




import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


# --- 1. Quantum-Inspired Sine Layer with Phase Modulation ---
class QuantumSineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0, use_phase=True, use_complex=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.use_phase = use_phase
        self.use_complex = use_complex

        # 🔥 必须先保存 in_features/out_features
        self.in_features = in_features
        self.out_features = out_features

        if use_complex:
            # Complex linear layer: W ∈ ℂ^(out×in), b ∈ ℂ^out
            self.weight = nn.Parameter(torch.complex(
                torch.randn(out_features, in_features) * 1e-3,
                torch.randn(out_features, in_features) * 1e-3
            ))
            if bias:
                self.bias = nn.Parameter(torch.complex(
                    torch.randn(out_features) * 1e-3,
                    torch.randn(out_features) * 1e-3
                ))
            else:
                self.bias = None
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        if use_phase:
            self.phase = nn.Parameter(torch.zeros(1, out_features))  # φ

        self.init_weights()  # 现在可以安全调用

    def init_weights(self):
        with torch.no_grad():
            if not self.use_complex:
                if self.is_first:
                    self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
                else:
                    self.linear.weight.uniform_(
                        -np.sqrt(6 / self.in_features) / self.omega_0,
                        np.sqrt(6 / self.in_features) / self.omega_0
                    )
                if self.linear.bias is not None:
                    self.linear.bias.zero_()

    def forward(self, x):
        if self.use_complex:
            # x: real input -> cast to complex
            x_c = torch.complex(x, torch.zeros_like(x))
            output = torch.matmul(x_c, self.weight.t())
            if self.bias is not None:
                output = output + self.bias
            # Extract magnitude and apply sine on phase
            mag = output.abs()
            phase = output.angle() + self.phase if self.use_phase else output.angle()
            return torch.sin(self.omega_0 * mag + phase).real
        else:
            linear_out = self.linear(x)
            if self.use_phase:
                return torch.sin(self.omega_0 * linear_out + self.phase)
            else:
                return torch.sin(self.omega_0 * linear_out)

    def forward_with_intermediate(self, x):
        if self.use_complex:
            x_c = torch.complex(x, torch.zeros_like(x))
            output = torch.matmul(x_c, self.weight.t()) + (self.bias if self.bias is not None else 0)
            mag = output.abs()
            phase = output.angle() + self.phase if self.use_phase else output.angle()
            sine_out = torch.sin(self.omega_0 * mag + phase).real
            return sine_out, mag.detach()
        else:
            linear_out = self.linear(x)
            if self.use_phase:
                activated = torch.sin(self.omega_0 * linear_out + self.phase)
            else:
                activated = torch.sin(self.omega_0 * linear_out)
            return activated, linear_out.detach()


# --- 2. Adaptive Omega Scheduler (Quantum Energy Level Transition) ---
class AdaptiveOmegaScheduler:
    """
    Adjusts omega_0 dynamically based on loss curvature (like energy level jumps).
    When loss decreases rapidly → reduce omega (lower frequency → stabilize).
    When stuck → increase omega (higher frequency → explore).
    """
    def __init__(self, layers, base_omega=30.0, min_omega=5.0, max_omega=60.0, patience=50):
        self.layers = layers
        self.base_omega = base_omega
        self.min_omega = min_omega
        self.max_omega = max_omega
        self.patience = patience
        self.loss_history = []
        self.step_count = 0

    def step(self, loss):
        self.loss_history.append(loss.item())
        self.step_count += 1

        if len(self.loss_history) < self.patience + 1:
            return

        # Compute recent loss trend
        recent = self.loss_history[-self.patience:]
        older = self.loss_history[-2*self.patience:-self.patience]

        recent_avg = np.mean(recent)
        older_avg = np.mean(older)

        # If not improving: increase omega (excite the system)
        if recent_avg >= older_avg:
            new_omega = min(self.base_omega * 1.2, self.max_omega)
        else:
            # If improving: decrease omega (relax to ground state)
            new_omega = max(self.base_omega * 0.9, self.min_omega)

        # Apply to all hidden layers
        for layer in self.layers:
            if hasattr(layer, 'omega_0'):
                layer.omega_0 = new_omega


# # --- 3. Quantum Tunneling Noise (Helps Escape Local Minima) ---
# def add_quantum_noise(model, coords, noise_scale=1e-3, prob=0.1):
#     """
#     Inject small random phase shift (tunneling effect) with low probability.
#     """
#     if torch.rand(1).item() < prob:
#         noise = noise_scale * torch.randn_like(coords)
#         return coords + noise
#     return coords


# --- 4. Final Quantum-Inspired SIREN ---
class QiSiren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=False, first_omega_0=30.0, hidden_omega_0=30.0,
                 use_phase=True, use_complex=False):
        super().__init__()

        self.net = nn.ModuleList()
        self.net.append(
            QuantumSineLayer(in_features, hidden_features, is_first=True,
                             omega_0=first_omega_0, use_phase=use_phase, use_complex=use_complex)
        )

        for _ in range(hidden_layers):
            self.net.append(
                QuantumSineLayer(hidden_features, hidden_features, is_first=False,
                                 omega_0=hidden_omega_0, use_phase=use_phase, use_complex=use_complex)
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0
                )
            self.net.append(final_linear)
        else:
            self.net.append(
                QuantumSineLayer(hidden_features, out_features, is_first=False,
                                 omega_0=hidden_omega_0, use_phase=use_phase, use_complex=use_complex)
            )

    def forward(self, coords):
        # x = add_quantum_noise(self, coords.requires_grad_(True))
        x = coords
        for layer in self.net:
            x = layer(x)
        return x, x.detach().requires_grad_(True)

    def get_hidden_layers(self):
        return [layer for layer in self.net if isinstance(layer, QuantumSineLayer)]