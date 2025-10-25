# Example: Using Scratch_Diffusion on a single 3x3 input
# This example uses only one 3x3 input. It shows how the functions work, but the model cannot learn true from a single example.

from Scratch_Diffusion import (
    cosineScheduler,
    time_embedding,
    forward_diffusion,
    forward_diffusion_for_backward,
    reverseDiffusion
)
import numpy as np

# Setup
T = 10
hidden_size = 8
x0 = np.ones((3, 3))  # dummy input
a, beta, alpha = cosineScheduler(T)

# Forward diffusion
x_forward = forward_diffusion(x0, beta)
print("Forward diffusion result (x_forward[-1]):\n", x_forward[-1])

# Prepare MLP weights for demonstration
input_size = x0.size + hidden_size
weights = [np.random.randn(input_size, 16)*0.1, np.random.randn(16, 9)*0.1]
biases = [np.zeros(16), np.zeros(9)]
activations = ['relu', 'linear']

# Forward for backward (get predicted noise)
x_t = x_forward[1]
epsilon_hat, _ = forward_diffusion_for_backward(x_t, t=1, weights=weights, biases=biases, activations=activations, hidden_size=hidden_size)
print("Predicted noise (epsilon_hat):\n", epsilon_hat)

# Reverse diffusion demo
x_rev = reverseDiffusion(x_forward[-1], beta, alpha, alpha, weights, biases, activations, hidden_size)
print("Reverse diffusion result (x_rev[-1]):\n", x_rev[-1])

