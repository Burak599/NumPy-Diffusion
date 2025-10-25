from Scratch_Diffusion import (
    cosineScheduler, time_embedding, forward_diffusion,
    forward_diffusion_for_backward, diffusion_backward,
    reverseDiffusion, AdamW
)
import numpy as np

def test_cosineScheduler():
    a, beta, alpha = cosineScheduler(10)
    assert len(a) == 11 and len(beta) == 10 and len(alpha) == 10

def test_time_embedding():
    emb = time_embedding(5, 8)
    assert emb.shape == (8,)

def test_forward_diffusion():
    x0 = np.ones((3,3))
    _, beta, _ = cosineScheduler(10)
    x_forward = forward_diffusion(x0, beta)
    assert x_forward.shape == (len(beta)+1, 3, 3)
    return x_forward

def test_forward_and_backward():
    x_forward = test_forward_diffusion()
    x_t = x_forward[1]
    hidden_size = 8
    a_len = x_t.flatten().shape[0] + hidden_size
    w0 = np.random.randn(a_len, 16) * 0.1
    w1 = np.random.randn(16, 9) * 0.1
    weights = [w0, w1]
    biases = [np.zeros(16), np.zeros(9)]
    activations = ['relu', 'linear']

    eps_hat, model_cache = forward_diffusion_for_backward(x_t, 1, weights, biases, activations, hidden_size)
    assert eps_hat.ndim == 1
    # dummy epsilon
    eps = np.random.randn(*x_t.flatten().shape)
    grads_W, grads_b = diffusion_backward(x_t.flatten(), eps, eps_hat, model_cache, weights, biases, activations)
    for g, w in zip(grads_W, weights):
        assert g.shape == w.shape

    # AdamW update
    w_new, m_new, u_new = AdamW(weights[0], grads_W[0], np.zeros_like(weights[0]), np.zeros_like(weights[0]), t_step=1, n=1e-3)
    assert w_new.shape == weights[0].shape

    # reverse diffusion
    _, beta, alpha = cosineScheduler(10)
    alpha_bar, _, _ = cosineScheduler(10)
    x_T = x_forward[-1]
    x_rev = reverseDiffusion(x_T, beta, alpha, alpha_bar, weights, biases, activations, hidden_size)
    assert x_rev.shape[1:] == x_T.shape

if __name__ == "__main__":
    test_cosineScheduler()
    test_time_embedding()
    test_forward_and_backward()
    print("All functions verified")