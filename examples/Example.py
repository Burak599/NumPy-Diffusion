#!!!!!!!!!!!!!!
# # Note: This example uses only a single 3x3 input. The model runs correctly but does not actually learn from this single sample.
# The purpose is to verify that all functions work correctly.
#!!!!!!!!!!!!!!

from Scratch_Diffusion import cosineScheduler, diffusion_backward, reverseDiffusion, time_embedding, forward_diffusion, forward_diffusion_for_backward, AdamW
import numpy as np


def test_cosineScheduler():
    T = 10
    a, beta, alpha = cosineScheduler(T)
    assert len(a) == T+1
    assert len(beta) == T
    assert len(alpha) == T
    assert np.all(a >= 0) and np.all(a <= 1)
    print("cosineScheduler ok")

def test_time_embedding():
    emb = time_embedding(5, 8)
    assert emb.shape == (8,)
    print("time_embedding ok")

def test_forward_diffusion():
    x0 = np.ones((3,3))
    _, beta, _ = cosineScheduler(10)
    x_forward = forward_diffusion(x0, beta)
    assert x_forward.shape == (len(beta)+1, 3, 3)
    print("forward_diffusion ok")
    return x_forward

def test_forward_diffusion_for_backward():
    x0 = np.ones((3,3))
    T = 10
    _, beta, _ = cosineScheduler(T)
    x_forward = forward_diffusion(x0, beta)
    x_t = x_forward[1]

    hidden_size = 8
    a_len = x_t.flatten().shape[0] + hidden_size
    # küçük MLP için ağırlıklar
    w0 = np.random.randn(a_len, 16) * 0.1
    w1 = np.random.randn(16, 9) * 0.1
    weights = [w0, w1]
    biases = [np.zeros(16), np.zeros(9)]
    activations = ['relu', 'linear']

    eps_hat, model_cache = forward_diffusion_for_backward(x_t, 1, weights, biases, activations, hidden_size)
    assert eps_hat.ndim == 1
    assert len(model_cache) == len(weights)
    print("forward_diffusion_for_backward ok")
    return x_t, eps_hat, model_cache, weights, biases, activations

def test_diffusion_backward(x_t, eps, eps_hat, model_cache, weights, biases, activations):
    grads_W, grads_b = diffusion_backward(x_t.flatten(), eps.flatten(), eps_hat, model_cache, weights, biases, activations)
    assert len(grads_W) == len(weights)
    assert len(grads_b) == len(biases)
    for g, w in zip(grads_W, weights):
        assert g.shape == w.shape
    print("diffusion_backward ok")
    return grads_W, grads_b

def test_AdamW(weights, grads_W):
    w = weights[0].copy()
    dw = grads_W[0]
    m_prev = np.zeros_like(w)
    u_prev = np.zeros_like(w)
    w_new, m_new, u_new = AdamW(w, dw, m_prev, u_prev, t_step=1, n=1e-3)
    assert w_new.shape == w.shape
    assert m_new.shape == m_prev.shape
    assert u_new.shape == u_prev.shape
    print("AdamW ok")

def test_reverseDiffusion(x_forward, weights, biases, activations):
    _, beta, alpha = cosineScheduler(10)
    alpha_bar, _, _ = cosineScheduler(10)  # note: main.cosineScheduler returns (a,beta,alpha) in user's code; using same for alpha_bar
    x_T = x_forward[-1]
    x_rev = reverseDiffusion(x_T, beta, alpha, alpha_bar, weights, biases, activations, hidden_size=8)
    assert x_rev.shape[1:] == x_T.shape
    print("reverseDiffusion ok")

if __name__ == "__main__":
    test_cosineScheduler()
    test_time_embedding()
    x_forward = test_forward_diffusion()
    x_t, eps_hat, model_cache, weights, biases, activations = test_forward_diffusion_for_backward()
    # create a dummy 'epsilon' consistent with shapes
    eps = np.random.randn(*x_t.flatten().shape)
    grads_W, grads_b = test_diffusion_backward(x_t, eps, eps_hat, model_cache, weights, biases, activations)
    test_AdamW(weights, grads_W)
    test_reverseDiffusion(x_forward, weights, biases, activations)

    print("ALL TESTS PASSED")
