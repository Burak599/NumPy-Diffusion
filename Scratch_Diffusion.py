import numpy as np

# Computes alpha and beta schedules for DDPM using a cosine schedule
def cosineScheduler(T, s=0.008):
    a = np.zeros(T+1)
    beta = np.zeros(T)

    for t in range(T+1):
        a[t] = np.cos(((t/T + s)/(1+s)) * (np.pi/2)) ** 2

    for t in range(T):
        beta[t] = 1 - a[t+1]/a[t]

    alpha = a[1:] / a[:-1]
    return a, beta, alpha

# Converts timestep t into a high-dimensional embedding for the MLP
def time_embedding(t, embedding_size):
    indices = np.arange(embedding_size) / 2
    div_term = 10000**(indices / embedding_size)
    t_vector = np.array([t]).reshape(-1, 1) 
    
    emb = np.empty(embedding_size)
    emb[0::2] = np.sin(t_vector / div_term)[0, 0::2]
    emb[1::2] = np.cos(t_vector / div_term)[0, 1::2]
    return emb

# Adds noise to the input x0 over T timesteps to simulate forward diffusion
def forward_diffusion(x0, beta):
    T = len(beta)
    x = np.zeros((T+1,) + x0.shape)
    x[0] = x0
    for t in range(1, T+1):
        noise = np.random.normal(0, 1, x0.shape)
        x[t] = np.sqrt(1 - beta[t-1]) * x[t-1] + np.sqrt(beta[t-1]) * noise
    return x

# Prepares input for MLP, returns predicted noise (epsilon_hat) and intermediate activations
def forward_diffusion_for_backward(x_t, t, weights, biases, activations, hidden_size):
    t_emb = time_embedding(t, hidden_size)
    h = np.concatenate([x_t.flatten(), t_emb]) # Combine input and time embedding
    model_cache=[]
    a = h
    for i in range(len(weights)):
        z = a @ weights[i] + biases[i]
        model_cache.append(a)

        # Apply activation functions
        if activations[i] == 'relu':
            a = np.maximum(0, z)
        elif activations[i] == 'tanh':
            a = np.tanh(z)
        else:
            a = z

    epsilon_hat = a # predicted noise
    return epsilon_hat, model_cache

T = 10
alpha_bar, beta, alpha = cosineScheduler(T)
x0 = np.ones((3, 3))
x_forward = forward_diffusion(x0, beta)
print("x_forward shape:", x_forward.shape)

# Computes gradients of weights and biases via backpropagation using MSE loss
def diffusion_backward(x_t, epsilon, epsilon_hat, model_cache, weights, biases, activations):
    batch_size = 1

    # Derivative of loss with respect to epsilon_hat
    dl_depsilon_hat = (2 * (epsilon_hat.flatten() - epsilon.flatten()) / batch_size).reshape(1, -1)

    grads_W = [] # weight gradients
    grads_b = [] # bias gradients
    delta = dl_depsilon_hat.copy() 

    for i in reversed(range(len(weights))):
        a_prev = model_cache[i].reshape(1, -1) 
        W = weights[i]
        b = biases[i]

        dW = a_prev.T @ delta  # Gradient of loss w.r.t weight
        db = np.sum(delta, axis=0) 

        grads_W.insert(0, dW)
        grads_b.insert(0, db)

        if i != 0:
            da_prev = delta @ W.T

            # Derivative of activations
            if activations[i-1] == "relu":
                dz = da_prev * (a_prev > 0)
            elif activations[i-1] == "tanh":
                dz = da_prev * (1 - a_prev**2)
            else:
                dz = da_prev

            delta = dz

    return grads_W, grads_b

# Performs AdamW update: combines Adam optimization with weight decay
def AdamW(weights, dl_dw, m_prev, u_prev, t_step, n, Y=0.01, b1 = 0.9, b2 = 0.999, eps=1e-6):
    weight_decay = Y * weights
    m_new = b1 * m_prev + (1-b1) * dl_dw
    u_new = b2 * u_prev + (1-b2) * (dl_dw ** 2)
    m_hat = m_new/ (1- (b1 ** t_step))
    u_hat = u_new/ (1- (b2 ** t_step))
    adam_update = n * (m_hat/ (np.sqrt(u_hat) + eps))
    weights_new =  weights- n * (adam_update + weight_decay)

    return weights_new, m_new, u_new

# MLP setup
input_size = 3*3 +8
hidden_sizes = [64,64,9] # neurons in hidden layers
activations = ["relu", "relu", "linear"]

# Initialize weights, biases, and AdamW moments
weights=[np.random.randn(input_size, hidden_sizes[0])*0.1,
         np.random.randn(hidden_sizes[0], hidden_sizes[1])* 0.1,
         np.random.randn(hidden_sizes[1], hidden_sizes[2])* 0.1]

biases = [np.zeros(h) for h in hidden_sizes]
m_prev = [np.zeros_like(w) for w in weights]
u_prev = [np.zeros_like(w) for w in weights]

T = 10
alpha_bar, beta, alpha = cosineScheduler(T)
x0 = np.ones((3,3)) # example input
n_steps = 10000
lr = 0.005

# Training loop
for step in range(1, n_steps+1):
    t = np.random.randint(1, T+1)
    x_forward = forward_diffusion(x0, beta)
    x_t = x_forward[t]

    epsilon = (x_t - np.sqrt(alpha_bar[t-1]) * x0) / np.sqrt(1 - alpha_bar[t-1])
    epsilon_hat, model_cache = forward_diffusion_for_backward(x_t, t, weights, biases, activations, hidden_size=8)

    grads_w, grads_b = diffusion_backward(x_t.flatten(), epsilon.flatten(), epsilon_hat, model_cache, weights, biases, activations)

    for i in range(len(weights)):
        weights[i], m_prev[i], u_prev[i] = AdamW(weights[i], grads_w[i], m_prev[i], u_prev[i], step, lr)
        biases[i], _, _ = AdamW(biases[i], grads_b[i], np.zeros_like(grads_b[i]), np.zeros_like(grads_b[i]), step, lr)

    # Print MSE loss every 500 steps
    if step % 500 == 0:
        loss = np.mean((epsilon_hat - epsilon.flatten()) ** 2)
        print(f"Step {step}, MSE Loss {loss:.4f}")

# Reconstruct x0 from x_T using predicted noise
def reverseDiffusion(x_T, beta, alpha, alpha_bar, weights, biases, activations, hidden_size):
    T = len(beta)
    x = np.zeros((T+1,) + x_T.shape)
    x[T] = x_T

    for t in range(T-1, -1, -1):
        z = np.random.normal(0, 1, x_T.shape)
        stochastic = z * np.sqrt(beta[t])
        eps_hat, _ = forward_diffusion_for_backward(x[t+1], t+1, weights, biases, activations, hidden_size)
        eps_hat = eps_hat.reshape(x_T.shape)
        x[t] = (1/np.sqrt(alpha[t])) * (x[t+1] - beta[t] / np.sqrt(1-alpha_bar[t]) * eps_hat) + stochastic

    return x

# Test reconstruction
x_T = x_forward[-1]
x_reverse = reverseDiffusion(
    x_T, beta, alpha, alpha_bar, 
    weights=weights, biases=biases, activations=activations, hidden_size=8
)

print("x0 (original):\n", x0)
print("x_T (noisy last step):\n", x_T)
print("x_reverse[-1] (after reverse diffusion):\n", x_reverse[-1])
    

