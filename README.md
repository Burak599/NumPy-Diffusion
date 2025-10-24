# Scratch_Diffusion

A complete, from-scratch implementation of a diffusion model using only NumPy.

## Overview
This project demonstrates the core mechanics of diffusion models:
- **Forward Diffusion**: Gradually adds Gaussian noise to an input image.
- **Reverse Diffusion**: Recovers the original image from the noisy version using a trained MLP.
- **Noise Prediction**: A small multi-layer perceptron predicts the noise (epsilon) at each timestep.
- **Time Embeddings**: Each timestep is embedded into a vector for the network to condition on time.
- **Cosine Noise Scheduler**: Implements a cosine schedule for beta values controlling noise magnitude.

## Features
- Fully implemented in NumPy (no deep learning frameworks required)
- Adjustable number of timesteps and hidden layer sizes
- Demonstrates the full training loop using a single example input
- AdamW optimizer implemented from scratch

## Usage
1. Set the input image `x0` (currently a 3x3 example image).
2. Adjust hyperparameters like `T` (timesteps), `hidden_sizes`, `n_steps`, and `lr`.
3. Run the script to see:
   - Forward diffusion adding noise
   - Network training to predict noise
   - Reverse diffusion recovering the original input
4. Monitor MSE loss printed every few steps.

## Learning Goals
- Understand diffusion model mechanics
- Learn forward and reverse processes
- Learn to implement training of MLP for noise prediction
- Explore custom optimizers and time embeddings

## Notes
- Currently trains on a single example; adding more images is required for generalization.
- Designed for educational purposes to visualize the full process.

> ⚠️ Note: This project is implemented from scratch for educational purposes. 
> It is not optimized for production use and does not scale to large datasets or real-world images.
