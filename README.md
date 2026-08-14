# 🎭 CelebA-VAE-Latent-Manipulator

[![Architecture](https://img.shields.io/badge/Architecture-Conv--VAE-blue)](#)
[![Dataset](https://img.shields.io/badge/Dataset-CelebA-orange)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
[![Loss](https://img.shields.io/badge/Loss-MSE%20%2B%20KL%20Divergence-purple)](#)
[![Reconstruction](https://img.shields.io/badge/MSE-0.036%20(%2B0.005%20imprv)-brightgreen)](#)

A deep generative **Convolutional Variational Autoencoder (Conv-VAE)** trained on the **CelebA** facial dataset. This project explores latent space geometry, vector arithmetic manipulation (e.g., adding smile, glasses, or age vectors), and loss-balancing tradeoffs between distribution regularization ($D_{KL}$) and image reconstruction quality (MSE).

---

## ✨ Key Capabilities

* **Deep Convolutional Architecture:** Features strided 2D convolutions in the encoder for spatial downsampling and transposed convolutions in the decoder for high-fidelity image reconstruction.
* **Latent Space Vector Arithmetic:** Extracts attribute direction vectors by computing mean latent representations across target feature groups, enabling controllable facial edits (e.g., $z_{\text{smiling\_face}} = z_{\text{neutral}} + \alpha \cdot v_{\text{smile}}$).
* **$\beta$-VAE Loss Optimization:** Balanced the trade-off between the Reconstruction Loss (MSE) and Kullback-Leibler (KL) Divergence to prevent posterior collapse while maintaining a smooth, continuous latent space.
* **Improved Image Fidelity:** Reduced reconstruction **MSE from 0.041 to 0.036**, producing sharp reconstructions and smooth, natural latent interpolations between distinct faces.

---

## 📊 Performance & Optimization

Evaluated on the held-out CelebA validation subset:

| Configuration | Reconstruction Loss (MSE) | Latent Smoothness / Disentanglement |
| :--- | :---: | :---: |
| **Standard Baseline VAE** | **0.041** | Blurry / Discontinuous Latent Space |
| **Conv-VAE + Optimized $\beta$ Schedule (Ours)** | **0.036 (-0.005)** | **Continuous / Linear Vector Operations** |

---

## 🧪 Latent Space Manipulation Flow
