Diffusion-based Depth and Optical Flow
======================================

Torchvision provides an experimental module for estimating depth and optical
flow with an iterative diffusion-style process. The implementation is inspired
by recent research on applying denoising diffusion models to geometric tasks.
It refines random depth and flow predictions conditioned on the input image.

Example usage::

    import torch
    from torchvision.models import DiffusionDepthFlowEstimator

    model = DiffusionDepthFlowEstimator(num_steps=3)
    img = torch.randn(1, 3, 64, 64)
    depth, flow = model(img)

The code is intentionally lightweight and is **not** a replacement for full
state-of-the-art implementations described in the latest literature. It serves
as a simple demonstration of the technique so users can experiment with
iterative refinement approaches.
