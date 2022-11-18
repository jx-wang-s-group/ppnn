# PPNN
PDE Preserved Neural Network

Manuscript on arXiv: [Predicting parametric spatiotemporal dynamics by
multi-resolution PDE structure-preserved deep learning](https://arxiv.org/pdf/2205.03990.pdf)

## Abstract
Pure data-driven deep learning models suffer from high training costs, error accumulation, and poor generalizability when predicting complex physical processes. A more promising way is to leverage our prior physics knowledge in scientific deep learning models, known as physicsinformed deep learning (PiDL). In most PiDL frameworks, the physics prior is utilized to regularize neural network training by incorporating governing equations into the loss function. The resulting physical constraint, imposed in a soft manner, relies heavily on a proper setting of hyperparameters that weigh each loss term. To this end, we propose a new direction to leverage physics prior knowledge by “baking” the mathematical structure of governing equations into the neural network architecture, namely PDE-preserved neural network (PPNN). The discretized
PDE is preserved in PPNN as convolutional residual networks formulated in a multi-resolution setting. This physics-inspired learning architecture endows PPNN with excellent generalizability and long-term prediction accuracy compared to the state-of-the-art black-box baselines. The effectiveness and merit of the proposed methods have been demonstrated over a handful of spatiotemporal dynamical systems governed by spatiotemporal PDEs, including reaction-diffusion, Burgers’, and Navier-Stokes equations

## Structure

<!-- ![structure](docs/demo/PDE_preserved_schematic2.png) -->
<p align="center"><img src="docs/demo/PDE_preserved_schematic2.png" alt="structure" align="center" width="600px"></p>

## Demo case
* Navier-Stokes equation 

 ![Navier-Stokes equation](docs/demo/ns.gif)