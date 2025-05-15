# PPNN
## PDE Preserved Neural Network

Published on Communications Physics: [Multi-resolution partial differential equations preserved learning framework for spatiotemporal dynamics](https://www.nature.com/articles/s42005-024-01521-z) | [arxiv version](https://arxiv.org/pdf/2205.03990.pdf)
<details>
<summary>Abstract</summary>

Traditional data-driven deep learning models often struggle with high training costs, error accumulation, and poor generalizability in complex physical processes. Physics-informed deep learning (PiDL) addresses these challenges by incorporating physical principles into the model. Most PiDL approaches regularize training by embedding governing equations into the loss function, yet this depends heavily on extensive hyperparameter tuning to weigh each loss term. To this end, we propose to leverage physics prior knowledge by “baking” the discretized governing equations into the neural network architecture via the connection between the partial differential
equations (PDE) operators and network structures, resulting in a PDE-preserved neural network (PPNN). This method, embedding discretized PDEs through convolutional residual networks in a multi-resolution setting, largely improves the generalizability and long-term prediction accuracy, outperforming conventional black-box models. The effectiveness and merit of the proposed methods have been demonstrated across various spatiotemporal dynamical systems governed by spatiotemporal PDEs, including reaction-diffusion, Burgers’, and Navier-Stokes equations.
</details>

<p align="center"><img src="docs/demo/PDE_preserved_schematic.png" alt="structure" align="center" width="600px"></p>

* results
    * Navier-Stokes equation 
<p align="center"><img src="docs/demo/ns.gif" alt="structure" align="center" width="600px"></p>



## Code
* requirements
   ```bash
   pytorch
   numpy
   matplotlib
   tensorboard

   deepxde # required by DeepONet only
   ```
* **data generation**
  
   To generate training and testing set please refer to the code in `src/operators.py`.
  
   The reference data used in the figures shown in the paper can be downloaded at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15397548.svg)](https://doi.org/10.5281/zenodo.15397548)
* training
   ```bash
   python src/train2D.py cases/CASE_NAME.yaml
   ```
##
* `src`: PPNN source codes
    * `opertors.py`: numerical operators, is used to generate dataset. Also works as the PDE-preserving part of PPNN
    * `rhs.py`: define various right hand side of PDEs
    * `train2D.py`: the main training script for RD and burgers case. It requires config files. Examples of config files are listed in the folder `cases`
    * `models.py`: deep learning neural networks

* `case`: contains yaml files that list configurations for different cases. 

* `Bv`: contains source code for parameterizing different boundary conditions, as disscussed in the first section in the supplementary informantion.

* `baselines`: source code for the baseline methods including FNO, PINN and DeepONet
* 
## Problems
If you find any bugs in the code or have trouble in running PPNN, you are very welcome to [create an issue](https://github.com/jx-wang-s-group/ppnn/issues) in this repository.

## Citation
If you find our work relevant to your research, please cite:
```
@article{liu2024multi,
  title={Multi-resolution partial differential equations preserved learning framework for spatiotemporal dynamics},
  author={Liu, Xin-Yang and Zhu, Min and Lu, Lu and Sun, Hao and Wang, Jian-Xun},
  journal={Communications Physics},
  volume={7},
  number={1},
  pages={31},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

