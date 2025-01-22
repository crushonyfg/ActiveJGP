# Jump Gaussian Process Model for Estimating Piecewise Continuous Regression Functions

This project implements the algorithms and models proposed in the papers "Jump Gaussian Process Model for Estimating Piecewise Continuous Regression Functions" and "Active Learning of Piecewise Gaussian Process Surrogates."

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [JumpGP_code_py.JumpGP_LD](#jumpgp_code_pyjumpgp_ld)
  - [Figures 3, 4, 5](#figures-3-4-5)
- [Important Files](#important-files)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to estimate piecewise continuous regression functions using a Jump Gaussian Process model. The model effectively handles regression problems with discontinuities and optimizes the learning process through active learning methods.

## Installation

Please ensure you have Python 3.x and the required dependencies installed. You can install the necessary libraries using the following command:

```python
pip install -r requirements.txt
```

## Usage

### JumpGP_code_py.JumpGP_LD

`JumpGP_code_py.JumpGP_LD` is the main class implementing the Jump Gaussian Process model. You can use it as follows:

```python
from JumpGP_code_py.JumpGP_LD import JumpGP_LD

mu_t, sig2_t, model, h = JumpGP_LD(x, y, xt, 'CEM', True)
```

### Figures 3, 4, 5, 6

Figures 3, 4, 5, and 6 illustrate the model's performance on different datasets. The code to reproduce these figures can be found in the `Figure_3_4_5_6.py` script. Running this script will generate visual results that help you understand the model's effectiveness as presented in the paper "Jump Gaussian Process Model for Estimating Piecewise Continuous Regression Functions."

## Important Files

- `JumpGP_code_py/`: Contains the code implementing the Jump Gaussian Process model.
- `requirements.txt`: Lists all the dependencies required for the project.

## Contributing

Contributions of any kind are welcome! Please submit issues or pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

