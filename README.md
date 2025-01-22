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
from JumpGP_code_py import JumpGP_LD

# Create an instance of the model
model = JumpGP_LD()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Figures 3, 4, 5

Figures 3, 4, and 5 illustrate the model's performance on different datasets. You can find the code to generate these figures in the `figures` folder. Running the corresponding scripts will produce visual results to help you understand the model's effectiveness.

## Important Files

- `JumpGP_code_py/`: Contains the code implementing the Jump Gaussian Process model.
- `figures/`: Contains the code for generating the figures presented in the papers.
- `requirements.txt`: Lists all the dependencies required for the project.

## Contributing

Contributions of any kind are welcome! Please submit issues or pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

