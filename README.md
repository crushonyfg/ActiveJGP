# Active Learning of Piecewise Gaussian Process Surrogates

This project implements the algorithms and models proposed in the paper:

> **Park et al. (2023)**  
> *Active Learning of Piecewise Gaussian Process Surrogates*  
> [arXiv:2301.08789](https://arxiv.org/abs/2301.08789)

## Installation

```bash
git clone https://github.com/crushonyfg/ActiveJGP.git
cd ActiveJGP
git clone https://github.com/crushonyfg/JumpGaussianProcess.git
pip install -r requirements.txt
```

## Results

![RMSE Comparison](test_rmse.png)

This figure shows a sample RMSE result of active learning using different methods. You can reproduce it using:

```bash
python run_main_simulation.py
```

The script uses synthetic data for benchmarking.

## Usage

```python
from ActiveJGP import ActiveJGP

xt_next, criteria, bias2_changes, var_changes, pred, pred_xt, pred_var, pred_bias = ActiveJGP(
    x, y, xc, mode, logtheta
)
```

### Arguments

- `x`: numpy array of shape `(N, D)` — Training inputs  
- `y`: numpy array of shape `(N, 1)` — Training outputs  
- `xc`: numpy array of shape `(M, D)` — Candidate test inputs  
- `mode`: str — Acquisition mode, one of `['MIN_IMSPE', 'MAX_MSPE', 'MAX_VAR', 'MIN_ALC']`  
- `logtheta`: (optional) Hyperparameter for GP kernel (default: `None`)

## Contributing

Contributions of any kind are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Citation

If you use this code, please cite the following paper:

```bibtex
@article{park2023active,
  title={Active learning of piecewise Gaussian process surrogates},
  author={Park, Chiwoo and Waelder, Robert and Kang, Bonggwon and Maruyama, Benji and Hong, Soondo and Gramacy, Robert},
  journal={arXiv preprint arXiv:2301.08789},
  year={2023}
}
```
