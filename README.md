# pybo
Bayesian optimization in python.

Currently in the `main.py` file, the cost function is the negative validation loss (cross entropy) of a ResNet-10 model trained on the KMNIST dataset with SGD.
The search space is the exponent of the SGD learning rate with base 10, e.g. `--lr_exp_min -4` and `--lr_exp_max -1` gives learning rates in the interval [1e-4, 1e-1].

To start, simply execute `python main.py --device "cuda"` (assuming you have cuda available in pytorch).
Default are 5 epochs for the model training, which makes `main.py` take ~8 minutes on a RTX2070 Super (set lower if you have a slow GPU).
To see the CLI arguments and their descriptions, run `python main.py -h`.

Depending on the cost function values, the plots (default path `./results/`) may get really ugly.
The lower and upper bound of the search space are currently set to make the plots look ok in earlier iterations.

Tested setup:
Ubuntu 18.04
Python 3.8.5
Pytorch 1.6.0
Sklearn 0.24.0
Numpy 1.19.1
Scipy 1.5.2
Torchvision 0.7.0
Pandas 1.1.3
Seaborn 0.11.1
