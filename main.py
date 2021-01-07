import numpy as np
import os
import torch
from os import path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from src.acquisition_functions import EI
from src.bayesian_optimization import bayesian_optimization
from src.datasets import KMNIST
from src.models import resnet10, train
from src.plotting import plot_bo


def main(args):
    """
    Main function applying the bayesian optimization framework on a hard-coded
    setting, here: ResNet-10 model optimized with SGD on the KMNIST dataset.
    This setting is specified in the form of a cost function.
    A plotting function does the logging during the hyperparameter tuning.
    No value will be returned, but the winning hyperparameter is stored in
    the given `save_directory` specified in the CLI.
    In general, all supported arguments are called in the CLI.

    Args:
        args (argparse.ArgumentParse):
            CLI argument parser, which includes the arguments 
            `save_directory`, `device`, `max_evaluations`, `epochs`,
            `batch_size`, `lr_exp_min`, `lr_exp_max`, and `lr_resolution`.
            For a detailed description of each, call "python main.py -h" in
            the command line.
    """

    # 1) INITIALIZATIONS / DEFINITIONS
    np.random.seed(args.seed)

    search_space = np.linspace(
        args.lr_exp_min, args.lr_exp_max, args.lr_resolution
    )

    def plot(*arguments):
        """Helper function used for logging in the bayesian opt. framework."""
        plot_bo(args.save_directory, *arguments)

    def cost_function(lr_exp):
        """
        Helper function used for calculating the cost value in each bayesian
        opt. iteration. Because the learning rate is bounded at 0, we search
        over its exponent with base 10.

        Args:
            lr_exp (float): Exponent of the learning rate of SGD (with base 10)

        Returns:
            float: Highest negative validation loss of all epochs. Used as the
                value to maximize over the given search space.
        """
        model = resnet10().to(device=args.device)
        dataset = KMNIST(args.batch_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=float(10**lr_exp))
        neg_val_loss = train(
            model, optimizer, dataset, args.epochs, args.device
        )

        return neg_val_loss

    # 2) RUN BAYESIAN OPTIMIZATION
    lambda_hat = bayesian_optimization(
        search_space,
        cost_function,
        acquisition_function=EI,
        predictive_model=GaussianProcessRegressor(),
        max_evaluations=args.max_evaluations,
        logging_function=plot
    )

    # 3) SAVE the winning lambda into the given directory
    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)

    torch.save(lambda_hat, path.join(args.save_directory, "lambda_hat.pt"))


if __name__ == "__main__":

    import argparse

    # From here on: CLI argument parsing and `main` function execution

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--save_directory",
        type=str,
        default="results/",
        help="directory where the results will be written to"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for pytorch model training"
    )
    parser.add_argument(
        "--max_evaluations",
        type=int,
        default=10,
        help="amount of evaluations of the bayesian optimization algorithm"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="amount of epochs to train the resnet model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size for pytorch model training"
    )
    parser.add_argument(
        "--lr_exp_min",
        type=float,
        default=-4,
        help="lower bound of lr exponent search space"
    )
    parser.add_argument(
        "--lr_exp_max",
        type=float,
        default=-1,
        help="upper bound of lr exponent search space"
    )
    parser.add_argument(
        "--lr_resolution",
        type=int,
        default=1000,
        help="grid size of the lr search space"
    )
    args = parser.parse_args()

    main(args)
    print("Done.")
