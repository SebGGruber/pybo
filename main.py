import numpy as np
import torch
from os import path
from sklearn.gaussian_process import GaussianProcessRegressor
from src.bayesian_optimization import bayesian_optimization
from src.models import resnet10, train
from src.datasets import KMNIST
from src.acquisition_functions import EI


def main(args):
    """
    """

    np.random.seed(args.seed)
    search_space = np.geomspace(args.lr_min, args.lr_max, args.lr_resolution)

    def cost_function(lr):
        model = resnet10().to(device=args.device)
        dataset = KMNIST(args.batch_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=float(lr))
        val_loss = train(model, optimizer, dataset, args.epochs, args.device)

        return val_loss

    lambda_hat = bayesian_optimization(
        search_space,
        cost_function,
        acquisition_function=EI,
        predictive_model=GaussianProcessRegressor(),
        max_evaluations=args.max_evaluations
    )

    torch.save(lambda_hat, path.join(args.save_directory, "lambda_hat.pt"))


if __name__ == "__main__":

    import argparse
    # argument parsing and algorithm execution

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dataset", type=str, default="KMNIST")
    parser.add_argument(
        "--save_directory",
        type=str,
        default=".",
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
        default=10,
        help="amount of epochs to train the resnet model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size for pytorch model training"
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-5,
        help="lower bound of lr search space"
    )
    parser.add_argument(
        "--lr_max",
        type=float,
        default=1e-0,
        help="upper bound of lr search space"
    )
    parser.add_argument(
        "--lr_resolution",
        type=int,
        default=1000,
        help="grid size of the lr search space"
    )
    args = parser.parse_args()

    print("Running bayesian optimization...")
    main(args)
    print("Done.")
