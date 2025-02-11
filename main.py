import pathlib as pl
import os

import click
from click_params import DecimalRange

import torch

from src.logger import logger
from src.task import TrainTask, TestTask
from src.model import ModelSelector
from src.optimization import OptimizationSelector
from src.data import DatasetSelector

PATCH_PATH = pl.Path(os.path.abspath("")) / pl.Path('data') / pl.Path('ndb_ufes') / pl.Path('patches')
IMAGE_PATH = pl.Path(os.path.abspath("")) / pl.Path('data') / pl.Path('ndb_ufes') / pl.Path('images') 
METADATA_PATH = pl.Path(os.path.abspath("")) / pl.Path('data') / pl.Path('ndb_ufes') / pl.Path('ndb-ufes.csv') 

@click.command()
@click.option("--train", is_flag=True, help="Train flag. When used along with test prevails over test flag", default=False)
@click.option("--test", is_flag=True, help="Test flag. When used along with train, train prevails.", default=False)
@click.option("--optimizer_name", default="sgd", help="Optimizer name", type=click.Choice(["adam", "sgd","adabelief"]), show_default=True)
@click.option("--learning_rate", default=0.001, help="Learning rate", type=float, show_default=True)
@click.option("--scheduler_name", default="reduce_lr_on_plateau", help="Scheduler name", type=click.Choice(["step_lr", "reduce_lr_on_plateau"]), show_default=True)
@click.option("--step_size", default=30, help="Step size for StepLR scheduler", type=int, show_default=True)
@click.option("--gamma", default=0.1, help="Gamma for StepLR scheduler", type=float, show_default=True)
@click.option("--mode", default="min", help="Mode for    scheduler", type=str, show_default=True)
@click.option("--factor", default=0.1, help="Factor for ReduceLROnPlateau scheduler", type=float, show_default=True)
@click.option("--patience", default=10, help="Patience for ReduceLROnPlateau scheduler", type=int, show_default=True)
@click.option("--min_lr", default=10e-6, help="Min lr for ReduceLROnPlateau scheduler", type=float, show_default=True)
@click.option("--loss_name", default="cross_entropy", help="Loss name", type=click.Choice(["cross_entropy"]), show_default=True)
@click.option("--use_weights_loss",is_flag=True, help="Use weights for loss", default=False, show_default=True)
@click.option("--dataset_name", default="cwru", help="Dataset name", type=click.Choice(["cwru"]), show_default=True) # MUST CHANGE
@click.option("--dataset_path", default=PATCH_PATH, help="Dataset path", type=str, show_default=True)
@click.option("--train_size", default=0.8, help="Train size", show_default=True, type=DecimalRange(0, 1))
@click.option("--k_folds", default=5, help="K folds", type=int, show_default=True)
@click.option("--model_name", default="densenet121", help="Model name", type=click.Choice(["densenet121"]), show_default=True)
@click.option("--model_weights_path", default='default', help="Model weights path", type=str, show_default=True)
@click.option("--epochs", default=200, help="Number of epochs", type=int, show_default=True)
@click.option("--batch_size", default=32, help="Batch size", type=int, show_default=True)
@click.option("--save_path", default="run_results", help="Save path", type=str, show_default=True)
@click.option("--device", default="auto", help="Device", type=str, show_default=True)
@click.option("--project_name", default="image-signal", help="Experiment Project Name", type=str, show_default=True)
@click.option("--run_name", default="image-signal", help="Experiment Run Name", type=str, show_default=True)
@click.option("--wandb_offline", is_flag=True, help="Wandb offline", default=False)
@click.option("--partial_fine_tunning", is_flag=True, help="Partial Fine Tunning", default=False)
def main(train, test, optimizer_name, learning_rate, scheduler_name, step_size, gamma, mode, factor, patience, min_lr, loss_name, \
         use_weights_loss, dataset_name, dataset_path, train_size, k_folds, model_name, model_weights_path, epochs,  \
         batch_size, save_path, device, project_name, run_name, wandb_offline,partial_fine_tunning):    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device

    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    # convert to Path
    dataset_path = pl.Path(dataset_path)
    save_path = pl.Path(save_path)
    model_weights_path = pl.Path(model_weights_path) if model_weights_path.lower() != "default" else model_weights_path

    train_size = float(train_size)
    dataset_selector = DatasetSelector(dataset_name, dataset_path, train_size=train_size, k_folds=k_folds)
    dataset = dataset_selector.get_dataset()
    num_classes = len(dataset.labels_names)

    # initialize model
    model_selector = ModelSelector(model_name, model_weights_path, num_classes=num_classes, device=device, freeze_conv=partial_fine_tunning)

    if train:
        logger.info(f"Training model {model_name} with dataset {dataset_name}")
        optimization_selector = OptimizationSelector(optimizer_name, scheduler_name, learning_rate, step_size=step_size, gamma=gamma, mode=mode, factor=factor, patience=patience, min_lr=min_lr)
        train_task = TrainTask(model_selector, optimization_selector, loss_name, use_weights_loss, dataset, epochs, batch_size, k_folds, save_path, device, project_name, run_name)
        train_task.run()
    elif test:
        logger.info(f"Testing model {model_name} with dataset {dataset_name}")
        test_task = TestTask(model_selector, loss_name, use_weights_loss, dataset, batch_size, k_folds, save_path, device, project_name, run_name)
        test_task.run()
    else:
        raise ValueError("You must use --train or --test flag")
    
if __name__ == "__main__":
    main()