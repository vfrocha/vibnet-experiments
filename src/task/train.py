import os
import pathlib as pl
import copy

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from src.pipeline import train, test

from src.loss import LossSelector

from src.logger import logger

class TrainTask:
    def __init__(self, model_selector, optimizer_selector, loss_name, use_loss_weights, dataset, epochs, batch_size, k_folds, save_path, device, project_name="ia_health",run_name="Experiment"):
        self.model_selector = model_selector
        self.optimizer_selector = optimizer_selector
        self.loss_name = loss_name
        self.use_loss_weights = use_loss_weights
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.k_folds = k_folds
        self.save_path = save_path
        self.device = device
        self.project_name = project_name
        self.save_results_path = None
        self.run_name = run_name

        self._init_wandb()

    def _init_wandb(self):
        config_dict = dict(
            model=self.model_selector.model_name,
            optimizer=self.optimizer_selector.optimizer_name,
            scheduler=self.optimizer_selector.scheduler_name,
            criterion=self.loss_name,
            learning_rate=self.optimizer_selector.learning_rate,
        )

        config_dict["optimization_kwargs"] = self.optimizer_selector.kwargs

        wandb_kwargs = dict(
            project=self.project_name,
            name=self.run_name,
            config=config_dict
        )

        wandb.init(**wandb_kwargs) 
                        
    def _make_save_dir(self):
        count = 1
        while True:
            save_dir = f"{self.model_selector.model_name}_{self.optimizer_selector.optimizer_name}_{self.dataset.name}_{count}"
            save_dir_path = pl.Path(self.save_path) / pl.Path(save_dir)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
                logger.info(f"Created directory to {save_dir_path}")
                break
            count += 1
        
        self.save_results_path = save_dir_path

    def _make_fold_save_dir(self, fold):
        os.makedirs(self.save_results_path / f"fold_{fold}")
        return self.save_results_path / f"fold_{fold}"

    def run(self):
        model = self.model_selector.get_model()
        self.save_path = pl.Path(self.save_path) / pl.Path("train")

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            logger.info(f"Created root directory to save all results {self.save_path}")

        # make saving dir with format model_name_optimizer_name_dataset_day_month_year
        save_dir = f"{self.model_selector.model_name}_{self.optimizer_selector.optimizer_name}_{self.dataset.name}"
        save_dir_path = pl.Path(self.save_path) / pl.Path(save_dir)
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
            self.save_results_path = save_dir_path
            logger.info(f"Created directory to {save_dir_path}")
        elif os.path.exists(save_dir_path):
            logger.info(f"Directory {save_dir_path} already exists")
            self._make_save_dir()

        # save fold results
        folds_paths_df = self.dataset.folds_df
        folds_paths_df.to_csv(self.save_results_path / "folds_paths.csv", index=False)

        test_dataset = self.dataset.test_dataset
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        for fold in range(self.k_folds):
            logger.info(f"Training model for fold {fold}")
            logger.info('------------------------------------------------------------------------------')
            model = copy.deepcopy(self.model_selector.get_model())
            train_dataset, val_dataset = self.dataset.get_k_fold_train_val_tuple(fold)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            optimizer_selector = copy.deepcopy(self.optimizer_selector)
            optimizer = optimizer_selector.get_optimizer(model.parameters())
            scheduler = optimizer_selector.get_scheduler()

            if self.use_loss_weights:
                class_weights = compute_class_weight('balanced', classes=np.array(list(self.dataset.labels_names.keys())), y=train_dataset.labels)
                class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
                loss_selector = LossSelector(self.loss_name, class_weights)
            else:
                loss_selector = LossSelector(self.loss_name)
            
            loss = loss_selector.get_loss()
        
            fold_dir = self._make_fold_save_dir(fold)
            train_losses, train_accs, vals_losses, vals_accs = train(model, optimizer, scheduler, loss, train_loader, val_loader, fold_dir, self.epochs, self.device,fold)
            logger.info('------------------------------------------------------------------------------')
            
            # save fold results
            fold_results = {
                "train_losses": train_losses,
                "train_accs": train_accs,
                "val_losses": vals_losses,
                "val_accs": vals_accs
            }

            fold_results_df = pd.DataFrame(fold_results)
            fold_results_df.to_csv(fold_dir / "fold_results.csv", index=False)

            # test model
            PATH_MODEL = fold_dir / "best_checkpoint.pth"
            model.load_state_dict(torch.load(PATH_MODEL))
            
            test_loss, _, y_pred, y_true = test(model, loss, test_loader, self.device)

            recall_score_val = recall_score(y_true, y_pred, average='weighted')
            precision_score_val = precision_score(y_true, y_pred, average='weighted')
            f1_score_val = f1_score(y_true, y_pred, average='weighted')
            accuracy_score_val = accuracy_score(y_true, y_pred)

            logger.info(f"Test loss: {test_loss} for fold {fold}")
            logger.info(f"Recall score: {recall_score_val*100:0.4f}% for fold {fold}")
            logger.info(f"Precision score: {precision_score_val*100:0.4f}% for fold {fold}")
            logger.info(f"F1 score: {f1_score_val*100:0.4f}% for fold {fold}")
            logger.info(f"Accuracy score: {accuracy_score_val*100:0.4f}% for fold {fold}")

            wandb.log({f"fold{fold}/test/loss": test_loss, f"fold{fold}/test/recall": recall_score_val, f"fold{fold}/test/precision": precision_score_val, f"fold{fold}/test/f1": f1_score_val, f"fold{fold}/test/accuracy": accuracy_score_val})

            results = {
                "test_loss": test_loss,
                "recall_score": recall_score_val,
                "precision_score": precision_score_val,
                "f1_score": f1_score_val,
                "accuracy_score": accuracy_score_val
            }

            # save metrics
            results_df = pd.DataFrame(results, index=[0])
            results_df.to_csv(fold_dir / "test_results.csv", index=False)
            logger.info(f"Saved test results to {fold_dir / 'test_results.csv'}")

            preds_true = {
                "preds": y_pred,
                "true": y_true
            }
            
            # save predictions
            preds_true_df = pd.DataFrame(preds_true)
            preds_true_df.to_csv(fold_dir / "preds_true.csv", index=False)
            logger.info(f"Saved predictions to {fold_dir / 'preds_true.csv'}")

            #wandb table
            table = wandb.Table(data=preds_true_df)
            wandb.log({f"fold{fold}/preds_true": table})