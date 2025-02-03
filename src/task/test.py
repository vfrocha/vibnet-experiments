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

class TestTask:
    def __init__(self, model_selector, loss_name, use_loss_weights, dataset, batch_size, k_folds, save_path, device, project_name="ia_health",run_name="Experiment"):
        self.model_selector = model_selector
        self.loss_name = loss_name
        self.use_loss_weights = use_loss_weights
        self.dataset = dataset
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
            criterion=self.loss_name,
        )

        wandb_kwargs = dict(
            project=self.project_name,
            name=self.run_name,
            config=config_dict
        )

        wandb.init(**wandb_kwargs) 

    def _make_save_dir(self):
        count = 1
        while True:
            save_dir = f"{self.model_selector.model_name}_{self.dataset.name}_{count}"
            save_dir_path = pl.Path(self.save_path) / pl.Path(save_dir)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
                logger.info(f"Created directory to {save_dir_path}")
                break
            count += 1
        
        self.save_results_path = save_dir_path

    def run(self):
        model = self.model_selector.get_model()
        self.save_path = pl.Path(self.save_path) / pl.Path("test")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            logger.info(f"Created root directory to save all results {self.save_path}")

        # make saving dir with format model_name_optimizer_name_dataset_day_month_year
        save_dir = f"{self.model_selector.model_name}_{self.dataset.name}"
        save_dir_path = pl.Path(self.save_path) / pl.Path(save_dir)
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
            self.save_results_path = save_dir_path
            logger.info(f"Created directory to {save_dir_path}")
        elif os.path.exists(save_dir_path):
            logger.info(f"Directory {save_dir_path} already exists")
            self._make_save_dir()

        test_dataset = self.dataset.test_dataset
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)      
  
        if self.use_loss_weights:
            class_weights = compute_class_weight('balanced', classes=np.array(list(self.dataset.labels_names.keys())), y=test_dataset.labels)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            loss_selector = LossSelector(self.loss_name, class_weights)
        else:
            loss_selector = LossSelector(self.loss_name)
        
        loss = loss_selector.get_loss()

        test_loss, _, y_pred, y_true = test(model, loss, test_loader, self.device)

        recall_score_val = recall_score(y_true, y_pred, average='weighted')
        precision_score_val = precision_score(y_true, y_pred, average='weighted')
        f1_score_val = f1_score(y_true, y_pred, average='weighted')
        accuracy_score_val = accuracy_score(y_true, y_pred)

        logger.info(f"Test loss: {test_loss}")
        logger.info(f"Recall score: {recall_score_val*100:0.4f}%")
        logger.info(f"Precision score: {precision_score_val*100:0.4f}%")
        logger.info(f"F1 score: {f1_score_val*100:0.4f}%")
        logger.info(f"Accuracy score: {accuracy_score_val*100:0.4f}%")

        wandb.log({f"test/loss": test_loss, f"test/recall": recall_score_val, f"test/precision": precision_score_val, f"test/f1": f1_score_val, f"test/accuracy": accuracy_score_val})

        results = {
            "test_loss": test_loss,
            "recall_score": recall_score_val,
            "precision_score": precision_score_val,
            "f1_score": f1_score_val,
            "accuracy_score": accuracy_score_val
        }

        # save metrics
        results_df = pd.DataFrame(results, index=[0])
        results_df.to_csv(self.save_results_path / "test_results.csv", index=False)
        logger.info(f"Saved test results to {self.save_results_path / 'test_results.csv'}")

        preds_true = {
            "preds": y_pred,
            "true": y_true
        }
        
        # save predictions
        preds_true_df = pd.DataFrame(preds_true)
        preds_true_df.to_csv(self.save_results_path / "preds_true.csv", index=False)
        logger.info(f"Saved predictions to {self.save_results_path / 'preds_true.csv'}")

        #wandb table
        table = wandb.Table(data=preds_true_df)
        wandb.log({f"preds_true": table})