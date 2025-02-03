import torch
import numpy as np
import pandas as pd
import time

import wandb
import pathlib as pl

from src.logger import logger
from src.early_stopper import EarlyStopper

def train_inner_loop(model, optimizer, loss_func, train_dataloader, device='cpu'):
    model.train()
    running_loss = 0
    pred_list, target_list = [], []
    counter = 0

    for i, (targets, labels) in enumerate(train_dataloader):
        targets, labels = targets.to(device), labels.to(device)

        outputs = model(targets)

        loss = loss_func(outputs, labels)
        running_loss += loss.item()

        _, pred = torch.max(outputs, axis=1)
        pred_list.append(pred.cpu().numpy())
        target_list.append(labels.cpu().numpy())
        running_accuracy = (pred == labels).sum().item() / len(labels)

        #wandb.log({"train/loss/step": loss.item(), "train/acc/step": running_accuracy})
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1
        
    # joining all the predictions and targets and flattening them
    pred_list  = np.concatenate(pred_list).ravel()
    target_list  = np.concatenate(target_list).ravel()

    train_acc = np.mean(pred_list == target_list)
    train_loss = running_loss / counter

    return train_loss, train_acc

def train(model, optimizer, scheduler, loss_func, train_dataloader, val_dataloader, saving_path, num_epoch=200, device='cpu',fold=0):
    checkpoint_name = pl.Path(saving_path) / "last_checkpoint.pth"

    model.train()
    train_accs, train_losses = [], []
    vals_accs, vals_losses = [], []
    best_loss = np.inf
    early_stopper = EarlyStopper(patience=20, min_delta=0.0001)
    
    logger.info('| Epoch | Train Loss | Train Acc | Validation Loss | Validation Acc |  Time  |')

    for epoch in range(num_epoch):
        start = time.time()

        train_loss, train_acc = train_inner_loop(model, optimizer, loss_func, train_dataloader, device=device)
        val_loss, val_acc, _, _ = test(model, loss_func, val_dataloader, device=device)
        scheduler.step(val_loss)
        
        end = time.time()
        logger.info(f'|  {epoch+1:03.0f}  |   {train_loss:.5f}  |    {train_acc*100:02.0f}%    |     {val_loss:.5f}     |       {val_acc*100:02.0f}%      | {end-start:.2f}s |')
        
        # logging to wandb
        wandb.log({f"fold{fold}/train/loss/epoch": train_loss, f"fold{fold}/train/acc/epoch": train_acc, 
                   f"fold{fold}/val/loss/epoch": val_loss, f"fold{fold}/val/acc/epoch": val_acc,
                   f"fold{fold}/lr": optimizer.param_groups[0]['lr']})

        # saving best and last checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            best_path_name = pl.Path(saving_path) / "best_checkpoint.pth"
            torch.save(model.state_dict(), best_path_name)
        
        torch.save(model.state_dict(), checkpoint_name)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        vals_accs.append(val_acc)
        vals_losses.append(val_loss)

        if early_stopper.early_stop(val_loss):             
            break

    df_loss = pd.DataFrame({"train_loss": train_losses, "val_loss": vals_losses})
    df_acc = pd.DataFrame({"train_acc": train_accs, "val_acc": vals_accs})

    table_loss = wandb.Table(dataframe=df_loss)
    table_acc = wandb.Table(dataframe=df_acc)

    wandb.log({   
        f"fold{fold}/loss": table_loss,
        f"fold{fold}/acc": table_acc,
        f"fold{fold}/loss": wandb.plot.line_series(
            xs=range(epoch),
            ys=[train_losses, vals_losses],
            keys=["Train loss", "Val loss"],
            title="Train x Val loss",
            xname="Epochs"),
        f"folds{fold}/acc": wandb.plot.line_series(
            xs=range(epoch),
            ys=[train_accs, vals_accs],
            keys=["Train accuracy", "Val accuracy"],
            title="Train x Val accuracy",
            xname="Epochs")
        }
    )

    return train_losses, train_accs, vals_losses, vals_accs

def test(model, loss_func, dataloader, device='cpu'):
    model.eval()
    with torch.no_grad():
        running_loss = 0
        pred_list, target_list = [], []
        counter = 0
        for target, labels in dataloader:
            target, labels = target.to(device), labels.to(device)

            outputs = model(target)
            loss = loss_func(outputs, labels)
            _, pred = torch.max(outputs, axis=1)
            
            loss = loss.item()
            running_loss += loss

            pred_list.append(pred.cpu().numpy())
            target_list.append(labels.cpu().numpy())

            counter += 1

        pred_list  = np.concatenate(pred_list).ravel()
        target_list  = np.concatenate(target_list).ravel()

        acc = np.mean(pred_list == target_list)
        loss = running_loss / counter

        return loss, acc, pred_list, target_list