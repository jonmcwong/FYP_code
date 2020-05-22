import os
from pathlib import Path
import copy
import torch
import glob


# keeps the top 5 best models.
def rotating_save_checkpoint(state, prefix, path="./checkpoints", nb=5, best=False):
    if not os.path.isdir(path):
        os.makedirs(path)
    filenames = []
    first_empty = None
    # best_filename = Path(path) / f"{prefix}_best.pth"

    # # # save latest as best 

    # torch.save(state, best_filename)

    # # save checkpoint with other top n as n-1
    for i in range(nb):
        filename = Path(path) / f"{prefix}_{i}.pth"

        # set first empty to the first empty file
        if not os.path.isfile(filename) and first_empty is None:
            first_empty = filename
        filenames.append(filename)
    
    # if there was an emtpy file name
    if first_empty is not None:
        torch.save(state, first_empty)
    else:
        first = filenames[0]
        os.remove(first)

        # shift every thing down
        for filename in filenames[1:]:
            os.rename(filename, first)
            first = filename
        # save latests model.
        torch.save(state, filenames[-1])

def build_checkpoint(exp_name, unique_id, tpe, model, optimizer, val_loss, train_loss, batch, amp):
    return {
        "exp_name": exp_name,
        "unique_id": unique_id,
        "type": tpe,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": val_loss,
        "train_loss": train_loss,
        "batch": batch,
        "amp": amp.state_dict()
    }
            
def restore_checkpoint(filename, model=None, optimizer=None):
    """restores checkpoint state from filename and load in model and optimizer if provided"""
    print(f"Extracting state from {filename}")

    state = torch.load(filename)
    if model:
        print(f"Loading model state_dict from state found in {filename}")
        model.load_state_dict(state["model"])
    if optimizer:
        print(f"Loading optimizer state_dict from state found in {filename}")
        optimizer.load_state_dict(state["optimizer"])
    return state

def restore_checkpoint_cpu(filename, model=None, optimizer=None):
    """restores checkpoint state from filename and load in model and optimizer if provided"""
    print(f"Extracting state from {filename}")

    state = torch.load(filename, map_location=torch.device('cpu'))
    if model:
        print(f"Loading model state_dict from state found in {filename}")
        model.load_state_dict(state["model"])
    if optimizer:
        print(f"Loading optimizer state_dict from state found in {filename}")
        optimizer.load_state_dict(state["optimizer"])
    return state

def restore_best_checkpoint(prefix, path="./checkpoints", model=None, optimizer=None):
    filename = Path(path) / f"{prefix}_best"
    return restore_checkpoint(filename, model, optimizer)


# def restore_best_checkpoint(exp_name, unique_id, tpe,
#                             model=None, optimizer=None, path="./checkpoints", extension="pth"):
#     filename = Path(path) / f"{exp_name}_{unique_id}_{tpe}_best.{extension}"
#     return restore_checkpoint(filename, model, optimizer)

