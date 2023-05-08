import time
import copy
from utils import colorstr
import logging
from tqdm import tqdm
import torch

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("Torch-Cls")

def train_model(model, criterion, optimizer, device, num_epochs, dataloaders):
    since = time.time()

    LOGGER.info(f"\n{colorstr('Device:')} {device}")
    LOGGER.info(f"\n{colorstr('Optimizer:')} {optimizer}")
    LOGGER.info(f"\n{colorstr('Loss:')} {type(criterion).__name__}")


    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_optim = copy.deepcopy(optimizer.state_dict())
    best_val_acc = 0.0

    model.to(device)
    for epoch in range(num_epochs):
        LOGGER.info(colorstr(f'\nEpoch {epoch}/{num_epochs-1}:'))

        for phase in ["train", "val"]:
            if phase == "train":
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) % 
                                ('Training:', 'gpu_mem', 'loss', 'acc'))
                model.train()
            else:
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) % 
                                ('Validation:','gpu_mem', 'loss', 'acc'))
                model.eval()

            running_items = 0
            running_loss = 0.0
            running_corrects = 0

            _phase = tqdm(dataloaders[phase],
                      total=len(dataloaders[phase]),
                      bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                      unit='batch')

            for inputs, labels in _phase:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
   
                running_items += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / running_items
                epoch_acc = running_corrects / running_items

                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                desc = ('%35s' + '%15.6g' * 2) % (mem, running_loss / running_items, running_corrects / running_items)
                _phase.set_description_str(desc)

            if phase == 'train':
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_optim = copy.deepcopy(optimizer.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s with {} epochs'.format(time_elapsed // 3600,time_elapsed % 3600 // 60, time_elapsed % 60, num_epochs))
    print('Best val Acc: {:4f}'.format(best_val_acc))

    model.load_state_dict(best_model_wts)
    optimizer.load_state_dict(best_model_optim)

    return model, best_val_acc.item()
