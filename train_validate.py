import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import time

import matplotlib.pyplot as plt

from my_yolo_model import YOLO
from config import (
    DEVICE,
    NUM_EPOCHS,
    PATH,
    NUM_CLASSES,
    CONF_THRESHOLD,
    GRID_SIZES,
    LOAD_MODEL,
    CHECKPOINT_FILE,
    LEARNING_RATE,
    ANCHORS,
    SCHEDULER_STEP,
    MODEL_NAME,
    NUM_IMAGES,
    DATASET_TYPE
)
from loss import YoloLoss
from utils import save_checkpoint, load_pretrained_model
from dataloader import get_data_loaders
from utils_f.metrics import check_metrics
from plotting import plot_predicted_imgs

#===================================
# Train Validate
#===================================


def train(
    model: nn.Module, loader: DataLoader, loss_fn, optimizer, scaler, scaled_anchors: torch.Tensor
) -> tuple[float, float]:
    """ф-я проводит обучение модели.
    возвращает loss на трейновом датасете"""

    model.train()
    total_loss = 0

    for x, y in tqdm(loader, desc="Training"):
        

        y_0, y_1, y_2 = y
        y_0 = y_0.to(DEVICE)
        y_1 = y_1.to(DEVICE)
        y_2 = y_2.to(DEVICE)
        x = x.to(DEVICE)

        optimizer.zero_grad()


        with torch.autocast(device_type=DEVICE.type):
            # Получаем выход модели: [out_1, out_2, out_3]
            output = model(x)
            output_0, output_1, output_2 = output

            loss = (
                loss_fn(output_0, y_0, scaled_anchors[0])
                + loss_fn(output_1, y_1, scaled_anchors[1])
                + loss_fn(output_2, y_2, scaled_anchors[2])
            )

        total_loss += loss.item()
        
        # Backpropagate the loss 
        scaler.scale(loss).backward()

        
        # Optimization step
        scaler.step(optimizer)
        
        # Update the scaler for next iteration
        scaler.update()

        

    total_loss /= len(loader)
    

    return total_loss 


@torch.inference_mode()
def valid(model: nn.Module, loader: DataLoader, loss_fn, scaled_anchors) -> tuple[float, dict]:
    """Ф-я расчитывает ф-ю ошибки и метрику на валидационной выборке"""
    model.eval()

    total_loss = 0
    
    metrics_history = {
        'class': {'acc': [], 'prec': [], 'rec': []},
        'obj': {'acc': [], 'prec': [], 'rec': []},
        'noobj': {'acc': [], 'prec': [], 'rec': []}
     }
    
    for x, y in tqdm(loader):
        x = x.to(DEVICE)
        y_0, y_1, y_2 = y
        y_0 = y_0.to(DEVICE)
        y_1 = y_1.to(DEVICE)
        y_2 = y_2.to(DEVICE)

        output = model(x)

        output_0, output_1, output_2 = output

        loss = (
            loss_fn(output_0, y_0, scaled_anchors[0])
            + loss_fn(output_1, y_1, scaled_anchors[1])
            + loss_fn(output_2, y_2, scaled_anchors[2])
        )

        total_loss += loss.item()
        # (all_class_metrics = {"acc":torch.Tensor, "prec":torch.Tensor, "rec":torch.Tensor},
        #  all_obj_metrics = {"acc":torch.Tensor, "prec":torch.Tensor, "rec":torch.Tensor},
        #  all_noobj_metrics = {"acc":torch.Tensor, "prec":torch.Tensor, "rec":torch.Tensor})
        batch_metrics = check_metrics(y, output, confidence_threshold=CONF_THRESHOLD)

         # Store batch metric

        
        # Store batch metrics
        for metric_type in metrics_history:
            for key in metrics_history[metric_type]:
                metrics_history[metric_type][key].append(batch_metrics[metric_type][key])

        # total_metric += metric(pred_boxes=output, true_boxes=y, num_classes=NUM_CLASSES).item()

    total_loss /= len(loader)
    
    # Average metrics
    for metric_type in metrics_history:
        for key in metrics_history[metric_type]:
            metrics_history[metric_type][key] = torch.stack(metrics_history[metric_type][key]).mean()


    return total_loss, metrics_history


def plot_stats(train: list[float], valid: list[float], title: str):
    plt.figure(figsize=(16, 8))

    plt.title(title)

    plt.plot(train, label="Train")
    plt.plot(valid, label="Valid")
    plt.legend()
    plt.grid()

    plt.show()


def plot_metrics(metrics: dict):
    
    metric_types = ['acc', 'prec', 'rec']
    
    for metric_type in metric_types:
        
        plt.figure(figsize=(16, 8))
        plt.title(f'{metric_type.upper()} Metrics')

        for category in metrics:
            plt.plot(metrics[category][metric_type], label=f'{category}')

        plt.legend()
        plt.grid()
        plt.show()



# Функция обучения и валидации на всех эпохах
def whole_train_valid_cycle(
    model: nn.Module,
    num_epochs: int,
    loss_fn,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    metric_name: str,
    title: str,
    scheduler,
    optimizer,
    scaler,
    model_epoch: int = 0,
):
    """Ф-я проводит для каждой эпох обучение на трейновой выборке, валидацию для валидной выборки
    и отрисовку графиков."""
    print(100 * "-", "\n", "Запуск процесса обучения/валидации \n", 100 * "-")
    train_loss_history = []
    valid_loss_history  = []
    
    valid_metrics_history = {
        'class': {'acc': [], 'prec': [], 'rec': []},
        'obj': {'acc': [], 'prec': [], 'rec': []},
        'noobj': {'acc': [], 'prec': [], 'rec': []}
     }

    valid_loss = 0
    # Scaling the anchors 
    scaled_anchors = (torch.tensor(ANCHORS) * torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1,3,2)).to(DEVICE) 

    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss = train(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            scaled_anchors=scaled_anchors,
        )
        
        # train_metric_history.append(train_metric)
        
        if epoch % 5 == 0:
            valid_loss, valid_metrics = valid(model, valid_loader, loss_fn, scaled_anchors=scaled_anchors)

        model_epoch += 1    
        if epoch % 30 ==0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                model_epoch=model_epoch,
                dataset_type=DATASET_TYPE,
                filename=f"{MODEL_NAME}_E_  {model_epoch}_LOSS_{valid_loss:.2f}.pt",
            )
        
        # add loss to history arrays
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        # Store epochs metrics
        for metric_type in valid_metrics_history:
            for key in valid_metrics_history[metric_type]:
                valid_metrics_history[metric_type][key].append(valid_metrics[metric_type][key].cpu())

        
        scheduler.step()

        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        print(
            f"step {epoch} |  Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | dt: {dt:.2} s"
        )

        
    save_checkpoint(
                model=model,
                optimizer=optimizer,
                model_epoch=model_epoch,
                dataset_type=DATASET_TYPE,
                filename=f"{MODEL_NAME}_E_{model_epoch}_LOSS_{valid_loss:.2f}.pt",

            )
    plot_stats(train_loss_history, valid_loss_history, title="Loss")

            
    plot_metrics(metrics=valid_metrics_history)



def main():
    t0 = time.time()
    print(100 * "-", "\n", "Инициализация модели \n", 100 * "-")

    model = YOLO(num_classes=NUM_CLASSES)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=0.9)
    loss_fn = YoloLoss(num_classes=NUM_CLASSES)

    # Зададим scaler для ускорения обучения на GPU
    scaler = torch.amp.GradScaler()

    # set the type of tesor as TF32
    # torch.set_float32_matmul_precision("high")

    # Работа с несколькими cuda
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"Имеется {torch.cuda.device_count()} GPU")
            model = nn.DataParallel(model)

    model_epoch = 0
    if LOAD_MODEL:
        model_epoch_list = []
        model, optimizer = load_pretrained_model(checkpoint_path=CHECKPOINT_FILE, model=model, optimizer=optimizer,
                                      lr=LEARNING_RATE, model_epoch_list=model_epoch_list)
        model_epoch = model_epoch_list[0]
        

    model.to(DEVICE)

    num_epochs = NUM_EPOCHS
    metric_name = "ACCURACY"
    title = MODEL_NAME
    

    print(100 * "-", "\n", "Загрузка даталоадера \n", 100 * "-")
    if DATASET_TYPE == 'PASCAL_VOC':
        train_loader, valid_loader = get_data_loaders(path=PATH, num_images=NUM_IMAGES, dataset_type="voc")
    elif DATASET_TYPE == "COCO":
        train_loader, valid_loader = get_data_loaders(path=PATH, num_images=NUM_IMAGES, dataset_type="coco")
    

    whole_train_valid_cycle(
        model=model,
        num_epochs=num_epochs,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        metric_name=metric_name,
        title=title,
        scheduler=scheduler,
        optimizer=optimizer,
        model_epoch=model_epoch,
        scaler=scaler,
    )
    # изображение с предсказаниями
    
    plot_predicted_imgs(
        loader=valid_loader, model=model, confidence_threshold=0.5, iou_threshold=0.5
    )

    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    print(f"Общее время обучения и валидации = {dt: .4f}s")


if __name__ == "__main__":
    main()
