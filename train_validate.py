import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import time

import matplotlib.pyplot as plt

from my_yolo_model import YOLO
from config import (DEVICE,
                    NUM_EPOCHS,
                    PATH, 
                    NUM_CLASSES, 
                    NUMBER_BLOCKS_LIST, 
                    BACKBONE_NUM_CHANNELS, 
                    LOAD_MODEL, 
                    CHECKPOINT_FILE,
                    LEARNING_RATE)
from loss import YoloLoss
from utils import get_data_loaders, save_checkpoint, load_checkpoint, mean_average_precision






def train(model: nn.Module,
         loader: DataLoader,
         loss_fn,
         metric, 
         device,
         optimizer) -> tuple[float, float]: 
    
    '''ф-я проводит обучение модели. 
    возвращает ошибку на трейновом датасете'''
    
    model.train()
    total_loss = 0
    total_metric = 0
    
    for x, y in tqdm(loader, desc="Train"):
        # y = [y1, y2, y3]
        
        y_1, y_2, y_3 = y
        y_1 = y_1.to(device)
        y_2 = y_2.to(device)
        y_3 = y_3.to(device)
        x = x.to(device)

        optimizer.zero_grad() 
        # Получаем выход модели: [out_1, out_2, out_3]
        output = model(x)
        output_1, output_2, output_3 = output
        
        loss = loss_fn(output_1, y_1) +  loss_fn(output_2, y_2) + loss_fn(output_3, y_3)
        
        total_loss += loss.item()
        
        loss.backward() 
    
        optimizer.step()
        
        # total_metric += metric(pred_boxes=output, true_boxes=y, num_classes=NUM_CLASSES).item()
        
    total_loss /= len(loader)
    # total_metric /= len(loader) 
    
    
    return total_loss#, total_metric

@torch.inference_mode() 
def valid(model: nn.Module,
         loader: DataLoader,
         loss_fn,
         metric) -> tuple[float, float] :
    
    '''Ф-я расчитывает ф-ю ошибки и метрику на валидационной выборке'''
    model.eval() 
    
    total_loss = 0
    total_metric = 0

    
    for x, y in tqdm(loader, desc="Evaluation"):
        x = x.to(DEVICE)
        y_1, y_2, y_3 = y
        y_1 = y_1.to(DEVICE)
        y_2 = y_2.to(DEVICE)
        y_3 = y_3.to(DEVICE)
        
        output = model(x) 
        
        output_1, output_2, output_3 = output
        
        loss = loss_fn(output_1, y_1) +  loss_fn(output_2, y_2) + loss_fn(output_3, y_3)
        
        total_loss += loss.item()
        
        # total_metric += metric(pred_boxes=output, true_boxes=y, num_classes=NUM_CLASSES).item()
        

    total_loss /= len(loader)
    # total_metric /= len(loader) 
    
    return total_loss#, total_metric 

def plot_stats(train: list[float], valid: list[float], title: str):
    plt.figure(figsize=(16, 8))

    plt.title(title)

    plt.plot(train, label='Train')
    plt.plot(valid, label='Valid')
    plt.legend()
    plt.grid()

    plt.show()


#Функция обучения и валидации на всех эпохах
def whole_train_valid_cycle(model: nn.Module,
                    num_epochs: int, 
                    loss_fn, 
                    train_loader: DataLoader, 
                    valid_loader: DataLoader,
                    metric,
                    metric_name: str,
                    title: str,
                    scheduler,
                    optimizer,
                    model_epoch: int = 0,
                    ):
    '''Ф-я проводит для каждой эпох обучение на трейновой выборке, валидацию для валидной выборки
    и отрисовку графиков. '''
    print(100 * "-", '\n', "Запуск процесса обучения/валидации \n", 100 * "-")
    train_loss_history, train_metric_history = [], []
    valid_loss_history, valid_metric_history = [], []
    
    valid_loss, valid_metric = 0, 0
    
    for epoch in range(num_epochs): 
        t0 = time.time()
        train_loss = train(model=model, loader=train_loader, loss_fn=loss_fn, metric=metric, device=DEVICE, optimizer=optimizer)
        train_loss_history.append(train_loss) 
        # train_metric_history.append(train_metric) 
        
        model_epoch += 1
        if epoch % 5 ==0:
            valid_loss = valid(model, valid_loader, loss_fn, metric)
            save_checkpoint(model=model, optimizer=optimizer, model_epoch=model_epoch, filename=f"models\\{title}_{epoch}.tar")
            

        valid_loss_history.append(valid_loss) 
        # valid_metric_history.append(valid_metric)
        
        scheduler.step() 

        t1 = time.time()
        dt = (t1 - t0) #time difference in seconds
        print(f"step {epoch} |  Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Valid {metric_name}: {valid_metric:.4f} | dt: {dt:.5}s")
               
        # print(f"step {epoch} | loss: {loss_accum.item():.6f} | lr: {lr: .4e} | norm: {norm: .4f} | dt: {dt:.5}s | tok/sec: {tokens_per_sec}")
        
    plot_stats(train_loss_history, valid_loss_history, title="Loss") 
    plot_stats(train_metric_history, valid_metric_history, metric_name) 
    
        

def main():
    print(100 * "-", '\n', "Инициализация модели \n", 100 * "-")
    model = YOLO(num_classes=NUM_CLASSES, n_blocks_list=NUMBER_BLOCKS_LIST, backbone_num_channels=BACKBONE_NUM_CHANNELS)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    loss_fn = YoloLoss(num_classes=NUM_CLASSES)

    # Работа с несколькими cuda
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f'Имеется {torch.cuda.device_count()} GPU') 
            model = nn.DataParallel(model)

    #set the type of tesor as TF32
    torch.set_float32_matmul_precision('high')
    model_epoch = 0

    # Загрузка предобученной модели
    if LOAD_MODEL:
        try:
            model_epoch_list = []
            load_checkpoint(
                CHECKPOINT_FILE, model, optimizer, LEARNING_RATE, model_epoch_list
            )
            model_epoch = model_epoch_list[0]
        except:
            print("Не удалось загрузить модель")

    model.to(DEVICE)

    num_epochs = NUM_EPOCHS
    metric = mean_average_precision
    metric_name = "IOU"
    title = 'YOLO_V1'
    

    print(100 * "-", '\n', "Загрузка даталоадера \n", 100 * "-")
    train_loader, valid_loader = get_data_loaders(path=PATH, num_images=1000)
    

    whole_train_valid_cycle(model=model, num_epochs=num_epochs, loss_fn=loss_fn, train_loader=train_loader, valid_loader=valid_loader, metric=metric, metric_name=metric_name, 
                            title=title,
                            scheduler=scheduler,
                            optimizer=optimizer,
                            model_epoch=model_epoch)


if __name__=="__main__":
    main()