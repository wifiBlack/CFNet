import warnings
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import *
from model.CFNet import *
from configs import *
from train import *
import os

def run():

    # Parse arguments:
    args = parse_arguments()
    log_path = args.log_dir
    data_path = args.data_dir
    gpus = str(args.gpu[0])[2]
    if len(args.gpu) > 1:
        for gpu in args.gpu[1:]:
            gpus = gpus + "," + str(gpu)[2]
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    lr = args.lr
    checkpoint = args.checkpoint
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Initialize tensorboard:
    writer = SummaryWriter(log_dir=log_path)

    # Inizialitazion of dataset and dataloader:
    train_data = MyDataset(data_path, "train")
    val_data = MyDataset(data_path, "val")
    print("Train on {} image-pairs, Eval on {} image-pairs".format(len(train_data),len(val_data)))
    
    train_loader = DataLoader(train_data, batch_size=batch_size,num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size,num_workers = num_workers, shuffle=True)

    # device setting for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CFNet(3, 3)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    # restart_from_checkpoint = False
    if checkpoint!=None:
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        print("Checkpoint {} succesfully loaded".format(checkpoint))

    # print number of parameters
    parameters_tot = 0
    for _, param in model.named_parameters():
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    print("Number of model parameters:{}".format(parameters_tot))

    scaler = torch.GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    warnings.filterwarnings("ignore", category=UserWarning)
    
    train(
        train_loader,
        val_loader,
        scaler,
        model,
        optimizer,
        scheduler,
        log_path,
        writer,
        epochs,
        device
    )
    writer.close()

if __name__ == "__main__":
    run()