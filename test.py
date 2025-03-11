import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs import *
from dataset.dataset import MyDataset
from confuse_matrix import *
from model.CFNet import CFNet
import torch.nn as nn
from torch.amp import autocast

def test():

    args = parse_arguments()
    data_path = args.data_dir
    gpus = str(args.gpu[0])[2]
    if len(args.gpu) > 1:
        for gpu in args.gpu[1:]:
            gpus = gpus + "," + str(gpu)[2]
    batch_size = args.batch_size
    num_workers = args.num_workers
    checkpoint = args.checkpoint
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    test_data = MyDataset(data_path, "test")
    print("Test on {} image-pairs".format(len(test_data)))
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CFNet(3, 3)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    if checkpoint!=None:
        model.load_state_dict(torch.load(checkpoint,weights_only=False))
        print("Checkpoint {} succesfully loaded".format(checkpoint))
        
    model = model.to(device)
    confuse_matrix = ConfuseMatrixFor2Classes()
    model.eval()

    loop = tqdm(test_loader,position=0,leave=False)
    with torch.no_grad():
        for x1, x2, target, _ in loop:
            x1 = x1.to(device).float()
            x2 = x2.to(device).float()
            target = target.to(device).float()
            with autocast(device.type+":"+str(device.index)):                
                y_change,_,_,_,_ = model(x1, x2,device)
            y_change = torch.where(y_change>0.5,torch.ones_like(y_change),torch.zeros_like(y_change))
            y_change = y_change.to(torch.float32)
            confuse_matrix.update(pred=y_change.to("cpu").numpy(), target=target.to("cpu").numpy())
    scores_dictionary = confuse_matrix.get_scores()
    test_res = 'F1 = {}, IoU = {}, Precision = {}, Recall = {}'.format(
            round(100*scores_dictionary['F1'],3),
            round(100*scores_dictionary["IoU"],3),
            round(100*scores_dictionary['Recall'], 3), 
            round(100*scores_dictionary['Precision'], 3))
            
    print(test_res)
    
if __name__ == "__main__":
    test()