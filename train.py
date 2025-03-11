import os
from tqdm import tqdm
from loss.loss import *
from torch.amp import autocast
from confuse_matrix import *


def train(
    train_loader,
    val_loader,
    scaler,
    model,
    optimizer,
    scheduler,
    logpath,
    writer,
    epochs,
    device
):

    model = model.to(device)
    current_IoU = 0
    current_F1 = 0
    
    for epc in range(epochs):
        loop = tqdm(train_loader,position=0,leave=False)
        train_loss = 0
        model.train()
        for x1,x2,target,_ in loop:          
            optimizer.zero_grad()
            x1 = x1.to(device).float()
            x2 = x2.to(device).float()
            target = target.to(device).float()
            with autocast(device.type+":"+str(device.index)):                
                y_change, Y1_change, Y2_change,Y1_unchange,Y2_unchange = model(x1, x2,device)
                loss = Loss()(y_change, Y1_change, Y2_change, Y1_unchange, Y2_unchange, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss=train_loss+loss.item()           
        train_loss = round(train_loss/len(train_loader),3)
        
        model.eval()
        eval_loss = 0
        confuse_matrix = ConfuseMatrixFor2Classes()
        loop = tqdm(val_loader,position=0,leave=False)
        with torch.no_grad():
            for x1, x2, target, _ in loop:
                x1 = x1.to(device).float()
                x2 = x2.to(device).float()
                target = target.to(device).float()
                with autocast(device.type+":"+str(device.index)):                
                    y_change, Y1_change, Y2_change,Y1_unchange,Y2_unchange = model(x1, x2,device)
                    loss = Loss()(y_change, Y1_change, Y2_change, Y1_unchange, Y2_unchange, target)
                eval_loss = eval_loss + loss.item()
                y_change = torch.where(y_change>0.5,torch.ones_like(y_change),torch.zeros_like(y_change))
                confuse_matrix.update(pred=y_change.to("cpu").numpy(), target=target.to("cpu").numpy())
        eval_loss = round(eval_loss/len(val_loader),3)
        
        scores_dictionary = confuse_matrix.get_scores()
        epoch_result = 'F1 = {}, IoU = {}, Recall = {}, Precision = {}'.format(
            round(100*scores_dictionary['F1'],3),
            round(100*scores_dictionary["IoU"],3),
            round(100*scores_dictionary['Recall'], 3), 
            round(100*scores_dictionary['Precision'], 3))
        
        
        print("Epoch {} : Train Loss = {}, Eval Loss = {}, {}".format(epc,train_loss,eval_loss,epoch_result))
        
        F1 = scores_dictionary['F1']
        IoU = scores_dictionary['IoU']
        writer.add_scalar("Train Loss/epoch", train_loss, epc)
        writer.add_scalar("Eval Loss/epoch", eval_loss, epc)
        writer.add_scalar("F1/epoch", F1, epc)
        writer.add_scalar("IoU/epoch", IoU, epc)
        writer.flush()
        if IoU > current_IoU or F1>current_F1:
            current_IoU = IoU
            current_F1 = F1
            torch.save(model.state_dict(), os.path.join(logpath, "epoch{}.pth".format(epc)))

        scheduler.step()