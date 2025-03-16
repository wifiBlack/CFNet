import torch
import torch.nn as nn
from model.CFNet import *
from loss.loss import *
from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':
    # start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x1 = torch.randn(1, 3, 256, 256).to(device)
    x2 = torch.randn(1, 3, 256, 256).to(device)
    target = torch.randn(1,256,256).to(device)
    # net = nn.DataParallel(CFNet(3, 3),device_ids=[0,1,2,3])
    net = CFNet(3, 3)
    net = net.to(device)
    with autocast(device.type+":"+str(device.index)):
        y_change, Y1_change, Y2_change,Y1_unchange,Y2_unchange = net(x1, x2,device=device)
    loss = Loss()
    res = loss(y_change, Y1_change, Y2_change, Y1_unchange, Y2_unchange, target)
    from fvcore.nn import FlopCountAnalysis,parameter_count_table
    flops = FlopCountAnalysis(net, (x1, x2))
    total = sum([param.nelement() for param in net.parameters()])
    print("Params_Num: %.2fM" % (total/1e6))
    # print(flops.total()/1e9)
    # print(parameter_count_table(net))
    # end = time.time()