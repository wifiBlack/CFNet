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
from torchvision.transforms import ToPILImage
import shutil  # 用于删除文件夹
from model.utils import save_feature_maps_with_label

    

def content():
    args = parse_arguments()
    data_path = args.data_dir
    content_dir = args.content_dir  # 指定保存content结果的文件夹
    gpus = str(args.gpu[0])[2]
    if len(args.gpu) > 1:
        for gpu in args.gpu[1:]:
            gpus = gpus + "," + str(gpu)[2]
    batch_size = args.batch_size
    num_workers = args.num_workers
    checkpoint = args.checkpoint

    # 如果 content_dir 存在，则删除它
    if os.path.exists(content_dir):
        shutil.rmtree(content_dir)
        print(f"Deleted existing directory: {content_dir}")
    
    # 重新创建 content_dir
    os.makedirs(content_dir)
    print(f"Created new directory: {content_dir}")

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    test_data = MyDataset(data_path, "test")
    print("Show Content on {} image-pairs".format(len(test_data)))
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CFNet(3, 3)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, weights_only=False))
        print("Checkpoint {} successfully loaded".format(checkpoint))

    model = model.to(device)
    model.eval()

    loop = tqdm(test_loader, position=0, leave=False)
    with torch.no_grad():
        for i, (x1, x2, target, filename) in enumerate(loop):
            x1 = x1.to(device).float()
            x2 = x2.to(device).float()
            target = target.to(device).float()
            with autocast(device.type + ":" + str(device.index)):
                # y_change, _, _, _, _ = model(x1, x2, device)
                Y1,Y2 = model(x1, x2, device,is_content=True)
                save_feature_maps_with_label(Y1,Y2,target,content_dir,filename)
                       


if __name__ == "__main__":
    content()
