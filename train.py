import argparse
import os
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from utils.datasets import create_dataloader
from utils.general import check_img_size, increment_path, labels_to_class_weights, strip_optimizer
from utils.torch_utils import select_device
from utils.loss import ComputeLoss
from models.yolo import Model
from utils.plots import plot_results

def train(hyp, opt, device):
    # Paths
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # Increment run if exists
    (save_dir / 'weights').mkdir(parents=True, exist_ok=True)

    # Load hyperparameters
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    train_path = data_dict['train']
    val_path = data_dict['val']
    nc = int(data_dict['nc'])  # number of classes

    # Model initialization
    model = Model(opt.cfg, ch=3, nc=nc).to(device)  # Create model
    img_size = check_img_size(opt.img_size, s=model.stride.max())  # check img_size

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hyp['lr0'])

    # DataLoader
    dataloader, dataset = create_dataloader(train_path, img_size, opt.batch_size, opt.workers, opt.rect, opt.cache_images, opt.single_cls, opt.img_weights, opt.augment, hyp, self.opt)
    val_loader = create_dataloader(val_path, img_size, opt.batch_size, opt.workers, opt.rect, False, False, opt.single_cls, hyp)[0]

    # Loss function
    compute_loss = ComputeLoss(model)

    # Train
    for epoch in range(opt.epochs):
        model.train()
        for i, (imgs, targets, paths, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Forward pass
            pred = model(imgs)

            # Loss computation
            loss, loss_items = compute_loss(pred, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{opt.epochs}] Loss: {loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % opt.save_period == 0 or (epoch + 1) == opt.epochs:
            torch.save(model.state_dict(), f"{save_dir}/weights/epoch_{epoch+1}.pt")

    # Final model
    torch.save(model.state_dict(), f"{save_dir}/weights/best.pt")

    # Strip optimizer from the saved model
    strip_optimizer(f"{save_dir}/weights/best.pt")

    # Plot results
    plot_results(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/breast_cancer.yaml', help='dataset.yaml path')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='yolov7-breast-cancer', help='save to project/name')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-period', type=int, default=10, help='save model every x epochs')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    # Hyperparameters
    with open('data/hyp.scratch.custom.yaml', errors='ignore') as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Select device
    device = select_device(opt.device)

    # Train the model
    train(hyp, opt, device)
