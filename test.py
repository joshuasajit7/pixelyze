import os
import csv
from tqdm import tqdm
import torch
import argparse
from tools.dataset import Dataset_train, Dataset_val
import cv2
from tools.utils import valid, train
from tools.patch_generate import divide
import matplotlib.pyplot as plt
from model.auto_vit import VAE
import imutils

min_loss = 1000.0

# Paths (make sure these are correct)
img_path = r".\image\splicing-02.png"
noise_path = r".\noise\splicing-02.mat"
save_path = r".\results\splicing-02.png"
mask_path = r""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=0, help='training batch size.')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate .')
    parser.add_argument('--epoch', type=int, default=20, help='number of training epoch.')
    parser.add_argument('--loss_interval', type=int, default=2)
    parser.add_argument('--stop_loss', type=float, default=0.0001)  # 0.003
    parser.add_argument('--threshold', default=0.5)
    args = parser.parse_args()

    # Device configuration: Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load image
    img_cv = cv2.imread(img_path)
    [m, n, c] = img_cv.shape

    # Determine patch size and stride based on image size
    patch_size = 64
    if m * n > 1000 * 1000:
        patch_stride = 16
        print(f"Input image size [{m} x {n}] large")
    else:
        patch_stride = 8
        print(f"Input image size [{m} , {n}] small")

    # Divide the image into patches
    data_all, qf = divide(img_path, noise_path, patch_size, patch_stride)

    # Datasets
    train_dataset = Dataset_train(data_all)
    valid_dataset = Dataset_val(data_all)

    # Determine batch size
    if args.batch_size == 0:
        len_train = len(train_dataset)
        batch_size = int(len_train / 20)
        print(f"Batch size = {batch_size}")
    else:
        batch_size = args.batch_size

    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model and move to the correct device (CPU/GPU)
    net = VAE().to(device)  # Move model to device (CPU or GPU)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-4, last_epoch=-1)

    # Training loop
    for epoch in range(0, args.epoch):
        net.train()
        label, min_loss = train(train_dataloader, epoch, args, optimizer, net, min_loss)
        if label:
            break

    print("Start generating map")

    # Evaluation (Heat Map Generation)
    net.eval()
    if not mask_path:
        map, map_threshold0_5, img_cv = valid(valid_dataloader, img_path, mask_path, args, optimizer, net, save_path)
        plt.figure()
        plt.title('Heat map')
        plt.imshow(imutils.opencv2matplotlib(img_cv), interpolation='none')
        plt.imshow(map, cmap='jet', alpha=0.5, interpolation='none')
        plt.show()
        if save_path:
            cv2.imwrite(save_path, map * 255)
    else:
        valid(valid_dataloader, img_path, mask_path, args, optimizer, net, save_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = VAE().to(device)
