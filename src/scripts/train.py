# Imports
import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model
import yaml
from datetime import datetime
import fileutils
from tensorboardX import SummaryWriter

# User and Group varaibles to for file access permissions
gid = int(os.environ['GID'])
uid = int(os.environ['UID'])

PROJECT_STORAGE_FOLDER = "/home/storage/torch_tutorial"
PROJECT_DATA_FOLDER = "/home/data/CIFAR10"

# Make starting directory
fileutils.make_directory(PROJECT_STORAGE_FOLDER, uid, gid)
fileutils.make_directory(PROJECT_DATA_FOLDER, uid, gid)

# Build instance folder
now = datetime.now()
BASE_PATH = PROJECT_STORAGE_FOLDER + now.strftime("/%m_%d_%Y_%H:%M:%S")
fileutils.make_directory(BASE_PATH, uid, gid)

# Select location of model + args and tensorboard saves
MODEL_STORE_PATH = BASE_PATH + "/model"
TENSORBOARD_STORE_PATH = BASE_PATH + "/tensorboard"
fileutils.make_directory(MODEL_STORE_PATH, uid, gid)
fileutils.make_directory(TENSORBOARD_STORE_PATH, uid, gid)

MODEL_PATH = MODEL_STORE_PATH + "/m.pwf"
OPTIMISER_PATH = MODEL_STORE_PATH + "/o.pwf"
MODEL_CONFIG_PATH = MODEL_STORE_PATH + "/config.yml"

# Tensorboard setup
writer = SummaryWriter(log_dir=TENSORBOARD_STORE_PATH)


# Parser flags
parser = argparse.ArgumentParser(description="Pytorch CIFAR10 Tutorial")
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before checkpointing')
parser.add_argument('--resume', type=str, default=None, metavar='DIR',
                    help='resume training from checkpoint')
args = parser.parse_args()

with open(MODEL_CONFIG_PATH, "w") as f:
    yaml.dump(vars(args), f)

# Select device
use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda' if use_cuda else 'cpu')

# Set the seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# Load the datasets
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='/home/data/CIFAR10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='/home/data/CIFAR10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
fileutils.recursive_chown(PROJECT_DATA_FOLDER, uid, gid)

# Create a new model
net = model.Net().to(device)

# Get the optimiser and criterion
optimiser = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

# Check if resume training
if args.resume:
    net.load_state_dict(torch.load(args.resume + "/m.pwf"))
    optimiser.load_state_dict(torch.load(args.resume + "/o.pwf"))

# Run the training loop
net.train()
for epoch in range(args.epochs):
    for i, (data, target) in enumerate(trainloader):
        data = data.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        optimiser.zero_grad()

        outputs = net(data)

        loss = F.nll_loss(outputs, target)
        loss.backward()
        optimiser.step()
        writer.add_scalar('loss', loss.item(), i + epoch*len(trainloader))

        if i % args.save_interval == 0:
            torch.save(net.state_dict(), MODEL_PATH)
            torch.save(optimiser.state_dict(), OPTIMISER_PATH)

writer.close()

net.eval()
test_loss, correct = 0, 0

with torch.no_grad():
    for data, target in testloader:
        data = data.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)
        outputs = net(data)
        test_loss += F.nll_loss(outputs, target, reduction='sum').item()
        predicted = outputs.argmax(1, keepdim=True)
        correct += predicted.eq(target.view_as(predicted)).sum().item()

test_loss /= (len(testloader)*args.batch_size)
acc = correct / (len(testloader)*args.batch_size)

info = {"ARGS":vars(args), "TEST":{"accuracy":acc, "loss":test_loss}}
with open(MODEL_CONFIG_PATH, "w") as f:
    yaml.dump(info, f)

fileutils.recursive_chown(BASE_PATH, uid, gid)
