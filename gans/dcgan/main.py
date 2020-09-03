import argparse
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms

from dcgan_generator import Generator
from dcgan_discriminator import Discriminator
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=bool, default=True, help='add checkpoint before training')
parser.add_argument('--checkpoint_path', type=str, default='gan_models/dcgan_149_43050.pt', help='file path of checkpoint')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--d_step', type=int, default=2)

opt = parser.parse_args()

batch = int(opt.batch)
nc = int(opt.nc)              # number of channels in training images
nz = int(opt.nz)            # latent vector
ngf = int(opt.ngf)            # Size of feature maps in generator
ndf = int(opt.ndf)            # Size of feature maps in discriminator


# Define device as availabe hardware
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Dataset and DataLoader creation
dataroot = 'E:/final_dataset/cover/train'
train_dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

torch.save(train_dataset, 'train_dataset_128_norm_mine.pt')
train_dataset = torch.load('train_dataset_128_norm_mine.pt')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch, drop_last=True)

torch.manual_seed(4)

# Models init
model_g = Generator(nz, ngf, nc)
model_d = Discriminator(nz, ndf, nc)
model_g.to(device)
model_d.to(device)
model_g.apply(utils.weights_init)
model_d.apply(utils.weights_init)

# Criterion init
criterion = nn.BCELoss()

# Optimizer init
lr_d = 2e-4
lr_g = 2e-4
weight_decay = 0
optimizer_D = torch.optim.Adam(model_d.parameters(), lr=lr_d, weight_decay=weight_decay, betas=(0.5, 0.99))
optimizer_G = torch.optim.Adam(model_g.parameters(), lr=lr_g, weight_decay=weight_decay, betas=(0.5, 0.99))

# One smooth labeling
real_label = 0.9
fake_label = 0

epochs = 20
cost_d=[]
cost_g=[]
img_list = []
epoch = 0
it = 0
if opt.checkpoint:
    checkpoint = torch.load(opt.checkpoint_path, map_location=lambda storage, loc: storage)
    model_d.load_state_dict(checkpoint['model_d'])
    model_g.load_state_dict(checkpoint['model_g'])
    cost_d = checkpoint['cost_d']
    cost_g = checkpoint['cost_g']
    epoch = checkpoint['epoch']
    it = len(cost_g)

for epoch in range(epoch, epoch+epochs):
    for i, img in enumerate(train_loader,1):
        it+=1
        img, _ = img
        # DISCRIMINATOR
        # Real batch
        model_d.zero_grad()
        z_d_real = model_d(img.to(device)).view(-1)
        real_batch = torch.full((batch,), real_label, device=device)
        loss_d_real = criterion(z_d_real, real_batch)
        #loss_d_real = -((1/batch)*torch.sum(torch.log(z_d_real)))
        loss_d_real.backward()

        # Fake batch
        noise = (torch.randn(batch, nz, 1, 1, device=device)-0.5)/0.5
        img_fake = model_g(noise.to(device))
        z_d_fake = model_d(img_fake.detach()).view(-1)
        fake_batch = torch.full((batch,), fake_label, device=device)
        loss_d_fake = criterion(z_d_fake, fake_batch)
        #loss_d_fake = -((1/batch)*torch.sum(torch.log(1-z_d_fake)))
        loss_d_fake.backward()
        loss_d = loss_d_fake + loss_d_real
        optimizer_D.step()

        # Generator
        model_g.zero_grad()
        z_g = model_d(img_fake).view(-1)
        loss_g = criterion(z_g.view(-1), real_batch)
        #loss_g = -((1/batch)*torch.sum(torch.log(z_g.view(-1))))
        loss_g.backward()
        optimizer_G.step()

        cost_d.append(loss_d)
        cost_g.append(loss_g)

        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, epochs+epoch, i, len(train_loader), loss_d.item(), loss_g.item()))


    # Check how the generator is doing by saving G's output on fixed_noise
    with torch.no_grad():
        fake = model_g(noise).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    vutils.save_image(img_list, 'gan_models/imgs_noise/imgs_'+str(epoch)+'_'+str(it)+'.jpg')

    torch.save(
        {'model_d': model_d.state_dict(),
         'model_g': model_g.state_dict(),
         'epoch': epoch,
         'optimizer_d': optimizer_D.state_dict(),
         'optimizer_g': optimizer_G.state_dict(),
         'cost_d': cost_d,
         'cost_g': cost_g,
         'iter': it}, 'gan_models/dcgan_'+str(epoch)+'_'+str(it)+'.pt')