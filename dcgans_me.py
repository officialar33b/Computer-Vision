from __future__ import print_function
import torch 
import torch.nn as nn 
import torch.nn.parallel as parallel 
import torch.backends.cudnn as cudnn
from torch import optim
from torchvision import datasets, transforms
import torch.utils.data 
import torchvision.utils as vutils
import random 
cudnn.benchmark = True 
from torch.autograd import Variable

manual_seed = random.randint(1, 10000)
print("Random Seed: ", manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

dataset = datasets.CIFAR10(root="./data", download=True,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5 , 0.5), (0.5, 0.5, 0.5))
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

#Checking Cuda.
device = "cuda" if torch.cuda.is_available() else 'cpu'
print(device)

num_classes=3
noise_dim = 100
num_gen_filters=64
num_disc_filters = 64

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') !=-1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_dim, num_gen_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_gen_filters * 8),
            nn.ReLU(True),
            # state size. (num_gen_filters*8) x 4 x 4
            nn.ConvTranspose2d(num_gen_filters * 8, num_gen_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_filters * 4),
            nn.ReLU(True),
            # state size. (num_gen_filters*4) x 8 x 8
            nn.ConvTranspose2d(num_gen_filters * 4, num_gen_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_filters * 2),
            nn.ReLU(True),
            # state size. (num_gen_filters*2) x 16 x 16
            nn.ConvTranspose2d(num_gen_filters * 2, num_gen_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_filters),
            nn.ReLU(True),
            # state size. (num_gen_filters) x 32 x 32
            nn.ConvTranspose2d(num_gen_filters, num_classes, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            return output
        
netG = Generator(1).to(device)
netG.apply(weights_init)

print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_classes, num_disc_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_disc_filters) x 32 x 32
            nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_disc_filters*2) x 16 x 16
            nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_disc_filters*4) x 8 x 8
            nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_disc_filters*8) x 4 x 4
            nn.Conv2d(num_disc_filters * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

netD = Discriminator(1).to(device)
netD.apply(weights_init)
#load weights to test the model 
#netD.load_state_dict(torch.load('weights/netD_epoch_24.pth'))
print(netD)

loss = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.99))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.99))

random_noise = torch.randn(128, noise_dim, 1, 1, device=device)
real_label = 1
fake_label = 0 

EPOCHS = 25
g_loss = []
d_loss = []

for epoch in range(EPOCHS):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = Variable(data[0].to(device))
        batch_size = real_cpu.size(0)
        label = Variable(torch.full((batch_size, ), real_label, device=device)).to(torch.float32)

        output = netD(real_cpu)
        errD_real = loss(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = loss(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = loss(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, EPOCHS, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        #save the output
        if i % 10 == 0:
            print('saving the output')
            vutils.save_image(real_cpu,'output/real_samples.png',normalize=True)
            fake = netG(noise)
            vutils.save_image(fake.detach(),'output/fake_samples_epoch_%03d.png' % (epoch),normalize=True)
    
    # Check pointing for every epoch
    torch.save(netG.state_dict(), 'weights/netG_epoch_%d.pth' % (epoch))
    torch.save(netD.state_dict(), 'weights/netD_epoch_%d.pth' % (epoch))