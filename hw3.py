import torch
import torchvision
import torchvision.transforms as transforms
from networks import DCGenerator, DCDiscriminator, weights_init, WGenerator, WDiscriminator, ACDiscriminator, ACGenerator
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import math

n_class = 10
nz = 100
ngf = 64
ndf = 32
nc = 3

def compute_gp(netD, real_data, fake_data):
        batch_size = real_data.size(0)
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        eps = eps.expand_as(real_data)
        
        # Interpolation between real data and fake data.
        interpolation = eps * real_data + (1 - eps) * fake_data
        
        # get logits for interpolated images
        interp_logits = netD(interpolation)
        grad_outputs = torch.ones_like(interp_logits)
        
        # Compute Gradients
        gradients = torch.autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)

def train_WGAN(device, train_data):
    netG = WGenerator().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)


    # Create the Discriminator
    netD = WDiscriminator().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    lr = .00005
    
    optim_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.95))
    optim_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.95))

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
  
    num_epochs = 100
    real_label = 1
    fake_label = -1 
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(train_data, 0):
             
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
          
            output = netD(real_cpu).view(-1)
       
            
       
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
         
            fake_output = netD(fake.detach()).view(-1)
            
            D_G_z1 = fake_output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            #errD = errD_real + errD_fake
            errD = -torch.mean(output)+ torch.mean(fake_output.detach())
            netD.zero_grad()
            errD.backward(retain_graph=True)
            # Update D
            optim_D.step()
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            label.fill_(real_label)  
            if i % 4 == 0:
                output = netD(fake).reshape(-1)
                # Calculate G's loss based on this output
                #errG = criterion(output, label)
                errG = -torch.mean(output)
                netG.zero_grad()
                #errG = torch.mean(output)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optim_G.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(train_data),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_data)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(train_data),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        print("Start Testing")
        real_batch = next(iter(train_data))

        # Plot the real images
        if epoch % 5 == 4:
            plt.figure(figsize=(15,15))
            plt.subplot(1,2,1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

            # Plot the fake images from the last epoch
            plt.subplot(1,2,2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(img_list[-1],(1,2,0)))
            plt.show()
        

def train_DCGAN(device, batch_size, trainloader):
    netG = DCGenerator().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    netD = DCDiscriminator().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    lr = 0.00015
    beta1 = 0.5

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    num_epochs = 30
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
      
            
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
            # Generate batch of latent vectors
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
         
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label) 
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(trainloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(trainloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(trainloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        print("Start Testing")
        real_batch = next(iter(trainloader))

        # Plot the real images
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1],(1,2,0)))
        plt.show()
        
def onehot_encode(label, device, n_class=n_class):  
    eye = torch.eye(n_class, device=device) 
    return eye[label].view(-1, n_class, 1, 1)   
 
def concat_image_label(image, label, device, n_class=n_class):
    B, C, H, W = image.shape   
    oh_label = onehot_encode(label, device=device)
    oh_label = oh_label.expand(B, n_class, H, W)
    return torch.cat((image, oh_label), dim=1)
 
def concat_noise_label(noise, label, device):
    oh_label = onehot_encode(label, device=device)
    return torch.cat((noise, oh_label), dim=1)
        
def train_ACGAN(device, trainloader, batch_size):
  image_size = 32
  netD = ACDiscriminator().to(device)
  netD.apply(weights_init)
  netG = ACGenerator().to(device)
  netG.apply(weights_init)
  s_criterion = nn.BCELoss()
  c_criterion = nn.NLLLoss()

  r_label = 0.7
  f_label = 0
  
  input = torch.tensor([batch_size, nc, image_size, image_size], device=device)
  noise = torch.tensor([batch_size, nz, 1, 1], device=device)
  
  fixed_noise = torch.randn(1, nz, 1, 1, device=device)
  fixed_label = torch.randint(0, n_class, (1, ), device=device)
  fixed_noise_label = concat_noise_label(fixed_noise, fixed_label, device)
  lr = 0.001
  beta1 = 0.5
  optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
  optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
  
  G_losses = []
  D_losses = []
  iters = 0
  num_epochs = 30

  print("Starting Training Loop...")
  for epoch in range(num_epochs):
      for i, data in enumerate(trainloader, 0):
          # prepare real image and label
          real_label = data[1].to(device)
          real_image = data[0].to(device)
          b_size = real_label.size(0)     
          
          # prepare fake image and label
          fake_label = torch.randint(n_class, (batch_size,), dtype=torch.long, device=device)
          noise = torch.randn(b_size, nz, 1, 1, device=device).squeeze(0)
          noise = concat_noise_label(noise, real_label, device)  
          fake_image = netG(noise)
          
          # target
          real_target = torch.full((b_size,), r_label, device=device)
          fake_target = torch.full((b_size,), f_label, device=device)
          
          #-----------------------
          # Update Discriminator
          #-----------------------
          netD.zero_grad()
          
          # train with real
          s_output, c_output = netD(real_image)
          
          s_errD_real = s_criterion(s_output.squeeze(), real_target)  # realfake
          c_errD_real = c_criterion(c_output, real_label)  # class
          errD_real = s_errD_real + c_errD_real
          errD_real.backward()
          D_x = s_output.data.mean()
  
          # train with fake
          s_output,c_output = netD(fake_image.detach())
          s_errD_fake = s_criterion(s_output.squeeze(), fake_target)  # realfake
          c_errD_fake = c_criterion(c_output, real_label)  # class
          errD_fake = s_errD_fake + c_errD_fake
          errD_fake.backward()
          D_G_z1 = s_output.data.mean()
          
          errD = s_errD_real + s_errD_fake
          optimizerD.step()        
  
          #-----------------------
          # Update Generator
          #-----------------------
          netG.zero_grad()
          
          s_output,c_output = netD(fake_image)
          s_errG = s_criterion(s_output, real_target)  # realfake
          c_errG = c_criterion(c_output, real_label)  # class
          errG = s_errG + c_errG
          errG.backward()
          D_G_z2 = s_output.data.mean()
          
          optimizerG.step()
  
          # Save Losses for plotting later
          G_losses.append(errG.item())
          D_losses.append(errD.item())
          
          iters += 1
  
      # scheduler.step(errD.item())
      
      print('[%d/%d][%d/%d]\nLoss_D: %.4f\tLoss_G: %.4f\nD(x): %.4f\tD(G(z)): %.4f / %.4f'
            % (epoch+1, num_epochs, i+1, len(dataloader),
               errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
      
      show_generated_img(num_show)
           

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 10

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    device = torch.device('cuda:15' if torch.cuda.is_available() else 'cpu')
    train_DCGAN(device, batch_size, trainloader)
    train_WGAN(device, trainloader)
    #train_ACGAN(device, trainloader, batch_size)

main()
