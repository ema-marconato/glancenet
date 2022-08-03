from contextlib import suppress
from grpc import method_handlers_generic_handler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

def vae_train(vae, optimizer, train_loader,test_loader ):
    vae.train()
    train_loss = 0
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.8)

    writer = SummaryWriter(log_dir='tensorboard_runs/VAE')
    for epoch in range(51):
        for batch_idx, (data, y) in enumerate(train_loader):
            ## BIN VERSION
            mask =  (y == 4) | (y == 5) #(y != 0) & (y != 1) & (y != 2) & (y != 3)  & (y != 4) & (y != 5)
            data = data[mask].cuda()
            y = y[mask].cuda()

            optimizer.zero_grad()

            recon_batch, mu, log_var, z = vae(data)
            pred, _ = vae.predict(z[:,:2])

            loss, losses = vae.loss_function(recon_batch, data, mu, log_var, z, pred, y)
            
            for key in losses.keys():
                writer.add_scalar("%s"%key, losses[key], epoch)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item() / len(data)))
        writer.flush()
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
        lr_scheduler.step()
        vae_test(vae, test_loader)
    writer.close()
    return vae

def vae_test(vae, test_loader):
    vae.eval()
    test_loss = 0
    test_dict = {'rec': 0, 'kld': 0, 'pred': 0, 'lat': 0}
    with torch.no_grad():
        for data, y in test_loader:
            mask =   (y == 4) | (y == 5) #(y != 0) & (y != 1) & (y != 2) & (y != 3)  & (y != 4) & (y != 5)
            data = data[mask].cuda()
            y = y[mask].cuda()
            recon, mu, log_var, z = vae(data)
            pred, _ = vae.predict(z[:,:2])

            # sum up batch loss
            p_loss, l_dict = vae.loss_function(recon, data, mu, log_var, z, pred, y)
            test_dict['rec'] += l_dict['rec'].item() / len(test_loader.dataset)
            test_dict['kld'] += l_dict['kld'].item() / len(test_loader.dataset)
            test_dict['lat'] += l_dict['lat'].item() / len(test_loader.dataset)
            test_dict['pred'] += l_dict['pred'].item() / len(test_loader.dataset)
            test_loss += p_loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print(test_dict)


def cbm_train(cbm, optimizer, train_loader, test_loader):
    cbm.train()
    train_loss = 0
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.75)

    writer = SummaryWriter(log_dir='tensorboard_runs/CBM')
    for epoch in range(21):
        for batch_idx, (data, y) in enumerate(train_loader):

            ## BIN VERSION
            mask = (y == 4) | (y == 5)  #(y != 0) & (y != 1) & (y != 2) & (y != 3)  & (y != 4) & (y != 5)
            data = data[mask].cuda()
            y = y[mask].cuda()
            
            optimizer.zero_grad()

            mu = cbm(data)
            pred, _ = cbm.predict(mu)

            loss, losses = cbm.loss_function(data, mu, pred, y)
            
            for key in losses.keys():
                writer.add_scalar("%s"%key, losses[key], epoch)
                
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item() / len(data)))
        writer.flush()
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
        lr_scheduler.step()
        cbm_test(cbm, test_loader)
    writer.close()
    return cbm

def cbm_test(cbm, test_loader):
    cbm.eval()
    test_loss = 0
    test_dict = {'rec': 0, 'kld': 0, 'pred': 0, 'lat': 0}
    with torch.no_grad():
        for data, y in test_loader:
            mask =   (y == 4) | (y == 5)  #(y != 0) & (y != 1) & (y != 2) & (y != 3)  & (y != 4) & (y != 5)
            data = data[mask].cuda()
            y = y[mask].cuda()
            mu = cbm(data)
            pred, _ = cbm.predict(mu)

            # sum up batch loss
            p_loss, l_dict = cbm.loss_function(data, mu, pred, y)

            test_dict['lat'] += l_dict['lat'].item() / len(test_loader.dataset)
            test_dict['pred'] += l_dict['pred'].item() / len(test_loader.dataset)
            test_loss += p_loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print(test_dict)


def osr_vae_train(osr_vae, optimizer, train_loader, test_loader):
    osr_vae.train()
    train_loss = 0
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.8)

    writer = SummaryWriter(log_dir='tensorboard_runs/OSR_VAE')
    for epoch in range(51):
        for batch_idx, (data, y) in enumerate(train_loader):
            ## BIN VERSION
            mask =  (y == 4) | (y == 5) 
            data = data[mask].cuda()
            y = y[mask].cuda()
            
            optimizer.zero_grad()
            
            recon_batch, mu, log_var, z = osr_vae(data)
            pred, _ = osr_vae.predict(z[:,:2])

            loss, losses = osr_vae.loss_function(recon_batch, data, mu, log_var, z=z,  pred=pred, y=y )
            
            for key in losses.keys():
                writer.add_scalar("%s"%key, losses[key], epoch)
            
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
        writer.flush()
        osr_vae_test(osr_vae, test_loader)
        if epoch > 6:
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
        lr_scheduler.step()
    writer.close()
    return osr_vae

def osr_vae_test(osr_vae, test_loader):
    osr_vae.eval()
    test_loss = 0
    test_dict = {'rec': 0, 'kld': 0, 'pred': 0, 'lat': 0}
    with torch.no_grad():
        for data, y in test_loader:
            mask =   (y == 4) | (y == 5) #(y != 0) & (y != 1) & (y != 2) & (y != 3)  & (y != 4) & (y != 5)
            data = data[mask].cuda()
            y = y[mask].cuda()

            #y_pass = torch.zeros(size=y.size()).cuda()
            #y_pass[y == 4] = torch.zeros(size=y.size(), dtype=torch.long)[y==4].cuda() 
            #y_pass[y == 5] = torch.ones(size=y.size(),  dtype=torch.long)[y==5].cuda()

            #y_hot = F.one_hot(y-4, num_classes=2).cuda()
            #mu_y = osr_vae.z_enc(y_hot.to(torch.float))

            recon, mu, log_var, z = osr_vae(data)
            pred, _ = osr_vae.predict(z[:,:2])

            # sum up batch loss
            p_loss, l_dict = osr_vae.loss_function(recon, data, mu, log_var, z=z,  pred=pred, y=y )

            test_dict['rec'] += l_dict['rec'].item() / len(test_loader.dataset)
            test_dict['kld'] += l_dict['kld'].item() / len(test_loader.dataset)
            test_dict['lat'] += l_dict['lat'].item() / len(test_loader.dataset)
            test_dict['pred'] += l_dict['pred'].item() / len(test_loader.dataset)
            test_loss += p_loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print(test_dict)




def train_parity(epoch, model, optimizer, dataloader, name='vae' ):
    model.train()

    train_loss = 0
    accuracy = 0

    counter = torch.zeros(10)

    for batch_idx, (data,y) in enumerate(dataloader):

        mask =  (y == 4) | (y == 5)

        data = data[~mask].cuda()
        y = y[~mask].cuda()

        optimizer.zero_grad()
        if name == 'vae':
            recon, mu, log_var, z = model(data, n_samples=1000)
            zs = []
            for latent in z:
                zs.append(latent[:,:2])
            pred = model.predict(zs, sampling=True)
            loss, _ = model.loss_function(recon, data, mu, log_var, zs, pred, y, only_class=True)
        elif name == 'cbm':
            mu = model(data)
            pred = model.predict(mu)
            loss, _ = model.loss_function(data, mu, pred, y, only_class=True)
        else:
            NotImplementedError('Wrong')
        
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Iter ',batch_idx,' Loss:', loss.item() )

        
    return model

def test_parity(model, test_loader, name='vae'):
    bs = test_loader.batch_size
    accuracy = 0
    log_vars = []
    for index, (x,y) in enumerate(test_loader):

        mask =(y == 4) | (y == 5) 

        x  = x[~mask].cuda()
        y = y[~mask].cuda()
        if name == 'vae':
            _, _, log_var, z = model(x, n_samples=100)
            log_vars.append(log_var)
            zs = []
            for latent in z:
                zs.append(latent[:,:2])
            preds = model.predict(zs, sampling=True)
            probs = []
            for pred in preds:
                probs.append(nn.Softmax(dim=1)(pred)[:,1])

            
        elif name == 'cbm':
            z = model(x)
            pred = model.predict(z)            
            probs = [nn.Softmax(dim=1)(pred)[:,1]]

        # CALCULATE PROB
        accuracy_term = 0
        tot = 0
        for prob in probs:

            for j, p in enumerate(prob):
                tot += 1
                if p > 0.5: res = 1
                else: res = 0

                if y[j] % 2 == 0: y_true = 0
                else: y_true = 1
                
                if res == y_true: 
                    accuracy_term += 1

            accuracy += accuracy_term/tot
    accuracy /= len(probs)
    torch.set_printoptions(2, sci_mode=False)

    print('# TEST -> Overall accuracy:', accuracy / (index + 1))

    if name=='vae':
        mean_var = torch.zeros(10).cuda()
        mean_log_var = torch.zeros(10).cuda()
        for log_var in log_vars:
            mean_log_var += torch.mean(log_var, dim=0)
            mean_var += torch.mean( torch.exp(log_var)/2, dim=0)
        print('Log expectations:', mean_log_var)
        print('Expectations:', mean_var)
    
    return accuracy / (index + 1)


def vae_leakage(epoch, model, optimizer, dataloader, name='vae' ):
    model.train()

    train_loss = 0
    accuracy = 0

    counter = torch.zeros(10)

    for batch_idx, (data,y) in enumerate(dataloader):

        mask = (y == 4) | (y == 5) 

        data = data[~mask].cuda()
        y = y[~mask].cuda()

        optimizer.zero_grad()
        if name == 'vae':
            recon, mu, log_var, z = model(data)
            pred, prob = model.leak_classifier(z)
            loss, _ = model.loss_function(recon, data, mu, log_var, z, pred, y, only_class=True)
        
        loss.backward()

        optimizer.step()



        # CALCULATE PROB
        accuracy_term = 0
        tot = 0
        for j, p in enumerate(prob):
            tot += 1
            if p > 0.5: res = 0
            else: res = 1

            if y[j] % 2 == 0: y_true = 0
            else: y_true = 1
            
            if res == y_true: 
                accuracy_term += 1

        #print('Accuracy', accuracy_term)
        #print('tot', tot)
        accuracy += accuracy_term/tot

        
    return model