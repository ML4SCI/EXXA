import torch
from torch import optim
from tqdm.auto import tqdm

def train_fn(model, train_loader, optimizer, device, criterion=None):
    model.train().to(device)
    it_loss = 0
    counter = 0
    for i, data in enumerate(tqdm(train_loader,total = len(train_loader))):
        imgs = data['img_data'].to(device)
        img_path = data['img_path']
        optimizer.zero_grad()
        
        if (criterion is None):
            mse_loss, ssim_loss, _, _ = model(imgs, img_path)
            loss = mse_loss + (1 - ssim_loss)
        else:
            reconstructed, _ = model(imgs)
            loss, _, _ = criterion(reconstructed, imgs)
            
        loss.backward()
        optimizer.step()
        it_loss += loss.item() * imgs.shape[0]
        counter += imgs.shape[0]
    return it_loss/counter

def test_fn(model, test_loader, device, criterion=None):
    model.eval().to(device)
    it_mse = 0
    it_ssim = 0
    counter = 0
    for i, data in enumerate(tqdm(test_loader,total = len(test_loader))):
        imgs = data['img_data'].to(device)
        img_path = data['img_path']
        with torch.no_grad():
            if (criterion is None):
                mse_loss, ssim_loss, _, _ = model(imgs, img_path)
            else:
                reconstructed, _ = model(imgs)
                _, mse_loss, ssim_loss = criterion(reconstructed, imgs)
                
            it_mse += mse_loss.item() * imgs.shape[0]
            it_ssim += ssim_loss.item() * imgs.shape[0]
            counter += imgs.shape[0]
    return it_ssim/counter, it_mse/counter

def run_epochs(model, train_loader, test_loader, save_dir, 
               epochs=30, lr=1.5e-4, optimizer='adam', scheduler=None, 
               weight_decay=0.01, criterion=None, data_type='images'):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if (optimizer == 'adam'):
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif (optimizer == 'adamw'):
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif (optimizer == 'radam'):
        optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay = weight_decay)
    elif (optimizer == 'sgd'):
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = weight_decay)
    elif (optimizer == 'rmsprop'):
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay = weight_decay)
    else:
        print("Unknown optimzer, pass optimizer object instead for this")
        return 

    
    if (scheduler.lower() == 'reducelronplateau'):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    elif (scheduler.lower() == 'cosineannealinglr'):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
    elif (scheduler.lower() == 'exponentiallr'):
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        print("Unknown scheduler, pass scheduler object instead for this")
        return 

    if (criterion is not None and criterion.lower() == 'hybridloss'):
        criterion = HybridLoss(data_type)

    best_loss = float('inf')
    best_ssim = 0.0
    best_mse = float('inf')
    for epoch in range(epochs):
        train_loss = train_fn(model, train_loader, optimizer, device, criterion)
        test_ssim, test_mse = test_fn(model, test_loader, device, criterion)
        total_loss = (1 - test_ssim) + test_mse
        if (scheduler is not None):
            scheduler.step(total_loss)

        if (total_loss < best_loss):
            best_loss = total_loss
            best_ssim = test_ssim
            best_mse = test_mse
            torch.save({
                'model_dict':model.state_dict(),
                'optimizer_dict':optimizer.state_dict()
            }, save_dir)

            print('MODEL SAVED!')
        print(f'Epoch {epoch+1}, Train loss: {train_loss:.6f}, Test MSE : {test_mse:.6f}, Test MS-SSIM : {test_ssim:.6f}')
    return best_ssim, best_mse, save_dir