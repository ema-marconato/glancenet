import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model  import LinearRegression, LogisticRegression, Lasso


def plot_train_scores(path, kind='VAE', n_latents=7):
    data = pd.read_csv(path+'train_runs/metrics.csv')
    iters = data['iter']
    epoch = data['epoch']
    
    
    x = iters+np.max(iters)*(epoch-1)
    


    for i, value in enumerate(data['latent_error']):
        if value == -1:
            if i == 0: data['latent_error'][0] =100
            else:
                data['latent_error'][i] = data['latent_error'][i-1]


    fig = plt.figure(figsize=(10,10))
    fig.suptitle('TRAIN SCORES')
    fig.add_subplot(2,1,1)
    
    if kind=='VAE':
        recon = data['reconstruction_error']
        kld = data['kld']
    else: 
        recon = np.zeros(len(iters))
        kld = np.zeros(len(iters))
    
    plt.plot(x,recon , label='reconstruction')
    plt.plot(x, kld, label='kld')
    plt.plot(x, data['latent_error'], label='latent')
    plt.plot(x, data['classification_error'], label='Cross-Entropy classification')
    
    
    top_acc =False
    if top_acc is not None:
        plt.plot(x, np.ones(len(x))*top_acc, label='Cross-Entropy classification' )

#    plt.xlim(-1000,5*10**4)
    plt.legend()

    fig.add_subplot(2,1,2)
    lines = ['-', '-.', '--']
    for i in range(n_latents):
        plt.plot(x, data['latent%i'%i].rolling(2).mean(), label='latent %i'%i, linestyle=lines[i%3])
        
        
#        plt.plot(x, data['latent%i'%i], label='latent %i'%i, linestyle=lines[i%3])        
        #plt.legend()

    
    
def plot_disentanglement_scores(path):
    data = pd.read_csv(path+'eval_results/dis_metrics.csv')
    x = np.linspace(0,1, len(data))
    print('Data keys:', data.keys())
    fig = plt.figure(figsize=(14,8))
    fig.suptitle('DISENTANGLEMENT SCORES')

    colors= ['red', 'blue', 'green', 'orange', 'purple', 'black']
    for i, key in enumerate(data.keys()):
        fig.add_subplot(2,3,i+1)
        y = data[key]
        plt.plot(x, y, label=key, color=colors[i])
        plt.legend()
    plt.show()
    
    
def plot_test_scores(path, kind='VAE', n_latents=7):
    ## TEST-SET ERRORS
    data = pd.read_csv(path+'eval_results/test_metrics.csv')
    
    iters = data['iter']/1620
    Accuracy = data['Acc']
    BCE = data['BCE']
    latent = data['latent']
    
    if kind=='VAE':
        recon = data['rec']
        kld = data['kld']
    else:
        recon = np.zeros(len(iters))
        kld = np.zeros(len(iters))

    f = plt.figure(figsize=(10,9))
    f.suptitle('TEST SCORES')

    f.add_subplot(2,3,1)
    plt.plot(iters, recon, label='recon', color='blue')
    plt.legend()

    f.add_subplot(2,3,2)
    plt.plot(iters, kld, label='kld', color='black')
    plt.legend()

    
    f.add_subplot(2,3,3)
    plt.plot(iters, latent, label='latent error', color='red')
    plt.legend()


    f.add_subplot(2,3,4)
    plt.plot(iters, BCE, label='CE prediction', color='gold')
    plt.legend()
    
    
    f.add_subplot(2,3,5)
    lines = ['-', '-.', '--']
    for i in range(n_latents):
        plt.plot(iters, data['latent%i'%i].rolling(2).mean(), label='latent %i'%i, linestyle=lines[i%3])
     
    
def plot_val_scores(path, kind='VAE', n_latents=7):
    ## TEST-SET ERRORS
    data = pd.read_csv(path+'train_runs/val_metrics.csv')

    iters = data['epoch']
    Accuracy = data['acc']
    BCE = data['bce']
    latent = data['latent']
    if kind=='VAE':
        recon = data['rec']
        kld = data['kld']
    else:
        recon = np.zeros(len(iters))
        kld = np.zeros(len(iters))

    f = plt.figure(figsize=(10,9))
    f.suptitle('VAL SCORES')

    f.add_subplot(2,2,1)
    plt.plot(iters, recon, label='recon', color='blue')
    plt.legend()

    f.add_subplot(2,2,2)
    plt.plot(iters, kld, label='kld', color='black')
    plt.legend()

    
    f.add_subplot(2,2,3)
    plt.plot(iters, latent, label='latent error', color='red')
    plt.legend()
    
    f.add_subplot(2,2,4)
    plt.plot(iters, BCE, label='CE prediction', color='gold')
    plt.legend()
    
  
    
    
def create_categories(z, g, all_labels):
    # pass a list of lists
    dec_z, dec_g = [], []

    l = len(all_labels)

    for labels in all_labels:

        enc = OneHotEncoder()
        range_labels = [[i] for i in range(len(labels))]
        encoder = enc.fit(range_labels)

        dec_z.append(encoder.inverse_transform(z[:,labels]))
        dec_g.append(encoder.inverse_transform(g[:,labels]))
        
    all_labels = sum(all_labels, [])
    
    if  l==0:
        pass

    elif l==1: 
        z = np.delete(z, all_labels, 1) 
        g = np.delete(g, all_labels, 1)

        dec_z = np.array(dec_z).reshape(-1,1)
        dec_g = np.array(dec_g).reshape(-1,1)

        z = np.concatenate((dec_z , z ), axis=1 )
        g = np.concatenate((dec_g , g ), axis=1 )

    else:
        z = np.delete(z, all_labels, 1) 
        g = np.delete(z, all_labels, 1)

        dec_z = np.array(dec_z).T
        dec_g = np.array(dec_g).T

        z = np.concatenate((dec_z, z ), axis=1 )
        g = np.concatenate((dec_g, g ), axis=1 )
    return z, g
    

def DCI(z, g, n_gens=40, all_labels=[[0,1,2],], rel_factors=10 ** 4, verbose=False):
    '''
    Compute the DCI scores of the latent factors given the generative ones.
    It can be done on a restricted number of entries.
    '''

    # We transform first three dimensions into one categorical
    
    z, g = create_categories(z, g, all_labels)

    D = len(z[0])
    K = len(g[0])

    coeff = np.zeros(shape=(D,K))
    MSE = 0
    score = 0
    for i in range(K):
        if i < len(all_labels):
            model = Lasso(fit_intercept=True, alpha=0.01)
        else:
            model =  LogisticRegression(fit_intercept=True, penalty='l1', solver='liblinear', C=100) #Cs=0.1,
           
           
        
        model.fit(z, g[:, i], )
        
        score += model.score(z,g[:,i]) / K
        
        coeff[:,i] = model.coef_
        mse = np.mean( (model.predict(z) - g[:,i])**2 )
        MSE += mse / K
        
    
    R = np.abs(coeff) + 10**-7
    if verbose:
        
        print('# Total accuracy of Regressor model:', score )

        print('Shape of R', np.shape(R))
    
    ## DISENTANGLEMENT ##
    
    R_z = np.sum(R, axis=1)
    
    P = np.zeros(np.shape(R))
    for i in range(D):
        for j in range(K):
            P[i, j] = R[i, j] / (R_z[i])#  + 10 ** -6)

    H_K = np.zeros(D)
    for j in range(D):
        for k in range(K):
            H_K[j] -=  (P[j,k] * np.log(P[j,k]))
    

    DIS = 1 - H_K / np.log(K)


    rho = np.sum(R, axis=1) / np.sum(R)# + 10 ** -6)
    DIS_tot = np.sum(rho * DIS)

    np.set_printoptions(suppress=True)

    if verbose: print('Disentanglement score:', DIS_tot)

    
    ## INTERPRETABILITY ##
    R_g = np.sum(R, axis=0)
    
    P = np.zeros(np.shape(R))
    for i in range(D):
        for j in range(K):
            P[i, j] = R[i, j] / (R_g[j])#  + 10 ** -6)

    H_D = np.zeros(K)
    for j in range(K):
        for d in range(D):
            H_D[j] -=  (P[d, j] * np.log(P[d, j]))


    I = 1 - H_D / np.log(D)


    rho = np.sum(R, axis=0) / np.sum(R)# + 10 ** -6)
    I_tot = np.sum(rho * I)
    if verbose: print('Completeness score:', I_tot)
    
    E = 1 - 6* MSE
    if verbose: print('Explicitness score:', E)
    return I, I_tot, R, DIS, DIS_tot, E
