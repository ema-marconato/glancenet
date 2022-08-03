from cProfile import label
import os.path
import torch
from torch import nn
import torch.optim as optim
from models.vae import VAE
from models.vae import VAEModel
from architectures import encoders, decoders
from common.ops import reparametrize
from common.utils import Accuracy_Loss, Interpretability
from common import constants as c
import torch.nn.functional as F
from common.utils import is_time_for
import logging

import numpy as np
import pandas as pd


class GlanceNet(VAE):
    """
    Graybox version of VAE, with standard implementation. The discussion on
    """

    def __init__(self, args):

        super().__init__(args)

        print('Initialized GrayVAE_Join model')

        # checks
        assert self.num_classes is not None, 'please identify the number of classes for each label separated by comma'

        # encoder and decoder
        encoder_name = args.encoder[0]
        decoder_name = args.decoder[0]

        encoder = getattr(encoders, encoder_name)
        decoder = getattr(decoders, decoder_name)

        # number of channels
        image_channels = self.num_channels
        input_channels = image_channels
        decoder_input_channels = self.z_dim

        ## add classification layer
        if args.z_class != 0:
            self.z_class = args.z_class
        else:
            self.z_class = self.z_dim    
        self.n_classes = args.n_classes
        self.classification_epoch = args.classification_epoch #TODO REMOVE THIS SHIT
        self.reduce_rec = args.reduce_recon                   #TODO REMOVE THIS SHIT

        # model and optimizer
        self.model = VAEModel(encoder(self.z_dim, input_channels, self.image_size),
                               decoder(decoder_input_channels, self.num_channels, self.image_size),
                               ).to(self.device)
                               
        self.classification = nn.Linear(args.z_class, args.n_classes, bias=True).to(self.device) ### CHANGED OUT DIMENSION
        if args.conditional_prior:
            self.push_cluster = True
            self.enc_z_from_y = nn.Linear(args.n_classes, args.z_dim, bias=True).to(self.device)
            self.optim_G = optim.Adam([*self.model.parameters(), 
                                       *self.classification.parameters(), 
                                       *self.enc_z_from_y.parameters()],
                                      lr=self.lr_G, betas=(self.beta1, self.beta2))
        else:    
            self.optim_G = optim.Adam([*self.model.parameters(), *self.classification.parameters()],
                                      lr=self.lr_G, betas=(self.beta1, self.beta2))

        # update nets
        self.net_dict['G'] = self.model
        self.optim_dict['optim_G'] = self.optim_G

        self.setup_schedulers(args.lr_scheduler, args.lr_scheduler_args,
                              args.w_recon_scheduler, args.w_recon_scheduler_args)        

        ## CHOOSE THE WEIGHT FOR CLASSIFICATION
        self.label_weight = args.label_weight
        
        ## CHOOSE THE WEIGHT FOR LATENTS
        if args.latent_weight is None:
            self.latent_weight = args.label_weight 
        else:
            self.latent_weight = args.latent_weight
        self.masking_fact = args.masking_fact
        self.latent_loss = args.latent_loss
        
        ## OTHER STUFF
        self.show_loss = args.show_loss
        self.wait_counter = 0
        self.save_model = True
        self.is_VAE = True

        self.dataframe_dis = pd.DataFrame() #columns=self.evaluation_metric)
        self.dataframe_eval = pd.DataFrame()
        self.validation_scores = pd.DataFrame()
## OSR MECHANISM
        self.zy = torch.zeros(size=(args.n_classes, args.z_dim), device=self.device)
        self.thr_rec = torch.zeros(size=(), device=self.device)
        self.thr_y = torch.zeros(size=(args.n_classes,), device=self.device)
        
        
    def update_osr(self):
        # to evaluate performances on disentanglement
        
        recons = torch.zeros(len(self.data_loader.dataset), device=self.device)
        z = torch.zeros(len(self.data_loader.dataset), self.z_dim, device=self.device)
        y = torch.zeros(len(self.data_loader.dataset), device=self.device)
    
        for internal_iter, (x_true1, label1, y_true1, examples) in enumerate(self.data_loader):
            
            x_true1 = x_true1.to(self.device)
            y_true1 = y_true1.to(self.device, dtype=torch.long)

            if self.dset_name == 'dsprites_full':
                label1 = label1[:, 1:].to(self.device)
            else:
                label1 = label1.to(self.device)

            losses, params = self.vae_classification(losses, x_true1, label1, y_true1, examples)

            ## ADD FOR EVALUATION PURPOSES
            z[internal_iter*self.batch_size:(internal_iter+1)*self.batch_size, :] = params['z']
            y[internal_iter*self.batch_size:(internal_iter+1)*self.batch_size] = y_true1
 
            for i in range(self.batch_size):
                x_recon = params['x_recon']
                x_true = x_true1
                rec = F.binary_cross_entropy(input=x_recon[i], target=x_true[i],reduction='sum') / self.w_recon
                recons[i + internal_iter*self.batch_size] = rec
            
        ## EVALUATE THE RECON THR ##    
        print('## Updating thr in rec')
        l = len(recons)
        r_min, r_max = torch.min(recons).item(), torch.max(recons).item()
        good_r = []
        
        for eta in np.linspace(r_min, r_max, 1000):
            mask = (recons < eta)
            if len( recons[mask] )/l > 0.945 and len( recons[mask] )/l < 0.955:
                good_r.append(eta)
        self.thr_rec = torch.mean(good_r).to(self.device)
        
        print('Updated the threshold on reconstruction')
        
        
        ## EVALUATE THE zy THR ##
        print('## Updating thr in latent representations')

        clusters = self.enc_z_from_y(torch.eye(self.n_classes, dtype=torch.float, device=self.device))
        good_dist = []

        for yclass in range(self.n_classes):
            mask = (y == yclass)
            l = len(y[mask])
            
            dist = [clusters[yclass] - k for k in z[mask] ]
            dist = np.array([torch.norm(value[:self.z_class]).item() for value in dist])
            
            dmin = np.min(dist)
            dmax = np.max(dist)
            
            for eta in np.linspace(dmin, dmax, 1000):
                conds = dist < eta 
            
                if conds.sum()/l > 0.945 and  conds.sum()/l < 0.955:
                    good_dist.append(eta)
            self.thr_y[yclass] = torch.mean(good_dist).to(device=self.device)

        print('Updated the threshold on latent representations')
        
        ## SAVE INFO TO FOLDER
        eta_rec  = pd.DataFrame.from_dict({'thr_rec': self.thr_rec.item() })
        eta_y    = pd.DataFrame(colums=['thr_y'], data=self.thr_y.detach().cpu().numpy())
        
        eta_rec.to_csv(os.path.join(self.out_path+'/train_runs', 'thr_rec.csv'), index=False)
        eta_y.to_csv(os.path.join(self.out_path+'/train_runs',   'thr_y.csv'), index=False)
        
            
    def rejection_mech(self, recon, z):
        mask_rec = (recon < self.thr_rec)
        
        mask_zy = torch.zeros(len(mask_rec), dtype=torch.bool, device=self.device )
  
        clusters = self.enc_z_from_y(torch.eye(self.n_classes, dtype=torch.float, device=self.device))
        
        for i in range(self.n_classes):
            
            #EVALUATE DISTANCE
            dist = [clusters[i] - k for k in z]
            dist = np.array([torch.norm(value[:self.z_class]).item() for value in dist])

            mask_zy = mask_zy | (dist < self.thr_y[i])
            
        return mask_rec & mask_zy
    
    def reject_test_set(self):
        
        passed_test_samples = torch.zeros(dim=())    
        for internal_iter, (x_true, label, y_true, _) in enumerate(self.test_loader):
            x_true = x_true.to(self.device)

            if self.dset_name == 'dsprites_full':
                label = label[:, 1:].to(self.device)
            else:
                label = label.to(self.device, dtype=torch.float)
            
            y_true =  y_true.to(self.device, dtype=torch.long)

            mu, logvar = self.model.encode(x=x_true, )
            
            z = reparametrize(mu, logvar)
            x_recon = self.model.decode(z=z,)   
            
            mask =self.rejection_mech(x_recon,z )
            passed_test_samples.cat(mask)
             
        return passed_test_samples

    def predict(self, **kwargs):
        """
        Predict the correct class for the input data.
        """
        input_x = kwargs['latent'].to(self.device)
        pred_raw = self.classification(input_x)
        pred = nn.Softmax(dim=1)(pred_raw)
        return  pred_raw, pred.to(self.device, dtype=torch.float32) #nn.Sigmoid()(self.classification(input_x).resize(len(input_x)))

    def reparametrize_many(self, mu, log_var, how_many=100):
        
        zs = []
        for i in range(how_many):
            z = reparametrize(mu, log_var)
            zs.append(z)
        return zs 
    
    def vae_classification(self, losses, x_true1, label1, y_true1, examples):

        mu, logvar = self.model.encode(x=x_true1,)

        z = reparametrize(mu, logvar)
        
        mu_processed = torch.tanh(z/2)
        
        x_recon = self.model.decode(z=z,)

        prediction, forecast = self.predict(latent=mu_processed[:,:self.z_class])
        rn_mask = (examples==1)
        n_passed = len(examples[rn_mask])

        loss_fn_args = dict(x_recon=x_recon, x_true=x_true1, mu=mu, logvar=logvar, z=z)
        if self.conditional_prior:
            y_onehot = F.one_hot(y_true1, self.n_classes).to(dtype=torch.float, device=self.device)
            mu_cluster = self.enc_z_from_y(y_onehot)     
               
            loss_fn_args.update(mu_target=mu_cluster)
            # PUSH THE CLUSTER CLOSER TO THE 
            concepts = torch.tanh(mu_cluster / 2)
            if self.cluster_dim == 0:
                cluster_dims = self.z_dim
            else:
                cluster_dims = self.cluster_dim
            
            if self.push_cluster and self.latent_loss=='MSE':
                loss_bin =  nn.MSELoss(reduction='mean')( concepts[rn_mask][:, :cluster_dims ], 2*label1[:,:cluster_dims][rn_mask]-1  ) * 100
            if self.push_cluster and self.latent_loss=='BCE':
                loss_bin = nn.BCELoss(reduction='mean')((1+concepts[rn_mask][:, :cluster_dims ])/2, label1[:, :cluster_dims ][rn_mask] ) * 100
            else:
                loss_bin = torch.tensor(0, dtype=torch.float, device=self.device)


        loss_dict = self.loss_fn(losses, reduce_rec=False, **loss_fn_args)
        losses.update(loss_dict)

        pred_loss = nn.CrossEntropyLoss(reduction='mean')(prediction, y_true1) *self.label_weight  # her efor celebA
        losses.update(prediction=pred_loss)
        losses[c.TOTAL_VAE] += pred_loss
        
        del loss_dict, pred_loss

        if n_passed > 0: # added the presence of only small labelled generative factors
            if self.latent_loss == 'MSE':                
                loss_bin += nn.MSELoss(reduction='mean')( mu_processed[rn_mask][:, :label1.size(1)], 2*label1[rn_mask]-1  )
                ## track losses
                err_latent = []
                for i in range(label1.size(1)):
                    err_latent.append(nn.MSELoss(reduction='mean')(mu_processed[rn_mask][:, i], 2 * label1[rn_mask][:,i] - 1).detach().item() )
                losses.update(true_values=self.latent_weight * loss_bin)
                losses[c.TOTAL_VAE] += self.latent_weight * loss_bin

            elif self.latent_loss == 'BCE':
                loss_bin += nn.BCELoss(reduction='mean')((1+mu_processed[rn_mask][:, :label1.size(1)])/2,
                                                            label1[rn_mask] )
                ## track losses
                err_latent = []
                for i in range(label1.size(1)):
                    err_latent.append(nn.BCELoss(reduction='mean')((1+mu_processed[rn_mask][:, i])/2,
                                                                    label1[rn_mask][:, i] ).detach().item())
                losses.update(true_values=self.latent_weight * loss_bin)
                losses[c.TOTAL_VAE] += self.latent_weight * loss_bin

            else:
                raise NotImplementedError('Not implemented loss.')

        else:
            losses.update(true_values=torch.tensor(-1))
            err_latent =[-1]*label1.size(1)

        return losses, {'x_recon': x_recon, 'mu': mu, 'z': z, 'logvar': logvar, "prediction": prediction,
                        'forecast': forecast, 'latents': err_latent, 'n_passed': n_passed}

    def train(self, **kwargs):

        if 'output'  in kwargs.keys():
            out_path = kwargs['output']
            track_changes=True
            self.out_path = out_path #TODO: Not happy with this thing
            print("## Initializing Train indexes")
            print("->path chosen::",out_path)

        else: track_changes=False;
            
        ## SAVE INITIALIZATION ##
        #self.save_checkpoint()

        Iterations, Epochs, Reconstructions, KLDs, True_Values, Accuracies, F1_scores = [], [], [], [], [], [], []  ## JUST HERE FOR NOW
        latent_errors = []
        epoch = 0
        max_lr = np.log10(self.optim_G.param_groups[0]['lr']) 
        lr_log_scale = np.logspace(-7, max_lr, 5)
        
        while not self.training_complete():
            # added annealing
            if epoch < 5:
                self.optim_G.param_groups[0]['lr'] = lr_log_scale[epoch] 
            print('lr:',  self.optim_G.param_groups[0]['lr'])
            epoch += 1
            self.net_mode(train=True)
            vae_loss_sum = 0
            # add the classification layer #
            if epoch>self.classification_epoch:
                print("## STARTING CLASSIFICATION ##")
                start_classification = True
            else: start_classification = False
            
            # to evaluate performances on disentanglement
            z = torch.zeros(len(self.data_loader.dataset), self.z_dim, device=self.device)
            g = torch.zeros(len(self.data_loader.dataset), self.z_dim, device=self.device)
            
            for internal_iter, (x_true1, label1, y_true1, examples) in enumerate(self.data_loader):

                losses = {'total_vae':0}

                x_true1 = x_true1.to(self.device)
                y_true1 = y_true1.to(self.device, dtype=torch.long)

                if self.dset_name == 'dsprites_full':
                    label1 = label1[:, 1:].to(self.device)
                else:
                    label1 = label1.to(self.device)
 
                losses, params = self.vae_classification(losses, x_true1, label1, y_true1, examples)
   
                ## ADD FOR EVALUATION PURPOSES
                z[internal_iter*self.batch_size:(internal_iter+1)*self.batch_size, :] = params['z']
                g[internal_iter*self.batch_size:(internal_iter+1)*self.batch_size, :label1.size(1)] = label1

                self.optim_G.zero_grad()

                if (self.iter%self.show_loss)==0: print(f"Losses: {losses}")

                if not start_classification:
                    losses[c.TOTAL_VAE].backward(retain_graph=False)
                    #losses['true_values'].backward(retain_graph=False)
                    self.optim_G.step()

                if start_classification:   # and (params['n_passed']>0):
                    losses['prediction'].backward(retain_graph=False)
                    self.optim_G.step()

                vae_loss_sum += losses[c.TOTAL_VAE]
                losses[c.TOTAL_VAE_EPOCH] = vae_loss_sum /( internal_iter+1) ## ADDED +1 HERE IDK WHY NOT BEFORE!!!!!

                ## Insert losses -- only in training set
                if track_changes and is_time_for(self.iter, self.test_iter):
                    #TODO: set the tracking at a given iter_number/epoch
                    print('tracking changes')
                    Iterations.append(self.iter + 1); Epochs.append(epoch)
                    Reconstructions.append(losses['recon'].item()); KLDs.append(losses['kld'].item()); True_Values.append(losses['true_values'].item())
                    latent_errors.append(params['latents']); Accuracies.append(losses['prediction'].item())
                    F1_scores.append(Accuracy_Loss()(params['prediction'], y_true1, dims=self.n_classes).item())
                    
                    if epoch >0:
                        sofar = pd.DataFrame(data=np.array([Iterations, Epochs, Reconstructions, KLDs, True_Values, Accuracies, F1_scores]).T,
                                             columns=['iter', 'epoch', 'reconstruction_error', 'kld', 'latent_error', 'classification_error', 'accuracy'], )
                        for i in range(label1.size(1)):
                            sofar['latent%i'%i] = np.asarray(latent_errors)[:,i]

                        sofar.to_csv(os.path.join(out_path+'/train_runs', 'metrics.csv'), index=False)
                        del sofar

                        # ADD validation step
                        val_rec, val_kld, val_latent, val_bce, val_acc, _, _, _ =self.test(validation=True, name=self.dset_name)
                        sofar = pd.DataFrame(np.array([epoch, val_rec, val_kld, val_latent, val_bce, val_acc]).reshape(1,-1), 
                                            columns=['epoch','rec', 'kld', 'latent', 'bce', 'acc'] )
                        self.validation_scores = self.validation_scores.append(sofar, ignore_index=True)
                        self.validation_scores.to_csv(os.path.join(out_path+'/train_runs', 'val_metrics.csv'), index=False)
                        del sofar
                    # validation check
                    if epoch > 10: 
                        print('Validation stop evaluation')
                        print(self.iter, self.epoch)
                        print(self.validation_scores)
                        self.validation_stopping()


                # TESTSET LOSSES
                if is_time_for(self.iter, self.test_iter):

                    #                    self.dataframe_eval = self.dataframe_eval.append(self.evaluate_results,  ignore_index=True)
                    # test the behaviour on other losses
                    trec, tkld, tlat, tbce, tacc, I, I_tot, err_latent = self.test(end_of_epoch=False,name=self.dset_name, 
                                                                        out_path=self.out_path )
                    factors = pd.DataFrame(
                        {'iter': self.iter+1, 'rec': trec, 'kld': tkld, 'latent': tlat, 'BCE': tbce, 'Acc': tacc,
                         'I': I_tot}, index=[0])

                    for i in range(len(err_latent)):
                        factors['latent%i' % i] = np.asarray(err_latent)[i]

                    self.dataframe_eval = self.dataframe_eval.append(factors, ignore_index=True)
                    self.net_mode(train=True)

                    if track_changes and not self.dataframe_eval.empty:
                        self.dataframe_eval.to_csv(os.path.join(out_path, 'eval_results/test_metrics.csv'),
                                                   index=False)

                    # include disentanglement metrics
                    dis_metrics = pd.DataFrame(self.evaluate_results, index=[0])
                    self.dataframe_dis = self.dataframe_dis.append(dis_metrics)
                    del dis_metrics

                    if track_changes and not self.dataframe_dis.empty:
                        self.dataframe_dis.to_csv(os.path.join(out_path, 'eval_results/dis_metrics.csv'),
                                                  index=False)
                        print('Saved dis_metrics')

                    
                if self.save_model:
                    self.log_save(input_image=x_true1, recon_image=params['x_recon'], loss=losses)
                else:
                    self.step()
                    pass_dict ={'input_image':x_true1, 'recon_image':params['x_recon'], 'loss':losses}
                    if is_time_for(self.iter, self.schedulers_iter):
                        self.schedulers_step(pass_dict.get(c.LOSS, dict()).get(c.TOTAL_VAE_EPOCH, 0),
                                            self.iter // self.schedulers_iter)
                    del pass_dict

            # end of epoch
            if self.save_model:
                print('Saved model at epoch', self.epoch)
            
            if out_path is not None and self.save_model: # and validation is None:
                with open( os.path.join(out_path,'train_runs/latents_obtained.npy'), 'wb') as f:
                    np.save(f, self.epoch)
                    np.save(f, z.detach().cpu().numpy()[:10000] )
                    np.save(f, g.detach().cpu().numpy()[:10000])
                del z, g
                
        self.pbar.close()
        self.update_osr()            

    def test(self, end_of_epoch=True, validation=False, name='dsprites_full', out_path=None):
        self.net_mode(train=False)
        rec, kld, latent, BCE, Acc = 0, 0, 0, 0, 0
        I = np.zeros(self.z_dim)
        I_tot = 0

        N = 10**4
        l_dim = self.z_dim
        g_dim = self.z_dim

        if validation: loader = self.val_loader
        else: loader = self.test_loader
        
        z_array = np.zeros( shape=(len(loader.dataset), l_dim))
        g_array = np.zeros( shape=(len(loader.dataset), g_dim))

        y_pred_list = []
        y_test = []
        for internal_iter, (x_true, label, y_true, _) in enumerate(loader):
            x_true = x_true.to(self.device)

            if self.dset_name == 'dsprites_full':
                label = label[:, 1:].to(self.device)
            else:
                label = label.to(self.device, dtype=torch.float)
            
            y_true =  y_true.to(self.device, dtype=torch.long)
           
            g_array = g_array[:,:label.size(1)]

            mu, logvar = self.model.encode(x=x_true, )
            z = reparametrize(mu, logvar)
        
            mu_processed = torch.tanh(z / 2)
            prediction, forecast = self.predict(latent=mu[:,:self.z_class] )
            x_recon = self.model.decode(z=z,)    
        
            zs = self.reparametrize_many(mu, logvar, 100)
            prediction = torch.zeros(size=prediction.size(), device=self.device)
            forecast   = torch.zeros(size=forecast.size(), device=self.device)            
            for z_prov in zs:
                z_concept = torch.tanh(z_prov / 2).to(self.device)
                pred, fore = self.predict(latent=z_concept[:,:self.z_class])
                prediction += pred / len(zs)
                forecast   += fore / len(zs)
        
            # create the relevant quantities for confusion matrix
            y_test.append(y_true.detach().cpu().numpy())
            y_test_pred = prediction
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
            
            z = np.asarray(nn.Sigmoid()(z).detach().cpu())
            g = np.asarray(label.detach().cpu())
            
            bs = len(z)
            z_array[self.batch_size*internal_iter:self.batch_size*internal_iter+bs, :] = z
            g_array[self.batch_size*internal_iter:self.batch_size*internal_iter+bs, :] = g

            rec+=(F.binary_cross_entropy(input=x_recon, target=x_true,reduction='sum').detach().item()/self.batch_size )
            if self.conditional_prior: 
                y_onehot = F.one_hot(y_true, self.n_classes).to(dtype=torch.float, device=self.device)
                mu_target = self.enc_z_from_y(y_onehot)
                kld += self._kld_loss_fn(mu, logvar, mu_target).detach().item()
                
            else: kld+=(self._kld_loss_fn(mu, logvar).detach().item())

            if self.latent_loss == 'MSE':
                loss_bin = nn.MSELoss(reduction='mean')(mu_processed[:, :label.size(1)], 2 * label.to(dtype=torch.float32) - 1)
                err_latent = []
                for i in range(label.size(1)):
                    err_latent.append(nn.MSELoss(reduction='mean')(mu_processed[:, i],
                                                                    2*label[:, i].to(dtype=torch.float32)-1 ).detach().item())
            elif self.latent_loss == 'BCE':
                loss_bin = nn.BCELoss(reduction='mean')((1+mu_processed[:, :label.size(1)])/2, label.to(dtype=torch.float32) )
                err_latent = []
                for i in range(label.size(1)):
                    err_latent.append(nn.BCELoss(reduction='mean')((1+mu_processed[:, i])/2,
                                                                    label[:, i] ).detach().item())
            elif self.latent_loss == 'None':
                loss_bin = torch.tensor(0)
                err_latent = []
            else:
                NotImplementedError('Wrong argument for latent loss.')

            latent+=(loss_bin.detach().item())
            del loss_bin

            BCE+=(nn.CrossEntropyLoss(reduction='mean')(prediction,
                                                        y_true).detach().item())

            Acc+=(Accuracy_Loss()(forecast,
                                y_true, dims=self.n_classes).detach().item() )

        if end_of_epoch:
            self.visualize_recon(x_true, x_recon, test=True)
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max),
                                    spacing=self.traverse_spacing,
                                    data=(x_true, label), test=True)
        
        
        if out_path is not None and self.save_model and not validation:
            if self.dset_name == 'dsprites_leakage':

                ## update the osr
                self.update_osr()

                y_hot = torch.tensor([[1., 0., 0.],
                                      [0., 1., 0.],
                                      [0., 0., 1.]], device=self.device)
                centroids = self.enc_z_from_y(y_hot)
                passed = self.reject_test_set()
                                
                with open( os.path.join(out_path,'eval_results/centroids.npy'), 'wb') as f:
                    np.save(f, centroids.cpu().detach().numpy())

                with open( os.path.join(out_path,'eval_results/passed.npy'), 'wb') as f:
                    np.save(f, passed.detach().cpu().numpy())

            with open( os.path.join(out_path,'eval_results/latents_obtained.npy'), 'wb') as f:
                print(np.shape(z_array[:20000]))
                print(np.shape(g_array[:20000]))
                np.save(f, self.epoch)
                np.save(f, z_array[:20000])
                np.save(f, g_array[:20000])
            with open(os.path.join(out_path,'eval_results/downstream_obtained.npy'), 'wb') as f:
                
                y_test      = np.array([a.squeeze().tolist() for a in y_test[:-1]])
                y_pred_list = np.array([a.squeeze().tolist() for a in y_pred_list[:-1]])
                np.save(f, self.epoch)
                np.save(f, y_test)
                np.save(f, y_pred_list) 
            
        nrm = internal_iter + 1
        return rec/nrm, kld/nrm, latent/nrm, BCE/nrm, Acc/nrm, I/nrm, I_tot/nrm, [err/nrm for err in err_latent]

    