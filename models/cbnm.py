import os.path
from tkinter import E
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

import numpy as np
import pandas as pd


class CBNM(VAE):
    """
    Graybox version of VAE, with standard implementation. The discussion on
    """

    def __init__(self, args):

        super().__init__(args)

        print('Initialized CBM_Join model')

        self.num_classes = args.n_classes
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

        # model and optimizer
        self.model = VAEModel(encoder(self.z_dim, input_channels, self.image_size),
                               decoder(decoder_input_channels, self.num_channels, self.image_size),
                               ).to(self.device)
        self.classification = nn.Linear(self.z_dim, args.n_classes, bias=False).to(self.device) 
        #self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))
        #self.optim_G_mse = optim.Adam(self.model.encoder.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        self.optim_G = optim.Adam([*self.model.encoder.parameters(), *self.classification.parameters()],
                                      lr=self.lr_G, betas=(self.beta1, self.beta2))
        self.class_G_all = self.optim_G
        # update nets
        self.net_dict['G'] = self.model
        self.optim_dict['optim_G'] = self.optim_G

        self.setup_schedulers(args.lr_scheduler, args.lr_scheduler_args,
                              args.w_recon_scheduler, args.w_recon_scheduler_args)

        ## add binary classification layer
        
        self.reduce_rec = args.reduce_recon

        ## CHOOSE THE WEIGHT FOR CLASSIFICATION
        self.label_weight = args.label_weight
        
        ## CHOOSE THE WEIGHT FOR LATENTS
        self.masking_fact = args.masking_fact
        if args.latent_weight is None:
            self.latent_weight = args.label_weight 
        else:
            self.latent_weight = args.latent_weight                
        self.latent_loss = args.latent_loss
        self.show_loss = args.show_loss

        self.wait_counter = 0
        self.save_model = True      
        self.is_VAE = False  
        

        self.dataframe_dis = pd.DataFrame() #columns=self.evaluation_metric)
        self.dataframe_eval = pd.DataFrame()
        self.validation_scores = pd.DataFrame()



    def predict(self, **kwargs):
        """
        Predict the correct class for the input data.
        """
        input_x = kwargs['latent'].to(self.device)
        pred_raw = self.classification(input_x)
        pred = nn.Softmax(dim=1)(pred_raw)
        return  pred_raw, pred.to(self.device, dtype=torch.float32) #nn.Sigmoid()(self.classification(input_x).resize(len(input_x)))

    def loss_fn(self, input_losses, reduce_rec=False, **kwargs):
        output_losses = dict()
        output_losses['total'] = input_losses.get('total', 0)
        return output_losses

    def cbm_classification(self, losses, x_true1, label1, y_true1, examples):

        # label_1 \in [0,1]
        mu, _ = self.model.encode(x=x_true1,)

        mu_processed = torch.tanh(mu/2)
        z = mu_processed
        #x_recon = self.model.decode(z=z,)

        # CHECKING THE CONSISTENCY

        prediction, forecast = self.predict(latent=mu_processed)
        rn_mask = (examples==1)
        
        if examples[rn_mask] is None:
            n_passed = 0
        else:
            n_passed = len(examples[rn_mask])

        losses.update(self.loss_fn(losses, reduce_rec=False,))

        pred_loss = nn.CrossEntropyLoss(reduction='mean')(prediction, y_true1) *self.label_weight  # her efor celebA
        losses.update(prediction=pred_loss)
        losses[c.TOTAL_VAE] += pred_loss

        if n_passed > 0: # added the presence of only small labelled generative factors

            if self.latent_loss == 'MSE':                
                loss_bin = nn.MSELoss(reduction='mean')( mu_processed[rn_mask][:, :label1.size(1)], 2*label1[rn_mask]-1  )
                losses.update(true_values=self.latent_weight * loss_bin)
                losses[c.TOTAL_VAE] += self.latent_weight * loss_bin
                ## track losses
                err_latent = []
                for i in range(label1.size(1)):
                    err_latent.append(nn.MSELoss(reduction='mean')(mu_processed[rn_mask][:, i], 2 * label1[rn_mask][:,i] - 1).detach().item() )
                
            elif self.latent_loss == 'BCE':
                loss_bin = nn.BCELoss(reduction='mean')((1+mu_processed[rn_mask][:, :label1.size(1)])/2,
                                                            label1[rn_mask] )

                losses.update(true_values=self.latent_weight * loss_bin)
                losses[c.TOTAL_VAE] += self.latent_weight * loss_bin
                ## track losses
                err_latent = []
                for i in range(label1.size(1)):
                    err_latent.append(nn.BCELoss(reduction='mean')((1+mu_processed[rn_mask][:, i])/2,
                                                                    label1[rn_mask][:, i] ).detach().item())
            else:
                raise NotImplementedError('Not implemented loss.')

        else:
            losses.update(true_values=torch.tensor(-1))
            err_latent =[-1]*label1.size(1)
    #            losses[c.TOTAL_VAE] += nn.MSELoss(reduction='mean')(mu[:, :label1.size(1)], label1).detach()


        return losses, {'mu': mu, 'z': z, "prediction": prediction, 'forecast': forecast,
                    'latents': err_latent, 'n_passed': n_passed}

    def train(self, **kwargs):

        out_path = None

        if 'output'  in kwargs.keys():
            out_path = kwargs['output']
            track_changes=True
            self.out_path = out_path #TODO: Not happy with this thing

        else: track_changes=False;

        if track_changes:
            print("## Initializing Train indexes")
            print("::path chosen ->",out_path+"/train_runs")

        Iterations, Epochs, True_Values, Accuracies, CE_class = [], [], [], [], []  ## JUST HERE FOR NOW
        F1_scores = []
        latent_errors = []
        epoch = 0
        
        self.class_G_all.param_groups[0]['lr'] = 0
        lr_log_scale = np.logspace(-7,-4, 10)
        while not self.training_complete():
            
            if epoch < 10:
                self.class_G_all.param_groups[0]['lr'] = lr_log_scale[epoch]

            epoch += 1
            self.net_mode(train=True)
            vae_loss_sum = 0
            # add the classification layer #
            z = torch.zeros(self.batch_size*len(self.data_loader), self.z_dim, device=self.device)
            g = torch.zeros(self.batch_size*len(self.data_loader), self.z_dim, device=self.device)

            for internal_iter, (x_true1, label1, y_true1, examples) in enumerate(self.data_loader):

                losses = {'total_vae':0}

                x_true1 = x_true1.to(self.device)
                
                #label1 = label1[:, 1:].to(self.device)
                if self.dset_name == 'dsprites_full':
                    label1 = label1[:, 1:].to(self.device)
                else:
                    label1 = label1.to(self.device)

                y_true1 = y_true1.to(self.device)

                ###configuration for dsprites

                losses, params = self.cbm_classification(losses, x_true1, label1, y_true1, examples)

                ## ADD FOR EVALUATION PURPOSES
                z[internal_iter*self.batch_size:(internal_iter+1)*self.batch_size, :] = params['z']
                g[internal_iter*self.batch_size:(internal_iter+1)*self.batch_size, :label1.size(1)] = label1


                self.class_G_all.zero_grad()

                if (internal_iter%self.show_loss)==0: print("Losses:", losses)

                losses[c.TOTAL_VAE].backward(retain_graph=False)
                self.class_G_all.step()

                vae_loss_sum += losses[c.TOTAL_VAE]
                losses[c.TOTAL_VAE_EPOCH] = vae_loss_sum /( internal_iter+1)

                ## Insert losses -- only in training set
                if track_changes and is_time_for(self.iter, self.test_iter):

                    Iterations.append(internal_iter + 1)
                    Epochs.append(epoch)

                    True_Values.append(losses['true_values'].item())
                    latent_errors.append(params['latents'])

                    CE_class.append(losses['prediction'].item())
                    f1_class = Accuracy_Loss().to(self.device)
                    F1_scores.append(f1_class(params['prediction'], y_true1, dims=self.num_classes).item())

                    del f1_class
                    
                    if epoch > 1:
                        sofar = pd.DataFrame(data=np.array([Iterations, Epochs,  True_Values, CE_class, F1_scores]).T,
                                                columns=['iter', 'epoch', 'latent_error', 'classification_error', 'accuracy'], )
                        for i in range(label1.size(1)):
                            sofar['latent%i'%i] = np.asarray(latent_errors)[:,i]

                        sofar.to_csv(os.path.join(out_path+'/train_runs', 'metrics.csv'), index=False)
                        del sofar

                        # ADD validation step
                        val_latent, val_bce, val_acc, _, _, _ =self.test(validation=True, name=self.dset_name, 
                                                                out_path=self.out_path)
                        sofar = pd.DataFrame(np.array([epoch, val_latent, val_bce, val_acc]).reshape(1,-1), 
                                            columns=['epoch', 'latent', 'bce', 'acc'] )
                        self.validation_scores = self.validation_scores.append(sofar, ignore_index=True)
                        self.validation_scores.to_csv(os.path.join(out_path+'/train_runs', 'val_metrics.csv'), index=False)
                        del sofar
                    if epoch > 12: self.validation_stopping()
                    
                if is_time_for(self.iter, self.test_iter):

                    #                    self.dataframe_eval = self.dataframe_eval.append(self.evaluate_results,  ignore_index=True)
                    # test the behaviour on other losses
                    trec, tkld = 0, 0
                    tlat, tbce, tacc, I, I_tot, err_latent = self.test(end_of_epoch=False, name=self.dset_name,
                                                                       out_path=self.out_path)
                    factors = pd.DataFrame(
                        {'iter': self.iter, 'rec': trec, 'kld': tkld, 'latent': tlat, 'BCE': tbce, 'Acc': tacc,
                         'I': I_tot}, index=[0])

                    for i in range(label1.size(1)):
                        factors['latent%i' % i] = np.asarray(err_latent)[i]

                    self.dataframe_eval = self.dataframe_eval.append(factors, ignore_index=True)
                    self.net_mode(train=True)

                    if track_changes and not self.dataframe_eval.empty:
                        self.dataframe_eval.to_csv(os.path.join(out_path, 'eval_results/test_metrics.csv'),
                                                   index=False)
                        print('Saved test_metrics')

                    # include disentanglement metrics
                    dis_metrics = pd.DataFrame(self.evaluate_results, index=[0])
                    self.dataframe_dis = self.dataframe_dis.append(dis_metrics)

                    if track_changes and not self.dataframe_dis.empty:
                        self.dataframe_dis.to_csv(os.path.join(out_path, 'eval_results/dis_metrics.csv'),
                                                  index=False)
                        print('Saved dis_metrics')

                    
                if self.save_model:
                    self.log_save(input_image=x_true1, recon_image=x_true1, loss=losses)
                else:
                    self.step()
                    pass_dict ={'input_image':x_true1, 'recon_image': x_true1, 'loss':losses}
                    if is_time_for(self.iter, self.schedulers_iter):
                        self.schedulers_step(pass_dict.get(c.LOSS, dict()).get(c.TOTAL_VAE_EPOCH, 0),
                                            self.iter // self.schedulers_iter)
                    del pass_dict

            
            # end of epoch
            if out_path is not None and self.save_model:
                with open( os.path.join(out_path,'train_runs/latents_obtained.npy'), 'wb') as f:
                    np.save(f, self.epoch)
                    np.save(f, z.detach().cpu().numpy())
                    np.save(f, g.detach().cpu().numpy())
                del z, g
            

        self.pbar.close()

    def test(self, end_of_epoch=True, validation=False, name='dsprites_full', out_path=None):
        self.net_mode(train=False)
        rec, kld, latent, BCE, Acc = 0, 0, 0, 0, 0
        I = np.zeros(self.z_dim)
        I_tot = 0

        N = 10 ** 4
        l_dim = self.z_dim
        g_dim = self.z_dim

        z_array = np.zeros(shape=(self.batch_size * len(self.test_loader), l_dim))
        g_array = np.zeros(shape=(self.batch_size * len(self.test_loader), g_dim))
        
        if validation: loader = self.val_loader
        else: loader = self.test_loader

        y_pred_list = []
        y_test = []
        for internal_iter, (x_true, label, y_true, _) in enumerate(loader):
            
            x_true = x_true.to(self.device)
            if self.dset_name == 'dsprites_full':
                    label = label[:, 1:].to(self.device)
            else:
                label = label.to(self.device)
            
            y_true = y_true.to(self.device, dtype=torch.long)

            mu, logvar = self.model.encode(x=x_true, )
            z = reparametrize(mu, logvar)

            mu_processed = torch.tanh(mu / 2)
            prediction, forecast = self.predict(latent=mu_processed)
            
            # create the relevant quantities for confusion matrix
            y_test.append(y_true.detach().cpu().numpy())
            y_test_pred = prediction
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
            

            z = np.asarray(nn.Sigmoid()(z).detach().cpu())
            g = np.asarray(label.detach().cpu())
            bs = len(label)
            z_array[self.batch_size * internal_iter:self.batch_size * internal_iter + bs, :] = z
            g_array[self.batch_size * internal_iter:self.batch_size * internal_iter + bs, :] = g

            #            I_batch , I_TOT = Interpretability(z, g)
            #           I += I_batch; I_tot += I_TOT

            if self.latent_loss == 'MSE':
                loss_bin = nn.MSELoss(reduction='mean')(mu_processed[:, :label.size(1)],
                                                        2 * label.to(dtype=torch.float32) - 1)
                err_latent = []
                for i in range(label.size(1)):
                    err_latent.append(nn.MSELoss(reduction='mean')(mu_processed[:, i], 2 * label[:,i] - 1).detach().item() )
                
            elif self.latent_loss == 'BCE':
                loss_bin = nn.BCELoss(reduction='mean')((1 + mu_processed[:, :label.size(1)]) / 2,
                                                        label.to(dtype=torch.float32))
                err_latent = []
                for i in range(label.size(1)):
                    err_latent.append(nn.BCELoss(reduction='mean')((1+mu_processed[:, i])/2,
                                                                    label[:, i] ).detach().item())
            else:
                NotImplementedError('Wrong argument for latent loss.')

            latent += (loss_bin.detach().item())
            del loss_bin

            BCE += (nn.CrossEntropyLoss(reduction='mean')(prediction,
                                                          y_true).detach().item())

            Acc += (Accuracy_Loss()(forecast,
                                    y_true, dims=self.num_classes).detach().item())

       
        if out_path is not None and self.save_model and not validation:
            with open( os.path.join(out_path,'eval_results/latents_obtained.npy'), 'wb') as f:
                np.save(f, self.epoch)
                np.save(f, z_array)
                np.save(f, g_array)
                
            with open(os.path.join(out_path,'eval_results/downstream_obtained.npy'), 'wb') as f:
                    y_test      = np.array([a.squeeze().tolist() for a in y_test])
                    y_pred_list = np.array([a.squeeze().tolist() for a in y_pred_list])
                    
                    np.save(f, y_test)
                    np.save(f, y_pred_list) 

        print('Done testing')

        nrm = internal_iter + 1
        return latent / nrm, BCE / nrm, Acc / nrm, I / nrm, I_tot / nrm, [err/nrm for err in err_latent]
    
    