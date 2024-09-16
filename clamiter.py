
import numpy as np
import matplotlib.pyplot as plt

import torch

from torch.nn.functional import relu
from torch.autograd import grad as a_grad

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj, is_undirected, dropout_edge, sort_edge_index, contains_self_loops, k_hop_subgraph, remove_isolated_nodes

from torch_geometric.data import Data

from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.cluster import KMeans
import time
import os
import json

from transformation import  RealNVP, relu_lightcone, relu_transform, uv_from_xt
from utils.plotting import *
from utils import utils
from utils import utils_pyg as up
from utils.utils import get_prob_graph, edges_by_coords, roc_of_omitted_dyads, k_minimal_neighborhoods
from utils.printing_utils import printd, print_dolphin, print_escher, print_end_fit
from datasets.import_dataset import import_dataset
# from tests import tests


from tqdm import tqdm
import datetime
import os
eps = 1e-6





class PCLAMIter(MessagePassing):
    '''class to do the pclam iterations in the form of a message passing neural network'''
    # i can initialize the node feats 
    #TODO: add l1 regularization
    def __init__(self, 
                 lorenz, 
                 vanilla, 
                 dim_feat,
                 dim_attr=0, 
                 attr_opt=False,
                 l1_reg=1, 
                 reg_inr=0, 
                 s_reg=0, 
                 T=1,
                 prior=None, 
                 inflation_flow_name=None, 
                 hidden_dim=64, 
                 num_coupling_blocks=32, 
                 num_layers_mlp=2,
                 lr=0.01,
                 aggr='add', 
                 device=torch.device('cpu')):
        
        super(PCLAMIter, self).__init__(aggr=aggr)
        self.dim_feat = dim_feat
        self.dim_attr = dim_attr
        self.attr_opt = attr_opt
        self.lorenz = lorenz
        self.vanilla = vanilla 
        self.l1_reg = l1_reg
        self.reg_inr = reg_inr
        self.s_reg = s_reg
        self.device = device
        self.T = T
        
        if self.vanilla and not self.lorenz:
            self.model_name = 'bigclam'
        elif self.vanilla and self.lorenz:
            self.model_name = 'iegam'
        elif not self.vanilla and not self.lorenz:
            self.model_name = 'pclam'
        elif not self.vanilla and self.lorenz:
            self.model_name = 'piegam'

        if attr_opt:
            #! bug here, we add the dim attr before we do the data transform...
            self.in_out_dim = dim_feat + dim_attr
        else:
            self.in_out_dim = dim_feat

        if self.lorenz:
            if not dim_feat//2 == dim_feat/2:
               raise ValueError('dim_feat should be even for p/iegam')
            self.B = 1/self.T*(torch.concatenate([torch.ones(dim_feat//2), -torch.ones(dim_feat//2)])).to(device) # GPU 50 mib
            self.dim_feat = dim_feat
            
            
            self.vanilla = vanilla
            self.num_s_comms = dim_feat//2
            self.num_t_comms = dim_feat//2
            self.s_reg = s_reg
            self.feat_bounding=relu_lightcone
        else:
            self.B = (1/self.T*torch.ones(dim_feat)).to(device)
            self.num_t_comms = dim_feat
            self.num_s_comms = 0
            self.s_reg = 0
            self.feat_bounding = relu_transform

        if not vanilla:    
            if prior is None or prior == 'None':
                self.prior = RealNVP(in_out_dim=self.in_out_dim, hidden_dim=hidden_dim, num_coupling_blocks=num_coupling_blocks, num_layers_mlp=num_layers_mlp, attr_opt=self.attr_opt, inflation_flow_name=inflation_flow_name, device=device)
            else:
                self.add_prior(prior=prior)
        
        else: 
            self.prior = None

        self.to(self.device) # GPU didn't really change, i think clamiter doesn't take much space
                            #! if there is a prior could be loaded twice


    def add_prior(self, hidden_dim=64, num_coupling_blocks=32, num_layers_mlp=2, prior=None):
        if prior is not None:
            self.prior = prior
        else:
            self.prior = RealNVP(in_out_dim=self.in_out_dim, hidden_dim=hidden_dim, num_coupling_blocks=num_coupling_blocks, num_layers_mlp=num_layers_mlp,device=self.device)
        self.vanilla =False
        self.prior.to(self.device)

        if self.model_name == 'bigclam':
            self.model_name = 'pclam'  
        elif self.model_name == 'iegam':
            self.model_name = 'piegam'

    def forward(self, graph, node_mask):
        '''first called, starts mpnn process by preprocessing then calling propagate'''

        t = time.time()
        prior_grad = torch.zeros_like(graph.x)
        if not self.vanilla:
            #todo: concatenate features graph.s with graph.x
            if self.attr_opt:
                feats_for_prior = torch.cat([graph.x, graph.attr], dim=1)
            else:
                feats_for_prior = graph.x
            feats_for_prior = feats_for_prior[node_mask]
            feats_for_prior.requires_grad_(True)

            #? for omittion calc the prior, omit the nodes, add all up, looks good
            log_prior_loss = self.prior.forward_ll(feats_for_prior)
            
            masked_prior_grad = a_grad(log_prior_loss, feats_for_prior, create_graph=False)[0]
            prior_grad[node_mask] = masked_prior_grad[:, :self.dim_feat]
            #the extra clone means the data of the tensor is different. tested, it doesn't take longer.
            graph.x = graph.x.detach().clone() 
            # attributes must not change
            
        with torch.no_grad():
            tbr = self.propagate(edge_index=graph.edge_index, x=graph.x, global_features=(prior_grad), edge_attr=graph.edge_attr)
        
            # tbr2 = self.propagate(edge_index=graph.edge_index, x=graph.x, global_features=(torch.tensor(0.0)), edge_attr=graph.edge_attr) + prior_grad

            # if not torch.allclose(tbr, tbr2):
                # raise ValueError('tbr and tbr2 not equal')

        tbr = tbr*node_mask.unsqueeze(-1).float()
        # ==================================
        
        return tbr

        
    def message(self, x_j, x_i, edge_attr):
        '''returns the message from node j to node i. this is only for edges. the global sum is preprocessed in the forward function, and will be added in the update function.             

        link prediction here will be either:'''
        #* x_i is the reciever and x_j is the sender.

        # todo> change clamiter and maybe the prior to mask some of the edges
        # how to mask the edges? for now not very important.. if this becomes a problem for large graphs i can do 2d edge attributes or something. hope it's not 
        x_inner_product = torch.einsum('ij,ij->i', x_i, self.B*x_j) + eps
        
        if (x_inner_product < 0).any():
            raise ValueError('x_inner_product is negative for neighbors')
        if (x_inner_product == 0).any():
            raise ValueError('x_inner_product is 0 for neighbors')
        #* this is the only change to clamiter class due to dyad omittion
        msg_1 = x_j / (1 - torch.exp(-x_inner_product) + eps).unsqueeze(1) #- self.reg_inr*x_j*(x_inner_product - 1).unsqueeze(1)
        msg_0 = x_j

        # edge attr is 0 for omitted dyads
        msg = edge_attr.unsqueeze(1)*msg_1 + (~edge_attr).unsqueeze(1)*msg_0 
        #? TESTED  torch.where(msg == x_j) == torch.where(edge_attr==0) 
        return msg


    def update(self, aggr_out, x, global_features, edge_attr):
        '''returns the gradient of the loss with respect to the node features
        regularization: self.s_reg and self.l1_reg
        global: global_features[0] is the sum of the node features, global_features[1] is the prior grad'''
        #! here there is addition of 2000
        global_term = torch.sum(x, dim=0)
        
        s_feats = x[:, self.num_t_comms:]
        s_feats = torch.concatenate([torch.zeros([x.shape[0], self.num_t_comms]).to(s_feats.device), s_feats], dim=1)
      
        
        update = self.B*(aggr_out - global_term + x) + global_features - self.s_reg*s_feats - self.l1_reg*torch.sign(x)


        return update

    def readout(self, graph):
        # no add noise here - the nodes are after the optimization and the net noise is applied in the backward xz functions
        if self.vanilla:
            #noise isn't added here because there is no prior
            #! does this calculate the omittion loss?
            readout = clam_loss(graph, lorenz=self.lorenz)
        else:
            readout = clam_loss_prior(graph, self.lorenz, self.prior)
            
        return readout
    
   
    def fit_feats_fixed_prior(self, graph, n_iter, lr, node_mask=None, omitted_dyads=None, cutoff=0.0, plot_every=1000, verbose=False, show_iter=None):
        #* try saying that 3 times!
        ''' optimize n_iter times with the prior fixed, then for every feat dimension, swap it with all the rest of the dimensions and caluclate the loss. for every ax choose the swap that gives the best loss'''
        #todo: cancel the dropout node thing, just slows everything down
        printd(f'\n in fit_feats_fixed_prior, using device {self.device}')
        if self.prior != None:
            self.prior.eval()
        
        
        losses_feats, _ , l2_norms= self.fit_feats(graph=graph, node_mask=node_mask,
                                                n_iter=n_iter, 
                                                lr=lr, 
                                                omitted_dyads=omitted_dyads, cutoff=cutoff, 
                                                verbose=verbose, show_iter=show_iter)
            
                
        if verbose:
            printd(f'\n in fit_feats_fixed_prior, plotting losses and state')
            self.plot_state(graph, community_affiliation=graph.y, calling_function_name='fit_feats_fixed_prior')
            plot_losses(losses_feats, l2_norms)
        
        return losses_feats, l2_norms






# 888888 88 888888     Yb        dP 88""Yb    db    88""Yb 88""Yb 888888 88""Yb 
# 88__   88   88        Yb  db  dP  88__dP   dPYb   88__dP 88__dP 88__   88__dP 
# 88""   88   88         YbdPYbdP   88"Yb   dP__Yb  88"""  88"""  88""   88"Yb  
# 88     88   88          YP  YP    88  Yb dP""""Yb 88     88     888888 88  Yb 
                                                                              
                                                                              
    def fit_wrapper(self, 
                    graph, 
                    n_iter, 
                    lr, 
                    task,
                    which_fit='fit_feats',
                    early_stop=0,
                    cutoff=0.0, 
                    verbose=False,
                    acc_every=10, 
                    plot_every=100000,
                    **kwargs
                    ):
            
            def iter_step_feat():
                clamiter_grad = self(graph, node_mask)
                self.debug_last_grad = clamiter_grad
                graph.x = torch.clamp(self.feat_bounding(graph.x, clamiter_grad, lr, node_mask, cutoff), -5000,5000)
                loss = self.readout(graph)
                return loss
            
            def iter_step_prior():
                optimizer.zero_grad()
                log_likelihood = self.prior.forward_ll(feat_for_prior, noise_amp,sum=False)
                ll_loss = -torch.sum(log_likelihood)
                loss = ll_loss
                loss.backward()
                optimizer.step()
                return loss
            
            '''optimize the features using iterations of clamiter.
            param: node_mask: a mask for the nodes to optimize, the rest should stay unchanged'''
            # kwargs are: 
            #       node_mask=None, 
            #       performance_metric=None,
            #       omitted_dyads=None, 
            

            # ASSERTIONS
            assert graph.is_undirected(), 'graph is directed!!!'
            assert not graph.has_self_loops(), 'graph contains self loops!!!'
            assert which_fit in ['fit_feats', 'fit_prior'], 'which_fit should be either fit_feats or fit_prior'
            # ==== end assertions =====
            
            if plot_every == -1:
                plot_every = n_iter

            if verbose:
                printd(f'\n in {which_fit}, using device {self.device}')

            losses = []

            acc_tracker = AccTrack(clamiter=self, 
                                graph=graph, 
                                task=task, 
                                acc_every=acc_every,
                                calling_function_name=which_fit,
                                lorenz=self.lorenz,
                                **kwargs)
                 
            # ======== end init early stop ======
            
            # NODE MASK
            node_mask = kwargs.get('node_mask', None)
            if node_mask is None:
                node_mask = torch.ones(graph.x.shape[0], dtype=torch.bool)
            node_mask = node_mask.to(self.device) 
            # ===== end node mask =======

            if self.prior:
                self.prior.eval() if which_fit == 'fit_feats' else self.prior.train()
            

            # PRE LOOP     
            if which_fit == 'fit_prior':
                optimizer = kwargs.get('optimizer', None)
                noise_amp = kwargs.get('noise_amp', None)
                if optimizer is None:
                    raise ValueError('fit_prior needs an optimizer')
                iter_step = iter_step_prior
                feat_for_prior = graph.x
                if self.attr_opt:
                    feat_for_prior = torch.cat([graph.x, graph.attr], dim=1)
                if node_mask is not None:
                    feat_for_prior = feat_for_prior[node_mask]

            elif which_fit == 'fit_feats':
                iter_step = iter_step_feat
            # ======== pre loop ============

            # OPTIMIZATION LOOPS
            # for i in tqdm(range(n_iter), desc="feat opt"):
            for i in range(n_iter):
                try:
                    loss = iter_step()
                    losses.append(loss.item())
                except ValueError as e:
                    printd(f'fit wrapper {which_fit} error in iter_step at iter {i}: {e}')
                    break
                
                # ACCURACY CALCULATION
                if (i+1)%acc_every == 0:
                    #! problem here, accuracies not in sync with losses.
                    # measurement_interval = 1 if i == 0 else acc_every
                    measurement_interval = acc_every
                    acc_tracker.get_intermediate_accuracies(losses=losses, measurement_interval=measurement_interval)
                    if verbose:
                        printd(f'\n fit wrapper {which_fit}, intermediate values iter {i}')
                        acc_tracker.print_accuracies()
                
                if (i+1)%plot_every == 0:
                    printd(f'\nfit wrapper {which_fit}, plotting state at iter {i}')
                    acc_tracker.plot_intermediate(**kwargs)    
            # ========================================== end for loop
           
            # LAST CALC ACCS            
            acc_tracker.get_intermediate_accuracies(losses=losses, measurement_interval=(i+1)%acc_every)
            
            if verbose:
                printd(f'\nfit_wrapper {which_fit}. optmization loop ended at iteration {i}')
                acc_tracker.print_accuracies()
                if which_fit == 'fit_feats':
                    print_dolphin()
                elif which_fit == 'fit_prior':
                    print_escher()
                
            # ===============================  end return acc  

            
            return losses, acc_tracker.accuracies_test, acc_tracker.accuracies_val
                                                                                
                                                                                

# 888888 88 888888     888888 888888    db    888888 .dP"Y8 
# 88__   88   88       88__   88__     dPYb     88   `Ybo." 
# 88""   88   88       88""   88""    dP__Yb    88   o.`Y8b 
# 88     88   88       88     888888 dP""""Yb   88   8bodP' 


    def fit_feats(self, 
                  graph, 
                  n_iter, 
                  lr, 
                  task,
                  node_mask=None, 
                  performance_metric=None,
                  omitted_dyads=None, 
                  early_stop=0,
                  cutoff=0.0, 
                  verbose=False,
                  acc_every=10, 
                  plot_every=100000,
                  **kwargs):
        
        '''optimize the features using iterations of clamiter.
        param: node_mask: a mask for the nodes to optimize, the rest should stay unchanged'''
        
        return self.fit_wrapper(graph=graph,
                                n_iter=n_iter, 
                                lr=lr, 
                                task=task,
                                which_fit='fit_feats',
                                node_mask=node_mask,
                                performance_metric=performance_metric,
                                omitted_dyads=omitted_dyads, 
                                early_stop=early_stop,
                                cutoff=cutoff, 
                                verbose=verbose,
                                acc_every=acc_every, 
                                plot_every=plot_every,
                                **kwargs
                                )
        
      
# 888888 88 888888     88""Yb 88""Yb 88  dP"Yb  88""Yb 
# 88__   88   88       88__dP 88__dP 88 dP   Yb 88__dP 
# 88""   88   88       88"""  88"Yb  88 Yb   dP 88"Yb  
# 88     88   88       88     88  Yb 88  YbodP  88  Yb 
    def fit_prior(self, 
                  graph, 
                  n_iter,
                  optimizer, # lr in optimizer 
                  node_mask=None, 
                  acc_every=10,
                  weight_decay=None,
                  task='anomaly', 
                  performance_metric=None,
                  omitted_dyads=None,
                  early_stop=0,
                  noise_amp=0.01, 
                  verbose=False,
                  lr=None, # this is not used, but it's needed for the configs dict...
                  plot_every=100000, 
                  **kwargs
                  ):
        '''
        this function optimizes the model. 
        noise is added to the feature optimization and to the input nodes when training.
        ''' 
        return self.fit_wrapper(graph=graph,
                                n_iter=n_iter, 
                                lr=optimizer.param_groups[0]['lr'], 
                                task=task,
                                which_fit='fit_prior',
                                noise_amp=noise_amp,
                                optimizer=optimizer,
                                node_mask=node_mask,
                                performance_metric=performance_metric,
                                omitted_dyads=omitted_dyads, 
                                early_stop=early_stop,
                                cutoff=0.0, 
                                verbose=verbose,
                                acc_every=acc_every, 
                                plot_every=plot_every,
                                **kwargs)
                                
        

            # 888888 88 888888 
            # 88__   88   88   
            # 88""   88   88   
            # 88     88   88   

    def fit(self, graph, 
            first_func_in_fit='fit_feats',
            optimizer=None, scheduler=None, 
            n_back_forth=0, 
            omitted_dyads=None, 
            prior_fit_mask=None,
            plot_every=1000,
            acc_every=20,
            early_stop_fit=0,
            early_stops=None,
            performance_metric=None,
            configs_dict=None,
            task='anomaly',
            verbose_in_funcs=False,
            verbose=False,
            **params,
            ):
        '''train the features and the prior back and forth.
        should have two modes'''
            
        if verbose:
            printd(f'\nfit, {task=}')

        if plot_every == -1:
            plot_every = n_back_forth

        


        #todo: rearange the parameters better. everything should be loaded in the mighty and the specialized hypers should not have everything in them
        prior_params = params.get('prior_params', {})
        feat_params = params.get('feat_params', {})
        back_forth_params = params.get('back_forth_params', {})
        
        if early_stops is not None:
            #! in general should be none
            if len(early_stops) != 2:
                printd(f'\nfit, early_stops should be a list of two integers, got {early_stops}. will use the ones in configs dict')
            prior_params['early_stop'] = early_stops[0]
            feat_params['early_stop'] = early_stops[1]


        fit_feats_func = lambda: self.fit_feats(
                            graph, 
                            omitted_dyads=omitted_dyads, 
                            verbose=verbose_in_funcs,
                            task=task,
                            acc_every=acc_every,
                            performance_metric=performance_metric,
                            **feat_params,
                            **params)
        
        fit_prior_func = lambda: self.fit_prior(
                            graph=graph, 
                            node_mask=prior_fit_mask,
                            optimizer=optimizer,
                            task=task,
                            acc_every=acc_every,
                            omitted_dyads=omitted_dyads,
                            performance_metric=performance_metric,
                            verbose=verbose_in_funcs, 
                            **prior_params,
                            **params)
        
        # FIRST AND SECOND PARAMS
    
        second_function_name = 'fit_prior' if first_func_in_fit == 'fit_feats' else 'fit_feats'
        printd(f'\nin fit,\n{first_func_in_fit=}\n{second_function_name=}')

        first_func = fit_feats_func if first_func_in_fit == 'fit_feats' else fit_prior_func
        second_func = fit_prior_func if first_func_in_fit == 'fit_feats' else fit_feats_func

        # number of blank spots in the losses array VV
        num_blanks_first = prior_params['n_iter'] if first_func_in_fit == 'fit_feats' else feat_params['n_iter'] 
        num_blanks_second = feat_params['n_iter'] if first_func_in_fit == 'fit_feats' else prior_params['n_iter']
        # prior eval and train:
        first_eval_style = self.prior.eval if first_func_in_fit == 'fit_feats' else self.prior.train
        second_eval_style = self.prior.eval if first_func_in_fit == 'fit_prior' else self.prior.train
        
            
        
        # ========================================

        losses_first = []
        losses_second = []

        #INIT accuracy tracking
        # acc_tracker updates the accuracies after every round
        acc_tracker = AccTrack(clamiter=self, 
                                graph=graph, 
                                task=task, 
                                acc_every=acc_every,
                                calling_function_name='fit',
                                **params)
        
        # BACK FORTH 0 -> RUN FIRST FUNCTION
        if n_back_forth == 0:
            
            if first_func_in_fit == 'fit_prior':
                printd(f'\n\nWARNING\n\WARNING: n_back_forth = 0, starting {first_func_in_fit} this is only optimizing prior, you should do "fit prior"\nWARNIING\n')

            printd(f'\n n_back_forth = 0, starting {first_func_in_fit}')
            first_eval_style() # either prior.eval() or prior.train()
            t = time.time()
            
            # FIRST FUNCTION
            try:
                losses_first, accuracies_test_first, accuracies_val_first = first_func()
            except ValueError as e:
                printd(f'\nfit func. error in {first_func_in_fit} at iter {i}:\n {e}')
                raise
            #  ===== end first function =====

            printd(f'\n{first_func_in_fit} took {time.time()-t} seconds')
            
            acc_tracker.append_accuracies(accuracies_test_first, [], accuracies_val_first, [])
            acc_tracker.losses = losses_first
            
            return losses, acc_tracker.accuracies_test, acc_tracker.accuracies_val
        # ====== back forth == 0 ========================
        
        # BACK AND FORTH
        # for i in tqdm(range(n_back_forth), desc='back and forth'):
        for i in range(n_back_forth):

            printd(f'\nback and forth {i+1}/{n_back_forth}')
            
            # --FIRST FUNCTION -- 
            first_eval_style() #* either self.prior.eval() or self.prior.train()
            t_first = time.time()
            try:
                losses_epoch_1st, accuracies_test_epoch_1st, accuracies_val_epoch_1st = first_func()
            except ValueError as e:
                printd(f'\nfit func. nerror in {first_func_in_fit} at iter {i}:\n {e}')
                break       
            if verbose_in_funcs:
                printd(f'\nin fit, {first_func_in_fit} took {time.time()-t_first} seconds')
            # ==== end first function ==
            
            # --SECOND FUNCTION --            
            t_second = time.time()
            second_eval_style() # either self.prior.eval() or self.prior.train()
            try:
                losses_epoch_2nd, accuracies_test_epoch_2nd, accuracies_val_epoch_2nd = second_func()
            except ValueError as e:
                printd(f'\nfit func. error in {second_function_name} at iter {i}:\n {e}')
                break
                
            if verbose_in_funcs:
                printd(f'\nin fit, {second_function_name} took {time.time()-t_second} seconds')
            # ==== end second function =======================
            
            # COLLECT RESULTS
            
            losses_first += losses_epoch_1st + [losses_epoch_1st[-1]]*num_blanks_first
            losses_second += num_blanks_second*[losses_epoch_2nd[0]] + losses_epoch_2nd
            

            acc_tracker.append_accuracies(accuracies_test_epoch_1st, 
                                          accuracies_test_epoch_2nd,
                                          accuracies_val_epoch_1st,
                                          accuracies_val_epoch_2nd)
            
            
            
            if verbose:
                if verbose_in_funcs:
                    print_end_fit()
                printd(f'\nfit, back and forth {i+1}/{n_back_forth} took {time.time()-t_first} seconds')
                acc_tracker.print_accuracies()
                a=0
          
                
                # ===== end anomaly collect acc ======
                        
    
            # SCHEDULER
            
            scheduler_updated = utils.scheduler_step(scheduler, optimizer, feat_params, prior_params, verbose)
            if scheduler_updated:
                a=0

            if (i+1)%plot_every == 0:
                printd(f'\nfit, plotting state at iter {i+1}.\ndataset: {graph.name}, ,model: {self.model_name}')
                acc_tracker.plot_intermediate(num_blanks_second, num_blanks_first, i+1, **params)
        # ========= end fit loop ================

       
            if i == n_back_forth-1:
                printd(f'\nfit end, no early stopping')
            
        losses = (losses_first, losses_second)
        
        return losses, acc_tracker.accuracies_test, acc_tracker.accuracies_val
        
  #todo: refactor fit to have an acc tracker, simething to take care of the val accuracy management since we don't use it and it takes up a lot of space



    def init_node_feats(self, init_type,num_nodes=None, graph_given=None, node_feats_given=None):
        return init_node_feats(num_nodes=num_nodes, 
                               num_feats=self.dim_feat, 
                               lorenz=self.lorenz, 
                               init_type=init_type,
                               graph_given=graph_given, 
                               node_feats_given=node_feats_given, 
                               device=self.device)
    
    def star_ll_nodes(
            self, 
            graph, 
            nodes, 
            with_prior=True, 
            prior=None,
            test_orig_sum=False):
        ''' calculates the star prob for a list of several nodes'''
        
        if with_prior:    
            if prior is not None:
                the_prior = prior
            elif self.prior is not None:
                the_prior = self.prior
            else:
                the_prior = None
        else:
            the_prior = None

        return star_ll_nodes(graph, self.lorenz, nodes, self.device, the_prior)
        
    

    def save_state(self, data, configs_dict, inner_folder='clamiter', suffix=''):
        '''save trainer feats prior and config'''    
        model_save_path = f'{inner_folder}/{suffix}'    
        utils.save_feats_prior_hypers(x=data.x, prior=self.prior,configs_dict=configs_dict,folder_path=model_save_path, overwrite=True)
    
    def plot_state(
            self,
            graph, 
            things_to_plot=['adj','feats', '2dgraphs', 'losses'],
            community_affiliation=None, 
            omitted_dyads=None, 
            anomalies_gt=None, 
            i=0,
            n_iter=0,
            calling_function_name=None,
            **kwargs):
        
        '''plot the state of the model'''
        # printd(f'\nin plot state, calling function is {calling_function_name}')
        plot_optimization_stage(
                            self.prior, 
                            graph, 
                            self.lorenz, 
                            things_to_plot,
                            community_affiliation, 
                            omitted_dyads, 
                            calling_function_name=calling_function_name,
                            **kwargs)
        

# 88 88b 88 88 888888     888888 888888    db    888888 .dP"Y8 
# 88 88Yb88 88   88       88__   88__     dPYb     88   `Ybo." 
# 88 88 Y88 88   88       88""   88""    dP__Yb    88   o.`Y8b 
# 88 88  Y8 88   88       88     888888 dP""""Yb   88   8bodP' 


def init_node_feats(num_feats, lorenz, init_type, device, num_nodes=None, graph_given=None, node_feats_given=None, node_mask=None):
        '''initializes node features according to init_type.
        if graph is given nodes need not be and vise versa'''
        if num_nodes is None and graph_given is None:
            raise ValueError('in init_node_feats: num_nodes and graph_given are None')
        if num_nodes is None:
            num_nodes = graph_given.num_nodes
        
        if init_type == 'random': # uniform [0,1)
            node_feats = torch.rand([num_nodes, num_feats], requires_grad=False)
        elif init_type == 'zero':
            node_feats = torch.zeros([num_nodes, num_feats], requires_grad=False)
        elif init_type == 'ones':
            node_feats = torch.ones([num_nodes, num_feats], requires_grad=False)
        elif init_type == 'small_gaus':
            if lorenz:
                #in a lightcone time can have only positive values and space can have both positive and negative.
                mean_space = torch.tensor([0.0] * (num_feats//2))
                std_space = 0.1
                mean_time = torch.tensor([0.9] * (num_feats//2))
                std_time = 0.1
                
                node_feats_time = torch.randn(num_nodes, num_feats//2, requires_grad=False)
                node_feats_time = node_feats_time * std_time + mean_time
                
                node_feats_space = torch.randn(num_nodes, num_feats//2, requires_grad=False)
                node_feats_space = node_feats_space * std_space + mean_space

                node_feats = torch.cat([node_feats_time, node_feats_space], dim=1)

            else:
                mean = torch.tensor([0.9] * num_feats)
                std = 0.1
                node_feats = torch.randn(num_nodes, num_feats, requires_grad=False)
                node_feats = relu(node_feats * std + mean)
            
        elif init_type == 'minimal_neigh':
            if graph_given is None:
                raise ValueError('in init_node_feats: graph_given is None')
            if lorenz == False:
                node_feats = 0.1*torch.ones([graph_given.num_nodes, num_feats]) 
                minimal_neighborhoods = k_minimal_neighborhoods(graph_given, k=num_feats)
                for i, neighborhood in enumerate(minimal_neighborhoods):
                    node_feats[neighborhood, i] = 1
            
            if lorenz == True:
                '''if it's lorenz, start s communities with minimal neigh'''
                node_feats_time = 0.1*torch.ones([num_nodes, num_feats//2], requires_grad=False)
                minimal_neighborhoods = k_minimal_neighborhoods(graph_given, k=num_feats//2)
                for i, neighborhood in enumerate(minimal_neighborhoods):
                    node_feats_time[neighborhood, i] = 1
                std = 0.1 * min(node_feats_time.std(), 1)
                node_feats_space = std*torch.randn(num_nodes, num_feats//2, requires_grad=False)
                node_feats = torch.cat([node_feats_time, node_feats_space], dim=1)

        elif init_type == 'with_bigclam':
            if lorenz == False:
                raise NotImplementedError("in init_node_feats: can't initilalize bigclam with bigclam. please use init_type='with_given' instead.")
            else:
                assert node_feats_given.shape[1] == num_feats//2, 'in init_node_feats: node_feats_given are clam and should be half the size of the gam nodes we define with them'
                assert node_feats_given.shape[0] == num_nodes, 'in init_node_feats: node_feats_given should have the same number of nodes'
                node_feats_time = node_feats_given
                std = 0.1 * min(node_feats_time.std(), 1)
                node_feats_space = std*torch.randn(num_nodes, num_feats//2, requires_grad=False)
                node_feats = torch.cat([node_feats_time, node_feats_space], dim=1)
        
        elif init_type == 'with_given':
            '''this will initialize the '''
            if node_feats_given is None:
                raise ValueError('in init_node_feats: node_feats_given is None')
            if node_feats_given.shape[1] != num_feats:
                raise ValueError(f'in init_node_feats: node_feats_given has {node_feats_given.shape[1]} features but should have {num_feats}')
            if node_feats_given.shape[0] != num_nodes:
                raise ValueError(f'in init_node_feats: node_feats_given has {node_feats_given.shape[0]} nodes but should have {num_nodes}')
            node_feats = node_feats_given
        else:
            raise NotImplementedError('init_type not implemented')
        
        return node_feats.to(device) 




#    db    88b 88  dP"Yb  8b    d8    db    88     Yb  dP 
#   dPYb   88Yb88 dP   Yb 88b  d88   dPYb   88      YbdP  
#  dP__Yb  88 Y88 Yb   dP 88YbdP88  dP__Yb  88  .o   8P   
# dP""""Yb 88  Y8  YbodP  88 YY 88 dP""""Yb 88ood8  dP    



# .dP"Y8 888888    db    88""Yb     88""Yb 88""Yb  dP"Yb  88""Yb 
# `Ybo."   88     dPYb   88__dP     88__dP 88__dP dP   Yb 88__dP 
# o.`Y8b   88    dP__Yb  88"Yb      88"""  88"Yb  Yb   dP 88""Yb 
# 8bodP'   88   dP""""Yb 88  Yb     88     88  Yb  YbodP  88oodP 

class StarProb(MessagePassing):

    def __init__(self, lorenz, device, prior=None):
        super(StarProb, self).__init__(aggr='add')
        self.lorenz = lorenz
        self.device = device
        self.prior = prior

    def forward(self, graph):
        '''calculate the probability of the star graph for all nodes'''
        
        dim_feat = graph.x.shape[1]
        if self.lorenz:
            self.B = (torch.concatenate([torch.ones(dim_feat//2), -torch.ones(dim_feat//2)])).to(self.device)
        else:
            self.B = (torch.ones(dim_feat)).to(self.device)

        with torch.no_grad():
            tbr = self.propagate(edge_index=graph.edge_index_original, x=graph.x, global_features=())
            #todo: make sure that the attr didn't change
            if self.prior is not None:
                self.prior.eval()
                if self.prior.attr_opt:
                    feats_for_prior = torch.cat([graph.x, graph.attr], dim=1)
                else:
                    feats_for_prior = graph.x
                    #todo: check here if tbr changes for star_prior
                tbr = tbr + self.prior.forward_ll(feats_for_prior, sum=False)

        return tbr

    def message(self, x_j, x_i):
        '''the operation from every neighbor to the central node.
        x_i is the central node and x_j is the neighbor'''
        
        x_inner_product = torch.einsum('ij,ij->i', x_i, self.B*x_j)
        M = torch.max(x_inner_product)
        msg = torch.log(torch.exp(x_inner_product - M) - torch.exp(-M) + eps).unsqueeze(1) + M
        # msg = torch.log(torch.exp(x_inner_product) - 1).unsqueeze(1)
        return msg


    def update(self, aggr_out, x):
        '''each x is also multiplied by the sum of all nodes'''
        aggr_out= aggr_out.squeeze()
        global_term = torch.sum(x, dim=0)
        term_inner_with_all = (global_term*self.B*x).sum(dim=1)
        norms =  (x*self.B*x).sum(dim=1)
        update = 0.5*(aggr_out - term_inner_with_all + norms)
        #todo: try and compare to real no trick probability
        return update
        
def star_ll_nodes(graph, lorenz, nodes=None, device=torch.device('cpu'), prior=None):
    ''' calcs the star prob with an mpnn for a list of nodes. wrapper function for the message passing class'''
    #todo: can be made cheaper if we mask the nodes in the mpnn process.
    if nodes is None:
        nodes = torch.arange(graph.x.shape[0]).to(device)
    tbr = StarProb(lorenz, device, prior)(graph)
    return tbr[nodes]



def all_types_classify(clamiter, ds_with_anomalies, ll_types=['vanilla_star', 'prior', 'prior_star'], prior=None, verbose=False):
    '''classify the anomalies using all the methods'''
    tbr = []
    if 'vanilla_star' in ll_types:
        roc_auc_vanilla_star = classify_anomalies(clamiter, ds_with_anomalies, ll_type='vanilla_star', prior=prior, ret_ap=False, verbose=verbose)
        tbr.append(roc_auc_vanilla_star)
    if 'prior' in ll_types:
        roc_auc_prior = classify_anomalies(clamiter, ds_with_anomalies, ll_type='prior', prior=prior, ret_ap=False, verbose=verbose)
        tbr.append(roc_auc_prior)
    if 'prior_star' in ll_types:
        roc_auc_prior_star = classify_anomalies(clamiter, ds_with_anomalies, ll_type='prior_star', prior=prior, ret_ap=False, verbose=verbose)
        tbr.append(roc_auc_prior_star)
    
    return tbr

def classify_anomalies(clamiter, ds_with_anomalies, ll_type='vanilla_star', prior=None, ret_ap=False, verbose=False):
    
    if prior is not None:
        the_prior = prior
    else:
        if clamiter.prior is not None:
            the_prior = clamiter.prior

    # MASKS AND INDICES
    anomalies_index = torch.where(~ds_with_anomalies.gt_nomalous)[0]
    normal_index = torch.where(ds_with_anomalies.gt_nomalous)[0] #! right now gt_nomalous==False for anomalies so anomalies index is actually normal index
    
    # CALCULATE LOG PROBABILITY
    if ll_type == 'vanilla_star':
        calc_ll = lambda index: clamiter.star_ll_nodes(
                                                ds_with_anomalies, 
                                                index,
                                                with_prior=False)
        
    elif ll_type == 'prior':
        the_prior.eval()
        #todo: check that attr is not changed in the optimization
        if the_prior.attr_opt:
            feats_for_prior = torch.cat([ds_with_anomalies.x, ds_with_anomalies.attr], dim=1)
        else:
            feats_for_prior = ds_with_anomalies.x
        calc_ll = lambda index: the_prior.forward_ll(
                                                feats_for_prior[index], 
                                                sum=False)
    elif ll_type == 'prior_star':
        the_prior.eval()
        calc_ll = lambda index: clamiter.star_ll_nodes(
                                                ds_with_anomalies, 
                                                index,
                                                prior=the_prior,
                                                with_prior=True)
    else:
        raise ValueError(f'll_type {ll_type} not recognized')
    
    with torch.no_grad():
        #! anomalies index is the 
        #* look something is working so that's good. don't ruin it...
        ll_normal = calc_ll(normal_index)
        ll_anomalies = calc_ll(anomalies_index)
        
    # ========================================

    # COMAPRE PROBABILITIES AND CLASSIFY
    #TODO: gt_anomalies are False for anomalies. should change that

    normals_gt_mask = ds_with_anomalies.gt_nomalous.cpu().numpy()
    test_scores = torch.zeros_like(ds_with_anomalies.gt_nomalous, dtype=torch.float32)
    test_scores[ds_with_anomalies.gt_nomalous] = ll_normal
    test_scores[~ds_with_anomalies.gt_nomalous] = ll_anomalies
    

    fpr, tpr, thresholds = roc_curve(normals_gt_mask, test_scores.cpu().numpy())
    roc_auc = roc_auc_score(normals_gt_mask, test_scores.cpu().numpy())

    if ret_ap:
        ap = average_precision_score(normals_gt_mask, test_scores.cpu().numpy())

    if verbose:
        min_dist, best_threshold = utils.plot_roc_curve(fpr, tpr, thresholds)
        print(f'classification method: {ll_type} \nap_score:{ap}, ROC AUC: {roc_auc}, min_dist: {min_dist}, best_threshold: {best_threshold}')
    # =============================================================
    if ret_ap:
        return roc_auc, ap
    else:
        return roc_auc



# 888888 88""Yb    db     dP""b8 88  dP 888888 88""Yb 
#   88   88__dP   dPYb   dP   `" 88odP  88__   88__dP 
#   88   88"Yb   dP__Yb  Yb      88"Yb  88""   88"Yb  
#   88   88  Yb dP""""Yb  YboodP 88  Yb 888888 88  Yb 
# make both acc tracker and fit_wrapper

class AccTrack:
    def __init__(self, 
                 clamiter, 
                 graph, 
                 task, 
                 calling_function_name, 
                 acc_every,
                 **kwargs 
                 ):
        '''should track all of the accuracies and losses during an optimization in one of the fit functions. should also save the best state, validation accuracies...'''
        #todo: IF THE TASK IS NONE, ACC TRACKER SHOULD TRACK LOSSES. 
        # trainer should be an option as well
        self.task = task # trainer knows the model and dataset. does it matter? i think not.  
        self.acc_every = acc_every
        
        self.clamiter = clamiter
        self.graph = graph
        self.calling_function_name = calling_function_name
        # Initialize tracking variables

        self.accuracies_test = None
        self.accuracies_val = None
        self.losses = []
        # different conditions if its a vanilla, not vanilla, feat, prior fit even...... 
        # what is the difference between the prior and feats when collecting data? vanilla stuff shouldn't change when optimizing the prior... but we can collect them the same way why not?
        if task == 'anomaly':
            
            if self.clamiter.prior is not None:

                # ANOMALY acc test init
                self.accuracies_test = {'vanilla_star': [], 'prior': [], 'prior_star': []}
                
                # LINK earlystop
                
            else: # VANILLA init auc early stop 
                self.accuracies_test = {'vanilla_star': []}
                
    
    
        elif task == 'link_prediction':
            #* the code needs to work on datasets on which the dyads to omit is not known, therefore, val and test dyads to omit need to be sent down separately as if the test nodes are really not known, so that there can be an option to check only the validation nodes...
            self.lorenz = kwargs.get('lorenz', None)
            self.omitted_dyads = kwargs.get('omitted_dyads', None) 
            if self.omitted_dyads is None or self.lorenz is None:
                raise ValueError('in AccTrack, omitted_dyads and lorenz should be given')
            
            self.accuracies_test = {'auc': []}                                                                      
            self.accuracies_val = {'auc': []}


        elif task == 'distance':
            self.d = kwargs.get('d', None)
            self.calculate_cut = kwargs.get('calculate_cut', False)
            self.metric_log_cut = lambda data : utils.cut_log_data(data, self.clamiter.lorenz, d=self.d, return_d=True)
            self.accuracies_test = {'log_cut': []}
            
            if self.calculate_cut:
                self.metric_cut = lambda data : utils.cut_distance_data(data, self.clamiter.lorenz)
                self.accuracies_test = {'cut': [], 'log_cut': []}

            self.accuracies_val = []

            self.patiance_steps = 0

        else:
            self.accuracies_test = {'losses': self.losses }
            self.accuracies_val = None
            
            
    def get_intermediate_accuracies(self, losses, measurement_interval=None):
        ''' get the accuracies after a certain number of iterations'''
        
        if measurement_interval is None:
            measurement_interval = self.acc_every

        self.losses = losses
   
        if self.task == 'anomaly':
            if self.clamiter.prior is not None:              
                auc_vanilla_star_anomaly, auc_prior_anomaly, auc_prior_star_anomaly = all_types_classify(self.clamiter, self.graph, ll_types=['vanilla_star', 'prior', 'prior_star'])
                
            else: # VANILLA
                #TEST accuracy
                auc_vanilla_star_anomaly = all_types_classify(self.clamiter, self.graph, ll_types=['vanilla_star'])[0]
                    
            for key in self.accuracies_test.keys():
                self.accuracies_test[key] += [locals()[f'auc_{key}_anomaly']]*measurement_interval
                self.accuracies_test[key][-1] += eps
            
                
        
                    
  
        elif self.task == 'link_prediction':
            # the test and val set are independent of each other as the test set is "given" and the validation is chosen 
            # todo: add another method for when there is a prior in which you multiply with the prior probability
            auc_score_val = roc_of_omitted_dyads(
                        self.graph.x, 
                        self.lorenz, 
                        self.omitted_dyads[0])['auc']
            
            auc_score_test = roc_of_omitted_dyads(
                        self.graph.x, 
                        self.lorenz, 
                        self.omitted_dyads[1])['auc']
            
            self.accuracies_val['auc'].append(auc_score_val)

            self.accuracies_test['auc'].append(auc_score_test)



        elif self.task == 'distance':
            l2_norm = utils.relative_l2_distance_data(self.graph, self.clamiter.lorenz, verbose=False).cpu().item()
            self.accuracies_val.append(l2_norm)
          
            
            # estimation on what has been
            log_cut = self.metric_log_cut(self.graph)
            self.accuracies_test['log_cut'] += [log_cut]*measurement_interval
            
            if self.calculate_cut:
                cut = self.metric_cut(self.graph)
                self.accuracies_test['cut'] += [cut]*measurement_interval

        else:
            self.accuracies_test = {'losses': self.losses}


    def best_features_from_val(self):
        ''' if you were keeping track of the best features, this will return them'''
        #! used only with val
        if self.task == 'anomaly':
            # ANOMALY DETECTION HAS NO VALIDATION SET
            if self.clamiter.prior is not None:
               pass
      
            

               
            else: # vanilla. should take the best one anyway
                pass
 
                            
        
        elif self.task == 'distance':
            best_val_distance = max(self.accuracies_val)
            if best_val_distance - self.first_val_dist < 0:
                printd(f'\n fit prior. no improvement, best distance == first distance')
                # graph.x = first_x.clone()
            # del first_x
    

    def append_accuracies(self, 
                          accuracies_test_1st, 
                          accuracies_test_2nd,
                          accuracies_val_1st=None,
                          accuracies_val_2nd=None):
        
        for key in self.accuracies_test.keys():
            self.accuracies_test[key] += accuracies_test_1st[key] + accuracies_test_2nd[key]
        if self.accuracies_val is not None:
            self.accuracies_val += accuracies_val_1st + accuracies_val_2nd

    def print_accuracies(self):
        '''print the accuracies at the end of the iteration'''
        if self.task is not None:
            print(self.task.upper(), end=' ')

        if self.accuracies_val is not None:    
            print(f'VAL:')
            print('latest:')
            print(self.get_latest_val_acc())
            print('best:')
            print(self.get_best_val_acc())

            
        print(f'TEST accuracy.')
        print('Latest:')
        for key, value in self.get_latest_test_acc().items():
            print(f'{key}: {value}', end=' ')
        print('\nBest:')
        best_accs, iters_best = self.get_best_test_acc()
        for key in best_accs.keys():
            print(f'{key}: {best_accs[key]} at iteration {iters_best[key]}')
            

    def get_best_test_acc(self):
        best_acc = {}
        best_acc_iter = {}
        for key, list in self.accuracies_test.items():
            try:
                best_acc[key] = max(list)
                best_acc_iter[key] = list.index(best_acc[key])
            except ValueError:
                best_acc[key] = None
                best_acc_iter[key] = None
                
        return best_acc, best_acc_iter


    def get_best_val_acc(self):
        best_acc = {}
        best_acc_iter = {}
        for key, list in self.accuracies_val.items():
            try:
                best_acc[key] = max(list)
                best_acc_iter[key] = list.index(best_acc[key])
            except ValueError:
                best_acc[key] = None
                best_acc_iter[key] = None
        return best_acc, best_acc_iter
    

    def get_latest_val_acc(self):
        latest_acc = {}
        for key, list in self.accuracies_val.items():
            try:
                latest_acc[key] = list[-1]
            except ValueError:
                latest_acc[key] = None
            return latest_acc
        

    def get_latest_test_acc(self):
        latest_acc = {}
        for key, list in self.accuracies_test.items():
            try:
                latest_acc[key] = list[-1]
            except ValueError:
                latest_acc[key] = None
        
        return latest_acc


    def plot_intermediate(self, 
                          n_iter_first=None, 
                          n_iter_second=None, 
                          n_back_forth=None,
                          **kwargs):
        
        if self.accuracies_test.keys() != ['losses']:
            plot_test_accuracies(self.accuracies_test, n_iter_first, n_iter_second, n_back_forth)

        things_to_plot = []
        if self.graph.x.shape[1] <= 6:
            things_to_plot.append('2dgraphs')
        if self.task == 'distance':
            things_to_plot.append('adj')
        self.clamiter.plot_state(
                self.graph, 
                things_to_plot=things_to_plot,
                community_affiliation=self.graph.y, 
                calling_function_name='fit_feats',
                **kwargs)
            

# 888888    db    88""Yb 88     Yb  dP .dP"Y8 888888  dP"Yb  88""Yb 
# 88__     dPYb   88__dP 88      YbdP  `Ybo."   88   dP   Yb 88__dP 
# 88""    dP__Yb  88"Yb  88  .o   8P   o.`Y8b   88   Yb   dP 88"""  
# 888888 dP""""Yb 88  Yb 88ood8  dP    8bodP'   88    YbodP  88     
class EarlyStop:
    def __init__(self, clamiter, graph_name, model_name, task, patience, **kwargs):
        self.clamiter = clamiter
        self.graph_name = graph_name
        self.model_name = model_name
        self.task = task
        self.patience = patience

        now = datetime.datetime.now()
        data_time = now.strftime("%m-%d-%Y_%H-%M-%S")
        self.save_folder = f'{self.task}/{self.graph_name}/{self.model_name}/{data_time}'
        self.delete_folder = os.path.dirname(os.path.abspath(__file__)) + '/checkpoints/' + self.save_folder

    def update(self, accuracies_val):
        pass

    def check(self):
        pass        

    def save(self, graph):
        pass


# if task == 'distance':
                
            #     accuracies_test['log_cut'] += accuracies_test['log_cut'] + accuracies_test_epoch_1st['log_cut'] + accuracies_test_epoch_2nd['log_cut']
                
            #     accuracies_test['cut'] += accuracies_test['cut'] + accuracies_test_epoch_1st['cut'] + accuracies_test_epoch_2nd['cut']

            #     accuracies_val = accuracies_val + accuracies_val_epoch_1st + accuracies_val_epoch_2nd

            #     last_val_dist = accuracies_val[-1]

                
            #     if last_val_dist < best_distance:
            #         best_distance = last_val_dist
            #         patiance = max(patiance - 1, 0)
            #     else:
            #         patiance += 1
            #         if early_stop_fit!=0:
            #             if patiance >= early_stop_fit:
            #                 printd(f'\nfit_prior early stopping at iteration {i}')
            #                 break
                
            # # ========= distance collect results  ====================

        
            # if task == 'link_prediction':
            #     # here omit some of the dyads like 20% of all of the dyads and do 
            #     auc_score = roc_of_omitted_dyads(
            #                 graph.x, 
            #                 self.lorenz, 
            #                 dyads_to_omit)['auc']
            #     accuracies_test.append(auc_score)
            # # ===== end link collect acc =====
            
            # elif task == 'anomaly':                
            #     if i%acc_every == 0:
            #         # calculate intermediate accuracies
                    
            #         if dyads_to_omit is not None:
            #             # LINK get best per round
            #             best_acc_val_2nd = max(accuracies_val_epoch_2nd['prior_auc'])
            #             auc_link = best_acc_val_2nd
                        
            #             auc_vanilla_star, auc_prior, auc_prior_star = all_types_classify(self, graph, ll_types=['vanilla_star', 'prior', 'prior_star'])

            #             # CHECK IMPROVE
            #             if auc_link - best_auc_link > 0:
            #                 best_auc_link = auc_link
            #                 iter_best_link = i
            #                 count_not_improved_link = max(count_not_improved_link - 1, 0)
            #                 utils.delete_file_by_str(delete_folder, 'vanilla_star_auc')
            #                 utils.delete_file_by_str(delete_folder, 'prior_auc')
            #                 utils.delete_file_by_str(delete_folder, 'prior_star_auc')

            #                 saved_auc_vanilla_star_anomaly = auc_vanilla_star
            #                 saved_auc_prior_anomaly = auc_prior
            #                 saved_auc_prior_star_anomaly = auc_prior_star

            #                 self.save_state(graph, inner_folder=f'{save_folder}', configs_dict=configs_dict, suffix=f'vanilla_star_auc_{auc_vanilla_star:.3f}')
            #                 self.save_state(graph, inner_folder=f'{save_folder}', configs_dict=configs_dict, suffix=f'prior_auc_{auc_prior:.3f}')
            #                 self.save_state(graph, inner_folder=f'{save_folder}', configs_dict=configs_dict, suffix=f'prior_star_auc_{auc_prior_star:.3f}')
            #             else:
            #                 count_not_improved_link += 1
            #         # === end CHECK IMPROVE ======
                        
            #             if verbose:
            #                 printd(f'\nfit function after checking improvement {i= }; \n {count_not_improved_link=}, {iter_best_link= }\n\n ANOMALY ACC:\n {auc_vanilla_star= }, {auc_prior= }, {auc_prior_star= }' )
                        
            #             # STOPPING CONDITION
            #             if early_stop_fit!=0 and count_not_improved_link >= early_stop_fit:
            #                 printd(f'\nfit. early stopping at iteration {i+1}')
            #                 break
                
            #     accuracies_test['vanilla_star'] += accuracies_test_epoch_1st['vanilla_star'] + accuracies_test_epoch_2nd['vanilla_star']
            #     accuracies_test['prior'] += accuracies_test_epoch_1st['prior'] + accuracies_test_epoch_2nd['prior']
            #     accuracies_test['prior_star'] += accuracies_test_epoch_1st['prior_star'] + accuracies_test_epoch_2nd['prior_star']

            #     if accuracies_val is not None:
            #         accuracies_val += accuracies_val_epoch_1st + accuracies_val_epoch_2nd
            


# 88      dP"Yb     db    8888b.      8b    d8  dP"Yb  8888b.  888888 88     
# 88     dP   Yb   dPYb    8I  Yb     88b  d88 dP   Yb  8I  Yb 88__   88     
# 88  .o Yb   dP  dP__Yb   8I  dY     88YbdP88 Yb   dP  8I  dY 88""   88  .o 
# 88ood8  YbodP  dP""""Yb 8888Y"      88 YY 88  YbodP  8888Y"  888888 88ood8 
   


def load_model(path_in_checkpoints, device=torch.device('cpu') ,verbose=False):
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoints_dir = os.path.join(script_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoints_dir, path_in_checkpoints)
    checkpoint_name = path_in_checkpoints.split('/')[-1]
    
    
    # DESCRIPTIVE CHARACTERISTICS
    model_name = checkpoint_name.split('_')[1]
    ds_name = checkpoint_name.split('_')[0]
    if model_name == 'bigclam':
        lorenz = False
        vanilla = True
    elif model_name == 'pclam':
        lorenz = False
        vanilla = False
    elif model_name == 'iegam':
        lorenz = True
        vanilla = True
    elif model_name == 'piegam':
        lorenz = True
        vanilla = False
    else:
        raise ValueError(f"in load model, Model name {model_name} not recognized")
    
    # LOAD CLAMITER
    #TODO: this should be a method of clamiter
    config_dict = json.load(open(os.path.join(checkpoint_path, 'config.json')))
    clamiter_params = config_dict['clamiter_init']

    clamiter = PCLAMIter(**clamiter_params, lorenz=lorenz, vanilla=vanilla, device=device)
    
    if vanilla == False:
        for file in os.listdir(checkpoint_path):
            if file.startswith('prior'):
                prior_file = file
        clamiter.prior.model.load_state_dict(
            torch.load(
                os.path.join(checkpoint_path, prior_file), 
                map_location=clamiter.device)) 
    # ================================================================

    # DATA
    #* x and masks are saved in a data object (everything but the edge index)
    # data_path = os.path.join(checkpoint_path, 'x_and_masks.pth')
    data_path = os.path.join(checkpoint_path, 'data.pth')
    data = torch.load(data_path, map_location=device)
    # x = torch.load(data_path, map_location=clamiter.device)
    # data = import_dataset(ds_name)
    
    # if type(x) == Data:
    #     feats, train_node_mask  = x
    #     data.x = feats
    #     data.train_node_mask = train_node_mask
    # else:    
    #     data.x = x

    #* plot the features with the prior
    if verbose:
        clamiter.plot_state(data, community_affiliation=data.y, calling_function_name='ci.load_model')

    return clamiter, data, config_dict, model_name, ds_name

#  dP""b8 88        db    8b    d8     88      dP"Yb  .dP"Y8 .dP"Y8 
# dP   `" 88       dPYb   88b  d88     88     dP   Yb `Ybo." `Ybo." 
# Yb      88  .o  dP__Yb  88YbdP88     88  .o Yb   dP o.`Y8b o.`Y8b 
#  YboodP 88ood8 dP""""Yb 88 YY 88     88ood8  YbodP  8bodP' 8bodP' 

def clam_loss(graph, lorenz):
        ''' loss of the vanilla bigclam model on the featoptimizer's edge array
        '''
        dim_feat=graph.x.shape[1]
        if lorenz:
            B = torch.concatenate([torch.ones(dim_feat//2), -torch.ones(dim_feat//2)]).to(graph.x.device)
        else:
            B = torch.ones(dim_feat).to(graph.x.device)

        sum_graph_feats = torch.sum(graph.x, dim=0)
        
        term_glob_per_node = sum_graph_feats@(B*graph.x).T
        # term_sq_nodes_per_node = (graph.x**2).sum(dim=1)
        norms = (graph.x*B*graph.x).sum(dim=1)

        edges_feats_0, edges_feats_1 = edges_by_coords(graph)
        
        fufv = (edges_feats_0*B*edges_feats_1).sum(dim=1) # inner prods of all nodes in a row
        if (fufv < -10**-6).any():
            #* cases in which this could happen: if there is a node drop that is big
            raise ValueError('fufv has negative vals - crossed the potential wall at 0')
        # this is needed for all edges
        #! so far this is true for all edges. now what must be did it to make the loss out of fufv of the edges
        fufv_non_omitted = fufv[graph.edge_attr==1]
        fufv_omitted = fufv[graph.edge_attr==0] 
        
        M = torch.max(fufv_non_omitted)
        term_neighbors_non_omitted = M + torch.log(torch.exp(fufv_non_omitted-M)- torch.exp(-M)+1e-10)
        # old_term_neighbors_non_omitted = torch.log(torch.exp(fufv_non_omitted)-1 + 1e-10)
        # if edges are omitted, all of the edges with edge_attr = 1 will be calculated with the normal loss, and all of the edges with attr 0 we just sum their term_per_edge
        loss = 0.5*(torch.sum(term_neighbors_non_omitted) - torch.sum(term_glob_per_node) + torch.sum(norms) + torch.sum(fufv_omitted))

        if torch.isnan(loss):
            raise ValueError('in clam_loss: loss is nan')
        if torch.isinf(loss):
            raise ValueError('in clam_loss: loss is inf')
        
        return loss


def clam_loss_prior(graph, lorenz, prior):
        '''loss with prior.
            in normflows package, the output of forward_kld is minus the log likelihood'''
        vanilla_loss = clam_loss(graph, lorenz)
        if prior.attr_opt:
            feats_for_prior = torch.cat([graph.x, graph.attr], dim=1)
        else:
            feats_for_prior = graph.x
        log_prior_loss = prior.forward_ll(feats_for_prior)
        
        loss = 0.5 * vanilla_loss + log_prior_loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            a=0

        return loss





# 888888 888888 .dP"Y8 888888 88 88b 88  dP""b8 
#   88   88__   `Ybo."   88   88 88Yb88 dP   `" 
#   88   88""   o.`Y8b   88   88 88 Y88 Yb  "88 
#   88   888888 8bodP'   88   88 88  Y8  YboodP 

def star_prob_direct_sum(graph, nodes, lorenz):
    '''test if the sum without the reindexing trick is the same as with the trick'''
    assert graph.is_undirected(), 'in star prob direct sum: graph is directed'

    if lorenz:
        B = torch.cat([torch.ones(graph.x.shape[1]//2), -torch.ones(graph.x.shape[1]//2)]).to(graph.x.device)
    else:
        B = torch.ones(graph.x.shape[1]).to(graph.x.device)
    star_losses = torch.zeros(len(nodes), dtype=torch.float32).to(graph.x.device)
    for i,node in enumerate(nodes):

        edge_index_neighborhood = graph.edge_index[:, graph.edge_index[0] == node]
        nodes_neighborhood = torch.unique(edge_index_neighborhood.flatten())
        
        nodes_neighborhood = torch.unique(edge_index_neighborhood.flatten()) # seems not right, like it would have the node itself in here
        nodes_neighborhood_mask = utils.mask_from_node_list(nodes_neighborhood, graph.x.shape[0])
        nodes_neighborhood_mask[node] = False

        x_neighborhood = graph.x[nodes_neighborhood_mask]
        x_node = graph.x[node]
        
        fufv = torch.einsum('j,ij->i', B*x_node, x_neighborhood)
        term_neighbors = torch.log(1-torch.exp(-fufv))
        
        strangers_mask = ~nodes_neighborhood_mask
        strangers_mask[node] = False
        x_strangers = graph.x[strangers_mask]
        term_strangers = torch.einsum('j, ij -> i', B*x_node, x_strangers) 
        # if test_orig_sum: # test sum without trick
        
        star_loss_vanilla_hard_code = 0.5*(torch.sum(term_neighbors) - torch.sum(term_strangers) )
        star_losses[i] = star_loss_vanilla_hard_code      
    return star_losses
 