import os
import numpy as np
import random
import torch
import torch.nn as nn
from model import *
import arguments
import utils.load_dataset
import utils.data_loader
import utils.metrics
from utils.early_stop import EarlyStopping, Stop_args, StopVariable
import time
import argparse
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def para(args): 
    if args.dataset == 'yahooR3': 
        args.training_args = {'batch_size': 1024, 'epochs': 2000, 'patience': 80, 'block_batch': [6000, 500]}
        args.InterD_model_args = {"emb_dim": 10, "learning_rate": 0.0005, "weight_decay": 0.01}
        args.MF_model_args = {"emb_dim": 10, "learning_rate": 5e-6, "weight_decay": 1, 'patience': 80}
        args.Auto_model_args = {"emb_dim": 10, "learning_rate": 0.0005, "weight_decay": 10, 'imputaion_lambda': 0.05, 'epoch': 500}
        args.weight1_model_args = { "learning_rate": 0.1, "weight_decay": 0.001}
        args.weight2_model_args = { "learning_rate": 0.001, "weight_decay": 0.01}
        args.imputation_model_args = { "learning_rate": 0.1, "weight_decay": 0.0001}
    elif args.dataset == 'coat':
        args.training_args = {'batch_size': 128, 'epochs': 2000, 'patience': 100, 'block_batch': [256, 256]}
        args.InterD_model_args = {"emb_dim": 10, "learning_rate": 0.01, "weight_decay": 0}
        args.MF_model_args = {"emb_dim": 10, "learning_rate": 0.001, "weight_decay": 0, 'patience': 80}
        args.Auto_model_args = {"emb_dim": 10, "learning_rate": 1e-5, "weight_decay": 0, 'imputaion_lambda': 0.01, 'epoch': 2000}
        args.weight1_model_args = { "learning_rate": 1e-5, "weight_decay": 1e-5}
        args.weight2_model_args = { "learning_rate": 1e-5, "weight_decay": 0}
        args.imputation_model_args = { "learning_rate": 0.0001, "weight_decay": 0.001}
        args.gama = 0.05
        args.gama2 = 1
        args.beta = 0.05
    else: 
        print('invalid arguments')
        os._exit()

def both_test(loader, model_name, testname, K = 5, dataset = "None"):
    test_users = torch.empty(0, dtype=torch.int64).to(device)
    test_items = torch.empty(0, dtype=torch.int64).to(device)
    test_pre_ratings = torch.empty(0).to(device)
    test_ratings = torch.empty(0).to(device)
    ndcg_ratings = torch.empty(0).to(device)
    ndcg_item = torch.empty(0).to(device)
    ut_dict={}
    pt_dict={}
    for batch_idx, (users, items, ratings) in enumerate(loader):
        pre_ratings = model_name(users, items)
        for i,u in enumerate(users):
            try:
                ut_dict[u.item()].append(ratings[i].item())
                pt_dict[u.item()].append(pre_ratings[i].item())
            except:
                ut_dict[u.item()]=[ratings[i].item()]
                pt_dict[u.item()]=[pre_ratings[i].item()]
        test_users = torch.cat((test_users, users))
        test_items = torch.cat((test_items, items))
        test_pre_ratings = torch.cat((test_pre_ratings, pre_ratings))
        test_ratings = torch.cat((test_ratings, ratings))

        pos_mask = torch.where(ratings>=torch.ones_like(ratings), torch.arange(0,len(ratings)).float().to(device), 100*torch.ones_like(ratings))
        pos_ind = pos_mask[pos_mask != 100].long()
        users_ndcg = torch.index_select(users, 0, pos_ind)
        ratings_ndcg = model_name.allrank(users_ndcg, bias_train)
        ndcg_ratings = torch.cat((ndcg_ratings, ratings_ndcg))
        items = torch.index_select(items.float(), 0, pos_ind)
        ndcg_item= torch.cat((ndcg_item, items))

    test_results = utils.metrics.evaluate(test_pre_ratings, test_ratings, ['MSE', 'AUC', 'Recall_Precision_NDCG@'], users=test_users, items=test_items, NDCG=(ndcg_ratings, ndcg_item), UAUC=(ut_dict, pt_dict))
    U = test_results['UAUC']
    N = test_results['NDCG']
    print(f'The performances of {testname[0]} on {testname[2]}ed test are UAUC: {str(U)}, NDCG: {str(N)}')
    return test_results, U, N

def train_and_eval_MF(bias_train, bias_validation, bias_test, unif_validation, unif_test, m, n, device = 'cuda', args=None):
    print('*************************Train biased model MF************************************')
    train_dense = bias_train.to_dense()
    # build data_loader. (block matrix data loader)
    train_loader = utils.data_loader.Block(bias_train, u_batch_size=args.training_args['block_batch'][0], i_batch_size=args.training_args['block_batch'][1], device=device)
    biasval_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(bias_validation), batch_size=args.training_args['batch_size'], shuffle=False, num_workers=0)
    biastest_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(bias_test), batch_size=args.training_args['batch_size'], shuffle=False, num_workers=0)

    val_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(unif_validation), batch_size=args.training_args['batch_size'], shuffle=False, num_workers=0)
    test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(unif_test), batch_size=args.training_args['batch_size'], shuffle=False, num_workers=0)
    
    # data shape
    # n_user, n_item = train_data.shape
    n_user, n_item = m, n

    
    # Base model and its optimizer. This optimizer is for optimize parameters in base model using the updated weights (true optimization).
    base_model = MF_MSE(n_user, n_item, dim=args.MF_model_args['emb_dim'], dropout=0).to(device)
    base_optimizer = torch.optim.SGD(base_model.params(), lr=args.MF_model_args['learning_rate'], weight_decay=0) # todo: other optimizer SGD

    
    # loss_criterion
    sum_criterion = nn.MSELoss(reduction='sum')

    # begin training
    stopping_args = Stop_args(patience=args.MF_model_args['patience'], max_epochs=args.training_args['epochs'] * 10)
    early_stopping = EarlyStopping(base_model, **stopping_args)

    for epo in range(args.training_args['epochs']*10):
        training_loss = 0
        lossf_sum = 0
        lossl_sum=0
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                users_train, items_train, y_train = train_loader.get_batch(users, items)
                base_model.train()
                # all pair
                all_pair = torch.cartesian_prod(users, items)
                users_all, items_all = all_pair[:,0], all_pair[:,1]
                # observation
                y_hat_obs = base_model(users_train, items_train)
                cost_obs = sum_criterion(y_hat_obs, y_train)
                loss = cost_obs + args.MF_model_args['weight_decay'] * base_model.l2_norm(users_all, items_all)
                
                base_optimizer.zero_grad()
                loss.backward()
                base_optimizer.step()

                training_loss += loss.item()

        base_model.eval()
        with torch.no_grad():
            # training metrics
            train_pre_ratings = torch.empty(0).to(device)
            train_ratings = torch.empty(0).to(device)
            for u_batch_idx, users in enumerate(train_loader.User_loader): 
                for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                    users_train, items_train, y_train= train_loader.get_batch(users, items)
                    pre_ratings = base_model(users_train, items_train)
                    train_pre_ratings = torch.cat((train_pre_ratings, pre_ratings))
                    train_ratings = torch.cat((train_ratings, y_train))

            train_results = utils.metrics.evaluate(train_pre_ratings, train_ratings, ['MSE'])

            # validation metrics on unifi
            val_pre_ratings = torch.empty(0).to(device)
            val_ratings = torch.empty(0).to(device)
            for batch_idx, (users, items, ratings) in enumerate(val_loader):
                pre_ratings = base_model(users, items)
                val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
                val_ratings = torch.cat((val_ratings, ratings))


            val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'AUC'])

        print('Epoch: {0:2d} / {1}, MF Traning log: {2}, Unbiased Validation: {3}'.format(epo, args.training_args['epochs']*10, ' '.join([key+':'+'%.3f'%train_results[key] for key in train_results]),' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))

        if epo>=20 and early_stopping.check([val_results['AUC']], epo):
            break
    # restore best model
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    base_model.load_state_dict(early_stopping.best_state)

    # test metrics on unbias
    print('#'*30)
    MF_unbias_result, U_MF_unbias, N_MF_unbias = both_test(test_loader, base_model, ('MF', 'MF', 'unbias'), K=5, dataset= args.dataset)

    # test metrics on bias
    MF_unbias_result, U_MF_bias, N_MF_bias = both_test(biastest_loader, base_model, ('MF', 'MF', 'bias'), K = 5, dataset= args.dataset)
    print('#'*30)
    return base_model, (U_MF_unbias, N_MF_unbias, U_MF_bias, N_MF_bias)

def train_and_eval_AutoDebias(bias_train, bias_validation, bias_test, unif_train, unif_validation, unif_test, m, n, device = 'cuda', args=None):
    print('*************************Train debiased model AutoDebias************************************')
    train_dense = bias_train.to_dense()
    users_unif = unif_train._indices()[0]
    items_unif = unif_train._indices()[1]
    y_unif = unif_train._values()
    
    # build data_loader. (block matrix data loader)
    train_loader = utils.data_loader.Block(bias_train, u_batch_size=args.training_args['block_batch'][0], i_batch_size=args.training_args['block_batch'][1], device=device)
    biasval_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(bias_validation), batch_size=args.training_args['batch_size'], shuffle=False, num_workers=0)
    biastest_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(bias_test), batch_size=args.training_args['batch_size'], shuffle=False, num_workers=0)

    val_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(unif_validation), batch_size=args.training_args['batch_size'], shuffle=False, num_workers=0)
    test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(unif_test), batch_size=args.training_args['batch_size'], shuffle=False, num_workers=0)
    
    n_user, n_item = m, n
    
    # Base model and its optimizer. This optimizer is for optimize parameters in base model using the updated weights (true optimization).
    base_model = MetaMF(n_user, n_item, dim=args.Auto_model_args['emb_dim'], dropout=0).to(device)
    base_optimizer = torch.optim.SGD(base_model.params(), lr=args.Auto_model_args['learning_rate'], weight_decay=0) # todo: other optimizer SGD

    # Weight model and its optimizer. This optimizer is for optimize parameters of weight model. 
    weight1_model = ThreeLinear(n_user, n_item, 2).to(device)
    weight1_optimizer = torch.optim.Adam(weight1_model.parameters(), lr=args.weight1_model_args['learning_rate'], weight_decay=args.weight1_model_args['weight_decay'])

    weight2_model = ThreeLinear(n_user, n_item, 2).to(device)
    weight2_optimizer = torch.optim.Adam(weight2_model.parameters(), lr=args.weight2_model_args['learning_rate'], weight_decay=args.weight2_model_args['weight_decay'])

    imputation_model = OneLinear(3).to(device)
    imputation_optimizer = torch.optim.Adam(imputation_model.parameters(), lr=args.imputation_model_args['learning_rate'], weight_decay=args.imputation_model_args['weight_decay'])
    
    # loss_criterion
    sum_criterion = nn.MSELoss(reduction='sum')
    none_criterion = nn.MSELoss(reduction='none')

    # begin training
    stopping_args = Stop_args(patience=60, max_epochs=500)
    early_stopping = EarlyStopping(base_model, **stopping_args)
    for epo in range(args.Auto_model_args['epoch']):
        training_loss = 0
        lossf_sum = 0
        lossl_sum=0
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                # data in this batch ~ 
                # training set: 1. update parameters one_step (assumed update); 2. update parameters (real update) 
                # uniform set: update hyper_parameters using gradient descent. 
                users_train, items_train, y_train = train_loader.get_batch(users, items)
                all_pair = torch.cartesian_prod(users, items)
                users_all, items_all = all_pair[:,0], all_pair[:,1]
                
                # calculate weight 1
                weight1_model.train()
                weight1 = weight1_model(users_train, items_train, (y_train==1) * 1)
                weight1 = torch.exp(weight1/5) # for stable training
                
                # calculate weight 2
                weight2_model.train()
                weight2 = weight2_model(users_all, items_all, (train_dense[users_all,items_all]!=0)*1)
                weight2 = torch.exp(weight2/5) #for stable training
                
                # calculate imputation values
                imputation_model.train()
                impu_f_all = torch.tanh(imputation_model((train_dense[users_all,items_all]).long()+1))

                # one_step_model: assumed model, just update one step on base model. it is for updating weight parameters
                one_step_model = MetaMF(n_user, n_item, dim=args.Auto_model_args['emb_dim'], dropout=0)
                one_step_model.load_state_dict(base_model.state_dict())

                # formal parameter: Using training set to update parameters
                one_step_model.train()
                # all pair data in this block
                y_hat_f_all = one_step_model(users_all, items_all)
                cost_f_all = none_criterion(y_hat_f_all, impu_f_all)
                loss_f_all = torch.sum(cost_f_all * weight2)
                # observation data
                y_hat_f_obs = one_step_model(users_train, items_train)
                cost_f_obs = none_criterion(y_hat_f_obs, y_train)
                loss_f_obs = torch.sum(cost_f_obs * weight1)
                loss_f = loss_f_obs + args.Auto_model_args['imputaion_lambda'] * loss_f_all + args.Auto_model_args['weight_decay'] * one_step_model.l2_norm(users_all, items_all)
                lossf_sum += loss_f
                
                # update parameters of one_step_model
                one_step_model.zero_grad()
                grads = torch.autograd.grad(loss_f, (one_step_model.params()), create_graph=True)
                one_step_model.update_params(args.Auto_model_args['learning_rate'], source_params=grads)

                # latter hyper_parameter: Using uniform set to update hyper_parameters
                y_hat_l = one_step_model(users_unif, items_unif)
                loss_l = sum_criterion(y_hat_l, y_unif)
                lossl_sum += loss_l

                # update hyper-parameters
                weight1_optimizer.zero_grad()
                weight2_optimizer.zero_grad()
                imputation_optimizer.zero_grad()
                loss_l.backward()
                if epo >= 20:
                    weight1_optimizer.step()
                    weight2_optimizer.step()
                imputation_optimizer.step()

                # 2# use new weights to update parameters (real update)       
                weight1_model.train()
                weight1 = weight1_model(users_train, items_train,(y_train==1)*1)
                weight1 = torch.exp(weight1/5)
                
                # calculate weight2
                weight2_model.train()
                weight2 = weight2_model(users_all, items_all,(train_dense[users_all,items_all]!=0)*1)
                weight2 = torch.exp(weight2/5) # for stable training
                
                # use new imputation to update parameters
                imputation_model.train()
                impu_all = torch.tanh(imputation_model((train_dense[users_all,items_all]).long()+1))

                # loss of training set
                base_model.train()
                # all pair
                y_hat_all = base_model(users_all, items_all)
                cost_all = none_criterion(y_hat_all, impu_all)
                loss_all = torch.sum(cost_all * weight2)
                # observation
                y_hat_obs = base_model(users_train, items_train)
                cost_obs = none_criterion(y_hat_obs, y_train)
                loss_obs = torch.sum(cost_obs * weight1)
                loss = loss_obs + args.Auto_model_args['imputaion_lambda'] * loss_all + args.Auto_model_args['weight_decay'] * base_model.l2_norm(users_all, items_all)
                
                base_optimizer.zero_grad()
                loss.backward()
                base_optimizer.step()

                training_loss += loss.item()

        base_model.eval()
        with torch.no_grad():
            # training metrics
            train_pre_ratings = torch.empty(0).to(device)
            train_ratings = torch.empty(0).to(device)
            for u_batch_idx, users in enumerate(train_loader.User_loader): 
                for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                    users_train, items_train, y_train= train_loader.get_batch(users, items)
                    pre_ratings = base_model(users_train, items_train)
                    train_pre_ratings = torch.cat((train_pre_ratings, pre_ratings))
                    train_ratings = torch.cat((train_ratings, y_train))
            
            # validation metrics
            val_pre_ratings = torch.empty(0).to(device)
            val_ratings = torch.empty(0).to(device)
            for batch_idx, (users, items, ratings) in enumerate(val_loader):
                pre_ratings = base_model(users, items)
                val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
                val_ratings = torch.cat((val_ratings, ratings))
        
        train_results = utils.metrics.evaluate(train_pre_ratings, train_ratings, ['MSE'])
        val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'AUC'])
        
        print('Epoch: {0:2d} / {1}, AutoDebias Traning log: {2}, Unbiased Validation: {3}'.format(epo, '500', ' '.join([key+':'+'%.3f'%train_results[key] for key in train_results]), 
                    ' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))
        if epo >= 50 and early_stopping.check([val_results['AUC']], epo):
            break
    
    # restore best model
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    base_model.load_state_dict(early_stopping.best_state)
    print('#'*30)
    Auto_unbias_result, U_Auto_unbias, N_Auto_unbias = both_test(test_loader, base_model, ('CF', 'Auto', 'unbias'), K=5, dataset=args.dataset)

    # test metrics on bias
    Auto_unbias_result, U_Auto_bias, N_Auto_bias = both_test(biastest_loader, base_model, ('CF', 'Auto', 'bias'), K=5, dataset=args.dataset)
    print('#'*30)
    return (base_model, weight1_model, weight2_model, imputation_model), (U_Auto_unbias, N_Auto_unbias, U_Auto_bias, N_Auto_bias)

def train_and_eval_InterD(bias_train, bias_validation, bias_test, unif_validation, unif_test, m, n, Trained_MF_model, Trained_AutoDebias_model, MF_metrics, Auto_metrics, device = 'cuda', gama=999, args=None):
    print('*************************Train InterD************************************')
    train_dense = bias_train.to_dense()
    
    # build data_loader. (block matrix data loader)
    train_loader = utils.data_loader.Block(bias_train, u_batch_size=args.training_args['block_batch'][0], i_batch_size=args.training_args['block_batch'][1], device=device)
    biasval_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(bias_validation), batch_size=args.training_args['batch_size'], shuffle=False, num_workers=0)
    biastest_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(bias_test), batch_size=args.training_args['batch_size'], shuffle=False, num_workers=0)

    val_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(unif_validation), batch_size=args.training_args['batch_size'], shuffle=False, num_workers=0)
    test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(unif_test), batch_size=args.training_args['batch_size'], shuffle=False, num_workers=0)

    n_user, n_item = m, n
    
    # Base model and its optimizer. This optimizer is for optimize parameters in base model using the updated weights (true optimization).
    CF_model = Trained_AutoDebias_model[0]
    F_model = Trained_MF_model
    weight1_model = Trained_AutoDebias_model[1]
    weight2_model = Trained_AutoDebias_model[2]
    imputation_model = Trained_AutoDebias_model[3]
    
    CFF_model = MF_MSE(n_user, n_item, dim=args.InterD_model_args['emb_dim'], dropout=0).to(device)
    if args.dataset == 'yahooR3':
        CFF_model.load_state_dict(CF_model.state_dict())
    else:
        CFF_model.load_state_dict(F_model.state_dict())
    InterD_optimizer = torch.optim.SGD(CFF_model.params(), lr=args.InterD_model_args['learning_rate'], weight_decay=0)
    # loss_criterion
    sum_criterion = nn.MSELoss(reduction='sum')

    # begin training
    stopping_args = Stop_args(stop_varnames=[StopVariable.AUC], patience=args.training_args['patience'], max_epochs=args.training_args['epochs'])
    early_stopping_cff = EarlyStopping(CFF_model, **stopping_args)

    for epo in range(args.training_args['epochs']):
        training_loss = 0
        lossf_sum = 0
        lossl_sum=0
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                users_train, items_train, y_train = train_loader.get_batch(users, items)
                CF_pred = CF_model.forward(users_train, items_train)
                F_pred = F_model.forward(users_train, items_train)
                weight1 = weight1_model(users_train, items_train,(y_train==1)*1)
                weight1 = torch.exp(weight1/5)
                Auto_loss = nn.MSELoss(reduction='none')(CF_pred, y_train)

                # all pair
                all_pair = torch.cartesian_prod(users, items)
                users_all, items_all = all_pair[:,0], all_pair[:,1]
                values_all = train_dense[users_all, items_all]
                obs_mask = torch.abs(values_all)

                weight2 = weight2_model(users_train, items_train,(train_dense[users_train,items_train]!=0)*1)
                weight2 = torch.exp(weight2/5)
                impu_train = torch.tanh(imputation_model((train_dense[users_train,items_train]).long()+1))
                cost_impu = nn.MSELoss(reduction='none')(CF_pred, impu_train)

                CF_loss = Auto_loss* weight1 + cost_impu* weight2
                F_loss = nn.MSELoss(reduction='none')(F_pred, y_train)

				#Imputation train
                users_no, items_no, values_all = users_all, items_all, train_dense[users_all, items_all]

                CF_pred_A = CF_model.forward(users_no, items_no)
                F_pred_A = F_model.forward(users_no, items_no)
                y_hat_obsA = CFF_model(users_no, items_no)
                Loss_FA = nn.MSELoss(reduction='none')(y_hat_obsA, F_pred_A)
                weight2A = weight2_model(users_no, items_no,(train_dense[users_no,items_no]!=0)*1)
                weight2A = torch.exp(weight2A/5)
                impu_trainA = torch.tanh(imputation_model((train_dense[users_no,items_no]).long()+1))
                # The outputs of missing pair from AutoDebias is generated by the learned imputation model.
                Loss_CFA = nn.MSELoss(reduction='none')(impu_trainA, y_hat_obsA)* weight2A

                W_CFA = torch.pow(Loss_FA, args.gama2) / (torch.pow(Loss_CFA, args.gama2) + torch.pow(Loss_FA, args.gama2))
                W_FA = torch.pow(Loss_CFA, args.gama2)  / (torch.pow(Loss_CFA, args.gama2) + torch.pow(Loss_FA, args.gama2))
                y_causal_trainA = W_CFA * CF_pred_A + W_FA * F_pred_A

                y_hat_obs_A = CFF_model(users_no, items_no)
                loss_A = nn.MSELoss(reduction='none')(y_hat_obs_A, y_causal_trainA)
                imp_mask = torch.ones_like(values_all)-torch.abs(values_all)
                loss_A = torch.sum(loss_A*imp_mask)

                #do causal fusion
                W_CF = torch.pow(F_loss, gama) / (torch.pow(CF_loss, gama) + torch.pow(F_loss, gama))
                W_F = torch.pow(CF_loss, gama)  / (torch.pow(CF_loss, gama) + torch.pow(F_loss, gama))
                y_causal_train = W_CF * CF_pred + W_F * F_pred


                CFF_model.train()
                y_hat_obs = CFF_model(users_train, items_train)
                cost_obs = sum_criterion(y_hat_obs, y_causal_train)
                loss = cost_obs + args.beta*loss_A + args.InterD_model_args['weight_decay'] * CFF_model.l2_norm(users_all, items_all)
                InterD_optimizer.zero_grad()
                loss.backward()
                InterD_optimizer.step()
                training_loss += loss.item()

        CFF_model.eval()
        with torch.no_grad():
            # training metrics
            train_pre_ratings = torch.empty(0).to(device)
            train_ratings = torch.empty(0).to(device)
            for u_batch_idx, users in enumerate(train_loader.User_loader): 
                for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                    users_train, items_train, y_train= train_loader.get_batch(users, items)
                    pre_ratings = CFF_model(users_train, items_train)
                    train_pre_ratings = torch.cat((train_pre_ratings, pre_ratings))
                    train_ratings = torch.cat((train_ratings, y_train))

            train_results = utils.metrics.evaluate(train_pre_ratings, train_ratings, ['MSE'], f=None)

            # validation metrics on unifi data
            un_val_pre_ratings = torch.empty(0).to(device)
            un_val_ratings = torch.empty(0).to(device)
            for batch_idx, (users, items, ratings) in enumerate(val_loader):
                pre_ratings = CFF_model(users, items)
                un_val_pre_ratings = torch.cat((un_val_pre_ratings, pre_ratings))
                un_val_ratings = torch.cat((un_val_ratings, ratings))

            un_val_results = utils.metrics.evaluate(un_val_pre_ratings, un_val_ratings, ['MSE','AUC'])

        print('Epoch: {0:2d} / {1}, InterD Traning log: {2}, Unbiased Validation: {3}'.format(epo, args.training_args['epochs'], ' '.join([key+':'+'%.3f'%train_results[key] for key in train_results]), ' '.join([key+':'+'%.3f'%un_val_results[key] for key in un_val_results])))

        if early_stopping_cff.check([un_val_results['AUC']], epo):
            break
    
    # restore best model
    print('Loading {}th epoch'.format(early_stopping_cff.best_epoch))
    CFF_model.load_state_dict(early_stopping_cff.best_state)

    # test metrics on unbias
    print('#'*30)
    CFF_unbias_result, U1, N1 = both_test(test_loader, CFF_model, ('InterD', 'CFF', 'unbias'))

    # test metrics on bias
    CFF_bias_result, U2, N2 = both_test(biastest_loader, CFF_model, ('InterD', 'CFF', 'bias'))
    print('#'*30)

    print('#'*15, 'The overall performances', '#'*15)
    print(f'MF ********** F1-UAUC : {str(round(2*MF_metrics[0]*MF_metrics[2]/(MF_metrics[0]+MF_metrics[2]),4))}, F1-NDCG: {str(round(2*MF_metrics[1]*MF_metrics[3]/(MF_metrics[1]+MF_metrics[3]),4))}')
    print(f'AutoDebias ** F1-UAUC : {str(round(2*Auto_metrics[0]*Auto_metrics[2]/(Auto_metrics[0]+Auto_metrics[2]),4))}, F1-NDCG: {str(round(2*Auto_metrics[1]*Auto_metrics[3]/(Auto_metrics[1]+Auto_metrics[3]),4))}')
    print(f'InterD ****** F1-UAUC : {str(round(2*U1*U2/(U1+U2),4))}, F1-NDCG: {str(round(2*N1*N2/(N1+N2),4))}')
    print('#'*30)


if __name__ == "__main__": 
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    args.exp_name = 'stable'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bias_train, bias_validation, bias_test, unif_train, unif_validation, unif_test, m, n = utils.load_dataset.load_dataset(data_name=args.dataset, type = args.type, seed = args.seed, device=device)

    Trained_MF_model, MF_metrics = train_and_eval_MF(bias_train+unif_train, bias_validation, bias_test, unif_validation, unif_test, m, n, args=args)
    Trained_AutoDebias_model, Auto_metrics = train_and_eval_AutoDebias(bias_train, bias_validation, bias_test, unif_train, unif_validation, unif_test, m, n, args=args)
    train_and_eval_InterD(bias_train+unif_train, bias_validation, bias_test, unif_validation, unif_test, m, n, Trained_MF_model, Trained_AutoDebias_model, MF_metrics, Auto_metrics, gama = args.gama, args=args)
