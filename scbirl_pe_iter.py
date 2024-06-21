import pickle

import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.utils as SIRLU
import SCBIRL_Global_PE.migrationProcess as SIRLP
import Analysis.priorKnow as PriorKnow

import jax
jax.config.update('jax_platform_name', 'cpu')

def train_model_one_traveler(who: int):
    data_dir = './data/user_data/{:09d}/'.format(who)
    model_dir = './model/{:09d}/'.format(who)
    
    iter_start_date = 20230501
    
    inputs, targets_action, pe_code, action_dim, state_dim = SIRLU.loadTrajChain(data_dir, type='before', start_date=iter_start_date)
    print(inputs.shape, targets_action.shape, pe_code.shape)
    model = SIRLT.avril(inputs, targets_action, pe_code, state_dim, action_dim, state_only=True)

    # model the model with no prior knowledge
    PriorKnow.experienceModel(model, data_dir, model_dir, start_date = iter_start_date)

    # NOTE: train the model before migration
    model.train(iters=1000, loss_threshold=0.001)
    model_save_path = model_dir + 'initial_model.pickle'
    model.modelSave(model_save_path)

    # NOTE: Compute rewards after migration
    SIRLP.afterMigrt(model, data_dir, model_dir, start_date = iter_start_date)

if __name__ =="__main__":
    '''
        Iteration Version
    '''
    # for who in who_list:
    #     train_model_one_traveler(who = who)

    '''
        Parallel Version
    '''
    # import multiprocessing as mp
    
    # MAX_CPU_COUNT = 6
    # done_who = [6854307, 21410711, 33816672, 37852495]
    # import os
    # file_list = os.listdir('./data/user_chains/')
    # pid_list = [s.rstrip('.csv').split('_')[-1] for s in file_list]
    # who_list = [int(pid) for pid in pid_list]
    # for who in done_who:
    #     who_list.remove(who)

    # with mp.Pool(MAX_CPU_COUNT) as pool:
    #     pool.map(train_model_one_traveler, who_list)


    '''
        Terminal Version
    '''
    train_model_one_traveler(who = 1102234)
