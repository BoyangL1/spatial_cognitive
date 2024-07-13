import pickle
import copy
import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.utils as SIRLU
import SCBIRL_Global_PE.migrationProcess as SIRLP
import Analysis.priorKnow as PriorKnow
from SCBIRL_Global_PE.utils import Traveler

import jax
jax.config.update('jax_platform_name', 'cpu')

def train_model_one_traveler(who: int):
    data_dir = './data/user_data/{:09d}/'.format(who)
    model_dir = './model/{:09d}/'.format(who)
    
    iter_start_date = SIRLU.load_traveler(who).iter_start_date
    # here the `iter_start_date` is a constant defined by utility module.
    inputs, targets_action, pe_code, action_dim, state_dim = SIRLU.loadTrajChain(data_dir, type='before', start_date=iter_start_date)
    print(inputs.shape, targets_action.shape, pe_code.shape)
    model = SIRLT.avril(inputs, targets_action, pe_code, state_dim, action_dim, state_only=True)

    # model the model with no prior knowledge
    PriorKnow.experienceModel(model, data_dir, model_dir, start_date = iter_start_date)
    # NOTE: Compute rewards after migration
    model_no_prior = copy.deepcopy(model)
    SIRLP.afterMigrt(model_no_prior, data_dir, model_dir, start_date = iter_start_date, iter_type='prior')

    # NOTE: train the model before migration
    model.train(iters=1000, loss_threshold=0.001)
    model_save_path = model_dir + 'initial_model.pickle'
    model.modelSave(model_save_path)

    # NOTE: Compute rewards after migration
    SIRLP.afterMigrt(model, data_dir, model_dir, start_date = iter_start_date, iter_type='recent')

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
    
    # MAX_CPU_COUNT = mp.cpu_count()
    # done_who = [1102234]
    # undone_who = [93854949, 102181433]
    # import os
    # file_list = os.listdir('./data/user_data/')
    # who_list = [int(pid) for pid in file_list]
    # for who in done_who:
    #     who_list.remove(who)
    # with mp.Pool(MAX_CPU_COUNT) as pool:
    #     pool.map(train_model_one_traveler, who_list)

    '''
        Terminal Version
    '''
    train_model_one_traveler(who = 102181433)
