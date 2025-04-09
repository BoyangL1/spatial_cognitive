import pickle
import copy
import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.utils as SIRLU
import SCBIRL_Global_PE.migrationProcess as SIRLP
import Analysis.priorKnow as PriorKnow
from SCBIRL_Global_PE.utils import Traveler, UserDataPart, convert_positions_to_utm
# from Analysis.comparison_models import avril_without_pe

import jax
jax.config.update('jax_platform_name', 'cpu')

def train_model_one_traveler(who: int, no_prior = True, initial = True, prior = True, recent = True):
    data_dir = UserDataPart + '{:09d}/'.format(who)
    model_dir = './model/{:09d}/'.format(who)
    
    iter_start_date = SIRLU.load_traveler(who).iter_start_date
    # here the `iter_start_date` is a constant defined by utility module.
    inputs, targets_action, positions, action_dim, state_dim = SIRLU.loadTrajChain(data_dir, type='before', start_date=iter_start_date)
    positions = convert_positions_to_utm(positions)
    # tabular rasa model
    model = SIRLT.avril(inputs, targets_action, positions, state_dim, action_dim, state_only=True)
    # model = avril_without_pe(inputs, targets_action,  state_dim, action_dim, state_only=True)

    # model the model with no prior knowledge
    if no_prior:
        PriorKnow.experienceModel(model, data_dir, model_dir, start_date = iter_start_date)
    # NOTE: Compute rewards after migration
    model_no_prior = copy.deepcopy(model)
    if prior:
        SIRLP.afterMigrt(model_no_prior, data_dir, model_dir, start_date = iter_start_date, iter_type='prior')

    # NOTE: train the model before migration
    if initial:
        model.train(iters=1000, loss_threshold=0.001)
        model_save_path = model_dir + 'initial_model.pickle'
        model.modelSave(model_save_path)

    # NOTE: Compute rewards after migration
    if recent:
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
    import multiprocessing as mp
    import os
    
    MAX_CPU_COUNT = mp.cpu_count() - 2
    done_who = []
    file_list = os.listdir(UserDataPart)
    who_list = [int(pid) for pid in file_list]
    who_list = [10013454]
    for who in done_who:
        who_list.remove(who)
    with mp.Pool(MAX_CPU_COUNT) as pool:
        args = [(who, False, False, False, True) for who in who_list]  # 最后一个 True 是 recent 的默认
        pool.starmap(train_model_one_traveler, args)

    '''
        Terminal Version
    '''
    # train_model_one_traveler(who = 1102234)
