import pickle

import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.utils as SIRLU
import SCBIRL_Global_PE.migrationProcess as SIRLP
import Analysis.priorKnow as PriorKnow

def train_model_one_traveler(who: int):
    data_dir = './data/user_data/{:08d}/'.format(who)
    model_dir = './model/{:08d}/'.format(who)
    save_dir = './data_pe/{:08d}/'.format(who)

    # Paths for data files
    before_migration_path = data_dir + 'before_migrt.json'
    after_migration_path = data_dir + 'after_migrt.json'
    full_trajectory_path = data_dir + 'all_traj.json'
    
    inputs, targets_action, pe_code, action_dim, state_dim = SIRLU.loadTrajChain(before_migration_path, full_trajectory_path)
    print(inputs.shape,targets_action.shape,pe_code.shape)
    model = SIRLT.avril(inputs, targets_action, pe_code, state_dim, action_dim, state_only=True)

    # model the model with no prior knowledge
    PriorKnow.experienceModel(model, after_migration_path, before_migration_path, full_trajectory_path, data_dir, model_dir)

    # NOTE: train the model before migration
    model.train(iters=1000, loss_threshold=0.001)
    model_save_path = model_dir + 'before_migrt_model.pickle'
    model.modelSave(model_save_path)

    # NOTE: Compute rewards after migration
    SIRLP.afterMigrt(model, after_migration_path, before_migration_path, full_trajectory_path, model_save_path, data_dir, save_dir)

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
    train_model_one_traveler(who = 39317715)
