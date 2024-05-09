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
    model.train(iters=1000,loss_threshold=0.001)
    model_save_path = model_dir + 'before_migrt_model.pickle'
    model.modelSave(model_save_path)

    # NOTE: Compute rewards after migration
    SIRLP.afterMigrt(model, after_migration_path, before_migration_path, full_trajectory_path, model_save_path, data_dir, save_dir)

if __name__ =="__main__":
    train_model_one_traveler(who = 6854307)