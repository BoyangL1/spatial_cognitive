import pickle

import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.utils as SIRLU
import SCBIRL_Global_PE.migrationProcess as SIRLP

if __name__ =="__main__":
    data_dir = './data/'
    model_dir = './model/'
    save_dir = './data_pe/'

    # Paths for data files
    before_migration_path = data_dir + 'before_migrt.json'
    after_migration_path = data_dir + 'after_migrt.json'
    full_trajectory_path = data_dir + 'all_traj.json'

    inputs, targets_action, pe_code, action_dim, state_dim = SIRLU.loadTrajChain(before_migration_path, full_trajectory_path)
    print(inputs.shape,targets_action.shape,pe_code.shape)
    model = SIRLT.avril(inputs, targets_action, pe_code, state_dim, action_dim, state_only=True)

    # NOTE: train the model before migration
    model.train(iters=1000,loss_threshold=0.001)
    model_save_path = model_dir + 'params_transformer_pe.pickle'
    model.modelSave(model_save_path)

    # NOTE: compute rewards and values before migration
    feature_file = data_dir + 'before_migrt_feature.csv'
    model.loadParams('./model/params_transformer_pe.pickle')
    SIRLP.computeRewardOrValue(model, feature_file, save_dir + 'before_migrt_reward.csv', attribute_type='reward')
    SIRLP.computeRewardOrValue(model, feature_file, save_dir + 'before_migrt_value.csv', attribute_type='value')

    # NOTE: Compute rewards after migration
    feature_file_all = data_dir + 'all_traj_feature.csv'
    output_reward_path = save_dir + 'after_migrt_reward.csv'
    SIRLP.afterMigrt(after_migration_path, before_migration_path, full_trajectory_path, feature_file_all, output_reward_path, model)