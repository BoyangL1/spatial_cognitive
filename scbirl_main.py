import pickle

import SCBIRL_Transformer.SCBIRLTransformer as SIRLT
import SCBIRL_Transformer.utils as SIRLU
import SCBIRL_Transformer.migrationProcess as SIRLP

if __name__ =="__main__":
    data_dir = './data/'
    model_dir = './model/'
    coords_file_path = './data/coords_grid_data.pkl'
    
    with open(coords_file_path, 'rb') as file:
        coords_grid_data = pickle.load(file)

    # Paths for data files
    before_migration_path = data_dir + 'before_migrt.json'
    after_migration_path = data_dir + 'after_migrt.json'
    full_trajectory_path = data_dir + 'all_traj.json'

    inputs, targets_action, grid_code, action_dim, state_dim = SIRLU.loadTrajChain(before_migration_path, full_trajectory_path, coords_grid_data)
    print(inputs.shape,targets_action.shape,grid_code.shape)
    model = SIRLT.avril(inputs, targets_action, grid_code, state_dim, action_dim, state_only=True)

    # NOTE: train the model before migration
    model.train(iters=1000,loss_threshold=0.001)
    model_save_path = model_dir + 'params_transformer.pickle'
    model.modelSave(model_save_path)

    # NOTE: compute rewards and values before migration
    feature_file = data_dir + 'before_migrt_feature.csv'
    model.loadParams('./model/params_transformer.pickle')
    SIRLP.computeRewardOrValue(model, feature_file, data_dir + 'before_migrt_reward.csv', coords_grid_data, attribute_type='reward')
    SIRLP.computeRewardOrValue(model, feature_file, data_dir + 'before_migrt_value.csv', coords_grid_data, attribute_type='value')

    # NOTE: Compute rewards after migration
    feature_file_all = data_dir + 'all_traj_feature.csv'
    output_reward_path = data_dir + 'after_migrt_reward.csv'
    SIRLP.afterMigrt(after_migration_path, before_migration_path, full_trajectory_path, coords_grid_data, feature_file_all, output_reward_path, model)