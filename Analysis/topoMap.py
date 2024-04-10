import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.utils as SIRLU



if __name__ == '__main__':
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

    model.loadParams('./model/params_transformer_pe.pickle')
