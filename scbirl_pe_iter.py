import pickle
import os
import copy
import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.utils as SIRLU
import SCBIRL_Global_PE.migrationProcess as SIRLP
import Analysis.priorKnow as PriorKnow
from SCBIRL_Global_PE.utils import Traveler, UserDataPart
# from Analysis.comparison_models import avril_without_pe

import jax
jax.config.update('jax_platform_name', 'cpu')

import dask
from dask.distributed import Client, LocalCluster
import numpy as np
from tqdm.auto import tqdm
import multiprocessing as mp

def train_model_one_traveler(who: int, no_prior = True, initial = True, prior = True, recent = True):
    data_dir = UserDataPart + '{:09d}/'.format(who)
    model_dir = './model/{:09d}/'.format(who)
    
    iter_start_date = SIRLU.load_traveler(who).iter_start_date
    # here the `iter_start_date` is a constant defined by utility module.
    inputs, targets_action, positions, action_dim, state_dim = SIRLU.loadTrajChain(data_dir, type='before', start_date=iter_start_date)
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

def train_models_parallel(who_list, n_workers=32, threads_per_worker=4):
    """
    Parallel model training using Dask for multiple travelers.
    No return values needed as results are saved to disk directly.
    Parameters:
    -----------
    who_list : list
        List of traveler IDs to process
    n_workers : int, default=32
        Number of worker processes to use
    threads_per_worker : int, default=4
        Number of threads per worker (total threads = n_workers * threads_per_worker)
    
    """
    # Set up Dask cluster
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit='4GB'  # Adjust based on your server's RAM
    )
    client = Client(cluster)
    print(f"Dashboard link: {client.dashboard_link}")
    
    try:
        # Create delayed objects for each traveler
        delayed_tasks = []
        for who in who_list:
            # Wrap the training function in delayed
            train_model_dask = dask.delayed(train_model_one_traveler)
            task = train_model_dask(who)
            delayed_tasks.append(task)
        
        # Compute all tasks in parallel with progress bar
        print(f"Training models for {len(who_list)} travelers...")
        
        # Use tqdm to show progress
        with tqdm(total=len(who_list), desc="Training Progress") as pbar:
            dask.compute(*delayed_tasks, scheduler='distributed')
            pbar.update(len(who_list))
            
    finally:
        # Clean up
        client.close()
        cluster.close()

def train_model_batch(who_list, batch_size):
    """
    Train models in batches to manage memory usage.
    Results are saved to disk directly by train_model_one_traveler.
    Parameters:
    -----------
    who_list : list
        List of traveler IDs to process
    batch_size : int, default=1000
        Number of travelers to process in each batch
    """
    
    # Split who_list into batches
    n_batches = (len(who_list) + batch_size - 1) // batch_size
    who_batches = np.array_split(who_list, n_batches)
    
    print(f"Processing {len(who_list)} travelers in {n_batches} batches")
    
    # Process each batch
    for i, batch in enumerate(who_batches):
        print(f"\nProcessing batch {i+1}/{n_batches}")
        train_models_parallel(batch.tolist())
        print(f"Completed batch {i+1}/{n_batches}")

def save_intermediate_results(results, filename):
    """
    Save intermediate results to avoid data loss.
    
    Parameters:
    -----------
    results : dict
        Results to save
    filename : str
        Path to save the results
    """
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

if __name__ =="__main__":
    '''
        Iteration Version
    '''
    # who_list = [1102234]
    # for who in who_list:
    #     train_model_one_traveler(who = who)
    '''
        Parallel Version
    '''
    # import multiprocessing as mp
    # import os
    
    # MAX_CPU_COUNT = mp.cpu_count() - 2
    # done_who = []
    # file_list = os.listdir(UserDataPart)
    # who_list = [int(pid) for pid in file_list]
    # for who in done_who:
    #     who_list.remove(who)
    # with mp.Pool(MAX_CPU_COUNT) as pool:
    #     pool.map(train_model_one_traveler, who_list)
    '''
        Professional Parallel Version
    '''
    # file_list = os.listdir(UserDataPart)
    # # Example who_list
    # who_list = [int(pid) for pid in file_list]
    
    # # Configure Dask for your hardware
    # n_workers = 32  # Number of CPU cores
    # threads_per_worker = 4  # Threads per worker (128/32 = 4)
    
    # # Train models with batch processing
    # results = train_model_batch(
    #     who_list,
    #     batch_size=n_workers  # Adjust based on memory requirements
    # )
    
    MAX_CPU_COUNT = mp.cpu_count() - 2
    done_who = []
    file_list = os.listdir(UserDataPart)
    who_list = [int(pid) for pid in file_list]
    who_list = [1102234]
    for who in done_who:
        who_list.remove(who)
    with mp.Pool(MAX_CPU_COUNT) as pool:
        args = [(who, False, False, False, True) for who in who_list]  # 最后一个 True 是 recent 的默认
        pool.starmap(train_model_one_traveler, args)

    '''
        Terminal Version
    '''
    # train_model_one_traveler(who = 1102234)
