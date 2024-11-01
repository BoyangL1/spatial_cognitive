# import module from the parent directory
import sys
import os
import numpy as np
working_directory = os.path.abspath('.')
sys.path.append(working_directory)

import SCBIRL_Global_PE.utils as SIRLU
from SCBIRL_Global_PE.utils import Traveler, UserDataPart

def travel_chain_counts(who):
     
    travelChains = SIRLU.loadTravelChainAll(who)
    
    return travelChains
    
    
if __name__ == "__main__":
    res = travel_chain_counts(who = 58124481)
    
    print("Done")