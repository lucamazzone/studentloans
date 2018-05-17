###############################################################
########## USEFUL FUNCTIONS ###################################
###############################################################

import numpy as np

###############################################################

def classifier(status,consumption):
    controls = np.where(status>1.0)    # status = 1 if root is found successfully
    cons_controls = np.where(consumption< 0.0) # consumption cannot be negative
    joined = cons_controls + controls
    return [x for xs in joined for x in xs]

