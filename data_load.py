# Scripts
from scripts.utils import (load_data)#, save_fig, create_subreport, save_report, filter_out_low_WAPS)
# from scripts.errors import compute_errors
# from scripts.plots import plot_pos_vs_time, plot_lat_vs_lon
# from scripts.models import (load_KNN, load_Random_Forest, load_SVM,
#                            load_Decision_Tree, threshold_variance, pca)

# Libraries
from time import time
# from sklearn.model_selection import train_test_split
# from matplotlib.pyplot import close, ioff, ion
# from pandas import DataFrame, concat

# Hyper-parameters / CONSTANTS
N = 520 # Number of WAPS - CONSTANT
MIN_WAPS = 9 # Required number of active WAPS per sample.
NO_SIGNAL_VALUE = -98 # Changed Null Value
QUANTITATIVE_COLUMNS = ['x', 'y'] # Regression Columns
DROP_VAL = True # if True, drops the validation dataset which may be corrupted
# Used to remove columns where information is missing the validation data.
DROP_COLUMNS = None #["SPACEID" ,"RELATIVEPOSITION", "USERID"]
SAVE_FIGS = False # Trigger to save/overwrite figures(saves 5 seconds if False)
SAVE_REPORT = False # Trigger to save/overwrite report
PRINT_SUB = False # Trigger to print sub reports or not.
DISPLAY_PLOTS = False # If true, the 20 figures will be created on screen.

################################## MAIN #######################################
    
if __name__ == "__main__":

    tic = time() # Start program performance timer
    
    # close("all") # Close all previously opened plots
    
    # ion() if DISPLAY_PLOTS else ioff()
    
    # Load and preprocess data with all methods that are independent of subset.
    place_name = 'OfficeP2'
    train_fname = 'OfficeP2_1_training'
    test_fname = 'OfficeP2_4_testing'

    x_train_o, y_train = load_data(
        train_fname,
        place_name,
        drop_columns=DROP_COLUMNS,
        dst_null= NO_SIGNAL_VALUE
    )

    x_test_o, y_test = load_data(
        test_fname,
        place_name,
        drop_columns=DROP_COLUMNS,
        dst_null= NO_SIGNAL_VALUE
    )

    Y_columns = y_test.columns

    print(x_train_o)
    print(y_train)