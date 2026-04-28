import os

DATAWRANGLING_PATH = "/home/pharaszti/ma-data"
# DATAWRANGLING_PATH = "/Users/peter/dev/ma-datawrangling-v2"
SRINGS_BASE_DIR = os.path.join(DATAWRANGLING_PATH, "springs")

SPRING_LIST_FILE = os.path.join(SRINGS_BASE_DIR, "springs_list.txt")
SPRING_LIST_FILE_TRAIN = os.path.join(SRINGS_BASE_DIR, "springs_list_train.txt")
SPRING_LIST_FILE_UNSEEN = os.path.join(SRINGS_BASE_DIR, "springs_list_unseen.txt")

RESULTS_DIR = "results"
RESULTS_PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")