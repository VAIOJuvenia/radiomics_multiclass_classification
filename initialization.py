import numpy as np
import os
import pandas as pd
import logging
import shutil

import radiomics
from radiomics import featureextractor
import scipy
import trimesh

import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sklearn
import xgboost
from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RepeatedKFold

import joblib
from sklearn.linear_model import LassoCV
#models

#model evaluation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score

#backup
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

def creat_dirs(parent_dir,subfolder_name):
    """
    This function creates structured folders that stores training, testing and prediction results.
    data_dir: the parent folder of the current folder containing py files.
    The parent folder contains the folder for images and masks
    subfolder_name: It can be defined as either "Train_Validate" or "Future_Application" based application scenario.
    returns: a list of structured folders.
    """
    data_dir = f"{parent_dir}/data"
    image_dir = f"{data_dir}/images"
    mask_dir = f"{data_dir}/masks"

    general_output_dir = os.path.join(parent_dir, subfolder_name)
    if not os.path.exists(general_output_dir):
        os.makedirs(general_output_dir)
    else:
        shutil.rmtree(general_output_dir)

    if subfolder_name == "train_validate":
        sub_output = ["radiomics_features", "LASSO", "models", "computation", "log", "code"]
        for i in sub_output:
            if not os.path.exists(f"{general_output_dir}/{i}"):
                os.makedirs(f"{general_output_dir}/{i}")

        output_features = f"{general_output_dir}/radiomics_features"
        output_models = f"{general_output_dir}/models"
        output_computation = f"{general_output_dir}/computation"
        output_log = f"{general_output_dir}/log"
        output_feature_selection = f"{general_output_dir}/LASSO"
        output_code = f"{general_output_dir}/code"

        return output_features,output_models,output_computation,output_log,output_feature_selection,output_code

    #When the variable "subfolder" is not defined as "Train_Validate", we assumed the pipeline is used for future application.
    #In this case, we only need subfolders like "radiomics_features",  "computation", "code".
    else:

        sub_output = ["radiomics_features",  "computation", "code","log"]
        for i in sub_output:
            if not os.path.exists(f"{general_output_dir}/{i}"):
                os.makedirs(f"{general_output_dir}/{i}")

        output_features = f"{general_output_dir}/radiomics_features"
        output_models = None
        output_computation = f"{general_output_dir}/computation"
        output_log = f"{general_output_dir}/log"
        output_feature_selection = None
        output_code = f"{general_output_dir}/code"

        return output_features, output_models, output_computation, output_log, output_feature_selection, output_code

def get_valid_case_list(image_dir, mask_dir):
    #The function is to check the data integrity between image_dir and mask_dir
    #The function is assuming image files and mask files are named the same for the each case
    #The input image_dir and mask_dir are path or path like.
    #The output is a list of all cases with both image and mask files
    image_cases = [x for x in os.listdir(image_dir) if x.endswith("nrrd")]
    mask_cases = [x for x in os.listdir(mask_dir) if x.endswith("nrrd")]

    if image_cases == mask_cases: 
        logging.info("All cases are with corresponding image and mask.")
        cases = image_cases
    else:
        logging.info("Unmatched image and mask found.")

        image_only = [x for x in image_cases if x not in mask_cases]
        mask_only = [x for x in mask_cases if x not in image_cases]

        if len(image_only) > 0:
            print(f"{len(image_only)} cases with only image files (max 5 files shown):\n {image_only[:5]}")

        if len(mask_only) > 0:
            print(f"{len(mask_only)} cases with only mask files (max 5 files shown):\n {mask_only[:5]}")

        cases = [x for x in image_cases if x in mask_cases]   
    
    return cases