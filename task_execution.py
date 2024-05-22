#The script is for training models and validation of models

from initialization import *
from extract_features import *
from data_cleaning import *
from feature_selection import *
from model_training_validation import *

from pathlib import Path
import shutil
import argparse
import sys

#load data and variable definition
parser = argparse.ArgumentParser(description='Radiomics project')
parser.add_argument('--setting', default='',help='Mode of application')
FLAGS = parser.parse_args()
mode = FLAGS.setting
print('-->setting: {}'.format(FLAGS.setting))


if mode == "train_validate":
    #1.0 Initilization
    print("Initilizing")
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    output_features,output_models,output_computation,output_log,output_feature_selection,output_code = creat_dirs(parent_dir,subfolder_name = mode)

    image_dir = f"{parent_dir}/data/images"
    mask_dir = f"{parent_dir}/data/masks"

    #2.0 Create patient list
    print("Creating list")
    cases = get_valid_case_list(image_dir,mask_dir)

    #3.0 Feature extraction
    extractor = initialize_extraction()
    failed_cases = extract_features(extractor,cases, image_dir, mask_dir,output_features)
    df_features = concatenate_features(output_features)

    #3.1 Post-extraction data cleaning
    print("Data cleaning")
    df_features,y_rename_dict,df_y_rename = get_label(df_features)
    df_y_rename.to_excel(f"{output_log}/y_rename_dict.xlsx",index = False)
    y_name = "Origin_num"
    visualize_Y(df_features,"Origin",output_computation)
    df_train,df_test = stratified_split(df = df_features, stratify_col = "Origin", test_size=0.2)

    #3.2 Data normalization
    radiomics_feature = get_radiomic_feature_name(df_features)

    df_train,df_test,scaler = normalize_data(df_train,df_test,radiomics_feature)
    joblib.dump(scaler, f'{output_models}/scaler.bin', compress=True)

    #3.3 Save normalized data
    df_train.to_excel(f"{output_log}/df_train_normalized.xlsx", index=False)
    df_test.to_excel(f"{output_log}/df_test_normalized.xlsx", index=False)
    df_features = pd.concat([df_train, df_test], axis=0)
    df_features.to_excel(f"{output_log}/df_normalized.xlsx", index=False)

    #4.0 Feature selection
    print("Selecting features")
    #4.1 Unsupervised feature selection
    df_train,df_test,non_related_features = select_feature_corr(df_train, df_test,radiomics_feature,correlation_threshold = 0.85)
    #4.2 Supervised LASSO-based feature selection
    bootstrap_LASSO(df_train,non_related_features,output_feature_selection, y_name,100)
    summarize_lasso_result(output_feature_selection)
    df_lasso_combined,selected_features = get_selected_feature_by_threshold(output_feature_selection,threshold = 0.8)
    pd.DataFrame(selected_features).to_excel(f"{output_log}/selected_features.xlsx",index = False)
    plot_feature_importance(df_lasso_combined,output_computation,threshold = 0.8)

    #5.0 Model training
    print("Model training")
    hyperparameter_dict,measure_to_tune = init_training_parameters()
    y_rename_dict_reverse = {int(value): key for key, value in y_rename_dict.items()}
    run_model_pipeline(hyperparameter_dict,df_train,df_test,y_name,selected_features,measure_to_tune,y_rename_dict_reverse,
                       output_models,output_computation,output_log,seed = 42)

    #6.0 Copy used code for tracking
    current_dir = Path.cwd()
    file_suffixes = ["py","txt","md"]
    files = [x for x in os.listdir(current_dir) if x.endswith(tuple(file_suffixes))]
    for file in files:
        shutil.copy(f'{current_dir}/{file}', f'{output_code}/{file}')

elif mode == "predict":
    # The script is for training models and validation of models
    # To use this, please copy the folder containing all these python script files into the parent folder containing masks and images

    # You may see
    # parent folder
    # +---images
    # +---masks
    # +---python scirpt

    # After this, you may choose to
    #      run Py7_Run_training_validation if you would like to train and validate the models
    #      run Py8_Future_translation if you would like to apply the trained models to futre unseen data
    #      For demo purpose, I assumed all cases in the subfolders masks and images are new cases

    # 1.0 Initilization
    print("Initilizing")
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    output_features, output_models, output_computation, output_log, output_feature_selection, output_code = creat_dirs(
        parent_dir, subfolder_name=mode)

    image_dir = f"{parent_dir}/data/images"
    mask_dir = f"{parent_dir}/data/masks"

    # 1.1 Check if the models have been trained
    models = [x for x in os.listdir(f"{parent_dir}/train_validate/models") if x.endswith("sav")]
    if len(models) ==0:
        print("No models trained before. Please train first")
        sys.exit()

    # 2.0 Create patient list
    print("Creating list")
    cases = get_valid_case_list(image_dir, mask_dir)

    # 3.0 Feature extraction
    extractor = initialize_extraction()
    failed_cases = extract_features(extractor, cases, image_dir, mask_dir, output_features)
    df_features = concatenate_features(output_features)

    # 3.1 Post-extraction data cleaning
    print("Data cleaning")
    df_features = add_ID_predict(df_features)
    selected_features = pd.read_excel(f"{parent_dir}/train_validate/log/selected_features.xlsx")["feature"]

    # 3.2 Data normalization
    radiomics_feature = get_radiomic_feature_name(df_features)
    scaler_path = f"{parent_dir}/train_validate/models/scaler.bin"  # Assumed you trained the scaler in previous task
    scaler = joblib.load(scaler_path)
    df_features = normalize_data_predict(df_features, radiomics_feature, scaler)

    # 3.3 Save normalized data
    df_features.to_excel(f"{output_log}/df_normalized.xlsx", index=False)

    # 4.0 Model training
    print("Model prediction")
    model_dir = f"{parent_dir}/train_validate/models"
    model_list = [x for x in os.listdir(model_dir) if x.endswith("sav")]
    for i in model_list:
        model = joblib.load(f"{model_dir}/{i}")
        y_rename_dict = pd.read_excel(f"{parent_dir}/train_validate/log/y_rename_dict.xlsx").T.to_dict()[0]
        y_rename_dict = {value: key for key, value in y_rename_dict.items()}
        df_output = make_prediction(model, df_features, selected_features, y_rename_dict)
        df_output.to_excel(f"{output_computation}/Prediction_{i[:-4]}.xlsx", index=False)
    df_summarized = summarize_prediction(output_computation)
    df_summarized.to_excel(f"{output_computation}/Summary.xlsx", index=False)

    # 5.0 Copy used code for tracking
    current_dir = Path.cwd()
    file_suffixes = ["py", "txt", "md"]
    files = [x for x in os.listdir(current_dir) if x.endswith(tuple(file_suffixes))]
    for file in files:
        shutil.copy(f'{current_dir}/{file}', f'{output_code}/{file}')

else:
    print("Wrong input for setting")

