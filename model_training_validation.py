from initialization import *
#Initialization of the model training
def init_training_parameters():
    """
    The function is to initialize training parameters
    return: 1. a dict containing parameter for hyperparameter tuning
            2. metric for hyperparameter tuning
    """

    # This is a simplified version for demonstration purposes: I only hypertuned one hyperparameters for each algorithm
    measure_to_tune = 'roc_auc_ovr'  # for multiclass classification

    hyperparameter_dict = {
        "Logistic": {},

        "Random forests": {'max_depth': [1, 2, 3, 4]},

        "Naive Bayes": {'var_smoothing': np.logspace(0, -9, num=100)},

        "XGBoost": {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]},

        "CART": {'ccp_alpha': [0.1, .01, .001]},

        "KNN": {'n_neighbors': (1, 10, 1)},

        "LDA": {'solver': ['svd', 'lsqr', 'eigen']},

        "SVM": {'C': [0.1, 1, 10, 100, 1000]},

        "Extra trees": {'n_estimators': [int(x) for x in np.linspace(start=50, stop=1000, num=10)]},

        "AdaBoost": {'n_estimators': [100, 200, 300]},

        "Neural network": {'hidden_layer_sizes': [(100, 100, 50), (50, 100, 50), (100, 50, 100)]}
    }

    return hyperparameter_dict,measure_to_tune

def prepare_model_pipeline(X_train,X_test,y_train,y_test,model_name,hyperparameter_dict,measure_to_tune,specific_seed):
    """
    The function is to prepare a pipeline of grid search, model training and model internal validation
    :param X_train, X_test, y_train, y_test: preprocessed data of radiomic features and Y label from training and test data
    :param model_name: name of the algorithm
    :param hyperparameter_dict: hyperparmeter dict for grid search tuning
    :param measure_to_tune: metric for hyperparameter tuning
    :param specific_seed: random seed for reproducibility
    return: 1. prediction from model
            2. probability of each class in numpy array
            3. trained model
            4. tuned hyperparemeter
    """
    try:
        parameters = hyperparameter_dict[f'{model_name}']
    except:
        print(f"\nNo parameters for Grid search!\n")
        parameters={}

    n_class = len(np.unique(y_test))
    #multiclass classification
    if model_name== 'Logistic':
        pipeline = LogisticRegression(random_state=specific_seed)
    elif model_name == 'Random forests':
        pipeline = RandomForestClassifier(random_state=specific_seed)
    elif model_name == 'Naive Bayes':
        pipeline = GaussianNB()  #ok
    elif model_name == 'XGBoost':
        pipeline = XGBClassifier()
    elif model_name== 'CART':
        pipeline = DecisionTreeClassifier(random_state=specific_seed)
    elif model_name== 'KNN':
        pipeline = KNeighborsClassifier()
    elif model_name== 'LDA':
        pipeline = LinearDiscriminantAnalysis()
    elif model_name== 'SVM':
        pipeline = SVC(probability=True)
    elif model_name== 'Extra trees':
        pipeline = ExtraTreesClassifier(random_state=specific_seed)
    elif model_name== 'AdaBoost':
        pipeline = AdaBoostClassifier(random_state=specific_seed)
    elif model_name== 'Neural network':
        pipeline = MLPClassifier(max_iter = 2000, random_state = 42)
    else:
        print("Wrong selection for the model name!")
        pass

    grid_pipeline = GridSearchCV(pipeline,parameters,scoring=measure_to_tune,cv=5,n_jobs = -1)
    # fit
    model=grid_pipeline.fit(X_train,y_train)
    model = model.best_estimator_
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)
    model_para=grid_pipeline.best_params_
    return y_pred,y_prob,model,model_para

def run_model_pipeline(hyperparameter_dict,df_train,df_test,y_name,selected_features,measure_to_tune,mapping_dict
                            ,output_dir_model,output_dir_result,output_dir_log
                            ,seed = 42):
    """
    The function is to run the predifined pipeline of grid search, model training and model internal validation
    :param hyperparameter_dict: hyperparmeter dict for grid search tuning
    :param df_train, df_test: preprocessed training and test data
    :param selected_features: selected features for model training
    :param measure_to_tune: metric for hyperparameter tuning
    :param output_dir_model: folder path to save trained model
    :param output_dir_result: folder path to save results
    :param output_dir_log: folder path to save log files
    :param seed: random seed for reproducibility
    """

    best_params_dict = {}
    counter = 0


    for model_i in [x for x in list(hyperparameter_dict.keys())]:
        print(f'---->Algorithms: {model_i}')

        X_train = df_train[selected_features]
        X_test = df_test[selected_features]
        y_train = df_train[y_name]
        y_test = df_test[y_name]

        y_pred, y_prob, model, model_para = prepare_model_pipeline(X_train, X_test, y_train, y_test, model_i,hyperparameter_dict,
                                                             measure_to_tune=measure_to_tune, specific_seed=seed)

        y_test_str, y_pred_str = prepare_data_for_confusion_matrix(y_test, y_pred, mapping_dict)
        plot_confusion_matrix(y_test_str, y_pred_str, y_name, output_dir_result,model_i)
        # data preprocessing within model_Customized deactiavted by pass empty list
        best_params_dict[model_i] = []
        best_params_dict[model_i].append(model_para)

        with open(f'{output_dir_model}/{model_i}.sav', 'wb') as f:
            joblib.dump(model, f)

        test_ID_col = y_name
        test_ID = pd.DataFrame(df_test[test_ID_col])

        decimal_point = 2

        if counter == 0:
            result_dict, prob_df = evaluate_performance_multi(model_i, test_ID, test_ID_col, y_test, y_pred, y_prob,
                                                            None, None, counter, decimal_point)
        else:
            result_dict, prob_df = evaluate_performance_multi(model_i, test_ID, test_ID_col, y_test, y_pred, y_prob,
                                                            result_dict, prob_df, counter, decimal_point)
        counter += 1

    output = pd.DataFrame.from_dict(result_dict)
    output.to_excel(f"{output_dir_result}/output.xlsx", index=False)
    prob_df.to_excel(f"{output_dir_result}/output_prob.xlsx", index=False)
    pd.DataFrame.from_dict(best_params_dict).to_excel(f"{output_dir_log}/tuned_para.xlsx", index=False)

    # get boostrapping
    prob_address = f"{output_dir_result}/output_prob.xlsx"

    # Bootstrap for 100 times for demonstration purpose. Normally, I adopted 2000 for actual calculation
    prob_df = pd.read_excel(prob_address)
    output_df, df_list, corresponse_dict_name = bootstrap_CI_multi(prob_df, 100, decimal_point)
    output_df.to_excel(f'{output_dir_result}/Summarized_CI.xlsx')

    for df, name in zip(df_list, corresponse_dict_name):
        df.to_excel(f'{output_dir_result}/{name}.xlsx')

def make_prediction(model, df, radiomics_features, y_rename_dict):
    """
    The function is to make prediction based on trained model for future translation
    :param model: trained model to make prediction
    :param df: preprocessed data to make prediction
    :param radiomics_features: column names for radiomic features
    :param y_rename_dict: dict for mapping between numeric Y label and Y label in string
    return df_output: dataframe to summarize the prediction results and probability for each class
    """

    X_test = df[radiomics_features]
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)

    df_output = pd.DataFrame()
    df_output["Case ID"] = df["Case"]
    df_output["Prediction"] = y_pred
    df_output["Prediction"] = df_output["Prediction"].map(y_rename_dict)

    prob_cols = [f"Prob_{y_rename_dict[x]}" for x in range(y_prob.shape[1])]
    df_prob = pd.DataFrame(y_prob,columns=prob_cols)

    df_prob.reset_index(inplace = True,drop = True)
    df_output.reset_index(inplace = True,drop = True)
    df_output = pd.concat([df_output,df_prob],axis = 1)

    return df_output

def summarize_prediction(input_dir):
    """
    The function is to summariz the individual prediction for each algorithm
    :param input_dir: folder path containing individual excel files
    return a dataframe summarizing predictin of all algorithms
    """

    dfs = []
    output_xlsx = [x for x in os.listdir(input_dir) if x.startswith("Prediction") and not x.startswith("._")]
    for i in output_xlsx:
        model_name = i.split(".")[0].split("_")[1]

        df_i = pd.read_excel(f"{input_dir}/{i}")
        df_i = df_i[["Case ID","Prediction"]]
        df_i.rename(columns={'Prediction': f'Prediction_{model_name}'}, inplace=True)
        dfs.append(df_i.set_index('Case ID'))

    df_output = pd.concat(dfs, axis=1, join='outer').reset_index()
    return df_output

def prepare_data_for_confusion_matrix(y_test, y_pred, mapping_dict):
    """
    The function is to convert y_test and y_pred coded in numbers into strings
    :param y_test: ground truth of Y label
    :param y_pred: predicted class
    return y_test and y_pred in strings
    """
    y_test = y_test.tolist()
    y_pred = y_pred.tolist()

    y_test_str = np.array([mapping_dict.get(int(item), "aa") for item in y_test])
    y_pred_str = np.array([mapping_dict.get(int(item), "aa") for item in y_pred])
    return y_test_str,y_pred_str

def plot_confusion_matrix(y_test, y_pred, class_name, output_dir,model_name):
    """
    The function is to plot confusion matrix for prediction perfromance evaluation
    :param y_test: ground truth of Y label
    :param y_pred: predicted class
    :param output_dir: folder path to save images
    :param model_name: name for the model
    """
    # Generate the confusion matrixs
    labels = np.unique(y_test)

    cm = confusion_matrix(y_test, y_pred,labels = labels)
    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(model_name)
    plt.savefig(f"{output_dir}/{model_name}.jpg",bbox_inches= "tight")


# compilation of all evaluation measures
def evaluate_performance_multi(model_i, test_ID, test_ID_col, y_test, y_pred, y_prob, result_dict, prob_df, counter,
                             decimal_point):
    """
    The function is to summarize evaluation metric for multiclass classification
    :param model_i: name of the model
    :param test_ID: dataframe storing Case ID
    :param test_ID_col: column name of Case ID
    :param y_test, y_pred, y_prob: ground truth, predicted class and probability for each class
    :param result_dict: result dict from previous for loop cycle, it can be None at the initialization
    :param prob_df: dataframe of probability from previous for loop cycle, it can be empty at the initialization
    :param counter: counter for documenting for loop
    :param decimal_point: decimal point for numeric rounding
    return 1. a dict that stores predictions
            2. dataframe to stores probability
    """
    if counter == 0:
        result_dict = {}
        result_dict["classification"] = []
        result_dict['Matthew_CC'] = []
        result_dict['Cohen_kappa'] = []
        result_dict['log_loss'] = []
        result_dict["auc_ovr_weighted"] = []
        result_dict["f1_weighted"] = []
        result_dict['Precision'] = []
        result_dict['Recall'] = []
        prob_df = pd.DataFrame()

    MCC = matthews_corrcoef(y_test, y_pred)
    cohen = cohen_kappa_score(y_test, y_pred)
    log_loss_value = log_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob, average="weighted", multi_class="ovr")
    cr = classification_report(y_test, y_pred, output_dict=True)
    precision = cr['weighted avg']['precision']
    recall = cr['weighted avg']['recall']
    f1_weighted = cr['weighted avg']['f1-score']
    confustion_mat = confusion_matrix(y_test, y_pred)

    print(" Matthew: ", format_float_value(MCC, decimal_point))
    print(" Cohen: ", format_float_value(cohen, decimal_point))
    print(" Log loss: ", format_float_value(log_loss_value, decimal_point))
    print(" AUC: ", format_float_value(auc, decimal_point))
    print(' F1: ', format_float_value(f1_weighted, decimal_point))
    print(" Precision: ", format_float_value(precision, decimal_point))
    print(" Recall: ", format_float_value(recall, decimal_point))
    print(' Confusion matrix: \n', confustion_mat)

    result_dict["classification"].append(model_i)
    result_dict['Matthew_CC'].append(MCC)
    result_dict['Cohen_kappa'].append(cohen)
    result_dict['log_loss'].append(log_loss_value)
    result_dict["auc_ovr_weighted"].append(auc)
    result_dict["f1_weighted"].append(f1_weighted)
    result_dict['Precision'].append(precision)
    result_dict['Recall'].append(recall)

    test_ID = test_ID.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    prob_df[f'$Case'] = np.array(list(test_ID[test_ID_col]))

    for i in [int(x) for x in np.unique(y_test)]:
        prob_df[f'{model_i}_{i}'] = y_prob[:, i]
    prob_df[f'{model_i}@Gt'] = y_test
    prob_df[f'{model_i}@Pred'] = y_pred

    return result_dict, prob_df

# Calculation of 95% by bootstrapping method
def bootstrap_CI_multi(prob_df, n_bootstrap, decimal_point):
    """
    The function is to calculate the 95% confidence interval by boostraping
    :param prob_df: dataframe for storage of probability of each class
    :param n_bootstrap: number of bootstraping cycle
    :param decimal_point: decimal point for numeric rounding
    return 1.output_df: a summary of model performance for each algorithm and each evaluation metric
           2.df_list: a list of dataframe of  for each evaluation metric
           3.corresponse_dict_name: a list of evaluation metric names
    """

    bootstrap_count = 0
    Matthew_dict = {}
    Cohen_kappa_dict = {}
    log_loss_dict = {}
    auc_dict = {}
    f1_weighted_dict = {}
    prec_dict = {}
    recal_dict = {}

    col_name = prob_df.columns.to_list()
    model_name_pre = [x for x in col_name if "$" not in x and "@" not in x]
    model_name = np.unique([x.split("_")[0] for x in model_name_pre])
    categories = np.unique([x.split("_")[1] for x in model_name_pre])

    dict_list = [Matthew_dict, Cohen_kappa_dict, log_loss_dict, auc_dict, f1_weighted_dict, prec_dict, recal_dict]
    corresponse_dict_name = ["MCC", "Cohen", "log_loss", "auc", "f1", "Precision", "Recall"]

    for metric_dict in dict_list:
        for model in model_name:
            metric_dict[model] = []

    for i in range(n_bootstrap):
        # i_bootstrap=np.random.sample(prob_df,size=len(prob_df))
        i_bootstrap = prob_df.sample(n=int(len(prob_df) * 0.8), random_state=i).reset_index()

        for model in model_name:
            if len(np.unique(i_bootstrap[f"{model}@Gt"])) == len(categories):  # make sure all classes were present
                model_prob = i_bootstrap[[f'{model}_{x}' for x in categories]]
                model_pred = i_bootstrap[f"{model}@Pred"]
                model_Gt = i_bootstrap[f"{model}@Gt"]

                MCC = matthews_corrcoef(model_Gt, model_pred)
                cohen = cohen_kappa_score(model_Gt, model_pred)
                log_loss_value = log_loss(model_Gt, model_prob)
                auc = roc_auc_score(model_Gt, model_prob, average="weighted", multi_class="ovr")
                cr = classification_report(model_Gt, model_pred, output_dict=True)
                precision = cr['weighted avg']['precision']
                recall = cr['weighted avg']['recall']
                f1_weighted = cr['weighted avg']['f1-score']

                Matthew_dict[model].append(MCC)
                Cohen_kappa_dict[model].append(cohen)
                log_loss_dict[model].append(log_loss_value)
                auc_dict[model].append(auc)
                f1_weighted_dict[model].append(f1_weighted)
                prec_dict[model].append(precision)
                recal_dict[model].append(recall)

        bootstrap_count += 1

    point_sum = {}
    for name, metric_dict in zip(corresponse_dict_name, dict_list):
        point_sum[name] = {}
        for model in model_name:

            try:
                point_sum[name][f'{model}'] = calculate_CI(metric_dict[model], decimal_point)
            except:
                point_sum[name][f'{model}'] = "NaN"

    output_df = pd.DataFrame.from_dict(point_sum)
    print(f'{bootstrap_count} Boostrapping successed!')
    df_list = [pd.DataFrame.from_dict(x) for x in dict_list]

    return output_df, df_list, corresponse_dict_name

def format_float_value(input, decimal_point):
    """
    The function is to format float value
    :param input: float value
    :param decimal_point: decimal point for numeric rounding
    return formatted float value
    """
    format_string = "{:." + str(decimal_point) + "f}"
    return format_string.format(round(input, decimal_point))


def calculate_CI(raw_value, decimal_point):
    """
    The function is to calculate confidence interval and formulate output
    :param raw_value: a list of numbers
    :param decimal_point: decimal point for numeric rounding
    return formulated output
    """

    average = np.average(raw_value)
    average_1 = round(average, decimal_point)
    raw_value.sort()
    confidence_lower = raw_value[int(0.025 * len(raw_value))]
    confidence_upper = raw_value[int(0.975 * len(raw_value))]

    confidence_lower_1 = round(confidence_lower, decimal_point)
    confidence_upper_1 = round(confidence_upper, decimal_point)
    point_output = '{} ({} - {})'.format(format_float_value(average_1, decimal_point),
                                         format_float_value(confidence_lower_1, decimal_point),
                                         format_float_value(confidence_upper_1, decimal_point))
    return point_output
