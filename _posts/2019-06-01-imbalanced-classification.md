layout : post
title : "Imbalanced Classification"
date : 2019-06-01
permalink: /imbalanced-classification/
---
![imbalanced-classification](/images/imbalanced-header.jpeg)





```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interp
from sklearn.preprocessing import scale
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
import itertools
import glmnet
import xgboost as xgb
import shap
import seaborn as sns
sns.set_style("ticks")
mpl.rcParams['axes.linewidth'] = 3 
mpl.rcParams['lines.linewidth'] =7
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
import warnings 
warnings.filterwarnings("ignore")
%matplotlib inline

# Unused Libraries here
# import keras
# from bayes_opt import BayesianOptimization
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from fancyimpute import SoftImpute, IterativeImputer
```


<style>.container { width:95% !important; }</style>


# Functions


```python
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _x_y_maker

def _x_y_maker(df, target, to_drop):
    """
    A function to receive the dataframe and make the X, and Y matrices:
    ----------
    parameters
    
    input: dataframe df
           string target (class label)
           list[string] to_drop (list of columns need to be dropped)
           
    output: dataframe Feature matrix (df_X)
            Target vector (Y)
    """    
    Targets = df[target]
    to_drop.append(target)
    Features = df.drop(to_drop, axis = 1)
    Scaled_Features = scale(Features.values)
    df_X = pd.DataFrame(data = Scaled_Features, columns = Features.columns.tolist())
    Y = Targets.values
    
    return df_X, Y

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _plot_roc_nfolds_xgboost

def _plot_roc_nfolds_xgboost(df_X, Y,
                                     n_folds = 10,
                                     n_estimators = 100,
                                     learning_rate = 0.05,
                                     max_depth = 3,
                                     min_child_weight = 5.0,
                                     gamma = 0.5,
                                     reg_alpha = 0.0,
                                     reg_lambda = 1.0,
                                     subsample = 0.9,
                                     objective = "binary:logistic",
                                     scale_pos_weight = 1.0,
                                     shuffle = True,
                                     random_state = 1367
                            ):
    """
    a function to plot k-fold cv ROC using xgboost
    input parameters:
                     df_X : features : pandas dataframe (numpy array will be built inside the function)
                     Y : targets
                     n_folds : number of cv folds (default = 10)
                     n_estimators = number of trees (default = 100)
                     learning_rate : step size of xgboost (default = 0.05)
                     max_depth : maximum tree depth for xgboost (default = 3)
                     min_child_weight : (default = 5.0)
                     gamma : (default = 0.5)
                     reg_alpha : lasso penalty (L1) (default = 0.0)
                     reg_lambda : ridge penalty (L2) (default = 1.0)
                     subsample : subsample fraction (default = 0.9)
                     objective : objective function for ML (default = "binary:logistic" for classification)
                     scale_pos_weight : (default = 1.0)
                     shuffle : shuffle flag for cv (default = True)
                     random_state : (default = 1367)
    
    """

    # Defining the data
    X = scale(df_X.values)
    y = Y
    n_samples, n_features = X.shape

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits = n_folds , shuffle = shuffle , random_state = random_state)
    classifier = XGBClassifier(learning_rate = learning_rate,
                               n_estimators = n_estimators,
                               max_depth = max_depth,
                               min_child_weight = min_child_weight,
                               gamma = gamma,
                               reg_alpha = reg_alpha,
                               reg_lambda = reg_lambda,
                               subsample = subsample,
                               objective = objective,
                               nthread = 4,
                               scale_pos_weight = 1.,
                               base_score = np.mean(y),
                               seed = random_state,
                               random_state = random_state)
    

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(18 , 13))
    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=3, alpha=0.5, 
                 label="ROC Fold %d (AUC = %0.2f)" % (i+1, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle="--", lw=3, color="k",
             label="Luck", alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color="navy",
             label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
             lw=4)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=.4,
                     label=r"$\pm$ 1 Standard Deviation")

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate" ,  fontweight = "bold" , fontsize=30)
    plt.ylabel("True Positive Rate",fontweight = "bold" , fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.legend( prop={"size":20} , loc = 4)
    # plt.savefig("./roc_xgboost.pdf" ,bbox_inches="tight")
    plt.show()

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _permute

def _permute(df, random_state):
    """
    a funtion to permute the rows of features and add them
    to the dataframe as noisy features to explore stability
    input parameters:
                    df : pandas dataframe
                    random_state : random seed for noisy features
    """
    normal_df = df.copy().reset_index(drop = True)
    noisy_df = df.copy()
    noisy_df.rename(columns = {col : "noisy_" + col for col in noisy_df.columns}, inplace = True)
    np.random.seed(seed = random_state)
    noisy_df = noisy_df.reindex(np.random.permutation(noisy_df.index))
    merged_df = pd.concat([normal_df, noisy_df.reset_index(drop = True)] , axis = 1)
    
    return merged_df
    
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _my_feature_selector

def _my_feature_selector(X, Y, n_iter = 1,
                                  num_boost_round = 1000,
                                  nfold = 10,
                                  stratified = True,
                                  metrics = ("auc"),
                                  early_stopping_rounds = 50,
                                  seed = 1367,
                                  shuffle = True,
                                  show_stdv = False,
                                  params = None,
                                  importance_type = "total_gain",
                                  callbacks = False,
                                  verbose_eval = False):
    """
    a function to run xgboost kfolds cv and train the model based on the best boosting round of each iteration.
    for different number of iterations. at each iteration noisy features are added as well. at each iteration
    internal and external cv calculated.
    NOTE: it is recommended to use ad-hoc parameters to make sure the model is under-fitted for the sake of feature selection
    input parameters:
                    X: features (pandas dataframe or numpy array)
                    Y: targets (1D array or list)
                    n_iter: total number of iterations (default = 1)
                    num_boost_rounds: max number of boosting rounds, (default = 1000)
                    stratified: stratificaiton of the targets (default = True)
                    metrics: classification/regression metrics (default = ("auc))
                    early_stopping_rounds: the criteria for stopping if the test metric is not improved (default = 20)
                    seed: random seed (default = 1367)
                    shuffle: shuffling the data (default = True)
                    show_stdv = showing standard deviation of cv results (default = False)
                    params = set of parameters for xgboost cv
                            (default_params = {
                                               "eval_metric" : "auc",
                                               "tree_method": "hist",
                                               "objective" : "binary:logistic",
                                               "learning_rate" : 0.05,
                                               "max_depth": 2,
                                               "min_child_weight": 1,
                                               "gamma" : 0.0,
                                               "reg_alpha" : 0.0,
                                               "reg_lambda" : 1.0,
                                               "subsample" : 0.9,
                                               "max_delta_step": 1,
                                               "silent" : 1,
                                               "nthread" : 4,
                                               "scale_pos_weight" : 1
                                               }
                            )
                    importance_type = importance type of xgboost as string (default = "total_gain")
                                      the other options will be "weight", "gain", "cover", and "total_cover"
                    callbacks = printing callbacks for xgboost cv
                                (defaults = False, if True: [xgb.callback.print_evaluation(show_stdv = show_stdv),
                                                             xgb.callback.early_stop(early_stopping_rounds)])
                    verbose_eval : a flag to show the result during train on train/test sets (default = False)
    outputs:
            outputs_dict: a dict contains the fitted models, internal and external cv results for train/test sets
            df_selected: a dataframe contains feature names with their frequency of being selected at each run of folds
    """
    # callback flag
    if(callbacks == True):
        callbacks = [xgb.callback.print_evaluation(show_stdv = show_stdv),
                     xgb.callback.early_stop(early_stopping_rounds)]
    else:
        callbacks = None
    
    # params
    default_params = {
                      "eval_metric" : "auc",
                      "tree_method": "hist",
                      "objective" : "binary:logistic",
                      "learning_rate" : 0.05,
                      "max_depth": 2,
                      "min_child_weight": 1,
                      "gamma" : 0.0,
                      "reg_alpha" : 0.0,
                      "reg_lambda" : 1.0,
                      "subsample" : 0.9,
                      "max_delta_step": 1,
                      "silent" : 1,
                      "nthread" : 4,
                      "scale_pos_weight" : 1,
                      "base_score" : np.mean(Y)
                     }
    # updating the default parameters with the input
    if params is not None:
        for key, val in params.items():
            default_params[key] = val
            
    # total results list
    int_cv_train = []
    int_cv_test = []
    ext_cv_train = []
    ext_cv_test = []
    bst_list = []
    
    # main loop
    for iteration in range(n_iter):
        
        # results list at iteration
        int_cv_train2 = []
        int_cv_test2 = []
        ext_cv_train2 = []
        ext_cv_test2 = []
        
        # update random state 
        random_state = seed * iteration 
    
        # adding noise to data
        X_permuted = _permute(df = X, random_state = random_state)
        cols = X_permuted.columns.tolist()
        Xval = X_permuted.values

        # building DMatrix for training/testing + kfolds cv
        cv = StratifiedKFold(n_splits = nfold , shuffle = shuffle , random_state = random_state)

        for train_index, test_index in cv.split(Xval, Y):
            Xval_train = Xval[train_index]
            Xval_test = Xval[test_index]
            Y_train = Y[train_index]
            Y_test = Y[test_index]

            X_train = pd.DataFrame(data = Xval_train, columns = cols)
            X_test = pd.DataFrame(data = Xval_test, columns = cols)
            
            dtrain = xgb.DMatrix(data = X_train, label = Y_train)
            dtest = xgb.DMatrix(data = X_test, label = Y_test)

            # watchlist during final training
            watchlist = [(dtrain,"train"), (dtest,"eval")]
            
            # a dict to store training results
            evals_result = {}
            
            # xgb cv
            cv_results = xgb.cv(params = default_params,
                                dtrain = dtrain,
                                num_boost_round = num_boost_round,
                                nfold = nfold,
                                stratified = stratified,
                                metrics = metrics,
                                early_stopping_rounds = early_stopping_rounds,
                                seed = random_state,
                                verbose_eval = verbose_eval,
                                shuffle = shuffle,
                                callbacks = callbacks)
            int_cv_train.append(cv_results.iloc[-1][0])
            int_cv_test.append(cv_results.iloc[-1][2])
            int_cv_train2.append(cv_results.iloc[-1][0])
            int_cv_test2.append(cv_results.iloc[-1][2])
            
            # xgb train
            bst = xgb.train(params = default_params,
                            dtrain = dtrain,
                            num_boost_round = len(cv_results) - 1,
                            evals = watchlist,
                            evals_result = evals_result,
                            verbose_eval = verbose_eval)
            # appending outputs
            bst_list.append(bst)
            ext_cv_train.append(evals_result["train"]["auc"][-1])
            ext_cv_test.append(evals_result["eval"]["auc"][-1])
            ext_cv_train2.append(evals_result["train"]["auc"][-1])
            ext_cv_test2.append(evals_result["eval"]["auc"][-1])
        
        print(F"-*-*-*-*-*-*-*-*- Iteration {iteration + 1} -*-*-*-*-*-*-*-*")
        print(F"-*-*- Internal-{nfold} Folds-CV-Train = {np.mean(int_cv_train2):.3} +/- {np.std(int_cv_train2):.3} -*-*- Internal-{nfold} Folds-CV-Test = {np.mean(int_cv_test2):.3} +/- {np.std(int_cv_test2):.3} -*-*")
        print(F"-*-*- External-{nfold} Folds-CV-Train = {np.mean(ext_cv_train2):.3} +/- {np.std(ext_cv_train2):.3} -*-*- External-{nfold} Folds-CV-Test = {np.mean(ext_cv_test2):.3} +/- {np.std(ext_cv_test2):.3} -*-*")
    
    # putting together the outputs in one dict
    outputs = {}
    outputs["bst"] = bst_list
    outputs["int_cv_train"] = int_cv_train
    outputs["int_cv_test"] = int_cv_test
    outputs["ext_cv_train"] = ext_cv_train
    outputs["ext_cv_test"] = ext_cv_test
    

    pruned_features = []
    for bst_ in outputs["bst"]:
        features_gain = bst_.get_score(importance_type = importance_type)
        for key, value in features_gain.items():
            pruned_features.append(key)

    unique_elements, counts_elements = np.unique(pruned_features, return_counts = True)
    counts_elements = [float(i) for i in list(counts_elements)]
    df_features = pd.DataFrame(data = {"columns" : list(unique_elements) , "count" : counts_elements})
    df_features.sort_values(by = ["count"], ascending = False, inplace = True)

    # plotting
    plt.figure()
    df_features.sort_values(by = ["count"]).plot.barh(x = "columns", y = "count", color = "pink" , figsize = (8,8))
    plt.show()

    plt.figure(figsize = (16,12))

    plt.subplot(2,2,1)
    plt.title(F"Internal {nfold} CV {metrics.upper()} - Train", fontsize = 20)
    plt.hist(outputs["int_cv_train"], bins = 20, color = "lightgreen")

    plt.subplot(2,2,2)
    plt.title(F"Internal {nfold} CV {metrics.upper()} - Test", fontsize = 20)
    plt.hist(outputs["int_cv_test"], bins = 20, color = "lightgreen")

    plt.subplot(2,2,3)
    plt.title(F"External {nfold} CV {metrics.upper()} - Train", fontsize = 20)
    plt.hist(outputs["ext_cv_train"], bins = 20, color = "lightblue")

    plt.subplot(2,2,4)
    plt.title(F"External {nfold} CV {metrics.upper()} - Test", fontsize = 20)
    plt.hist(outputs["ext_cv_test"], bins = 20, color = "lightblue")

    plt.show()
    
    return outputs, df_features
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: Confidence Interval calculator for 95% significance

def _ConfidenceInterval(score, n):
    """
    confidence interval for 95% significance level
    input: score such as accuracy, auc
           the size of test set n
    """
    
    CI_range = 1.96 * np.sqrt( (score * (1 - score)) / n)
    CI_pos = score + CI_range
    CI_neg = score - CI_range
    if(CI_neg < 0):
        CI_neg = 0
        
    return CI_neg, CI_pos

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _bst_shap

def _bst_shap(X, Y, num_boost_round = 1000,
                                    nfold = 10,
                                    stratified = True,
                                    metrics = ("auc"),
                                    early_stopping_rounds = 50,
                                    seed = 1367,
                                    shuffle = True,
                                    show_stdv = False,
                                    params = None,
                                    callbacks = False):
    """
    a function to run xgboost kfolds cv with the params and train the model based on best boosting rounds
    input parameters:
                    X: features
                    Y: targets
                    num_boost_rounds: max number of boosting rounds, (default = 1000)
                    stratified: stratificaiton of the targets (default = True)
                    metrics: classification/regression metrics (default = ("auc))
                    early_stopping_rounds: the criteria for stopping if the test metric is not improved (default = 20)
                    seed: random seed (default = 1367)
                    shuffle: shuffling the data (default = True)
                    show_stdv = showing standard deviation of cv results (default = False)
                    params = set of parameters for xgboost cv ( default_params = {
                                                                                  "eval_metric" : "auc",
                                                                                  "tree_method": "hist",
                                                                                  "objective" : "binary:logistic",
                                                                                  "learning_rate" : 0.05,
                                                                                  "max_depth": 2,
                                                                                  "min_child_weight": 5,
                                                                                  "gamma" : 0.0,
                                                                                  "reg_alpha" : 0.0,
                                                                                  "reg_lambda" : 1.0,
                                                                                  "subsample" : 0.9,
                                                                                  "silent" : 1,
                                                                                  "nthread" : 4,
                                                                                  "scale_pos_weight" : 1
                                                                                 }
                                                                    )
                   callbacks = printing callbacks for xgboost cv (defaults: [xgb.callback.print_evaluation(show_stdv = show_stdv),
                                                                             xgb.callback.early_stop(early_stopping_rounds)])
    outputs:
            bst: the trained model based on optimum number of boosting rounds
            cv_results: a dataframe contains the cv-results (train/test + std of train/test) for metric
    """
    # callback flag
    if(callbacks == True):
        callbacks = [xgb.callback.print_evaluation(show_stdv = show_stdv),
                     xgb.callback.early_stop(early_stopping_rounds)]
    else:
        callbacks = None
    
    # params
    default_params = {
                      "eval_metric" : "auc",
                      "tree_method": "hist",
                      "objective" : "binary:logistic",
                      "learning_rate" : 0.05,
                      "max_depth": 2,
                      "min_child_weight": 1,
                      "gamma" : 0.0,
                      "reg_alpha" : 0.0,
                      "reg_lambda" : 1.0,
                      "subsample" : 0.9,
                      "max_delta_step": 1,
                      "silent" : 1,
                      "nthread" : 4
                     }
    # updating the default parameters with the input
    if params is not None:
        for key, val in params.items():
            default_params[key] = val

    # building DMatrix for training
    dtrain = xgb.DMatrix(data = X, label = Y)
    print("-*-*-*-*-*-*-* CV Started *-*-*-*-*-*-*-")
    cv_results = xgb.cv(params = default_params,
                        dtrain = dtrain,
                        num_boost_round = num_boost_round,
                        nfold = nfold,
                        stratified = stratified,
                        metrics = metrics,
                        early_stopping_rounds = early_stopping_rounds,
                        seed = seed,
                        shuffle = shuffle,
                        callbacks = callbacks)
    print(F"*-*- Boosting Round = {len(cv_results) - 1} -*-*- Train = {cv_results.iloc[-1][0]:.3} -*-*- Test = {cv_results.iloc[-1][2]:.3} -*-*")
    print("-*-*-*-*-*-*-* Train Started *-*-*-*-*-*-*-")
    bst = xgb.train(params = default_params,
                    dtrain = dtrain,
                    num_boost_round = len(cv_results) - 1,
                   )
    
    return bst, cv_results

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _plot_xgboost_cv_score

def _plot_xgboost_cv_score(cv_results):
    """
    a function to plot train/test cv score for 
    """

    import matplotlib as mpl
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] = 2
    plt.figure(figsize=(10,8))
    plt.errorbar(range(cv_results.shape[0]), cv_results["train-auc-mean"],
                yerr=cv_results["train-auc-std"], fmt = "--", ecolor="lightgreen", c = "navy", label = "Train (9-Folds)")

    plt.errorbar(range(cv_results.shape[0]), cv_results["test-auc-mean"],
                yerr=cv_results["test-auc-std"], fmt = "--", ecolor="lightblue", c = "red", label = "Test (1-Fold)")


    plt.xlabel("# of Boosting Rounds" ,  fontweight = "bold" , fontsize=30)
    plt.ylabel("AUC",fontweight = "bold" , fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(loc = 4, prop={'size': 20})
    plt.show()
    
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _plot_importance

def _plot_importance(bst, importance_type = "total_gain", color = "pink", figsize = (10,10)):
    """
    a function to plot feature importance in xgboost
    """
    from xgboost import plot_importance
    from pylab import rcParams
    rcParams['figure.figsize'] = figsize
    plot_importance(bst, importance_type = importance_type, color = color, xlabel = importance_type)
    plt.show()

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _plot_confusion_matrix
def _plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize = 20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize = 20)
    plt.xlabel('Predicted label', fontsize = 20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _clf_xgboost

def _clf_xgboost(df_X, Y, test_size, n_folds):
    X = df_X.values
    clf = XGBClassifier(learning_rate = 0.05,
                                n_estimators = 100,
                                max_depth = 3,
                                min_child_weight = 5.,
                               # max_delta_step= 1,
                                gamma = 0.5,
                                reg_alpha = 0.0,
                                reg_lambda = 1.0,
                                subsample = 0.9,
                                colsample_bytree = 0.9,
                                objective = "binary:logistic",
                                nthread = 4,
                                #scale_pos_weight = (len(Y) - sum(Y)) / sum(Y),
                                base_score = np.mean(Y),
                                seed = 1367,
                                random_state = 1367)
    
    X_train, X_test, y_train, y_test = train_test_split(df_X, Y, test_size = test_size, shuffle = True, random_state = 1367, stratify = Y)
    clf.fit(X_train, y_train, eval_metric="auc", early_stopping_rounds = 50, verbose = False, eval_set=[(X_test, y_test)])
    accuracy_score = clf.score(X_test, y_test)
    cv_roc_score = cross_val_score(clf, X, Y, cv = n_folds, scoring = "roc_auc")
    y_pred_prob = clf.predict_proba(X_test)
    roc_score = roc_auc_score(y_test, y_pred_prob[:,1])
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
    optimal_idx = np.argmax(np.abs(tpr - fpr))
    optimal_threshold = thresholds[optimal_idx]
    
    return accuracy_score, roc_score, cv_roc_score, y_pred_prob, X_test, y_test, optimal_threshold

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _xgboost_auc_hist
def _xgboost_auc_hist(df_X, Y, n_iterations = 100, alpha = 0.95):
    """
    plot histogram of ROC AUC for N stratified classification for 95% significance level
    """

    n_size = int(len(Y) * 0.30)

    # Main Loop
    scores = list()
    for i in range(n_iterations):
        # split the data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(df_X, Y, test_size = n_size, shuffle = True, random_state = 1367 * i , stratify = Y)
        # fit the xgboost classifier
        clf = XGBClassifier(learning_rate = 0.05,
                                    n_estimators = 100,
                                    max_depth = 3,
                                    min_child_weight = 5.,
                                   # max_delta_step= 1,
                                    gamma = 0.5,
                                    reg_alpha = 0.0,
                                    reg_lambda = 1.0,
                                    subsample = 0.9,
                                    colsample_bytree = 0.9,
                                    objective = "binary:logistic",
                                    nthread = 4,
                                    #scale_pos_weight = (len(Y) - sum(Y)) / sum(Y),
                                    base_score = np.mean(Y),
                                    seed = 1367 * i ,
                                    random_state = 1367 * i)
        clf.fit(X_train, y_train, eval_metric = "auc", early_stopping_rounds = 50, verbose = False, eval_set = [(X_test, y_test)])
        # evaluate model
        predictions = clf.predict_proba(X_test)[:,1]
        pred_auc = roc_auc_score(y_test, predictions)
        scores.append(pred_auc)

    # Ploting the scores
    plt.figure(figsize=(10,5))
    plt.hist(scores, color = "pink")
    plt.xlabel("ROC AUC", fontsize = 30)
    plt.tick_params(axis = "both", which = "major", labelsize = 15)
    plt.show()

    # confidence intervals
    alpha = 0.95
    p = ((1.0 - alpha)/2.0) * 100
    lower = max(0.0, np.percentile(scores, p))
    p = (alpha + ((1.0 - alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(scores, p))
    print(F"{alpha*100}% Significance Level - ROC AUC Confidence Interval= [{lower:.3f} , {upper:.3f}]")
    
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _glmnet

def _glmnet(X, Y, alpha = 0.1, n_splits = 10, scoring = "roc_auc"):
    """
    a function for standard glmnet
    input parameters: pandas dataframe X
                      class labels Y
                      alpha = 0.1
                      n_splits = 10
                      scoring = "roc_auc"
    output: glmnet model 

    """  
    model = glmnet.LogitNet(alpha = alpha,
                     n_lambda = 100,
                     n_splits = n_splits,
                     cut_point = 1.0,
                     scoring = scoring,
                     n_jobs = -1,
                     random_state = 1367)
    model.fit(X, Y)
    
    return model

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _plot_glmnet_cv_score

def _plot_glmnet_cv_score(model):
    """
    a function to plot cv scores vs lambda
    parameters:
               model : a fitted glmnet object
    """
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] =2
    plt.figure(figsize=(10,6))
    plt.errorbar(-np.log(model.lambda_path_), model.cv_mean_score_, yerr=model.cv_standard_error_ , c = "r", ecolor="k", marker = "o" )
    plt.vlines(-np.log(model.lambda_best_), ymin = min(model.cv_mean_score_) - 0.05 , ymax = max(model.cv_mean_score_) + 0.05, lw = 3, linestyles = "--", colors = "b" ,label = "best $\lambda$")
    plt.vlines(-np.log(model.lambda_max_), ymin = min(model.cv_mean_score_) - 0.05 , ymax = max(model.cv_mean_score_) + 0.05, lw = 3, linestyles = "--", colors = "c" ,label = "max $\lambda$")
    plt.tick_params(axis='both', which='major', labelsize = 12)
    plt.grid(True)
    plt.ylim([min(model.cv_mean_score_) - 0.05, max(model.cv_mean_score_) + 0.05])
    plt.legend(loc = 4, prop={'size': 20})
    plt.xlabel("$-Log(\lambda)$" , fontsize = 20)
    plt.ylabel(F"Mean {model.n_splits} Folds CV {(model.scoring).upper()}", fontsize = 20)
    plt.title(F"Best $\lambda$ = {model.lambda_best_[0]:.2} with {len(np.nonzero(  model.coef_)[1])} Features" , fontsize = 20)
    plt.show()

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _plot_glmnet_coeff_path

def _plot_glmnet_coeff_path(model, df):
    """
    a function to plot coefficients vs lambda
    parameters:
               model : a fitted glmnet object
               df: in case that the input is the pandas dataframe,
                   the column names of the coeff. will appear as a legend
    """
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] =2
    plt.figure(figsize=(10,6))
    if not df.empty:
        for i in list(np.nonzero(np.reshape(model.coef_, (1,-1)))[1]):
            plt.plot(-np.log(model.lambda_path_) ,(model.coef_path_.reshape(-1,model.coef_path_.shape[-1]))[i,:], label = df.columns.values.tolist()[i]);
        plt.legend(loc= "right", bbox_to_anchor=(1.2 , .5), ncol=1, fancybox=True, shadow=True)

    else:
        for i in list(np.nonzero(np.reshape(model.coef_, (1,-1)))[1]):
            plt.plot(-np.log(model.lambda_path_) ,(model.coef_path_.reshape(-1, model.coef_path_.shape[-1]))[i,:]);
            
    plt.tick_params(axis='both', which='major', labelsize = 12)    
    plt.ylabel("Coefficients", fontsize = 20)
    plt.xlabel("-$Log(\lambda)$", fontsize = 20)
    plt.title(F"Best $\lambda$ = {model.lambda_best_[0]:.2} with {len(np.nonzero(model.coef_)[1])} Features" , fontsize = 20)
    plt.grid(True)
    plt.show()
    
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _df_glmnet_coeff_path

def _df_glmnet_coeff_path(model, df):
    """
    a function to build a dataframe for nonzero coeff.
    parameters:
               model : a fitted glmnet object
               df: in case that the input is the pandas dataframe,
                   the column names of the coeff. will appear as a legend
                   
    """
    idx = list(np.nonzero(np.reshape(model.coef_, (1,-1)))[1])
    dct = dict( zip([df.columns.tolist()[i] for i in idx], [model.coef_[0][i] for i in idx]))
        
    return pd.DataFrame(data = dct.items() , columns = ["Features", "Coeffs"]).sort_values(by = "Coeffs", ascending = False).reset_index(drop = True)

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: _glmnet_best_score_alpha

def _glmnet_best_score_alpha(X, Y):
    """
    a function to run glmnet for different alpha
    """
    
    def logitnet_nonzero_coef(g):
        idx = np.where(g.coef_[0] != 0)[0]
        return pd.DataFrame({"idx": idx, "coef": g.coef_[0][idx]})

    alpha_list = np.arange(0.0, 0.5, 0.05)
    mean_auc = []
    std_auc = []
    n_features = []
    for alpha in alpha_list:
        model = _glmnet(X, Y, alpha = alpha)
        mean_auc.append(model.cv_mean_score_[model.lambda_best_inx_])
        std_auc.append(model.cv_standard_error_[model.lambda_best_inx_])
        fe = logitnet_nonzero_coef(model)
        n_features.append(fe.shape[0])
        
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] =2
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.errorbar(alpha_list, mean_auc, yerr = std_auc, c = "r", ecolor="k", marker = "o" )
    plt.tick_params(axis='both', which='major', labelsize = 12)
    plt.xlabel(r"$\alpha$" , fontsize = 20)
    plt.ylabel("Mean 10 Folds CV AUC", fontsize = 20)
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(alpha_list, n_features, c = "r",  marker = "o" )
    plt.tick_params(axis='both', which='major', labelsize = 12)
    plt.xlabel(r"$\alpha$" , fontsize = 20)
    plt.ylabel("Number of Features", fontsize = 20)
    plt.grid(True)

    plt.show()

   
    
```

### First, I loaded the data into a pandas dataframe to get some idea.


```python
# readin the data into a dataframe
dateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
df_raw = pd.read_csv("./device_failure.csv",
                     parse_dates = ["date"],
                     date_parser = dateparser,
                     encoding = "cp1252")
```


```python
print("Shape: {}".format(df_raw.shape))
print("Prevalence = {:.3f}%".format(df_raw["failure"].sum()/df_raw.shape[0] * 100))
```

    Shape: (124494, 12)
    Prevalence = 0.085%



```python
df_raw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>device</th>
      <th>failure</th>
      <th>attribute1</th>
      <th>attribute2</th>
      <th>attribute3</th>
      <th>attribute4</th>
      <th>attribute5</th>
      <th>attribute6</th>
      <th>attribute7</th>
      <th>attribute8</th>
      <th>attribute9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01</td>
      <td>S1F01085</td>
      <td>0</td>
      <td>215630672</td>
      <td>56</td>
      <td>0</td>
      <td>52</td>
      <td>6</td>
      <td>407438</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-01</td>
      <td>S1F0166B</td>
      <td>0</td>
      <td>61370680</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>403174</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-01</td>
      <td>S1F01E6Y</td>
      <td>0</td>
      <td>173295968</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>237394</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-01</td>
      <td>S1F01JE0</td>
      <td>0</td>
      <td>79694024</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>410186</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-01</td>
      <td>S1F01R2B</td>
      <td>0</td>
      <td>135970480</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>313173</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### Printing out the statistical description of the data. It would help us to have idea about the variation of the each column in the data.


```python
df_raw.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>failure</th>
      <th>attribute1</th>
      <th>attribute2</th>
      <th>attribute3</th>
      <th>attribute4</th>
      <th>attribute5</th>
      <th>attribute6</th>
      <th>attribute7</th>
      <th>attribute8</th>
      <th>attribute9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>124494.000000</td>
      <td>1.244940e+05</td>
      <td>124494.000000</td>
      <td>124494.000000</td>
      <td>124494.000000</td>
      <td>124494.000000</td>
      <td>124494.000000</td>
      <td>124494.000000</td>
      <td>124494.000000</td>
      <td>124494.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.000851</td>
      <td>1.223881e+08</td>
      <td>159.484762</td>
      <td>9.940455</td>
      <td>1.741120</td>
      <td>14.222669</td>
      <td>260172.657726</td>
      <td>0.292528</td>
      <td>0.292528</td>
      <td>12.451524</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.029167</td>
      <td>7.045933e+07</td>
      <td>2179.657730</td>
      <td>185.747321</td>
      <td>22.908507</td>
      <td>15.943028</td>
      <td>99151.078547</td>
      <td>7.436924</td>
      <td>7.436924</td>
      <td>191.425623</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>6.128476e+07</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>221452.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>1.227974e+08</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>249799.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>1.833096e+08</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>310266.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>2.441405e+08</td>
      <td>64968.000000</td>
      <td>24929.000000</td>
      <td>1666.000000</td>
      <td>98.000000</td>
      <td>689161.000000</td>
      <td>832.000000</td>
      <td>832.000000</td>
      <td>18701.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Printing out the number of Null values in each column. These numbers should be imputed before feeding into the classifier.


```python
df_raw.isnull().sum()
```




    date          0
    device        0
    failure       0
    attribute1    0
    attribute2    0
    attribute3    0
    attribute4    0
    attribute5    0
    attribute6    0
    attribute7    0
    attribute8    0
    attribute9    0
    dtype: int64



### Checking the data types of the columns to make sure all the columns have continuous values as it is mentioned in the meta data, except date and device


```python
df_raw.dtypes
```




    date          datetime64[ns]
    device                object
    failure                int64
    attribute1             int64
    attribute2             int64
    attribute3             int64
    attribute4             int64
    attribute5             int64
    attribute6             int64
    attribute7             int64
    attribute8             int64
    attribute9             int64
    dtype: object



### Checking the pair-wise correlation between features
### methods = {'pearson', 'kendall', 'spearman'} 


```python
df_raw.corr(method = "kendall")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>failure</th>
      <th>attribute1</th>
      <th>attribute2</th>
      <th>attribute3</th>
      <th>attribute4</th>
      <th>attribute5</th>
      <th>attribute6</th>
      <th>attribute7</th>
      <th>attribute8</th>
      <th>attribute9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>failure</th>
      <td>1.000000</td>
      <td>0.001611</td>
      <td>0.053361</td>
      <td>0.003208</td>
      <td>0.057295</td>
      <td>0.003475</td>
      <td>0.002337</td>
      <td>0.096926</td>
      <td>0.096926</td>
      <td>0.004633</td>
    </tr>
    <tr>
      <th>attribute1</th>
      <td>0.001611</td>
      <td>1.000000</td>
      <td>-0.001004</td>
      <td>0.001960</td>
      <td>0.001289</td>
      <td>-0.003713</td>
      <td>-0.001779</td>
      <td>-0.002030</td>
      <td>-0.002030</td>
      <td>-0.002662</td>
    </tr>
    <tr>
      <th>attribute2</th>
      <td>0.053361</td>
      <td>-0.001004</td>
      <td>1.000000</td>
      <td>-0.018322</td>
      <td>0.219415</td>
      <td>-0.022125</td>
      <td>-0.062126</td>
      <td>0.107491</td>
      <td>0.107491</td>
      <td>-0.027759</td>
    </tr>
    <tr>
      <th>attribute3</th>
      <td>0.003208</td>
      <td>0.001960</td>
      <td>-0.018322</td>
      <td>1.000000</td>
      <td>0.117459</td>
      <td>0.089091</td>
      <td>0.056736</td>
      <td>-0.009622</td>
      <td>-0.009622</td>
      <td>0.369092</td>
    </tr>
    <tr>
      <th>attribute4</th>
      <td>0.057295</td>
      <td>0.001289</td>
      <td>0.219415</td>
      <td>0.117459</td>
      <td>1.000000</td>
      <td>-0.017980</td>
      <td>0.009706</td>
      <td>0.160216</td>
      <td>0.160216</td>
      <td>0.046189</td>
    </tr>
    <tr>
      <th>attribute5</th>
      <td>0.003475</td>
      <td>-0.003713</td>
      <td>-0.022125</td>
      <td>0.089091</td>
      <td>-0.017980</td>
      <td>1.000000</td>
      <td>0.057295</td>
      <td>-0.016603</td>
      <td>-0.016603</td>
      <td>0.026998</td>
    </tr>
    <tr>
      <th>attribute6</th>
      <td>0.002337</td>
      <td>-0.001779</td>
      <td>-0.062126</td>
      <td>0.056736</td>
      <td>0.009706</td>
      <td>0.057295</td>
      <td>1.000000</td>
      <td>-0.013186</td>
      <td>-0.013186</td>
      <td>0.069585</td>
    </tr>
    <tr>
      <th>attribute7</th>
      <td>0.096926</td>
      <td>-0.002030</td>
      <td>0.107491</td>
      <td>-0.009622</td>
      <td>0.160216</td>
      <td>-0.016603</td>
      <td>-0.013186</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-0.017097</td>
    </tr>
    <tr>
      <th>attribute8</th>
      <td>0.096926</td>
      <td>-0.002030</td>
      <td>0.107491</td>
      <td>-0.009622</td>
      <td>0.160216</td>
      <td>-0.016603</td>
      <td>-0.013186</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-0.017097</td>
    </tr>
    <tr>
      <th>attribute9</th>
      <td>0.004633</td>
      <td>-0.002662</td>
      <td>-0.027759</td>
      <td>0.369092</td>
      <td>0.046189</td>
      <td>0.026998</td>
      <td>0.069585</td>
      <td>-0.017097</td>
      <td>-0.017097</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Number of unique devices: this can be used as feauture based on the demand
df_raw["device"].nunique()
```




    1169



### Preprocessing: attribute7 and attribute8 are totally correlated (the same feature). For now, I am gonna drop date and device as well. Device has 1169 unique values that can be used as features.


```python
df = df_raw.copy()
target = "failure"
to_drop = ["date", "device", "attribute8"]
df_X, Y = _x_y_maker(df, target, to_drop)
```

### Printing the first 5 rows of the sclaed features. 


```python
df_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>attribute1</th>
      <th>attribute2</th>
      <th>attribute3</th>
      <th>attribute4</th>
      <th>attribute5</th>
      <th>attribute6</th>
      <th>attribute7</th>
      <th>attribute9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.323358</td>
      <td>-0.047478</td>
      <td>-0.053516</td>
      <td>2.193905</td>
      <td>-0.515755</td>
      <td>1.485268</td>
      <td>-0.039335</td>
      <td>-0.028479</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.865998</td>
      <td>-0.073170</td>
      <td>-0.037365</td>
      <td>-0.076004</td>
      <td>-0.515755</td>
      <td>1.442263</td>
      <td>-0.039335</td>
      <td>-0.065047</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.722517</td>
      <td>-0.073170</td>
      <td>-0.053516</td>
      <td>-0.076004</td>
      <td>-0.139414</td>
      <td>-0.229738</td>
      <td>-0.039335</td>
      <td>-0.065047</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.605942</td>
      <td>-0.073170</td>
      <td>-0.053516</td>
      <td>-0.076004</td>
      <td>-0.515755</td>
      <td>1.512983</td>
      <td>-0.039335</td>
      <td>-0.065047</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.192770</td>
      <td>-0.073170</td>
      <td>-0.053516</td>
      <td>-0.076004</td>
      <td>0.048757</td>
      <td>0.534543</td>
      <td>-0.039335</td>
      <td>-0.049375</td>
    </tr>
  </tbody>
</table>
</div>



### Checking the correlation again.


```python
df_X.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>attribute1</th>
      <th>attribute2</th>
      <th>attribute3</th>
      <th>attribute4</th>
      <th>attribute5</th>
      <th>attribute6</th>
      <th>attribute7</th>
      <th>attribute9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>attribute1</th>
      <td>1.000000</td>
      <td>-0.004250</td>
      <td>0.003701</td>
      <td>0.001836</td>
      <td>-0.003376</td>
      <td>-0.001522</td>
      <td>0.000151</td>
      <td>0.001121</td>
    </tr>
    <tr>
      <th>attribute2</th>
      <td>-0.004250</td>
      <td>1.000000</td>
      <td>-0.002617</td>
      <td>0.146593</td>
      <td>-0.013999</td>
      <td>-0.026350</td>
      <td>0.141367</td>
      <td>-0.002736</td>
    </tr>
    <tr>
      <th>attribute3</th>
      <td>0.003701</td>
      <td>-0.002617</td>
      <td>1.000000</td>
      <td>0.097452</td>
      <td>-0.006696</td>
      <td>0.009027</td>
      <td>-0.001884</td>
      <td>0.532366</td>
    </tr>
    <tr>
      <th>attribute4</th>
      <td>0.001836</td>
      <td>0.146593</td>
      <td>0.097452</td>
      <td>1.000000</td>
      <td>-0.009773</td>
      <td>0.024870</td>
      <td>0.045631</td>
      <td>0.036069</td>
    </tr>
    <tr>
      <th>attribute5</th>
      <td>-0.003376</td>
      <td>-0.013999</td>
      <td>-0.006696</td>
      <td>-0.009773</td>
      <td>1.000000</td>
      <td>-0.017049</td>
      <td>-0.009384</td>
      <td>0.005949</td>
    </tr>
    <tr>
      <th>attribute6</th>
      <td>-0.001522</td>
      <td>-0.026350</td>
      <td>0.009027</td>
      <td>0.024870</td>
      <td>-0.017049</td>
      <td>1.000000</td>
      <td>-0.012207</td>
      <td>0.021152</td>
    </tr>
    <tr>
      <th>attribute7</th>
      <td>0.000151</td>
      <td>0.141367</td>
      <td>-0.001884</td>
      <td>0.045631</td>
      <td>-0.009384</td>
      <td>-0.012207</td>
      <td>1.000000</td>
      <td>0.006861</td>
    </tr>
    <tr>
      <th>attribute9</th>
      <td>0.001121</td>
      <td>-0.002736</td>
      <td>0.532366</td>
      <td>0.036069</td>
      <td>0.005949</td>
      <td>0.021152</td>
      <td>0.006861</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Running my feature selector along with noise with 10-folds CV for multiple iterations to make sure about the selected features. This would help to have a simpler model with the same performance


```python
# choosing params to under-fit the model for feature selection + noisy features. This would help to have simpler model
params = {
          "learning_rate" : 0.05,
          "max_depth": 2,
          "min_child_weight": 5,
          "gamma" : 0.5
          
         }
o_, d_ = _my_feature_selector(df_X, Y, n_iter = 10,
                                       params = params,
                                       seed = 1367,
                                       callbacks = False)
```

    [21:19:05] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:05] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:05] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:05] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:19:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:09] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:09] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:20:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:07] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:24] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:21:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:21] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    -*-*-*-*-*-*-*-*- Iteration 1 -*-*-*-*-*-*-*-*
    -*-*- Internal-10 Folds-CV-Train = 0.934 +/- 0.00757 -*-*- Internal-10 Folds-CV-Test = 0.897 +/- 0.00767 -*-*
    -*-*- External-10 Folds-CV-Train = 0.934 +/- 0.00823 -*-*- External-10 Folds-CV-Test = 0.897 +/- 0.0423 -*-*
    [21:22:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:22:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:23:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:54] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:24:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:25:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    -*-*-*-*-*-*-*-*- Iteration 2 -*-*-*-*-*-*-*-*
    -*-*- Internal-10 Folds-CV-Train = 0.934 +/- 0.00786 -*-*- Internal-10 Folds-CV-Test = 0.896 +/- 0.00774 -*-*
    -*-*- External-10 Folds-CV-Train = 0.933 +/- 0.00855 -*-*- External-10 Folds-CV-Test = 0.891 +/- 0.0502 -*-*
    [21:26:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:40] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:26:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:20] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:27:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:00] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:02] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:02] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:02] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:02] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:02] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:02] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:02] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:02] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:02] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:02] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:50] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:50] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:50] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:50] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:50] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:50] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:50] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:50] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:50] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:28:50] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:14] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    -*-*-*-*-*-*-*-*- Iteration 3 -*-*-*-*-*-*-*-*
    -*-*- Internal-10 Folds-CV-Train = 0.931 +/- 0.0051 -*-*- Internal-10 Folds-CV-Test = 0.896 +/- 0.00629 -*-*
    -*-*- External-10 Folds-CV-Train = 0.93 +/- 0.006 -*-*- External-10 Folds-CV-Test = 0.89 +/- 0.0632 -*-*
    [21:29:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:29:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:14] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:30:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:21] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:31:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:27] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:27] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:32:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    -*-*-*-*-*-*-*-*- Iteration 4 -*-*-*-*-*-*-*-*
    -*-*- Internal-10 Folds-CV-Train = 0.935 +/- 0.00574 -*-*- Internal-10 Folds-CV-Test = 0.897 +/- 0.00864 -*-*
    -*-*- External-10 Folds-CV-Train = 0.933 +/- 0.00606 -*-*- External-10 Folds-CV-Test = 0.894 +/- 0.0556 -*-*
    [21:33:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:33:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:37] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:34:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:19] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:35:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:16] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:40] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:36:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:14] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    -*-*-*-*-*-*-*-*- Iteration 5 -*-*-*-*-*-*-*-*
    -*-*- Internal-10 Folds-CV-Train = 0.935 +/- 0.00767 -*-*- Internal-10 Folds-CV-Test = 0.897 +/- 0.0053 -*-*
    -*-*- External-10 Folds-CV-Train = 0.934 +/- 0.00798 -*-*- External-10 Folds-CV-Test = 0.893 +/- 0.0474 -*-*
    [21:37:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:37] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:37:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:03] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:24] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:26] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:26] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:26] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:26] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:26] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:26] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:26] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:26] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:26] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:26] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:53] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:38:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:29] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:54] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:39:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:20] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:23] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    -*-*-*-*-*-*-*-*- Iteration 6 -*-*-*-*-*-*-*-*
    -*-*- Internal-10 Folds-CV-Train = 0.932 +/- 0.00752 -*-*- Internal-10 Folds-CV-Test = 0.898 +/- 0.00819 -*-*
    -*-*- External-10 Folds-CV-Train = 0.93 +/- 0.00746 -*-*- External-10 Folds-CV-Test = 0.892 +/- 0.0489 -*-*
    [21:40:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:40:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:18] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:41:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:12] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:54] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:42:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:17] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:35] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:36] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:51] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:52] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:52] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:52] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:52] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:52] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:52] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:52] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:53] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:53] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:43:53] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    -*-*-*-*-*-*-*-*- Iteration 7 -*-*-*-*-*-*-*-*
    -*-*- Internal-10 Folds-CV-Train = 0.933 +/- 0.00717 -*-*- Internal-10 Folds-CV-Test = 0.897 +/- 0.0057 -*-*
    -*-*- External-10 Folds-CV-Train = 0.932 +/- 0.00791 -*-*- External-10 Folds-CV-Test = 0.893 +/- 0.0352 -*-*
    [21:44:10] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:10] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:10] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:10] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:10] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:10] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:10] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:30] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:47] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:44:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:10] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:12] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:12] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:12] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:12] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:12] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:12] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:12] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:12] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:12] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:12] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:29] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:30] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:30] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:52] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:54] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:54] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:54] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:54] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:45:55] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:13] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:44] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:48] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:48] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:48] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:48] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:48] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:48] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:48] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:48] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:48] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:46:48] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:24] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:24] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:24] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:24] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:24] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:24] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:24] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:24] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:24] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:24] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    -*-*-*-*-*-*-*-*- Iteration 8 -*-*-*-*-*-*-*-*
    -*-*- Internal-10 Folds-CV-Train = 0.933 +/- 0.00859 -*-*- Internal-10 Folds-CV-Test = 0.898 +/- 0.00904 -*-*
    -*-*- External-10 Folds-CV-Train = 0.932 +/- 0.009 -*-*- External-10 Folds-CV-Test = 0.89 +/- 0.0797 -*-*
    [21:47:47] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:47] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:47] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:47] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:47] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:47] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:47] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:47] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:47] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:47] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:47:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:48:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:09] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:11] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:32] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:47] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:49:49] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:08] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:24] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:25] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:26] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:44] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:50:46] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:03] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    -*-*-*-*-*-*-*-*- Iteration 9 -*-*-*-*-*-*-*-*
    -*-*- Internal-10 Folds-CV-Train = 0.932 +/- 0.00833 -*-*- Internal-10 Folds-CV-Test = 0.895 +/- 0.01 -*-*
    -*-*- External-10 Folds-CV-Train = 0.931 +/- 0.00866 -*-*- External-10 Folds-CV-Test = 0.891 +/- 0.0605 -*-*
    [21:51:05] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:05] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:05] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:05] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:05] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:06] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:21] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:22] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:40] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:41] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:51:42] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:02] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:04] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:05] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:05] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:26] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:28] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:29] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:29] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:43] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:44] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:44] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:44] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:44] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:44] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:44] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:44] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:45] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:52:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:01] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:31] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:34] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:57] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:53:59] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:54:14] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:54:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:54:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:54:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:54:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:54:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:54:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:54:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:54:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:54:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:54:15] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [21:54:37] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    -*-*-*-*-*-*-*-*- Iteration 10 -*-*-*-*-*-*-*-*
    -*-*- Internal-10 Folds-CV-Train = 0.933 +/- 0.00738 -*-*- Internal-10 Folds-CV-Test = 0.899 +/- 0.00686 -*-*
    -*-*- External-10 Folds-CV-Train = 0.932 +/- 0.0068 -*-*- External-10 Folds-CV-Test = 0.877 +/- 0.0567 -*-*



    <Figure size 432x288 with 0 Axes>



![png](output_25_2.png)



![png](output_25_3.png)



```python
params = {
          "learning_rate" : 0.05,
          "max_depth": 2,
          "min_child_weight": 5.,
          "gamma" : 0.5
         }
bst, cvr = _bst_shap(df_X, Y, params = params, callbacks = True)
```

    -*-*-*-*-*-*-* CV Started *-*-*-*-*-*-*-
    [22:09:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [22:09:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [22:09:38] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [22:09:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [22:09:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [22:09:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [22:09:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [22:09:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [22:09:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [22:09:39] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    [0]	train-auc:0.5	test-auc:0.5
    Multiple eval metrics have been passed: 'test-auc' will be used for early stopping.
    
    Will train until test-auc hasn't improved in 50 rounds.
    [1]	train-auc:0.5	test-auc:0.5
    [2]	train-auc:0.5	test-auc:0.5
    [3]	train-auc:0.5	test-auc:0.5
    [4]	train-auc:0.5	test-auc:0.5
    [5]	train-auc:0.5	test-auc:0.5
    [6]	train-auc:0.5	test-auc:0.5
    [7]	train-auc:0.5	test-auc:0.5
    [8]	train-auc:0.5	test-auc:0.5
    [9]	train-auc:0.5	test-auc:0.5
    [10]	train-auc:0.5	test-auc:0.5
    [11]	train-auc:0.5	test-auc:0.5
    [12]	train-auc:0.5	test-auc:0.5
    [13]	train-auc:0.5	test-auc:0.5
    [14]	train-auc:0.5	test-auc:0.5
    [15]	train-auc:0.5	test-auc:0.5
    [16]	train-auc:0.5	test-auc:0.5
    [17]	train-auc:0.5	test-auc:0.5
    [18]	train-auc:0.502591	test-auc:0.499988
    [19]	train-auc:0.507799	test-auc:0.499972
    [20]	train-auc:0.515655	test-auc:0.499924
    [21]	train-auc:0.515655	test-auc:0.499924
    [22]	train-auc:0.521354	test-auc:0.504899
    [23]	train-auc:0.527633	test-auc:0.509397
    [24]	train-auc:0.544239	test-auc:0.51922
    [25]	train-auc:0.547894	test-auc:0.519192
    [26]	train-auc:0.562889	test-auc:0.528576
    [27]	train-auc:0.562888	test-auc:0.528573
    [28]	train-auc:0.575345	test-auc:0.551115
    [29]	train-auc:0.589815	test-auc:0.570037
    [30]	train-auc:0.589813	test-auc:0.570032
    [31]	train-auc:0.589815	test-auc:0.570038
    [32]	train-auc:0.594991	test-auc:0.579028
    [33]	train-auc:0.603815	test-auc:0.588018
    [34]	train-auc:0.607991	test-auc:0.587983
    [35]	train-auc:0.607994	test-auc:0.587977
    [36]	train-auc:0.609034	test-auc:0.587952
    [37]	train-auc:0.611603	test-auc:0.58792
    [38]	train-auc:0.611606	test-auc:0.58792
    [39]	train-auc:0.621348	test-auc:0.592728
    [40]	train-auc:0.630541	test-auc:0.59751
    [41]	train-auc:0.641352	test-auc:0.610898
    [42]	train-auc:0.645456	test-auc:0.619846
    [43]	train-auc:0.649023	test-auc:0.624774
    [44]	train-auc:0.649023	test-auc:0.629293
    [45]	train-auc:0.652603	test-auc:0.638277
    [46]	train-auc:0.656164	test-auc:0.638207
    [47]	train-auc:0.656163	test-auc:0.638208
    [48]	train-auc:0.660153	test-auc:0.638036
    [49]	train-auc:0.660154	test-auc:0.638033
    [50]	train-auc:0.668212	test-auc:0.637722
    [51]	train-auc:0.683198	test-auc:0.651949
    [52]	train-auc:0.68676	test-auc:0.65183
    [53]	train-auc:0.702173	test-auc:0.664918
    [54]	train-auc:0.705216	test-auc:0.673904
    [55]	train-auc:0.716932	test-auc:0.688166
    [56]	train-auc:0.746947	test-auc:0.705966
    [57]	train-auc:0.766238	test-auc:0.732417
    [58]	train-auc:0.76625	test-auc:0.732301
    [59]	train-auc:0.781952	test-auc:0.749171
    [60]	train-auc:0.785332	test-auc:0.753368
    [61]	train-auc:0.788794	test-auc:0.753104
    [62]	train-auc:0.788767	test-auc:0.753255
    [63]	train-auc:0.789759	test-auc:0.757569
    [64]	train-auc:0.789792	test-auc:0.757407
    [65]	train-auc:0.791717	test-auc:0.762459
    [66]	train-auc:0.791741	test-auc:0.762543
    [67]	train-auc:0.794139	test-auc:0.771179
    [68]	train-auc:0.79414	test-auc:0.771128
    [69]	train-auc:0.798393	test-auc:0.78537
    [70]	train-auc:0.799311	test-auc:0.790212
    [71]	train-auc:0.799362	test-auc:0.790179
    [72]	train-auc:0.799436	test-auc:0.790156
    [73]	train-auc:0.804061	test-auc:0.789724
    [74]	train-auc:0.809716	test-auc:0.793472
    [75]	train-auc:0.81023	test-auc:0.793571
    [76]	train-auc:0.810291	test-auc:0.793676
    [77]	train-auc:0.810273	test-auc:0.794129
    [78]	train-auc:0.815368	test-auc:0.793779
    [79]	train-auc:0.818557	test-auc:0.798271
    [80]	train-auc:0.818553	test-auc:0.798253
    [81]	train-auc:0.818556	test-auc:0.798244
    [82]	train-auc:0.819048	test-auc:0.798204
    [83]	train-auc:0.819046	test-auc:0.798256
    [84]	train-auc:0.819084	test-auc:0.798343
    [85]	train-auc:0.819094	test-auc:0.798331
    [86]	train-auc:0.826427	test-auc:0.810713
    [87]	train-auc:0.827297	test-auc:0.815134
    [88]	train-auc:0.827317	test-auc:0.820203
    [89]	train-auc:0.82735	test-auc:0.820224
    [90]	train-auc:0.835594	test-auc:0.836472
    [91]	train-auc:0.839335	test-auc:0.83642
    [92]	train-auc:0.839334	test-auc:0.836343
    [93]	train-auc:0.844798	test-auc:0.840614
    [94]	train-auc:0.845237	test-auc:0.840554
    [95]	train-auc:0.845206	test-auc:0.840589
    [96]	train-auc:0.845202	test-auc:0.840541
    [97]	train-auc:0.845634	test-auc:0.840411
    [98]	train-auc:0.846109	test-auc:0.840341
    [99]	train-auc:0.8461	test-auc:0.840213
    [100]	train-auc:0.846579	test-auc:0.840174
    [101]	train-auc:0.847455	test-auc:0.840033
    [102]	train-auc:0.848847	test-auc:0.839937
    [103]	train-auc:0.848889	test-auc:0.839934
    [104]	train-auc:0.851638	test-auc:0.839058
    [105]	train-auc:0.851658	test-auc:0.839094
    [106]	train-auc:0.851696	test-auc:0.839025
    [107]	train-auc:0.851761	test-auc:0.839107
    [108]	train-auc:0.855548	test-auc:0.838203
    [109]	train-auc:0.855952	test-auc:0.838216
    [110]	train-auc:0.859843	test-auc:0.837271
    [111]	train-auc:0.859896	test-auc:0.837218
    [112]	train-auc:0.862772	test-auc:0.844197
    [113]	train-auc:0.866681	test-auc:0.847177
    [114]	train-auc:0.866765	test-auc:0.847206
    [115]	train-auc:0.867214	test-auc:0.847115
    [116]	train-auc:0.868626	test-auc:0.855585
    [117]	train-auc:0.869564	test-auc:0.859661
    [118]	train-auc:0.869588	test-auc:0.859604
    [119]	train-auc:0.870008	test-auc:0.859496
    [120]	train-auc:0.870083	test-auc:0.859439
    [121]	train-auc:0.870515	test-auc:0.859276
    [122]	train-auc:0.871057	test-auc:0.859007
    [123]	train-auc:0.871131	test-auc:0.859277
    [124]	train-auc:0.871211	test-auc:0.859541
    [125]	train-auc:0.871218	test-auc:0.859699
    [126]	train-auc:0.871588	test-auc:0.863687
    [127]	train-auc:0.871767	test-auc:0.863805
    [128]	train-auc:0.87185	test-auc:0.864008
    [129]	train-auc:0.871903	test-auc:0.863906
    [130]	train-auc:0.871946	test-auc:0.863755
    [131]	train-auc:0.871992	test-auc:0.863533
    [132]	train-auc:0.87198	test-auc:0.863675
    [133]	train-auc:0.872139	test-auc:0.863535
    [134]	train-auc:0.872082	test-auc:0.86365
    [135]	train-auc:0.872188	test-auc:0.863516
    [136]	train-auc:0.872232	test-auc:0.86354
    [137]	train-auc:0.872228	test-auc:0.863652
    [138]	train-auc:0.872266	test-auc:0.863696
    [139]	train-auc:0.872255	test-auc:0.863683
    [140]	train-auc:0.872303	test-auc:0.863636
    [141]	train-auc:0.872336	test-auc:0.863547
    [142]	train-auc:0.872402	test-auc:0.863523
    [143]	train-auc:0.872441	test-auc:0.863512
    [144]	train-auc:0.876281	test-auc:0.865899
    [145]	train-auc:0.876307	test-auc:0.865908
    [146]	train-auc:0.876324	test-auc:0.86593
    [147]	train-auc:0.876275	test-auc:0.870426
    [148]	train-auc:0.876311	test-auc:0.8704
    [149]	train-auc:0.876318	test-auc:0.870282
    [150]	train-auc:0.876344	test-auc:0.870302
    [151]	train-auc:0.876323	test-auc:0.87028
    [152]	train-auc:0.876347	test-auc:0.870184
    [153]	train-auc:0.876402	test-auc:0.87025
    [154]	train-auc:0.876392	test-auc:0.870155
    [155]	train-auc:0.876437	test-auc:0.870074
    [156]	train-auc:0.876463	test-auc:0.870078
    [157]	train-auc:0.884578	test-auc:0.870697
    [158]	train-auc:0.884583	test-auc:0.87077
    [159]	train-auc:0.888683	test-auc:0.867303
    [160]	train-auc:0.88869	test-auc:0.867468
    [161]	train-auc:0.89149	test-auc:0.863448
    [162]	train-auc:0.891448	test-auc:0.86335
    [163]	train-auc:0.89525	test-auc:0.865754
    [164]	train-auc:0.895239	test-auc:0.865826
    [165]	train-auc:0.902117	test-auc:0.873953
    [166]	train-auc:0.902968	test-auc:0.872901
    [167]	train-auc:0.906869	test-auc:0.881087
    [168]	train-auc:0.908143	test-auc:0.880633
    [169]	train-auc:0.910264	test-auc:0.890346
    [170]	train-auc:0.911123	test-auc:0.888702
    [171]	train-auc:0.912856	test-auc:0.890256
    [172]	train-auc:0.912859	test-auc:0.890457
    [173]	train-auc:0.914933	test-auc:0.885244
    [174]	train-auc:0.917162	test-auc:0.888691
    [175]	train-auc:0.919827	test-auc:0.893754
    [176]	train-auc:0.921293	test-auc:0.898306
    [177]	train-auc:0.921269	test-auc:0.898513
    [178]	train-auc:0.922025	test-auc:0.900192
    [179]	train-auc:0.922283	test-auc:0.898478
    [180]	train-auc:0.922729	test-auc:0.896203
    [181]	train-auc:0.922026	test-auc:0.89636
    [182]	train-auc:0.922845	test-auc:0.89517
    [183]	train-auc:0.922994	test-auc:0.89481
    [184]	train-auc:0.923685	test-auc:0.893755
    [185]	train-auc:0.924463	test-auc:0.89445
    [186]	train-auc:0.925612	test-auc:0.895708
    [187]	train-auc:0.925986	test-auc:0.896097
    [188]	train-auc:0.926308	test-auc:0.898049
    [189]	train-auc:0.926701	test-auc:0.89994
    [190]	train-auc:0.926804	test-auc:0.900076
    [191]	train-auc:0.927169	test-auc:0.899289
    [192]	train-auc:0.927667	test-auc:0.898406
    [193]	train-auc:0.927906	test-auc:0.898717
    [194]	train-auc:0.928189	test-auc:0.897986
    [195]	train-auc:0.928687	test-auc:0.897934
    [196]	train-auc:0.928905	test-auc:0.897401
    [197]	train-auc:0.929011	test-auc:0.899055
    [198]	train-auc:0.929382	test-auc:0.898587
    [199]	train-auc:0.929621	test-auc:0.897067
    [200]	train-auc:0.930354	test-auc:0.897016
    [201]	train-auc:0.930419	test-auc:0.895062
    [202]	train-auc:0.930577	test-auc:0.894952
    [203]	train-auc:0.93098	test-auc:0.89376
    [204]	train-auc:0.930982	test-auc:0.893814
    [205]	train-auc:0.931073	test-auc:0.893492
    [206]	train-auc:0.931204	test-auc:0.893421
    [207]	train-auc:0.931199	test-auc:0.892662
    [208]	train-auc:0.931229	test-auc:0.893241
    [209]	train-auc:0.931645	test-auc:0.892992
    [210]	train-auc:0.932143	test-auc:0.893321
    [211]	train-auc:0.932168	test-auc:0.89339
    [212]	train-auc:0.932059	test-auc:0.892893
    [213]	train-auc:0.932261	test-auc:0.892952
    [214]	train-auc:0.932471	test-auc:0.892118
    [215]	train-auc:0.932655	test-auc:0.892261
    [216]	train-auc:0.932859	test-auc:0.891813
    [217]	train-auc:0.932999	test-auc:0.892246
    [218]	train-auc:0.933072	test-auc:0.892004
    [219]	train-auc:0.933236	test-auc:0.891923
    [220]	train-auc:0.93351	test-auc:0.891635
    [221]	train-auc:0.93367	test-auc:0.891286
    [222]	train-auc:0.933816	test-auc:0.891612
    [223]	train-auc:0.934	test-auc:0.891521
    [224]	train-auc:0.934035	test-auc:0.890987
    [225]	train-auc:0.934061	test-auc:0.890868
    [226]	train-auc:0.934116	test-auc:0.890957
    [227]	train-auc:0.934202	test-auc:0.890346
    [228]	train-auc:0.934334	test-auc:0.890627
    Stopping. Best iteration:
    [178]	train-auc:0.922025+0.00538528	test-auc:0.900192+0.0420388
    
    *-*- Boosting Round = 178 -*-*- Train = 0.922 -*-*- Test = 0.9 -*-*
    -*-*-*-*-*-*-* Train Started *-*-*-*-*-*-*-
    [22:09:58] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.



```python
_plot_xgboost_cv_score(cvr)
```


![png](output_27_0.png)



```python
_plot_importance(bst, figsize=(10,6), importance_type = "total_gain")
```


![png](output_28_0.png)



```python
Tree_explainer = shap.TreeExplainer(bst)
shap_Treevalues = Tree_explainer.shap_values(df_X)
shap.summary_plot(shap_Treevalues, df_X, plot_type = "bar", color = "pink", max_display = 10)
```


![png](output_29_0.png)


### As seen, attribute3 has not played an important role through the feature selection process. Therefore, I dropped it and build the final model based on that.


```python
df_pruned = df_X.drop(["attribute3"], axis = 1)
```

### As seen, the total gain for attribute4, attirbute7, and attribute 2 are higher than the rest. This would say that, for linear models such as regulirized linear models including elastic net, these features will be pruned. Now, we can see, how these features perform:


```python
_plot_roc_nfolds_xgboost(df_pruned, Y)
```


![png](output_33_0.png)



```python
test_size = 0.3
n_folds = 10
accuracy_score, roc_score, cv_roc_score, y_pred_prob, X_test, y_test, optimal_threshold = _clf_xgboost(df_pruned, Y, test_size, n_folds)
CI_neg, CI_pos = _ConfidenceInterval(roc_score, X_test.shape[0])
print(F"Accuracy for test size {test_size}  = {accuracy_score:.5f}")
print(F"ROC for test size {test_size} = {roc_score:.3f}")
print(F"Mean ROC for {n_folds} folds CV = {cv_roc_score.mean():.3f}")
print(F"Confidence Interval Accuracy = [{CI_neg:.3f} , {CI_pos:.3f}]")

class_names = [0, 1]
#optimal threshold based on Youden Index
y_pred = (y_pred_prob[:,1] > optimal_threshold).astype(int)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure(figsize=(6,6))
_plot_confusion_matrix(cnf_matrix, classes=class_names, normalize = True)
plt.show()
```

    Accuracy for test size 0.3  = 0.99914
    ROC for test size 0.3 = 0.890
    Mean ROC for 10 folds CV = 0.896
    Confidence Interval Accuracy = [0.887 , 0.893]
    Normalized confusion matrix
    [[0.91 0.09]
     [0.25 0.75]]



![png](output_34_1.png)


# Discussion:
### As seen, the xgboost approach showed a good performance in terms of classification of the individuals in terms of Failure and Non-failure. The classification problem is an imbalanced problem (prevalence < 1%). Therefore, the classification accuracy by itself cannot be trusted and the other classification metrics such as area under ROC curve and Precision-Recalls or F-1 score which are robust to imbalanced classes should be used. The other challenge is the size of the dataset. In fact, we are looking for a model that can be generalized to a larger population. In this regard, it is also better to report the classification error with a confidence interval at a statistical significance level. For instance, to have 95% significance, we can calculate the classification error from the following formula using the Z-score of 1.96:

$ConfidenceInterval = Score \pm 1.96 \times \sqrt{ \frac{Score \times (1 - Score)}{N}}$

### For instance, for the approach , we have N= 37349 (30% of the data) individuals in the testing set (unseen individuals) with an AUC of 0.89 which is within the calculated 95% confidence interval. Now, we can add the prediction pribabilities for both class 0 and 1 to the test data.


```python
X_test["pred_proba_class_0"] = pd.Series(y_pred_prob[:,0], index = X_test.index)
X_test["pred_proba_class_1"] = pd.Series(y_pred_prob[:,1], index = X_test.index)
```


```python
print(F"Test Size = {X_test.shape}")
X_test.head()
```

    Test Size = (37349, 9)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>attribute1</th>
      <th>attribute2</th>
      <th>attribute4</th>
      <th>attribute5</th>
      <th>attribute6</th>
      <th>attribute7</th>
      <th>attribute9</th>
      <th>pred_proba_class_0</th>
      <th>pred_proba_class_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22999</th>
      <td>0.394775</td>
      <td>-0.073170</td>
      <td>-0.076004</td>
      <td>-0.202137</td>
      <td>-0.414557</td>
      <td>-0.039335</td>
      <td>-0.065047</td>
      <td>0.999829</td>
      <td>0.000171</td>
    </tr>
    <tr>
      <th>26237</th>
      <td>-1.097957</td>
      <td>0.844409</td>
      <td>-0.076004</td>
      <td>0.613269</td>
      <td>-2.623549</td>
      <td>-0.039335</td>
      <td>-0.065047</td>
      <td>0.997224</td>
      <td>0.002776</td>
    </tr>
    <tr>
      <th>123435</th>
      <td>-1.179217</td>
      <td>-0.073170</td>
      <td>-0.076004</td>
      <td>-0.515755</td>
      <td>0.573101</td>
      <td>-0.039335</td>
      <td>-0.065047</td>
      <td>0.999710</td>
      <td>0.000290</td>
    </tr>
    <tr>
      <th>120834</th>
      <td>0.404221</td>
      <td>-0.073170</td>
      <td>-0.076004</td>
      <td>-0.515755</td>
      <td>1.461849</td>
      <td>-0.039335</td>
      <td>-0.065047</td>
      <td>0.999801</td>
      <td>0.000199</td>
    </tr>
    <tr>
      <th>31117</th>
      <td>0.243279</td>
      <td>-0.073170</td>
      <td>-0.076004</td>
      <td>1.428676</td>
      <td>0.208837</td>
      <td>-0.039335</td>
      <td>-0.065047</td>
      <td>0.999160</td>
      <td>0.000840</td>
    </tr>
  </tbody>
</table>
</div>



### Now we can repeat this process for N times (let's do 100 times) and check the histograms of auc for 95% significance level.


```python
_xgboost_auc_hist(df_pruned, Y, n_iterations = 100, alpha = 0.95)
```


![png](/notebooks/imbalanced-classification_files/output_39_0.png)


    95.0% Significance Level - ROC AUC Confidence Interval= [0.849 , 0.934]


### Note that, in this example I have chosen the hyper-parameters based on rule of thumb and my experience. However, for finalizing the model, an exhaustive GridSearch or Bayesian Optimization is needed. I have not done it here due to my short time to finish this project. However, for all of my model that went to production so far, I do use hyper-parameter tuning.

# 2nd Approach: Regularized Linear Model


```python
# Ridge
alpha = 0.0

glmnet_model = _glmnet(df_X, Y, alpha = alpha)
_plot_glmnet_coeff_path(glmnet_model, df_X)
_plot_glmnet_cv_score(glmnet_model)
_df_glmnet_coeff_path(glmnet_model, df_X)
```


![png](/notebooks/imbalanced-classification_files/output_42_0.png)



![png](/notebooks/imbalanced-classification_files/output_42_1.png)




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>Coeffs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>attribute7</td>
      <td>0.001592</td>
    </tr>
    <tr>
      <th>1</th>
      <td>attribute4</td>
      <td>0.000901</td>
    </tr>
    <tr>
      <th>2</th>
      <td>attribute2</td>
      <td>0.000707</td>
    </tr>
    <tr>
      <th>3</th>
      <td>attribute5</td>
      <td>0.000030</td>
    </tr>
    <tr>
      <th>4</th>
      <td>attribute1</td>
      <td>0.000027</td>
    </tr>
    <tr>
      <th>5</th>
      <td>attribute9</td>
      <td>0.000022</td>
    </tr>
    <tr>
      <th>6</th>
      <td>attribute6</td>
      <td>-0.000007</td>
    </tr>
    <tr>
      <th>7</th>
      <td>attribute3</td>
      <td>-0.000013</td>
    </tr>
  </tbody>
</table>
</div>




```python
_glmnet_best_score_alpha(df_X, Y)
```


![png](/notebooks/imbalanced-classification_files/output_43_0.png)



```python
# Elastic-Net
alpha = 0.1

glmnet_model = _glmnet(df_X, Y, alpha = alpha)
_plot_glmnet_coeff_path(glmnet_model, df_X)
_plot_glmnet_cv_score(glmnet_model)
_df_glmnet_coeff_path(glmnet_model, df_X)
```


![png](/notebooks/imbalanced-classification_files/output_44_0.png)



![png](/notebooks/imbalanced-classification_files/output_44_1.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>Coeffs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>attribute7</td>
      <td>0.052871</td>
    </tr>
    <tr>
      <th>1</th>
      <td>attribute4</td>
      <td>0.018768</td>
    </tr>
  </tbody>
</table>
</div>



### As seen, glmnet has added more bias to the model (we lost around 4% AUC). However, the model is way more simpler. Generally, both methods along with each other would result in a more generalized model. 

# Notes could be considered:
### Other than, hyper parameter optimization (using grid search or bayesian optimization), I have not implemented any neural network models here that could be used as well (neural network without hyper-parameter optimizaiton is technically useless). Another thing that could be used is the time at event. For example, any useful information such as some specific day-of-the-month that failure happened can be engineered as a new feature (attribute). In addition to this, the device include 1169 unique values where failure happened for 106 of them.


```python
g = df_raw.groupby("device").agg({"failure" : "mean"}).sort_values(by = ["failure"], ascending = False)
g = g.reset_index(level=["device"])
print(F"Number of groups with non-zero failure rate = {g[g['failure'] > 0].shape[0]}")
g.head(10)
```

    Number of groups with non-zero failure rate = 106





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>device</th>
      <th>failure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S1F0RRB1</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S1F10E6M</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S1F11MB0</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S1F0CTDN</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Z1F1AG5N</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>5</th>
      <td>W1F0PNA5</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>6</th>
      <td>W1F13SRV</td>
      <td>0.076923</td>
    </tr>
    <tr>
      <th>7</th>
      <td>W1F03DP4</td>
      <td>0.071429</td>
    </tr>
    <tr>
      <th>8</th>
      <td>W1F1230J</td>
      <td>0.071429</td>
    </tr>
    <tr>
      <th>9</th>
      <td>W1F0T034</td>
      <td>0.058824</td>
    </tr>
  </tbody>
</table>
</div>



### For instance, the rows that include the device "S1F0RRB1" has 20% failure rate. In better words, these sets of devices (106) can be used as binary features. However, I was not sure that this is the goal or not. In fact, I am not sure, if would ruin the model since it brings the information from future with knowing the nature of the device. The other note would be for k-folds cross-validation. This devices can be used to group the data into stratified-group-k-folds for classificaiton. In better words, all the rows for specific groups go into the same folds.
