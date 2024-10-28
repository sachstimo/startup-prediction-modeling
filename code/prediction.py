# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#Import libraries
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import xgboost as xgb
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

path = 'data cannot be provided due to sensitivity'
df_raw = pd.read_excel(path, sheet_name='Dataset', index_col=0)

y = df_raw['tvpi_EUR']

mean_tvpi = y.mean()
std_tvpi = y.std()
z_cutoff = 200
df_raw['z_tvpi_EUR'] = (y - mean_tvpi) / std_tvpi
df = df_raw.copy()
df = df[(df['z_tvpi_EUR'] < z_cutoff) & (df['z_tvpi_EUR'] > (z_cutoff * -1))]

print(df.info())

#Creating a list containing the feature variables
predictors = [
    'is_b2b', 'is_europe', 'is_latam', 'fit_2019', 'fit_2020', 'is_fem_entrepren',
    'instr_is_CLN', 'instr_is_EQ', 'is_CLNconvtoEQ', 'is_reinvested', 'maturity',
    'inv_valuation_EUR', 'stake_ini_pct', 'total_invamt_EUR',

    #Sector dummy variables
    'is_ehealth', 'is_fintech', 'is_custser', 'is_telco', 'is_bigdata', 'is_gaming',
    'is_mktmedia', 'is_hr', 'is_edtech','is_enterprappl','is_iot','is_ecommerce',
    'is_cybersec','is_trvlltourism', 'is_logistics ', 'is_cloud','is_arvr','is_ai'
    ]

X = df[predictors] #Subsetting the whole dataframe for the selected variables
print(len(predictors)) #checking the number of predictive features (= 32)

y2 = df['is_outperf'] #final outcome variable to be predicted

# Calculating the VIF
x_temp = sm.add_constant(X)
vif = pd.DataFrame()
vif['features'] = x_temp.columns
vif['VIF_factor'] = [variance_inflation_factor(x_temp.values, i) for i in range(x_temp.values.shape[1])]
#print(vif.round(1))
#vif.to_excel('VIF.xlsx', 'VIF')
#corr_matrix = df[predictors].corr().round(3)
#corr_matrix.to_excel('MultiCorr.xlsx', 'Corr_Matrix')
#print(corr_matrix)


#Compare both models without splitting & scoring the data

#LogReg with statsmodel
logit = sm.Logit(y2, sm.add_constant(X)).fit() #adding the predictor variables and fitting the model
features = predictors.copy() #copying column names to later reference the variables
features.insert(0, 'const.') #name tag for logit intercept constant

print(logit.summary(xname=features)) #printing out the model summary with coefficients
resultFile = open('Logit_Results.csv', 'w') #open new file to store results
resultFile.write(logit.summary().as_csv()) #write results to CSV file
resultFile.close() #save and close output file

#LogReg with sklearn
scaler = preprocessing.StandardScaler().fit(X) #fit the scaler model for logistic regression
X_scaled = scaler.transform(X) #transform the input variable to be readable by the model
logit = LogisticRegression(penalty='none', fit_intercept=True) #set model parameters
logit.fit(X_scaled, y2) #fitting the model
coeff = pd.DataFrame(zip(predictors, logit.coef_[0].round(4)), columns=['var', 'coeff']) #printing results

#Running the Models in the loop to test for accuracy
i = 1 #count variable for number of runs
seed = 1 #random seed to allow for replicability
runs = 20 #number of validation runs
testsize = 0.2 #percentage split of the training data

lr_scores = [] #creating empty dataframe to store the score values
lr_tprs = [] #creating empty dataframe to store the tprs
lr_precisions = [] #creating empty dataframe to store precision scores
lr_recalls = [] #creating empty dataframe to store recall stores
mean_lr_fpr = np.linspace(0, 1, 100) #setting a mean fpr going from 0 to 1 with 100 increments
lr_aucs = [] #creating empty dataframe to store the aucs

#Running the Logit Model:
for i in range(1, runs):

    X_train, X_test, y_train, y_test = train_test_split(#splitting the training/test data
        X_scaled, y2,
        test_size=testsize,
        shuffle = True, random_state = seed)

    logit = LogisticRegression(penalty ='none', fit_intercept= True) #setting model parameters
    logit.fit(X_train, y_train) #training the model
    logit_y_pred = logit.predict(X_test) #predicting the startup outcome
    lr_score = metrics.accuracy_score(y_test, logit_y_pred) #calculate accuracy score
    lr_rec = metrics.recall_score(y_test, logit_y_pred) #calculate recall score
    lr_prec = metrics.precision_score(y_test, logit_y_pred) #calculate precision score
    lr_scores.append(lr_score) #accuracy score is appended to dataframe
    lr_recalls.append(lr_rec) #append recall score to dataframe
    lr_precisions.append(lr_prec) #append precision score to dataframe
    seed = seed + 1
    i = i + 1

    logit_y_probs = logit.predict_proba(X_test)  #predicting startup outcome probability
    lr_precision, lr_recall, lr_thresholds = metrics.precision_recall_curve(y_test, logit_y_probs[:, -1])
    lr_auc = metrics.auc(lr_recall, lr_precision)

    lr_fpr, lr_tpr, _ = metrics.roc_curve(y_test, logit_y_probs[:, -1], drop_intermediate = False) #calculating the tpr
    logit_roc_auc = metrics.auc(lr_fpr, lr_tpr) #calculating the auc for run(i)
    interp_lr_tpr = np.interp(mean_lr_fpr, lr_fpr, lr_tpr) #to smoothen the curve, the value are interpolated
    interp_lr_tpr[0] = 0.0 #setting first datapoint of the tpr to 0
    lr_tprs.append(interp_lr_tpr) #tpr of run(i) is appended to dataframe
    lr_aucs.append(logit_roc_auc) #auc of run(i) is appended to dataframe

lr_cm = metrics.confusion_matrix(y_test, logit_y_pred) #creating a confusion matrix with the results of the run

#Running the SVM Model

svm_scores = []
svm_tprs = []
svm_precisions = [] #creating empty dataframe to store precision scores
svm_recalls = [] #creating empty dataframe to store recall stores
mean_svm_fpr = np.linspace(0, 1, 100)
svm_aucs = []


for i in range(1, runs):

    X_train, X_test, y_train, y_test = train_test_split(#splitting the training/test data
        X_scaled, y2,
        test_size=testsize,
        shuffle = True, random_state = seed)

    svc_mod = svm.SVC(kernel='rbf', #using the radial basis function
                      gamma = 'scale', #Setting gamma to autoscale itself to fit the model
                      C = 1, #Setting the regularization parameter to 1
                      probability = True #allowing for the underlying decision probability to be calculated
                      )

    svc_mod.fit(X_train, y_train)
    svm_y_pred = svc_mod.predict(X_test)
    svm_score = metrics.accuracy_score(y_test, svm_y_pred)
    svm_rec = metrics.recall_score(y_test, svm_y_pred) #calculate recall score
    svm_prec = metrics.precision_score(y_test, svm_y_pred) #calculate precision score
    svm_scores.append(svm_score)
    svm_recalls.append(svm_rec) #append recall score to dataframe
    svm_precisions.append(svm_prec) #append precision score to dataframe
    seed = seed + 1
    i = i + 1

    #Precision Metrics
    svm_y_probs = svc_mod.predict_proba(X_test)
    svm_precision, svm_recall, svm_thresholds = metrics.precision_recall_curve(y_test, svm_y_probs[:, -1])
    svm_auc = metrics.auc(svm_recall, svm_precision)

    svm_fpr, svm_tpr, _ = metrics.roc_curve(y_test, svm_y_probs[:, -1], drop_intermediate = False)
    svm_roc_auc = metrics.auc(svm_fpr, svm_tpr)
    interp_svm_tpr = np.interp(mean_svm_fpr, svm_fpr, svm_tpr)
    interp_svm_tpr[0] = 0.0
    svm_tprs.append(interp_svm_tpr)
    svm_aucs.append(svm_roc_auc)

svm_cm = metrics.confusion_matrix(y_test, svm_y_pred)

#Running the XGBoost Model

xgb_scores = []
xgb_tprs = []
xgb_precisions = [] #creating empty dataframe to store precision scores
xgb_recalls = [] #creating empty dataframe to store recall stores
mean_xgb_fpr = np.linspace(0, 1, 100)
xgb_aucs = []

data_dmatrix = xgb.DMatrix(data = X, label = y2)

for i in range(1, runs):

    X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=testsize, shuffle = True, random_state = seed)

    xgb_mod = xgb.XGBClassifier(
                                use_label_encoder = False,
                                objective = 'reg:logistic',
                                colsample_bytree = 0.3,
                                learning_rate = 0.1,
                                max_depth = 10,
                                alpha = 5,
                                n_estimators = 100
    )

    xgb_mod.fit(X_train, y_train)
    xgb_y_pred = xgb_mod.predict(X_test)
    xgb_score = metrics.accuracy_score(y_test, xgb_y_pred) #calculate accuracy score
    xgb_rec = metrics.recall_score(y_test, xgb_y_pred) #calculate recall score
    xgb_prec = metrics.precision_score(y_test, xgb_y_pred) #calculate precision score
    xgb_scores.append(xgb_score)
    xgb_recalls.append(xgb_rec) #append recall score to dataframe
    xgb_precisions.append(xgb_prec) #append precision score to dataframe
    seed = seed + 1
    i = i + 1

    # Calculate precision metrics

    xgb_y_probs = xgb_mod.predict_proba(X_test)
    xgb_precision, xgb_recall, xgb_thresholds = metrics.precision_recall_curve(y_test, xgb_y_probs[:, -1])
    xgb_auc = metrics.auc(xgb_recall, xgb_precision)

    xgb_fpr, xgb_tpr, _ = metrics.roc_curve(y_test, xgb_y_probs[:, -1], drop_intermediate=False)
    xgb_roc_auc = metrics.auc(xgb_fpr, xgb_tpr)
    interp_xgb_tpr = np.interp(mean_xgb_fpr, xgb_fpr, xgb_tpr)
    interp_xgb_tpr[0] = 0.0
    xgb_tprs.append(interp_xgb_tpr)
    xgb_aucs.append(xgb_roc_auc)

xgb_cm = metrics.confusion_matrix(y_test, xgb_y_pred)

xgb.plot_importance(xgb_mod)


#Plotting confusion matrices

plt.rc('font', size=16)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure titl


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
cmap_col = 'YlGnBu'
fig.suptitle('Confusion matrices for "' + str(y2.name) + '"')
sns.heatmap(lr_cm, annot = True, ax = ax1, square = True, fmt = 'd', cbar = False, cmap = cmap_col)
ax1.set_xlabel('predicted')
ax1.set_ylabel('actual')
ax1.set_title('Logit')
sns.heatmap(svm_cm, annot = True, ax = ax2, square = True, fmt = 'd', cbar = False, cmap = cmap_col)
ax2.set_xlabel('predicted')
ax2.set_ylabel('actual')
ax2.set_title('SVM')
sns.heatmap(xgb_cm, annot = True, ax = ax3, square = True, fmt = 'd', cbar = False, cmap = cmap_col)
ax3.set_xlabel('predicted')
ax3.set_ylabel('actual')
ax3.set_title('XGBoost')


#Plotting the Accuracy Score Distribution

mean_logit_score = sum(lr_scores) / len(lr_scores)
mean_svm_score = sum(svm_scores) / len(svm_scores)
mean_xgb_score = sum(xgb_scores) / len(xgb_scores)

mean_lr_precision = sum(lr_precisions) / len(lr_precisions)
mean_svm_precision = sum(svm_precisions) / len(svm_precisions)
mean_xgb_precision = sum(xgb_precisions) / len(xgb_precisions)

mean_lr_recall = sum(lr_recalls) / len(lr_recalls)
mean_svm_recall = sum(svm_recalls) / len(svm_recalls)
mean_xgb_recall = sum(xgb_recalls) / len(xgb_recalls)

baseline = 1-(y2.sum()/len(y2))

print('Mean Logit Score:' + str(mean_logit_score))
print('Mean Logit Recall:' + str(mean_lr_recall))
print('Mean Logit Precision:' + str(mean_lr_precision))
print('Mean SVM Score:' + str(mean_svm_score))
print('Mean SVM Recall:' + str(mean_svm_recall))
print('Mean SVM Precision:' + str(mean_svm_precision))
print('Mean XGBoost Score:' + str(mean_xgb_score))
print('Mean XGBoost Recall:' + str(mean_xgb_recall))
print('Mean XGBoost Precision:' + str(mean_xgb_precision))

#Actual Plotting

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True, sharey = True)

sns.histplot(lr_scores, ax = ax1, kde = True, color ='lightblue', label = 'Logit', element = 'step')
sns.histplot(svm_scores, ax = ax2, kde = True, color = 'lightgrey', label = 'SVM', element = 'step')
sns.histplot(xgb_scores, ax = ax3, kde = True, color = 'orange', label = 'XGB', element = 'step')

ax1.annotate('N =' + str(len(y_test)), xy = (0.725, runs / 8))
ax1.annotate('y = ' + str(y2.name), xy = (0.725, (runs / 8 - 30)))
ax1.annotate('# runs = ' + str(runs), xy = (0.725, (runs / 8 - 60)))

ax1.set_title('Logit Accuracy Scores')
ax1.axvline(baseline, linestyle = '--', color = 'black', lw = 1)
ax1.axvline(mean_logit_score, color ='darkblue')
ax1.annotate('Mean Accuracy Score = ' + str(round(mean_logit_score * 100, 1)) + '%', xy = (mean_logit_score + 0.01, runs * 0.15), color ='darkblue', fontsize = 'medium')

ax2.set_title('SVM Classifier Accuracy Scores')
ax2.axvline(baseline, linestyle = '--', color = 'black', lw = 1)
ax2.axvline(mean_svm_score, color ='dimgrey')
ax2.annotate('Mean Accuracy Score = ' + str(round(mean_svm_score * 100, 1)) + '%', xy = (mean_svm_score + 0.01, runs * 0.15), color ='dimgrey', fontsize = 'medium')

ax3.set_title('XGBoost Accuracy Scores')
ax3.axvline(baseline, linestyle = '--', color = 'black', lw = 1)
ax3.annotate('B= ' + str(round(baseline * 100, 1)) + '%', xy = (baseline - 0.005, runs * 0.05), rotation = 90, fontsize = 'medium')
ax3.axvline(mean_xgb_score, color ='orangered')
ax3.annotate('Mean Accuracy Score = ' + str(round(mean_xgb_score * 100, 1)) + '%', xy = (mean_xgb_score + 0.01, runs * 0.15), color ='orangered', fontsize = 'medium')

ax3.set_xlabel('Accuracy scores')
ax3.set_xticks([0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925])
plt.xlim([0.7, 0.95])


#Plotting ROC Curves for all models

fig, ax = plt.subplots(3, 2, sharex = True, sharey = True)
fig.suptitle('ROC-AUC and Precision-Recall Curves')

ax[0, 0].set_title('Logit Regression | ROC-AUC', fontsize = 'medium', loc = 'left')
ax[1, 0].set_title('SVM Classifier Model | ROC-AUC',fontsize = 'medium', loc = 'left')
ax[2, 0].set_title('XGBoost Model | ROC-AUC', fontsize = 'medium', loc = 'left')
ax[0, 1].set_title('Logit Regression | Precison-Recall', fontsize = 'medium', loc = 'left')
ax[1, 1].set_title('SVM Classifier Model | Precision-Recall',fontsize = 'medium', loc = 'left')
ax[2, 1].set_title('XGBoost Model | Precision-Recall', fontsize = 'medium', loc = 'left')

#Plotting logit ROC curves

mean_lr_tpr = np.mean(lr_tprs, axis = 0)
mean_lr_tpr[-1] = 1.0
std_lr_tpr = np.std(lr_tprs, axis = 0)
lr_tprs_upper = np.minimum(mean_lr_tpr + std_lr_tpr, 1)
lr_tprs_lower = np.maximum(mean_lr_tpr - std_lr_tpr, 0)
mean_lr_auc = metrics.auc(mean_lr_fpr, mean_lr_tpr)
std_lr_auc = np.std(lr_aucs)

ax[0, 0].plot(mean_lr_fpr, mean_lr_tpr, color = 'darkorange', label = r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_lr_auc, std_lr_auc))
ax[0, 0].fill_between(mean_lr_fpr, lr_tprs_lower, lr_tprs_upper, color = 'grey', alpha = 0.2, label = r"$\pm$ 1 std. dev.")
ax[0, 0].plot([0, 1], [0, 1], color="b", linestyle="--", label = 'no skill', alpha = 0.8)
ax[0, 0].set_ylabel('True Positive Rate')
ax[0, 0].legend(loc = 'lower right')

#Plotting SVM ROC curves

mean_svm_tpr = np.mean(svm_tprs, axis = 0)
mean_svm_tpr[-1] = 1.0
std_svm_tpr = np.std(svm_tprs, axis = 0)
svm_tprs_upper = np.minimum(mean_svm_tpr + std_svm_tpr, 1)
svm_tprs_lower = np.maximum(mean_svm_tpr - std_svm_tpr, 0)
mean_svm_auc = metrics.auc(mean_svm_fpr, mean_svm_tpr)
std_svm_auc = np.std(svm_aucs)

ax[1, 0].plot(mean_svm_fpr, mean_svm_tpr, color = 'darkorange', label = r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_svm_auc, std_svm_auc))
ax[1, 0].fill_between(mean_svm_fpr, svm_tprs_lower, svm_tprs_upper, color = 'grey', alpha = 0.2, label = r"$\pm$ 1 std. dev.")
ax[1, 0].plot([0, 1], [0, 1], color="b", linestyle="--", label = 'no skill', alpha = 0.8)
ax[1, 0].set_ylabel('True Positive Rate')
ax[1, 0].legend(loc = 'lower right')


#Plotting XGB ROC curves

mean_xgb_tpr = np.mean(xgb_tprs, axis = 0)
mean_xgb_tpr[-1] = 1.0
std_xgb_tpr = np.std(xgb_tprs, axis = 0)
xgb_tprs_upper = np.minimum(mean_xgb_tpr + std_xgb_tpr, 1)
xgb_tprs_lower = np.maximum(mean_xgb_tpr - std_xgb_tpr, 0)
mean_xgb_auc = metrics.auc(mean_xgb_fpr, mean_xgb_tpr)
std_xgb_auc = np.std(xgb_aucs)

ax[2, 0].plot(mean_xgb_fpr, mean_xgb_tpr, color = 'darkorange', label = r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_xgb_auc, std_xgb_auc))
ax[2, 0].fill_between(mean_xgb_fpr, xgb_tprs_lower, xgb_tprs_upper, color = 'grey', alpha = 0.2, label = r"$\pm$ 1 std. dev.")
ax[2, 0].plot([0, 1], [0, 1], color="b", linestyle="--", label = 'no skill', alpha = 0.8)
ax[2, 0].set_ylabel('True Positive Rate')
ax[2, 0].legend(loc = 'lower right')
ax[2, 0].set_xlabel('False Positive Rate')


#Plot Precision / Recall Graphs for all three models

ax[0, 1].plot(lr_thresholds, lr_precision[: -1], "b--", label="Precision")
ax[0, 1].plot(lr_thresholds, lr_recall[: -1], "r--", label="Recall")
ax[0, 1].set_ylabel("Precision, Recall")

ax[1, 1].plot(svm_thresholds, svm_precision[: -1], "b--", label="Precision")
ax[1, 1].plot(svm_thresholds, svm_recall[: -1], "r--", label="Recall")
ax[1, 1].set_ylabel("Precision, Recall")

ax[2, 1].plot(xgb_thresholds, xgb_precision[: -1], "b--", label="Precision")
ax[2, 1].plot(xgb_thresholds, xgb_recall[: -1], "r--", label="Recall")
ax[2, 1].set_ylabel("Precision, Recall")

ax[2, 1].set_xlabel("Threshold")
ax[2, 1].legend(loc="lower right")
ax[2, 1].set_ylim([0,1])

plt.show()