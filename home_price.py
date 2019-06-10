#install library
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jupyter_core
import notebook
from ipykernel import kernelapp as app
import sklearn
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error


#=====================================================
#read data in
train = pd.read_csv("D:/work/kaggle/train.csv")
test = pd.read_csv("D:/work/kaggle/test.csv")

##have a look of the data
###train: 1460 samples, 81 features, one more saleprice column
train.head()
train.info()
train.dtypes.value_counts()
###test: 1459 samples, 80 features
test.head()
test.info()
test.dtypes.value_counts()

#===============================================
#check the distribution of the outcome and have a look of the plot
distplot = sns.distplot(train["SalePrice"])
#do the logtransformation of the saleprice in training data set
train_labels=train.pop("SalePrice")
train_labels = np.log(train_labels)
total = pd.concat([train,test],keys=["train","test"])

#=====================================================
#check for missingness of the data
missing_dat = pd.concat([train.isnull().sum(),test.isnull().sum()],axis=1,keys=["train","test"],sort=False)
missing_dat = missing_dat[(missing_dat["train"]!=0) | (missing_dat["test"]!=0)]

#how to deal with missingness in this data?
#There are two types of missingness, 
#1st is for some specific variables, missingness means this house does not have this equipment
#2nd is for some variables, they are the "true" missingness that the house cannot exist if there is no such equipment

#1st missingness, replace categorical missingness with Null
syscatmiss = ["MasVnrType","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","BsmtQual","GarageType","GarageFinish","GarageQual","GarageCond","Alley","FireplaceQu","MiscFeature","PoolQC","Fence"]
for syscatmissing in syscatmiss:
    total[syscatmissing]=total[syscatmissing].fillna('None')

#1st missingness, replace continuous misssingness with 0
sysconmiss = ["MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath","GarageYrBlt","GarageCars","GarageArea"]
for sysconmissing in sysconmiss:
    total[sysconmissing]=total[sysconmissing].fillna(0)

#2nd missingness, replace categorical missingness with most common value of building type category and neighborhood category
rancatmiss = ["MSZoning","Utilities","Exterior1st","Exterior2nd","Electrical","KitchenQual","Functional","SaleType"]
for rancatmissing in rancatmiss:
    total[rancatmissing]=total.groupby(["MSSubClass","Neighborhood"])[rancatmissing].apply(lambda x: x.fillna(x.mode()[0]))

#2nd missingness, replace continuous missingness with median value of neighborhood
total["LotFrontage"]=total.groupby(["Neighborhood"])["LotFrontage"].apply(lambda x: x.fillna(x.median()))


#%%
distplot_log = sns.distplot(train_labels)
#do the standardization for the numeric variables in the total data set
standardize_features=total.loc[:,["LotFrontage","LotArea","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","GrLivArea","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch"]]
standardize_features_values=(standardize_features-standardize_features.mean())/standardize_features.std()
#pairplot=sns.pairplot(standardize_features_values)
#remove the variable of pool area, 35snPorch, ScreenPorch
total=total.drop(columns=["PoolArea","3SsnPorch","ScreenPorch"])

#================================================
#%%
#transform category data to dummy 
#condition1, condition2 are two same variables
#conditions = set([x for x in total["Condition1"]] + [x for x in total["Condition2"]])
#dummies = pd.DataFrame(data=np.zeros((len(total.index), len(conditions))),
#                       index=total.index, columns=conditions)
#for i, cond in enumerate(zip(total['Condition1'], total['Condition2'])):
#    dummies.ix[i, cond] = 1
#total = pd.concat([total, dummies.add_prefix('Condition_')], axis=1)
#total.drop(['Condition1', 'Condition2'], axis=1, inplace=True)

#exterior1st and Exterior2nd are two same variables
#exteriors = set([x for x in total["Exterior1st"]] + [x for x in total#["Exterior2nd"]])
#dummies = pd.DataFrame(data=np.zeros((len(total.index),len(exteriors))),index=total.index,columns=exteriors)
#for i, cond in enumerate(zip(total["Exterior1st"],total["Exterior2nd"])):
#    dummies.ix[i,cond]=1
#total = pd.concat([total,dummies.add_prefix("Exterior_")],axis=1)
#total.drop(["Exterior1st","Exterior2nd"],axis=1,inplace=True)

for col in total.dtypes[total.dtypes == 'object'].index:
    for_dummy = total.pop(col)
    total = pd.concat([total, pd.get_dummies(for_dummy, prefix=col)], axis=1)

total_standardized = total.copy()
total_standardized.update(standardize_features_values)

#%%
#seprate train set and test set
train_features = total.loc['train'].drop('Id', axis=1)
test_features = total.loc['test'].drop('Id', axis=1)

train_features_st = total_standardized.loc['train'].drop('Id', axis=1)
test_features_st = total_standardized.loc['test'].drop('Id', axis=1)

#within train set, separate into train set and validation set
x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=200)
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_features_st, train_labels, test_size=0.1, random_state=200)

#%%
def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))

def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    # Printing estimator
    print(estimator)
    # Printing train scores
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    # Printing test scores
    print("Test")
    get_score(prediction_test, y_tst)

#gradient boost
GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)
train_test(GBest, x_train, x_test, y_train, y_test)

#%%
GB_model = GBest.fit(train_features, train_labels)
#%%
test_labels = np.exp(GB_model.predict(test_features))
#%%
pd.DataFrame({'Id': test.Id, 'SalePrice': test_labels}).to_csv('D:/work/kaggle/house_price.csv', index =False)


#%%
