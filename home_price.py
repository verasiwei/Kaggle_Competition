#install library
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jupyter_core
import notebook

#=====================================================
#read data in
train = pd.read_csv("D:/work/kaggle/train.csv")
test = pd.read_csv("D:/work/kaggle/test.csv")
total = pd.concat([train,test],keys=["train","test"])
##have a look of the data
###train: 1460 samples, 81 features, one more saleprice column
train.head()
train.info()
train.dtypes.value_counts()
###test: 1459 samples, 80 features
test.head()
test.info()
test.dtypes.value_counts()

total.info()

#=====================================================
#check for missingness of the data
missing_dat = pd.concat([train.isnull().sum(),test.isnull().sum()],axis=1,keys=["train","test"],sort=False)
missing_dat = missing_dat[(missing_dat["train"]!=0) | (missing_dat["test"]!=0)]

#how to deal with missingness in this data?
#There are two types of missingness, 
#1st is for some specific variables, missingness means this house does not have this equipment
#2nd is for some variables, they are the "true" missingness that the house cannot exist if there is no such equipment

#1st missingness, replace categorical missingness with NA
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

#===============================================
#check the distribution of the outcome and have a look of the plot
distplot = sns.distplot(train["SalePrice"])
#do the logtransformation of the saleprice in training data set
logSalePrice = np.log(train["SalePrice"])
train["LogSalePrice"]=logSalePrice
#%%
distplot_log = sns.distplot(train["LogSalePrice"])
#do the standardization for the numeric variables in the total data set
standardize_features=total.loc[:,["LotFrontage","LotArea","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","GrLivArea","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch"]]
standardize_features_values=(standardize_features-standardize_features.mean())/standardize_features.std()
pairplot=sns.pairplot(standardize_features_values)
#remove the variable of pool area, 35snPorch, ScreenPorch
total=total.drop(columns=["PoolArea","3SsnPorch","ScreenPorch"])





#%%
