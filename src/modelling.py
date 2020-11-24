from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from utils import save_fig, true_false_plot
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform 


data_dir = Path("data/")
img_dir = Path("../img")
columns_to_use = [
    "Lot Area",
    "Overall Qual",
    "Total Bsmt SF",
    "Garage Area",
    "Bedroom AbvGr",
]


all_data = pd.read_csv(data_dir / "housing-data.csv", index_col="Order")
target_column = "SalePrice"

X_train, X_test, y_train, y_test = train_test_split(
    all_data.drop(columns=target_column), all_data[target_column]
)

#X_train = X_train[columns_to_use]
#X_test = X_test[columns_to_use]

imputer = SimpleImputer(
    missing_values=np.nan, strategy="constant", fill_value=0
)

linear_model = LinearRegression()

#chosen_model = Pipeline([("imputer", imputer), ("model", linear_model)])

#Just a quick checkup to look at the data:
#print(all_data.describe())


#Creating imputers
numberimputer = SimpleImputer(missing_values = np.nan, strategy= "mean")
classimputer = SimpleImputer(missing_values = np.nan, strategy="constant", fill_value="Nan")

#Creating pipelines
encoder = OneHotEncoder(handle_unknown= 'ignore')
numberpipe = Pipeline(steps=[('imputer',numberimputer),
                             ('scalar', StandardScaler())])

classpipe = Pipeline(steps=[('imputer', classimputer),
                             ('onehot', encoder)])


numbers = list(all_data.drop(columns = target_column).select_dtypes(include=["float64", "int64"]))
categories = list(all_data.select_dtypes(exclude=["float64", "int64"]))

Transformer = ColumnTransformer(transformers=[('num', numberpipe, numbers),
                                              ('cat', classpipe, categories)])

ridge_model = Ridge(alpha=1, solver="auto", random_state=42)
lasso = Lasso(alpha=0.1)


lasso_model = Pipeline([("preprocessing", Transformer), ("model", lasso)])
lasso_model.fit(X_train, y_train)




#Hyperparameter tuning for Lasso regression
LassoGrid = {'model__alpha': [0.0005,0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.75, 1, 1.5, 2, 5, 10, 15, 20, 25, 30, 35, 40] ,
             'model__normalize': [True, False], 
             'model__selection': ['cyclic','random']}


#Parameter optimization is commented out because it takes a long time to run
'''
gridSearch = GridSearchCV(estimator=lasso_model,
                                param_grid=LassoGrid,
                                n_jobs=-1,
                                verbose = 1,
                                return_train_score = True,
                                )
gridSearch.fit(X_train, y_train)

LassoBest = gridSearch.best_params_
'''

#The best hyperparameters for the lasso regression are:
'''
{'model__alpha': 15, 'model__normalize': True, 'model__selection': 'cyclic'}
'''


#implementing the optimized hyperparameters in the lasso regression:
BestLassoModel = Lasso(max_iter = 50000, alpha = 15, normalize = True, selection = 'cyclic')





ridge_model = Pipeline([("preprocessing", Transformer), ("model", ridge_model)])
ridge_model.fit(X_train, y_train)


#Hyperparameter tuning for Ridge regression, commented out to save running time
'''
RidgeGrid = {'model__alpha': [0.0005,0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.75, 1, 1.5, 2, 5, 10, 15, 20, 25, 30, 35, 40],
             'model__normalize': [True, False], 
             'model__solver': ['auto','sparse_cg']}
gridSearchRidge = GridSearchCV(estimator=ridge_model,
                                param_grid=RidgeGrid,
                                n_jobs=-1,
                                verbose = 1,
                                return_train_score = True,
                                )
gridSearchRidge.fit(X_train, y_train)
'''
#The best parameters for Ridge regression (gridSearchRidge.best_params_) turned out to be:
'''
    {'model__alpha': 0.5, 'model__normalize': True, 'model__solver': 'auto'}
    
'''

#Implementing the optimized hyperparameters into ridge regression
BestRidgeModel = Ridge(alpha=0.5, solver="auto", normalize=True, random_state=42)
ridge_model = Pipeline([("preprocessing", Transformer), ("model", BestRidgeModel)])
ridge_model.fit(X_train, y_train)




#A function to plot learning curves
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=10
    )
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    


#Finally, we create and optimize an ElasticNet model. commented out to preserve time
'''
elastic = ElasticNet()
elastic_model = Pipeline([("preprocessing", Transformer), ("model", elastic)])
ElasticGrid = {'model__alpha': [15, 10, 7.5, 5, 2.5, 1, 0.5, 0.1, 0.05, 0.01, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075],
               'model__l1_ratio': [0.001, 0.01, 0.05, .1, .3, .5, .7, .9, .95, .99],
               'model__normalize': [True, False]}

gridSearchElastic = GridSearchCV(estimator=elastic_model,
                                param_grid=ElasticGrid,
                                n_jobs=-1,
                                verbose = 1,
                                return_train_score = True,
                                )
gridSearchElastic.fit(X_train, y_train)
'''

#The best hyperparameters for the ElasticNet model turn out to be:
'''
{'model__alpha': 0.0025, 'model__l1_ratio': 0.95, 'model__normalize': True}
'''

#implementing these in the model:
BestElastic = ElasticNet(alpha = 0.0025, l1_ratio = 0.95, normalize = True)
BestElasticModel = Pipeline([("preprocessing", Transformer), ("model", BestElastic)])


    
#Plotting the learning curves:
'''
plot_learning_curves(chosen_model, X_train, y_train)
plot_learning_curves(ridge_model, X_train, y_train)
plot_learning_curves(BestElasticModel, X_train, y_train)
'''

#Drawing true vs predicted graphs:
#ypred = chosen_model.predict(X_test)
#true_false_plot(y_test, ypred, "FirstLasso")


#The model with the best score is still the optimized Lasso model. So that will be our chosen_model:
chosen_model = Pipeline([("preprocessing", Transformer), ("model", BestLassoModel)])
chosen_model.fit(X_train, y_train)










