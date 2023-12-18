# Import appropriate Python Libraries

import streamlit as st
import numpy
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas import read_csv


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error


# Read the csv file into a Python dataframe
# Load dataset
filename = 'Marriage_Divorce_DB.csv'
df = read_csv(filename)

# Set the title and brief description of the app
st.title("Marriage and Divorce")
st.write("This program analyzes the dataset on marriage and divorce and aims to"
         "identify patterns and relationships between various factors and the likelihood of divorce.")
st.write("")
st.write("Aims to answer for the following questions below.")
st.write("Q. What are the top five factors that indicate the highest probability of divorce?")
st.write("Q. What are the top five factors that indicate the lowest probability of divorce?")
st.write("Q. Among the individual factors, which factor has the greatest impact on the likelihood of divorce?")
st.write("Q. Among the relationship factors, which factor has the greatest impact on the likelihood of divorce?")
st.write("Q. Is there any correlation between educational level and the probability of divorce?")
st.write("")
st.write("")
st.write("")
st.write("")

# Read first 5 and last 5 rows
st.title("[ Exploratory data analysis ]")
st.write("")
st.subheader('Head and tail of the dataset')
st.write("First five rows")
st.write(df.head())
st.write("Last five rows")
st.write(df.tail())
st.write("")

# Read the data shape (a number of rows and columns)
st.subheader('Data shape')
st.write("A number of rows and columns. This data contains 31 columns (100x31).")
st.write(df.shape)
st.write("")

# Read the types of data attributes
st.subheader('Type of data attributes')
st.write(df.dtypes)
st.write("")

# Read name of the attributes
st.subheader('Data attributes')
st.write('The first 30 columns are features (Inputs).' 
        ' The 31st column is Divorce Probability (Target).')
st.write(df.columns)
st.write("")

# Read data description
st.subheader('Data description')
st.write(df.describe())
st.write("")


# Read data correlation
st.subheader('Data correlation')
st.write(df.corr(method='pearson')) # generate a correlation matrix for a pandas DataFrame
st.write("")
st.write("Q. What are the top five factors that indicate the highest probability of divorce?"
         "\n\nA. Addiction > Love > Age Gap > Mental Health > Independency")
st.write("")
st.write("Q. What are the top five factors that indicate the lowest probability of divorce?"
         "\n\nA. Education > Good Income > Social Similarities > Relationship with the Spouse Family > Previous Trading")
st.write("")


# Read data correlation for the individual factors / and generate correlation matrix
st.subheader('Data correlation (Individual factors)')
st.write("These factors are related to the characteristics of each individual partner")
st.write(df[['Education','Height Ratio','Good Income','Independency','Self Confidence',
             'Mental Health','Addiction','Divorce Probability']].corr(method='pearson'))
st.write("")
corr_matrix_individual = df[['Education','Height Ratio','Good Income','Independency','Self Confidence',
             'Mental Health','Addiction','Divorce Probability']].corr()
fig_individual, ax = plt.subplots()
sns.heatmap(corr_matrix_individual, cmap="coolwarm", annot=False, ax=ax)
st.pyplot(fig_individual)
st.write("")
st.write("Q. Among the individual factors, which factor has the greatest impact on the likelihood of divorce?"
         "\n\nA. Addiction > Mental Health > Independency > Self Confidence > Height Ratio > Good Income > Education"
         "\n\nWithin this category, 'Addiction' had the highest impact, while 'Education level' had the lowest impact.")
st.write("")


# Read data correlation for the relationship factors / and generate correlation matrix
st.subheader('Data correlation (Relationship factors)')
st.write("These factors are related to the dynamics and quality of the relationship between partners")
st.write(df[['Economic Similarity', 'Cultural Similarities', 'Common Interests', 'Religion Compatibility',
             'Relationship with the Spouse Family','Divorce Probability']].corr(method='pearson'))
st.write("")
corr_matrix_relationship = df[['Economic Similarity', 'Cultural Similarities', 'Common Interests', 'Religion Compatibility',
             'Relationship with the Spouse Family','Divorce Probability']].corr()
fig_relationship, ax = plt.subplots()
sns.heatmap(corr_matrix_relationship, cmap="coolwarm", annot=False, ax=ax)
st.pyplot(fig_relationship)
st.write("")
st.write("Q. Among the relationship factors, which factor has the greatest impact on the likelihood of divorce?"
         "\n\nA. Economic Similarity > Cultural Similarities > Religion Compatibility > Common Interests> Relationship with the Spouse Family"
         "\n\nWithin this category, 'Economic Similarity' had the highest impact,"
         "while 'Relationship with the Spouse Family' had the lowest impact.")
st.write("")


# Illustrate scatter plot
st.subheader('Scatter plot')
st.write("This scatter plot describes the relationship between education level and probability of divorce")
x = df[['Education']]
y = df[['Divorce Probability']]
fig_scatter, ax = plt.subplots()
ax.scatter(x, y)
ax.set_xlabel('Education')
ax.set_ylabel('Divorce Probability')
st.pyplot(fig_scatter)
st.write("")
st.write("Q. Is there any correlation between educational level and the probability of divorce?"
         "\n\nA. Although we can see a very slight negative correlation between the two variables,"
         " it is hard to say whether there is any significant correlation.")
st.write("")


# Illustrate histogram
st.subheader('Histogram')
fig_histogram, ax = plt.subplots(figsize=(25,25))
df.plot(kind='hist', subplots=True, layout=(7,5), sharex=False, sharey=False, ax=ax)
st.pyplot(fig_histogram)
st.write("")


# Illustrate density plot
st.subheader('Density plot')
fig_density, ax = plt.subplots(figsize=(25, 25))
df.plot(kind='density', subplots=True, layout=(8,4), sharex=False,sharey=False,ax=ax)
st.pyplot(fig_density)
st.write("")


# Illustrate box plot for detecting outliers
st.subheader('Box plot')
fig_box, ax = plt.subplots(figsize=(15, 25))
df.plot(kind='box', subplots=True, layout=(8,4), sharex=False, sharey=False, color='deeppink', ax=ax)
st.pyplot(fig_box)
st.write("")



st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.title("[ Prediction Modelling ]")
st.write("")

# Data Preparation for Model development
# Split the data into training and test dataset
st.subheader('Split the data')
array = df.values
X = array[:,0:29] # 1 to 30 columns as an independent variable
Y = array[:,30] # the last column (what needs to be predicted)
test_size = 0.20 # 20% data for testing, that is, 80% is for training to predict the model
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
st.write(X_train.shape,X_test.shape)
st.write("")


# Baseline evaluation metric during model development
# Setting up test harness
# Negative mean squared error
num_folds = 5 # creating batch
seed = 7
scoring = 'neg_mean_squared_error'

# Choosing scikit-learn prediction algorithms suite for regression modelling
# Linear Algorithms (3): Linear Regression, LASSO, ElasticNet
# Nonlinear Algorithms (3): Decision Trees (CART), Support Vector Regression(SVR),k-Nearest Neighbour (kNN)
# Algorithms Pipeline setup
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('GB', GradientBoostingRegressor()))
models.append(('SVR', SVR()))


# Build the model with training subset with each algorithm
# And evaluate each model using baseline performance metric
st.subheader('Model evaluation')
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    st.write(msg)
st.write("")


# Comparison of Algorithms Performance for Training Set
fig_algorithm = pyplot.figure()
fig_algorithm.suptitle('Algorithm Comparison')
ax = fig_algorithm.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
st.pyplot(fig_algorithm)
st.write("")


# Explore potential improvements with pre-processing
# Standardize the dataset
st.subheader('Model evaluation (scaled)')
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeRegressor())])))
pipelines.append(('ScaledGB', Pipeline([('Scaler', StandardScaler()), ('GB',GradientBoostingRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
  kfold = KFold(n_splits=num_folds,shuffle=True, random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  st.write(msg)
st.write("")


# Compare scaled algorithms
fig_scaled_algorithm = pyplot.figure()
fig_scaled_algorithm.suptitle('Scaled Algorithm Comparison')
ax = fig_scaled_algorithm.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
st.pyplot(fig_scaled_algorithm)
st.write("")


# Explore further performance improvement with Algorithm tuning
# kNN algorithm
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
st.write("")


# Display best performance for different tuned parameter values
st.write("Best performance for different tuned parameter values")
st.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  st.write("%f (%f) with: %r" % (mean, stdev, param))
st.write("")


# Explore further improvements with ensemble methods
# Boosting Methods: AdaBoost (AB) and Gradient Boosting (GBM).
# Bagging Methods: Random Forests (RF) and Extra Trees (ET).
# ensembles
st.subheader("Improvements with ensemble methods")
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB',AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF',RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET',ExtraTreesRegressor())])))
results = []
names = []
for name, model in ensembles:
  kfold = KFold(n_splits=num_folds,shuffle=True,  random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  st.write(msg)
st.write("")


# Compare Algorithms Visually
fig_algorithm_visually = pyplot.figure()
fig_algorithm_visually.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig_algorithm_visually .add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
st.pyplot(fig_algorithm_visually)
st.write("")


# Performance improved of Ensemble methods using parameter tuning
# Tune scaled GBM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds,shuffle=True, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)


# Best performing ensemble model after tuning of parameters
st.write("Best performing ensemble model after tuning of parameters")
st.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  st.write("%f (%f) with: %r" % (mean, stdev, param))
st.write("")


# finalise Model
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed,n_estimators=50)
model.fit(rescaledX, Y_train)


# Model Evaluation with independent test set
# pre-process test set before evaluation
st.write("Model Evaluation with independent test set")
rescaledTestX = scaler.transform(X_test)
predictions = model.predict(rescaledTestX)
st.write(mean_squared_error(Y_test, predictions))
st.write(model.score(X_test,Y_test))
st.write("")


# Model Evaluation prediction report
st.subheader("Model evaluation prediction report")
for x in range(len(predictions)):
    #st.write("\nPredicted: ", predictions[x], "Actual: ", Y_test[x], "Data: ", X_test[x],)
    st.write("\nPredicted: ", predictions[x], '\t\t' "Actual: ", Y_test[x], )
st.write("")


# Best Model export for deployment
import pickle
# save the model to disk
model_filename = 'best_regr_model.h5'
pickle.dump(model, open(model_filename, 'wb'))


# Check by Reloading saved model from disk using load function of pickle
st.write("R sqaured value of test data")
with open('best_regr_model.h5','rb') as file:
    loaded_model = pickle.load(file)
# Validate the R sqaured value of test data, it should be same of the original model
st.write(str(loaded_model.score(X_test,Y_test)))