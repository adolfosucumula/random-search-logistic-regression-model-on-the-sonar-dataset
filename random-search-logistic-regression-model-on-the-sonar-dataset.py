
from pandas import read_csv

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
#split into input and output elements
data = dataframe.values
X, y = data[:,:-1], data[:,-1]

#print(X.shape,y.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

#Define model 
model = LogisticRegression()

#Define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

print("")
print("=================== USING RANDOM SEARCH FOR CLASSIFICATION ===================")
print("")
#define search space
#Using Random Search for Classification
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']

from scipy.stats import loguniform
space['C'] = loguniform(1e-5, 100)

#define search by Random Search for classification

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)

#execute search
result = search.fit(X, y)

print('Best Score: %s', result.best_score_)
print('Best Hyperparameters: %s', result.best_params_)


print("")
print("=================== USING GRID SEARCH FOR CLASSIFICATION ===================")
print("")

space2 = dict()
space2['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space2['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space2['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

#Define search by Grid  Search for classification
from sklearn.model_selection import GridSearchCV

search2 = GridSearchCV(model, space2, scoring='accuracy', n_jobs=1, cv=cv)

#execute search
result = search2.fit(X, y)

#Summarize result
print('Best Score: %s', result.best_score_)
print('Best Parameter: %s', result.best_params_)


