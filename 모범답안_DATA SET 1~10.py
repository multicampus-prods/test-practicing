# -*- coding: utf-8 -*-
"""
Created on Sat May 22 14:21:08 2021

@author: cwjeo
"""

#%%

# Dataset 1_1

import pandas as pd

data = pd.read_csv("j:\실기\data\data files\Dataset_1_1_original.csv")
data

#1
data.isnull().sum()

#2
data.corr()

#3

from sklearn.linear_model import LinearRegression

data
data = data.dropna()

y = data.Sales
X = data[['TV', 'Radio', 'Social_Media']]

lm = LinearRegression()
lm.fit(X, y)

lm.coef_


#%%

# Dataset 1_2
import pandas as pd


data = pd.read_csv("j:\실기\data\data files\Dataset_1_2.csv")

#4
data.describe()
data
isFemale = data['Sex'] == 'F' 
isBP_High = data['BP'] == 'HIGH'
isChol_Norm = data['Cholesterol'] == 'NORMAL'

temp = data[isFemale & isBP_High & isChol_Norm]
len(temp) / len(data)



#5
import scipy.stats as stats

data['Age_gr'] = pd.cut(data.Age, bins = [0, 10, 20, 30, 40, 50, 60], labels = ['10', '20', '30', '40', '50', '60'])
data['Na_K_gr'] = pd.cut(data.Na_to_K, bins = [0, 10, 20, 30,100], labels = ['Lv1', 'Lv2', 'Lv3', 'Lv4'])

crosstab01 = pd.crosstab(data['Age_gr'], data['Drug'])
stats.chi2_contingency(crosstab01)

crosstab02 = pd.crosstab(data['Sex'], data['Drug'])
stats.chi2_contingency(crosstab02)

crosstab03 = pd.crosstab(data['BP'], data['Drug'])
stats.chi2_contingency(crosstab03)

crosstab04 = pd.crosstab(data['Cholesterol'], data['Drug'])
stats.chi2_contingency(crosstab04)

crosstab05 = pd.crosstab(data['Na_K_gr'], data['Drug'])
stats.chi2_contingency(crosstab05)


#6

data['Sex_cd'] = 0
data.Sex_cd[data['Sex']=='M'] = 0
data.Sex_cd[data['Sex']=='F'] = 1

data['BP_cd'] = 0
data.BP_cd[data['BP']=='LOW'] = 0
data.BP_cd[data['BP']=='NORMAL'] = 1
data.BP_cd[data['BP']=='HIGH'] = 2

data['Ch_cd'] = 0
data.Ch_cd[data['Cholesterol']=='HIGH'] = 1
data.Ch_cd[data['Cholesterol']=='NORMAL'] = 0


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydot

X = data[['Age', 'Na_to_K', 'Sex_cd', 'BP_cd', 'Ch_cd']]
y = data['Drug']

dTree = DecisionTreeClassifier(random_state = 123)
dTree.fit(X, y)

export_graphviz(dTree, out_file = 'dataset_1_2_6.dot', 
                class_names = sorted(list(set(list(y)))),
                feature_names = X.columns, rounded = True, filled = True)

(graph,) = pydot.graph_from_dot_file('dataset_1_2_6.dot', encoding = 'utf8')

graph.write_png('dataset_1_2_6.png')


#####
sorted(list(set(list(y))))

import graphviz

with open("dataset_1_2_6.dot") as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph)
dot.format = 'png'
dot.render(filename='tree', directory='./', cleanup=True)
#####



#%%

# Dataset 2_1
import pandas as pd


data = pd.read_csv("j:\실기\data\data files\Dataset_2_1.csv")
data

#1

data['forehead_ratio'] = data['forehead_width_cm'] / data['forehead_height_cm']
avg = data['forehead_ratio'].mean()
std = data['forehead_ratio'].std()
data['isOutlier'] = 0
data.isOutlier[data['forehead_ratio']< avg - 3 * std] = 1
data.isOutlier[data['forehead_ratio']> avg + 3 * std] = 1

data.loc[data['forehead_ratio']> avg + 3 * std, 'isOutlier'] = 1

data['isOutlier'].sum()




#2

import scipy.stats as stats

data.info()
data.gender

female = data.loc[data['gender'] == 'Female', 'forehead_ratio']
male = data.loc[data['gender'] == 'Male', 'forehead_ratio']
stats.ttest_ind(male, female, equal_var = False)



#3

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


X_train, X_test, y_train, y_test = train_test_split(data[['long_hair', 'forehead_width_cm', 'forehead_height_cm', 'nose_wide', 'nose_long', 'lips_thin', 'distance_nose_to_lip_long']], data['gender'], test_size = 0.3, random_state = 123)

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

y_pred = model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

metrics.precision_score(y_test, y_pred)

metrics.confusion_matrix(y_test, y_pred)
FF, FM, MF, MM = metrics.confusion_matrix(y_test, y_pred).ravel()

precision = MM / (MM + FM)
precision



#%%

# Dataset 2-2
import pandas as pd
import numpy as np



data = pd.read_csv("j:\실기\data\data files\Dataset_2_2.csv")
data


#1


data1 = data[data['LOCATION'] == 'KOR']
new = data1.pivot_table(index = 'TIME', values = 'Value', aggfunc = np.sum)
new = new.reset_index()
new.corr()


#2

from scipy.stats import ttest_rel

left = data[data['LOCATION'] == 'KOR']
right = data[data['LOCATION'] == 'JPN']

new2 = pd.merge(left, right, on=('TIME', 'SUBJECT'))

new3 = new2[new2.SUBJECT == 'BEEF']
ttest_rel(new3.Value_x, new3.Value_y)

new3 = new2[new2.SUBJECT == 'PIG']
ttest_rel(new3.Value_x, new3.Value_y)

new3 = new2[new2.SUBJECT == 'POULTRY']
ttest_rel(new3.Value_x, new3.Value_y)

new3 = new2[new2.SUBJECT == 'SHEEP']
ttest_rel(new3.Value_x, new3.Value_y)



#3

from sklearn.linear_model import LinearRegression

new3 = new2[new2.SUBJECT == 'BEEF']

y = new3.Value_x

X = np.array(new3.TIME).reshape(-1, 1)

lm = LinearRegression()
lm.fit(X, y)

lm.score(X, y)


new3 = new2[new2.SUBJECT == 'PIG']

y = new3.Value_x

X = np.array(new3.TIME).reshape(-1, 1)

lm = LinearRegression()
lm.fit(X, y)

lm.score(X, y)


new3 = new2[new2.SUBJECT == 'POULTRY']

y = new3.Value_x

X = np.array(new3.TIME).reshape(-1, 1)

lm = LinearRegression()
lm.fit(X, y)

lm.score(X, y)


new3 = new2[new2.SUBJECT == 'SHEEP']

y = new3.Value_x

X = np.array(new3.TIME).reshape(-1, 1)

lm = LinearRegression()
lm.fit(X, y)

lm.score(X, y)
pred = lm.predict(X)

np.mean(np.abs((y - pred) / y)) * 100





# Dataset 3-1
import pandas as pd


data = pd.read_csv("j:\실기\data\data files\Dataset_3_1.csv")

#1

data.info()
data.isnull().sum()



#2

data1 = data.dropna()
pd.crosstab(data1['Gender'], data1['Segmentation'])

from scipy.stats import chi2_contingency

result = chi2_contingency(pd.crosstab(data1['Gender'], data1['Segmentation']))
result
print('Chi2 Statistic: {}, p-value: {}'.format(result[0], result[1]))



#3

from sklearn.model_selection import train_test_split
from sklearn import metrics

data2 = data1[(data1['Segmentation'] == 'A') | (data1['Segmentation'] == 'D')]

X_train, X_test, y_train, y_test = train_test_split(data2[['Age_gr', 'Gender', 'Work_Experience', 'Family_Size', 'Ever_Married', 'Graduated', 'Spending_Score']], data2['Segmentation'], test_size = 0.3, random_state = 123)


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydot


dTree = DecisionTreeClassifier(random_state = 123, max_depth=7)
dTree.fit(X_train, y_train)

export_graphviz(dTree, out_file = 'dataset_3_1_3.dot', 
                class_names = sorted(list(set(list(y_train)))),
                feature_names = X_train.columns, rounded = True, filled = True)

(graph,) = pydot.graph_from_dot_file('dataset_3_1_3.dot', encoding = 'utf8')

graph.write_png('dataset_3_1_3.png')

y_pred = dTree.predict(X_test)


metrics.accuracy_score(y_test, y_pred)

metrics.precision_score(y_test, y_pred)

metrics.confusion_matrix(y_test, y_pred)
FF, FM, MF, MM = metrics.confusion_matrix(y_test, y_pred).ravel()

Accuracy = (FF + MM) / (MM + FM + FF + MF)
Accuracy




# Dataset 3-2
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("j:\실기\data\data files\Dataset_3_2.csv")


#4

avg_price_noview = data.price[data['waterfront'] == 0].mean()
avg_price_view = data.price[data['waterfront'] == 1].mean()
abs(avg_price_view - avg_price_noview)


#5

data1 = data[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'yr_built']]
result = data1.corr().iloc[0, 1:]
result
result.max()
result.min()


#6

X = data1[data1.columns.difference(['ID', 'date', 'price'])] 
y = data1.price


from statsmodels.formula.api import ols

model = ols('price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + waterfront + view + condition + grade + sqft_above + sqft_basement + yr_built + yr_renovated  + sqft_living15 + sqft_lot15', data)

result = model.fit()

result.summary()




# Dataset 4-1
import pandas as pd


data = pd.read_csv("j:\실기\data\data files\Dataset_4_1.csv")


#1

data1 = data[['GRE', 'TOEFL', 'CGPA', 'Chance_of_Admit']]
data1.corr()


#2

GRE_avg = data['GRE'].mean()
data['GRE_gr'] = 'under'
data.loc[data['GRE'] >= GRE_avg, 'GRE_gr'] = 'over'

import scipy.stats as stats


over = data.loc[data['GRE_gr'] == 'over', 'CGPA']
under = data.loc[data['GRE_gr'] == 'under', 'CGPA']
stats.ttest_ind(over, under)



#3
from sklearn.linear_model import LogisticRegression


data.info()
X = data[data.columns.difference(['Serial_No', 'GRE_gr','Chance_of_Admit'])]
X.info()

data['Target'] = 0
data[data['Chance_of_Admit'] > 0.5] = 1

y = data['Target']
y.sum()

model = LogisticRegression(solver = 'liblinear',random_state=123)
result = model.fit(X, y)
print(model.score(X, y))

result.coef_
result.get_params()




# Dataset 4-2

import pandas as pd
import numpy as np

data = pd.read_csv("f:\실기\data\data files\Dataset_4_2.csv")


#4

tab4 = pd.pivot_table(data, index = 'State', aggfunc='count', margins=True)
ratio1 = tab4.iloc[0,0] / tab4.iloc[3,0] 
ratio2 = tab4.iloc[1,0] / tab4.iloc[3,0]
ratio3 = tab4.iloc[2,0] / tab4.iloc[3,0]

ratio1
ratio2
ratio3


#5

tab5 = pd.pivot_table(data, index = 'State', values = 'Profit', aggfunc='mean')

tab5

tab5.Profit.max() - tab5.Profit.min()


#6

from sklearn.linear_model import LinearRegression

data


X_CA = data.loc[data['State'] == 'California', ['RandD_Spend', 'Administration', 'Marketing_Spend']]
X_CA
y_CA = data.loc[data['State'] == 'California', 'Profit']
y_CA

lm = LinearRegression()
lm.fit(X_CA, y_CA)

pred_CA = lm.predict(X_CA)

CA = np.mean(np.abs((y_CA - pred_CA) / y_CA)) * 100



X_NY = data.loc[data['State'] == 'New York', ['RandD_Spend', 'Administration', 'Marketing_Spend']]
X_NY
y_NY = data.loc[data['State'] == 'New York', 'Profit']
y_NY

lm = LinearRegression()
lm.fit(X_NY, y_NY)

pred_NY = lm.predict(X_NY)

NY = np.mean(np.abs((y_NY - pred_NY) / y_NY)) * 100




X_FL = data.loc[data['State'] == 'Florida', ['RandD_Spend', 'Administration', 'Marketing_Spend']]
X_FL
y_FL = data.loc[data['State'] == 'Florida', 'Profit']
y_FL

lm = LinearRegression()
lm.fit(X_FL, y_FL)

pred_FL = lm.predict(X_FL)

FL = np.mean(np.abs((y_FL - pred_FL) / y_FL)) * 100


CA
NY
FL

FL = np.mean(np.abs((y_FL - pred_FL) / y_FL)) * 100





# Dataset 5-1

import pandas as pd
import numpy as np

data = pd.read_csv("j:\실기\data\data files\Dataset_5_1.csv")


#1

data
data.isnull().sum()



#2
import scipy.stats as stats

data1 = data.dropna()
data
data1

data1['Age_gr'] = pd.cut(data.Age, bins = [0, 20, 30, 40, 50, 60, 100], labels = ['10', '20', '30', '40', '50', '60'])
data1

crosstab01 = pd.crosstab(data1['Age_gr'], data1['satisfaction'])
stats.chi2_contingency(crosstab01)

crosstab02 = pd.crosstab(data1['Gender'], data1['satisfaction'])
stats.chi2_contingency(crosstab02)

crosstab03 = pd.crosstab(data1['Customer_Type'], data1['satisfaction'])
stats.chi2_contingency(crosstab03)

crosstab04 = pd.crosstab(data1['Class'], data1['satisfaction'])
stats.chi2_contingency(crosstab04)

data1.loc[data1['Age_gr'] == '10', 'Age'].size
data1.loc[data1['Age_gr'] == '20', 'Age'].size
data1.loc[data1['Age_gr'] == '30', 'Age'].size
data1.loc[data1['Age_gr'] == '40', 'Age'].size
data1.loc[data1['Age_gr'] == '50', 'Age'].size
data1.loc[data1['Age_gr'] == '60', 'Age'].size



#3

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(data1[['Flight_Distance', 'Seat_comfort', 'Food_and_drink','Inflight_wifi_service','Inflight_entertainment', 'Onboard_service', 'Leg_room_service', 'Baggage_handling', 'Cleanliness','Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes']], data1['satisfaction'], test_size = 0.3, random_state = 123)


model = LogisticRegression(solver = 'liblinear',random_state=123)
result = model.fit(X_train, y_train)

y_pred = model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

metrics.precision_score(y_test, y_pred)

metrics.confusion_matrix(y_test, y_pred)
FF, FM, MF, MM = metrics.confusion_matrix(y_test, y_pred).ravel()

f1_dissatisfied = 2 / ((1/(258/(79+258))) + (1/(258/(69+258))))
f1_dissatisfied

f1_satisfied = 2 / ((1/(193/(79+193))) + (1/(193/(69+193))))
f1_satisfied

result.coef_
result.get_params()






# Dataset 5-2

import pandas as pd
import numpy as np

data = pd.read_csv("j:\실기\data\data files\Dataset_5_2.csv")


#4

data1 = data.loc[(data['previous_owners'] == 1) & (data['engine_power'] == 51), ('model','km','age_in_days')]
data1

data3 = pd.pivot_table(data1, ('km', 'age_in_days'), 'model', aggfunc='mean')
data3['km_per_day'] = data3.km / data3.age_in_days
data3.km_per_day.min() / data3.km_per_day.max()



#5
data['km_per_day'] = data.km / data.age_in_days
lounge = data[data['model'] == 'lounge']
sport = data[data['model'] == 'sport']

import scipy.stats as stats

stats.ttest_ind(lounge.km_per_day, sport.km_per_day, equal_var = True)



#6

from sklearn.linear_model import LinearRegression

data6 = data[data['model'] == 'pop']

y = data6.price
X = data6[['engine_power', 'age_in_days', 'km']]

lm = LinearRegression()
lm.fit(X, y)

X_pred = np.array([51, 400, 9500])

lm.predict(X_pred.reshape(1,-1))


