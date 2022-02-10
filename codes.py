import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv(r"/Users/burakgun/Desktop/Social_Network_Ads.csv")

y = data.Purchased.values

x = data.drop("Purchased",axis = 1)
x = x.drop("User ID",axis = 1)
x["new_Gender"] = [1 if each == "Male" else 0 for each in x["Gender"]]
x.drop("Gender",axis =1 ,inplace = True)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.05)

model = LogisticRegression()
model.fit(x_train,y_train)

predictions = model.predict(x_test)

accuracy_score = metrics.accuracy_score(predictions,y_test)

print("accuracy_score : {}".format(accuracy_score))
