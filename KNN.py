
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix as CM
from collections import  Counter
import matplotlib.pyplot as plt
import seaborn as sns

dataSet = pd.read_csv('Iris_DataSet.csv')  # loading the dataset


X_Features = dataSet[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
Y_Feature = dataSet['species'].values



Y_Feature=LE.fit_transform(LE,Y_Feature)        #transforming our string data into integers where setosa=0, versicolor=1
                                                #virginica=2

X_Features_Train, X_Features_Test, Y_Feature_Train,  Y_Feature_Test, =tts(X_Features,Y_Feature,test_size=0.3)

def distance(x1, y1):
    return np.sqrt(np.sum(x1-y1)**2)


def Knn(k, x, y, predict) :
    X_Train = x
    Y_Train = y

    labels = []
    for i in predict:
        dis = [distance(i, x) for x in X_Train]         #finding the distance of test item from traininig point 
        neigbhours = np.argsort(dis)
        k_nearest_neighbor=[]
        for i in range(k):
            k_nearest_neighbor.append(neigbhours[i])   #looking for k nearest elements
        k_nearest_Labels = [Y_Train[j] for j in k_nearest_neighbor]  #finding the label of the all neighbor's class
        common = Counter(k_nearest_Labels).most_common(1)   #finding the most occuring neighbor of same class
        labels.append(common[0][0])

    return np.array(labels)


result=Knn(3,X_Features_Train, Y_Feature_Train,X_Features_Test)

print("Accuracy is: ",end="")
print((np.sum(result==Y_Feature_Test)/len(Y_Feature_Test))*100) #calculating accuracy

print("Predicting result on the following input data: ", end="")
p=np.array([[6.7,3.3,5.7,2.1]])
print(p)
prediction=Knn(3,X_Features_Train, Y_Feature_Train,p)

if(prediction[0]==0):
    print("Model Prdicted a Setosa")
elif(prediction[0]==1):
    print("Model Prdicted a VersiColor")
elif(prediction[0]==2):
    print("Model Prdicted a Virginica ")
else:
    print("Not able to predict")

confusionMatrix= CM(Y_Feature_Test,result)
print("confusionMatrix is: ")
print(confusionMatrix)




