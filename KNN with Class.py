
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from collections import Counter
import matplotlib.pyplot as plt 


def numpy_distance(x,y):
    return np.linalg.norm(x-y)


class KNN:
    def __init__(self, k=3):
        self.K= k
        self.prediction=[]
        self.coorect_count=0
        self.accuracy=0
        
    def fit(self, A, B, C, D):
        self.X_test=A
        self.X_train=B
        self.y_test=C        
        self.y_train=D
        
        

    def predict(self, X):
        self.prediction=[]
        self.accuracy=0
        self.coorect_count=0
        for i in range(len(X)):  
            distances = []
            for j in range(len(self.X_train)):
                dist = numpy_distance(X[i], self.X_train[j])        
                distances.append((self.X_train[j], dist, self.y_train[j]))  
            
            distances.sort(key=lambda x: x[1])
            neighbors = distances[:10] ## Considering k closest points
            class_counter = Counter()  ## a counter to check which labels appeared how many times
            for neighbor in neighbors:
                class_counter[neighbor[2]] += 1
            self.prediction.append((Counter(class_counter).most_common(1)[0][0],neighbors[0][0]))
            if(self.y_test[i] == self.prediction[i][0]):  ## if prediction is correct than increase correct count
                self.coorect_count = self.coorect_count + 1
        
        acc = self.coorect_count/float(len(self.X_test))*100  ## accuracy
        self.accuracy=acc
        return self.prediction
                
    def get_accuracy(self, X):
        Y=X.reshape(132)
        return (np.sum(self.prediction[0][1] == Y) / len(Y)) * 100
        
    
    
    


data = pd.read_csv("C:\\Users\\malis\\Desktop\\sem5\\AI_Lab\\__Project\\dataSet.csv")  

X = data.to_numpy()   
X = X[:,0:132]

Y = data.to_numpy()  
Y = Y[:,132]


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=111) 


clf=KNN(5)
clf.fit(X_test, X_train, y_test, y_train)

pre=clf.predict(X_test)
print(clf.accuracy)
print(clf.prediction[0][0])

X_test = np.array([[0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0]])
pre=clf.predict(X_test)
print(clf.prediction[0][0])
A=clf.get_accuracy(X_test)
print(A)

    
    
    
    