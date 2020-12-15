
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





def plot_ConfusionMatrix(matrix):
    plt.figure()
    sns.heatmap(matrix, annot=True, cmap='Purples')
    plt.title("Confusion Matrix")
    plt.savefig("ConfusionMatrix.png")
    plt.show()



def plot_Corelatoin(dataset):
    plt.figure()
    sns.heatmap(dataset, annot=True, cmap='Purples')
    plt.title("Corelatoin Map")
    plt.savefig("Corelatoin.png")
    plt.show()
    
    
    
    
def plot_Class():
    plt.figure()
    plt.title("Class Wise Plotting")
    sns.scatterplot(x="sepal_length", y="sepal_width", data=dataSet, hue="species")
    plt.savefig("Class_SL_SW.png")
    plt.show()    
    
    plt.figure()
    plt.title("Class Wise Plotting")
    sns.scatterplot(x="petal_length", y="petal_width", data=dataSet, hue="species")
    plt.savefig("Class_PL_PW.png")
    plt.show()    
    
    



from sklearn.neighbors import KNeighborsClassifier 
from sklearn import datasets
from sklearn.model_selection import learning_curve 
dataset = datasets.load_iris()
X, y = dataset.data, dataset.target 
sizes, training_scores, testing_scores = learning_curve(KNeighborsClassifier(), X, y, cv=10, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 60)) 

def train_test_Plot():
    testError=[]
    for i in range(2,20):
        r=Knn(i,X_Features_Train, Y_Feature_Train,X_Features_Train)
        cm=CM(Y_Feature_Train,r)
        testError.append((cm[0][1]+cm[1][0])/cm.sum())
    plt.plot(range(2,20),testError,label='Training Error Graph',color="b")
    plt.xlabel('K Value')
    plt.ylabel('Error')
    plt.legend()
    #plt.savefig("training_graph.png")
    plt.show()
    
def validation_test_Plot():
    Standard_Deviation_testing = np.std(testing_scores, axis=1) 
    plt.plot(sizes, Standard_Deviation_testing, color="g", label="Cross-validation score") 
    plt.title("Validatoin Graph") 
    plt.xlabel("Training  Size"), plt.ylabel("Accuracy "), plt.legend(loc="best") 
    plt.tight_layout() 
    plt.savefig("Validation_Error_graph.png")
    plt.show()
    
        
def Testing_graph():
    testError=[]
    for i in range(2,20):
        r=Knn(i,X_Features_Train, Y_Feature_Train,X_Features_Test)
        cm=CM(Y_Feature_Test,r)		
        testError.append((cm[0][1]+cm[1][0])/cm.sum())
    plt.plot(range(2,20),testError,label='Testing Error Graph', color="g")
    plt.xlabel('K Value')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig("testing_graph.png")
    plt.show()

    
    
def accuracy():
    accuracy=[]
    k_value=[]
    for i in range(2,20):        
        r=Knn(i,X_Features_Train, Y_Feature_Train,X_Features_Test)    
        accuracy.append((np.sum(r==Y_Feature_Test)/len(Y_Feature_Test))*100)
        k_value.append(i)
    
    plt.figure()
    sns.lineplot(x=k_value, y=accuracy)
    plt.title('Testing Accuracy Graph')
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.savefig("Testing_Accuracy_graph.png")
    plt.show()
    
    
    
        
plot_ConfusionMatrix(confusionMatrix)
plot_Corelatoin(dataSet.corr())
plot_Class()    
train_test_Plot()
validation_test_Plot()
accuracy()
Testing_graph()
