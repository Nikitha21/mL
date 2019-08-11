# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:52:28 2017

@author: NIKITHA RAO
"""

'''from csv import reader
with open('F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/train_data.csv','rb') as csvFile:
     csvReader = reader(csvFile, delimiter=' ', quotechar='|')
     for row in csvReader:
         print ', '.join(row)'''
         
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.metrics import cohen_kappa_score
AnnAccuracy = []
csv1 = np.genfromtxt ('F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/train_data.csv',usecols=range(1,28),delimiter=",")
#second = csv[1::26,1]
#third = csv[1::26,3]
#five = csv[1::26,5]
#twenty_eight = csv[1::26,27]

#Input Array
#X=np.array([second,third,five,twenty_eight])
#X=np.array([csv[::26]])
#X=csv1[::]
#X=X.T
#print X

csv2 = np.genfromtxt ('F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/train_data.csv',usecols=-1, delimiter=",")
#Output
#y=np.array([csv2[::]])
#y=y.T

#X = normalize(X, axis=0, norm='max')
#X = X / X.max(axis=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#lm = linear_model.LinearRegression()

#model = lm.fit(X_train, y_train)
#predictions = lm.predict(X_test)

#Variable initialization
epoch=5000 #Setting training iterations
lr=0.1 #Setting learning rate
#inputlayer_neurons = X_train.shape[1] #number of features in data set
hiddenlayer_neurons = 15 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
'''wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))'''

#print "Initial weights"
#print wh,wout

#training the model
iteration = 26
accuracy = 0
for j in range(iteration):
    '''second = csv1[j::26,2]
    third = csv1[j::26,6]
    five = csv1[j::26,7]
    nine = csv1[j::26,8]
    ten = csv1[j::26,9]
    eleven = csv1[j::26,10]
    twelve = csv1[j::26,11]
    thirteen = csv1[j::26,12]
    twenty_three=csv1[j::26,23]
    twenty_eight = csv1[j::26,26]
    fourteen=csv1[j::26,13]
    eighteen=csv1[j::26,17]
    fifteen=csv1[j::26,14]
    twenty_two=csv1[j::26,21]
    sixteen=csv1[j::26,15]
    nineteen=csv1[j::26,18]
    twenty_three=csv1[j::26,22]
    seventeen=csv1[j::26,16]
    X=np.array([second,third,five,twenty_eight,nine,ten,eleven,twelve,thirteen,twenty_three,twenty_eight,
             fourteen,eighteen,fifteen,twenty_two,sixteen , nineteen, twenty_three,seventeen ])
    X=X.T'''

    X=csv1[j::26]
    y=np.array([csv2[j::26]])
    y=y.T
    X = normalize(X, axis=0, norm='max')
    '''print "\nX in %d:", j
    print X
    print y
    cv = cross_validation.KFold(len(X), n_folds=5, indices=False)
    results = [] '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    inputlayer_neurons = X_train.shape[1]
    wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
    bh=np.random.uniform(size=(1,hiddenlayer_neurons))
    wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
    bout=np.random.uniform(size=(1,output_neurons))
    
    for i in range(epoch):
        #Forward Propogation
        hidden_layer_input1=np.dot(X_train,wh)
        hidden_layer_input=hidden_layer_input1 + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1=np.dot(hiddenlayer_activations,wout)
        output_layer_input= output_layer_input1+ bout
        output = sigmoid(output_layer_input)

        #Backpropagation
        E = y_train-output
        slope_output_layer = derivatives_sigmoid(output)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        d_output = E * slope_output_layer
        Error_at_hidden_layer = d_output.dot(wout.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        wout += hiddenlayer_activations.T.dot(d_output) *lr
        bout += np.sum(d_output, axis=0,keepdims=True) *lr
        wh += X_train.T.dot(d_hiddenlayer) *lr
        bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
        #if(i%1000)==0:
          #  print "\nError:", str(np.mean(np.abs(E))) 

    #print "weights after training: \n"
    #print wh,wout
    #print "\n output of trained data:"
    #print output
    print "\nAccuracy of train data:", j
    print accuracy_score(y_train, output.round())

    #testing on the test_data
    hidden_layer_input1=np.dot(X_test,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    y_pred = sigmoid(output_layer_input)

    #print "\n output of test data \n"
    #print y_pred

    from sklearn.metrics import f1_score

    print "\nConfusion Matrix:\n"
    matrix = confusion_matrix(y_test, y_pred.round())
    print(matrix)
    print "\nAccuracy of test data:", j 
    print accuracy_score(y_test, y_pred.round())
    print "\nF1 Score of test data:", j
    AnnAccuracy.append(f1_score(y_test, y_pred.round()))
    accuracy += accuracy_score(y_test, y_pred.round())
    
print "\nTotal Accuracy="
#print str(np.average(np.abs(accuracy_score(y_test, y_pred.round())))) 
print (accuracy/26)*100

fig = plt.figure(figsize=(6, 6))
fig.canvas.set_window_title('ANN')
plt.plot(range(1, 27), AnnAccuracy, 'g-', label='ANN')
plt.title('ANN')
plt.xlabel('Voice Sample')
plt.ylabel('F1_Score')
plt.legend(loc = 'lower left')
# Let matplotlib improve the layout
plt.tight_layout()
 
    # ==================================
    # Display the plot in interactive UI
plt.show()
 
    #To save the plot to an image file, use savefig()
plt.savefig('ANN.png')
 
    # Closing the figure allows matplotlib to release the memory used.
plt.close()

#print cohen_kappa_score(y_test, y_pred.round())
#print "\nAccuracy:",str(np.mean(y_pred))
'''plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')'''