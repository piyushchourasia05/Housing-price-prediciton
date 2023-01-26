import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

df = pd.read_csv("USA_Housing.csv")

df = df.iloc[:,:-1]


x_train = df.to_numpy()
print(f"x_train before edit: {x_train}\n")

# storing price in y_train (copying last column of each row from x_train to y_train)
y_train = np.zeros(len(x_train))
for i in range(len(x_train)):
    y_train[i] = x_train[i,-1]
    
print(f"y_train : {y_train}")    

# Now, removing the last column from x_train
x_train = np.delete(x_train,-1,1)
print(f"x_train after edit: {x_train}\n")

#checking the size of x_train and y_train
print(f"input size: {x_train.shape}")
print(f"output size: {y_train.shape}")

#Visualizing the data
f = plt.figure()    
f, axes = plt.subplots(nrows = 2, ncols = 3, sharey = True, constrained_layout=True)

axes[0][0].scatter(x_train[:,0], y_train, c = np.random.rand(len(x_train[:,0]),3), marker = ".", s=1)
axes[0][0].set_xlabel('Average Area Income', labelpad = 5)
axes[0][0].set_ylabel('Price of House(in milliions)', labelpad = 5)

axes[0][1].scatter(x_train[:,1], y_train, c = np.random.rand(len(x_train[:,1]),3), marker = ".", s=1)
axes[0][1].set_xlabel('Average House Age', labelpad = 5)
axes[0][1].set_ylabel('Price of House(in millions)', labelpad = 5)

axes[0][2].scatter(x_train[:,2], y_train, c = np.random.rand(len(x_train[:,2]),3), marker = ".", s=1)
axes[0][2].set_xlabel('Average Area Number of Rooms', labelpad = 5)
axes[0][2].set_ylabel('Price of House(in millions)', labelpad = 5)

axes[1][0].scatter(x_train[:,3], y_train, c = np.random.rand(len(x_train[:,3]),3), marker = ".", s=1)
axes[1][0].set_xlabel('Averagae area number of Bedrooms', labelpad = 5)
axes[1][0].set_ylabel('Price of House(in millions)', labelpad = 5)

axes[1][1].scatter(x_train[:,4], y_train, c = np.random.rand(len(x_train[:,4]),3), marker = ".", s=1)
axes[1][1].set_xlabel('Area Population', labelpad = 5)
axes[1][1].set_ylabel('Price of House(in millions)', labelpad = 5)


def zscore_normalize_features(X):
     # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma);

def cost_function(x,y,w,b):
    
    m,n = x.shape
    cost=0
    for i in range(m):
        f_wb = np.dot(w,x[i]) + b
        cost += (f_wb - y[i])**2
    cost = cost / (2*m)
    
    return cost

def compute_gradient(x,y,w,b):
    
    m,n = x.shape
    
    # required 2 variable that we need to compute
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = np.dot(w,x[i]) + b;
        temp = f_wb - y[i]
        for j in range(n):
            dj_dw += temp*x[i,j]
        dj_db += temp
    dj_db = dj_db / m
    dj_dw = dj_dw / m
    
    return dj_dw, dj_db
def gradient_descent(x,y,w,b,aplha,num_iter):
    
    m,n = x.shape
    count = num_iter / 10
    for i in range(num_iter):
        
        # Calculate the gradient and update the parameters
        dj_dw, dj_db = compute_gradient(x,y,w,b)
        
        # Update Parameters using w, b, alpha and gradient
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        
        if (i == count):
            print(f"iteration {count}: w: {w} and b: {b}\n")
            print(f"cost: {cost_function(x,y,w,b)}\n\n")
            count += (num_iter / 10)
        
    
    return w,b
# Normalizing our input dataset
x_norm, x_mu, x_sigma =  zscore_normalize_features(x_train)
print(x_norm)
# finding w,b using above model
w = np.zeros(5)
b = 0
alpha = 3.0e-1
w_final,b_final = gradient_descent(x_norm, y_train, w, b, alpha, 10000)
print(f"\nw_final: {w_final} and b_final: {b_final}")

def predict():
    x = np.zeros(5)
    pred_cost = 0
    
    x[0] = (input("Enter Average area income: "))
    x[1] = (input("Enter Average House Age: "))
    x[2] = (input("Enter Number of Rooms: "))
    x[3] = (input("Enter Number of Bedrooms: "))
    x[4] = (input("Enter Area Population: "))
    
    x_norm_pred = (x-x_mu)/x_sigma;
    pred_cost = np.dot(x_norm_pred,w_final)+b_final
    print("\nThe predicted cost of house is: ", pred_cost)

    predict()