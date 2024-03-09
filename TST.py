from distutils.log import error
from pickletools import optimize
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import random
def sigmoid(x):
    return 1 / (1+math.exp(-x))
X = [0.3,-0.78,1.26,0.03,1.11,0.24,-0.24,-0.47,-0.77,-0.37,0.85,-0.41,-0.27,0.02,-0.76,2.66]
Y = [12.27,14.44,11.87,18.75,16.37,19.78,19.51,12.65,14.74,10.72,21.94,12.83,15.51,17.14,14.42]

a = tf.Variable(random.random())
b = tf.Variable(random.random())
c = tf.Variable(random.random())
d = tf.Variable(random.random())

def compute_loss():
    y_pred = a * X*X*X + b * X*X + c*X+d
    loss = tf.reduce_mean((Y-y_pred)**2)
    return loss

optimizer = tf.keras.optimizers.Adam(lr=0.07)
for i in range(1000):
    optimizer.minimize(compute_loss,var_list=[a,b,c,d])

    if i % 100 == 99:
        print(i,'a:',a.numpy(),'b:',b.numpy(),'c:',c.numpy(),'d:',d.numpy(),'loss:',compute_loss().numpy())
line_x = np.arange(min(X),max(X),0.01)
line_y = a * line_x*line_x*line_x+b+line_x*line_x+c*line_x+d

plt.plot(line_x,line_y,'r-')
plt.plot(X,Y,'bo')
plt.show()
# x = range(20)
# y = tf.random.normal([20],0,1)
# plt.plot(x,y)
# plt.show()
# x = np.array([[1,1],[1,0],[0,1],[0,0]])
# y = np.array([[0],[1],[1],[0]])

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=2,activation='sigmoid',input_shap/e=(2,0)),
#     tf.keras.layers.Dense(units=1,activation='sigmoid')

# ])
# model.complie(optimize=tf.keras.optimizers.SGD(lr=0.1),loss='mse')

# model.summary()
# w = tf.random.normal([2],0,1)
# b = tf.random.normal([1],0,1)
# b_x = 1

# for i in range(2000):
#     error_sum = 0 ↑
#     for j in range(4):
#         output = sigmoid(np.sum(x[j]*w)+b_x*b)
#         error = y[j][0] - output
#         w = w + x[j] * 0.1 *error
#         b = b + b_x * 0.1 *error
#         error_sum+=error
#     if i % 200 == 199:
#         print(i,error_sum)
# x = 1
# y = 0
# w = tf.random.normal([1],0,1)

# for i in range(1000):
#     output = sigmoid(x*w)
#     error = y-output
#     w = w + x *0.1*error

#     if i %100==99:
#         print(i,error,output)
