import numpy as np

x = np.array([2.0, 3.0, -1.0])
ytrue = 5.0

w = np.array([0.5, -0.5, 1.0])
b = 0.0
lr = 0.001

def forward(x, w, b):
  return np.dot(w, x) + b

def loss(ypred, ytrue):
  return (ypred - ytrue) ** 2

def gradients(w, b, x, ytrue):
  ypred = forward(x, w, b)
  dL_dy = 2 * (ypred - ytrue)

  dL_dw = dL_dy * x
  dL_db = dL_dy

  return dL_dw, dL_db

epochs = 200
for i in range(epochs):
  ypred = forward(x, w, b)
  dw, db = gradients(w, b, x, ytrue)

  w = w - lr*dw
  b = b - lr*db


  print(f"loss = {loss(ypred, ytrue)}")

