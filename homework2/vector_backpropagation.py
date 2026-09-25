import numpy as np

def sig_d(z):
    s = 1 / (1 + np.exp(-z))
    return s * (1 - s)

def l2(W, x):

    # Forward pass
    wx = W @ x         
    y = 1 / (1 + np.exp(-wx))
    f = np.sum(y**2)

    # Backward pass
    df_dy = 2 * y
    df_dz = df_dy * sig_d(wx)
    df_dx = W.T @ df_dz
    df_dW = df_dz[:, None] * x[None, :]

    return f, df_dx, df_dW


W = np.array([[0.1, 0.2, 0.6],[0.22, 0.8, 0.2],[0.5, 0.7, 0.3]])

x = np.array([2, 1, 5])

f, df_dx, df_dW = l2(W, x)

print("Forward pass: L2 loss f =", f)

print("Gradient W =\n", df_dW)
print("Gradient x =\n", df_dx)
