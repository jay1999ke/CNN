for h in range(n_H):
    for w in range(n_W):
        x_slice = X[h:h+f, w:w+f]
        H[h,w] = np.sum(x_slice * W)
        
for h in range(n_H):
    for w in range(n_W):
        dX[h:h+f, w:w+f] += W * dH(h,w)
        dW += X[h:h+f, w:w+f] * dH(h,w)