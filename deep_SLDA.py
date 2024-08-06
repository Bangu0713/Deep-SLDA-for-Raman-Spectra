import torch
import scipy.linalg as slinalg
import torch.nn as nn
from functools import partial
import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import svds
# from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch.autograd import Variable

class EigValsH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        A, B = args
        device = A.device
        A = A.detach().data.cpu().numpy()
        B = B.detach().data.cpu().numpy()
        w, v = eigsh(A, k=29, M=B)
        w = torch.from_numpy(w).to(device)
        v = torch.from_numpy(v).to(device)
        ctx.save_for_backward(w, v)
        return w,v
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        w, v = ctx.saved_tensors
        dw = grad_outputs[0]
        gA = torch.matmul(v, torch.matmul(torch.diag(dw), torch.transpose(v, 0, 1)))
        gB = -torch.matmul(v, torch.matmul(torch.diag(dw * w), torch.transpose(v, 0, 1)))
        return gA, gB


def eigh(A, B):
    device = A.device  # GPU/CPU
    A = A.detach().data.cpu().numpy()
    B = B.detach().data.cpu().numpy()
    w, v = eigsh(A, k=29, M=B)  # eigenvalues/eigenvectors
    w = torch.from_numpy(w).to(device)
    v = torch.from_numpy(v).to(device)
    return w, v


def slda_loss(range_evals, null_evals, n_classes):

    n_components = n_classes - 1
    range_evals = range_evals[-n_components:]
    range_mean_value = 0

    total = torch.sum(range_evals)
    proportions = [value / total for value in range_evals]
    
    if len(range_evals)==1:
        range_mean_value += -torch.log(range_evals[0])
    else:
        for i in range(len(range_evals)):
            range_mean_value += -((1-proportions[i]) ** (n_classes - len(null_evals))) * torch.log(range_evals[i])

    range_mean_value = range_mean_value/len(range_evals)

    return range_mean_value


class SLDA:
    def __init__(self, n_classes, FVE_threshold=0.95):
        super(SLDA, self).__init__()
        self.FVE_threshold = FVE_threshold
        self.n_classes = n_classes

    def _solve_SLDA(self, X, y, phase):
        n, p = X.shape
        labels, counts = torch.unique(y, return_counts=True)
        nc = len(labels)
        mu = torch.mean(X, axis=0) # global mean
        xbar = torch.empty((nc,p),dtype=X.dtype, device=X.device, requires_grad=False) # class means
        Res = torch.empty_like(X, dtype=X.dtype, device=X.device, requires_grad=False) # within
        
        for i, Nc in zip(labels, counts):
            idx = (y==labels[i])
            xbar[i,] = torch.mean(X[idx,], axis=0)
            Res[idx,] = X[idx,] - xbar[i,]
            xbar[i,] -= mu

        U, d, V = svd(Res.detach().data.cpu().numpy())
        U, d, V= torch.from_numpy(U).to(X.device), torch.from_numpy(d).to(X.device),torch.from_numpy(V).to(X.device)
        FVE = torch.cumsum(d**2, dim=0) / torch.sum(d**2)
        noeig = torch.argmax((FVE > self.FVE_threshold).int()) + 1
        phi = V[0:(noeig),]
		
        # range space
        Gamma_1 = torch.empty((nc,noeig),dtype=X.dtype, device=X.device, requires_grad=False)
        for i in range(nc):
            Gamma_1[i,] = torch.matmul(xbar[i,], phi.T)
        S_B = torch.matmul(Gamma_1.T,Gamma_1)/(nc-1)

        M = torch.diag(d[0:(noeig)]**2)
        
        if phase == "train":
            w,v = EigValsH.apply(S_B, M) 
        else:
            w,v = eigh(S_B, M) 

        range_eigenvalue = w
        VR = torch.matmul(phi.T, v)

		# null space
        phi= phi.detach().data.cpu().numpy()
        xbar = xbar.detach().data.cpu().numpy()
        for i in range(nc):
            xbar[i,] -= np.dot(np.dot(phi, xbar[i,]).T, phi)
        
        U, d, V = svds(xbar, nc-1)
        d = torch.from_numpy(d.copy()).to(X.device)
        null_eigenvalue = d**2
        V = torch.from_numpy(V.copy()).to(X.device)
        psi = V 
        dirs = torch.vstack((psi,VR.T))
            
        return dirs, range_eigenvalue, null_eigenvalue

    def fit(self, X, y, phase):
        
        if phase == "train":
            dirs, range_eigenvalue, null_eigenvalue = self._solve_SLDA(X, y,phase)
            return dirs, range_eigenvalue, null_eigenvalue
        else:
            dirs, range_eigenvalue, null_eigenvalue = self._solve_SLDA(X, y, phase)
            return range_eigenvalue, null_eigenvalue
