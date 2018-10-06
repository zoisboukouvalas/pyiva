# -*- coding: utf-8 -*-

import numpy as np
import os.path

#--------------------------------------------------------------------
def iva_laplace(X, A=[], whiten=False, verbose=True, initW=[],
              maxIter= 2*512, terminationCriterion='ChangeInCost',
              termThreshold=1e-6, alpha0=0.1):

    """
    Required arguments:

        X : numpy array of shape (N, K, T) containing data observations from K data sets.

    Optional keyword arguments:

        A : [], true mixing matrices A, automatically sets verbose
        whiten : Boolean, default = True
        verbose : Boolean, default = False : enables print statements
        W_init : [], ... % initial estimates for demixing matrices in W
        maxIter : 2*512, ... % max number of iterations
        terminationCriterion : string, default = 'ChangeInCost' : criterion for terminating iterations, either 'ChangeInCost' or 'ChangeInW'
        termThreshold : float, default = 1e-6, : termination threshold
        alpha0 : float, default = 0.1 : initial step size scaling

    Output:
        W : the estimated demixing matrices
    """

    alphaMin = 0.1
    alphaScale = 0.9
    outputISI = False

    if isinstance(A, np.ndarray):
        supplyA = A.any()

    elif isinstance(A, list):
        supplyA = not (not A)

    [K, N, T] = np.shape(X)

    if whiten:
        [X, V, yy] = whiten(X)

    if (initW != []) and (initW.any()):
        W = initW
        if np.shape(W)[2] == 1 and np.shape(W)[0] == N and np.shape(W)[2] == N:
            W = np.tile(W, [1, 1, K])
        if whiten:
            for k in range(0, K):
                W[k,:,:] = np.dot(W[k,:,:]), np.linalg.pinv(V[k,:,:])
    else:
        W = np.random.randn(K, N, N)


    if supplyA:
        verbose = True

        if False:
            outputISI = True
            isi = np.empty((1, maxIter)).flatten()
            isi[:] = np.nan

        if whiten:
            for k in range(0, K):
                A[k,:,:] = np.matmul(V[k,:,:],A[k,:,:])

    isi = 0
    cost = np.empty((1, maxIter)).flatten()
    cost[:] = np.nan

    Y = X*0
    for iter in range(0, maxIter):
        termCriterion = 0
        for k in range(0, K):
            Y[k,:,:] = np.matmul(W[k,:,:], X[k,:,:])

        sqrtYtY = np.sqrt(np.sum(np.power(np.abs(Y), 2), axis=0))
        sqrtYtYInv = np.power(sqrtYtY, -1)

        if supplyA:
            [amari_avg_isi, amari_joint_isi, bla, blabla] = bss_isi(W, A)
            if outputISI:
                isi[iter] = amari_joint_isi

        W_old = W
        cost[iter] = 0
        for k in range(0, K):
            temp = np.abs(np.linalg.det(W[k,:,:]))
            if(temp != 0):
                cost[iter] = cost[iter] + np.log(temp)
        """   cost(iter)=sum(sum(sqrtYtY))/T-cost(iter);
              cost(iter)=cost(iter)/(N*K);"""

        cost[iter] = np.sum(np.sum(sqrtYtY, axis = 0), axis = 0) / T - cost[iter]
        cost[iter] = cost[iter]/(N*K)

        dW = W*0
        for k in range(0, K):
            phi = np.multiply(sqrtYtYInv, Y[k,:,:])
            dW[k,:,:] = W[k,:,:] - np.matmul(np.matmul(phi, Y[k,:,:].T) / T, W[k,:,:])

        W = W + alpha0*dW


        if terminationCriterion.lower() == "ChangeInW".lower():
                termCriterion = max(termCriterion, max((1 - np.abs(np.diag(np.matmul(W_old[k,:,:], W[k,:,:]), 1)))))
        elif terminationCriterion.lower() == "ChangeInCost".lower():
            if iter == 0:
                termCriterion = 1
            else:
                termCriterion = np.abs(cost[iter-1]-cost[iter]) / np.abs(cost[iter])

        else:
            print("Uknown termination method. Line 574")

        if termCriterion < termThreshold or iter == maxIter:
            break
        elif np.isnan(cost[iter]):
            for k in range(0, K):
                W[k,:,:] = np.eye(N) + 0.1*np.random.randn(N)
            if verbose:
                print('\n W blowup, restart with new initial value.')
        elif iter > 1 and cost[iter] > cost[iter - 1]:
            alpha0 = np.maximum(alphaMin, alphaScale*alpha0)

        if verbose:
            if supplyA:
                print("\n Step ", iter, ": W changes: ", termCriterion, ", Cost: ", cost[iter], ", Avg ISI: ", amari_avg_isi, ", Joint ISI: ", amari_joint_isi)
            else:
                print("\n Step ", iter, ": W change: ", termCriterion, ", Cost: ", cost[iter])


    # Finish Display
    if iter == 1 and verbose:
        if supplyA:
            print("\n Step ", iter, ": W change: ", termCriterion, ", Cost: ",
                  cost[iter], ", Avg ISI: ", amari_avg_isi, ", Joint ISI: ", amari_joint_isi)
        else:
            print('\n Step ', iter, ": W change ", termCriterion, ", Cost: ", cost[iter] )

    if verbose:
        print("\n")

    if whiten:
        for k in range(0, K):
            W[k,:,:] = np.matmul(W[k,:,:], V[k,:,:])

    return W

Exprnd_available = True
Randraw_available = None


#--------------------------------------------------------------------
def vecmag(vec=None, *argv):
    if(not vec is None) and (len(argv) == 0):
        mag = np.sqrt(np.sum(np.multiply(vec, np.conjugate(vec)),axis=0))
    else:
        mag = np.multiply(vec, np.conjugate(vec))
        for ii in range(0, len(argv)):
            mag = mag + np.multiply(argv[ii], np.conjugate(argv[ii]))
        mag = np.sqrt(mag)

    return mag

#--------------------------------------------------------------------
def vecnorm(vec):
    [n,m] = vec.shape
    if n == 1:
        print('vecnorm operates on column vectors, input appears to have dimension of 1')
    uvec = np.zeros([n,m])
    mag = vecmag(vec)
    for ii in range(0, np.shape(vec)[0]):
        uvec[ii, :] = np.divide(vec[ii, :], mag)

    return [uvec, mag]

#--------------------------------------------------------------------
def whiten(x):
    if type(x) == np.ndarray:
        K,N,T = x.shape
        if K == 1:
            x = np.apply_along_axis(np.subtract, x, np.mean(x, 1))
            covar = np.dot(np.matmul(x, x.getH()), np.linalg.pinv(T))
            [eigval, eigvec] = np.linalg.eig(covar)
            V = np.linalg.lstsq(np.sqrt(eigval), eigvec)[0]
            U = np.matmul(eigvec, np.sqrt(eigval))
            z = np.matmul(V, x)
        else:
            K = x.shape[0]
            z = np.zeros((K,N,T))
            V = np.zeros((K,N,N))
            U = np.zeros((K,N,N))
            for k in range(0, K):
                xk = x[k,:,:] - np.mean(x[k,:,:],1).reshape(np.mean(x[k,:,:],1).size, 1)
                covar= np.matmul(xk, np.matrix(xk).getH())/T
                [eigval, eigvec] = np.linalg.eig(covar)
                V[k,:,:] = np.linalg.lstsq(np.sqrt(np.diag(eigval)), eigvec)[0]
                U[k,:,:] = np.matmul(eigvec, np.sqrt(eigval))
                z[k,:,:] = np.matmul(V[k,:,:], xk)
    """
    else:
        K = x.size
        sizex = x.shape
        V = np.ndarray(sizex)
        U = np.ndarray(sizex)
        z = np.ndarray(sizex)
        for k in range(0, K):
            T = x[k].shape[1]
            xk = np.apply_along_axis(np.subtract, x[k,:,:], np.mean(x[k,:,:], 1))
            covar = np.dot(T, np.linalg.pinv(np.matmul(xk, xk.getH())))
            [eigval, eigvec] = np.linalg.eig(covar)
            V[k] = np.linalg.lstsq(np.sqrt(eigval), eigvec)[0]
            U[k] = np.matmul(eigvec, np.sqrt(eigval))
            z[k] = np.matmul(V[k], xk)
"""
    return [z, V, U]

#--------------------------------------------------------------------
def bss_isi(W=None, A=None, s=None, Nuse=None, *argv):
    gen_perm_inv_flag = False
    success = True
    Wcell = (type(W) != np.ndarray)

    if A is None:
        Acell = False
    else:
        Acell = (type(A) != np.ndarray)

    if not Wcell and not Acell:
        if np.ndim(W) == 2 and np.ndim(A) == 2:
            if not W is None and not A is None and s is None:
                G = np.matmul(W, A)
                [N, M] = np.shape(G)
                Gabs = np.abs(G)
                if gen_perm_inv_flag:
                    max_G = Gabs.max(1)
                    Gabs = np.multiply(np.tile(np.divide(1, max_G), np.shape(G)[1]), Gabs)
            elif not W is None and not A is None and not s is None and Nuse is None:
                y = np.matmul(np.matmul(W,A), s)
                D = np.diag(np.divide(1, np.std(s, 1, ddof=1)))
                U = np.diag(np.divide(1, np.std(y, 1, ddof=1)))
                G = np.dot(np.matmul(np.matmul(U, W), A), np.linalg.pinv(D))
                [N, M] = np.shape(G)
                Gabs = np.abs(G)
            else:
                print("Not Acceptable.")

            isi = 0
            for n in range(0, N):
                isi = isi + np.dot(np.sum(Gabs[n, :], axis=0)), np.linalg.pinv(Gabs[n, :].max(0)) - 1
            for m in range(0, M):
                isi = isi + np.dot(np.sum(Gabs[:, m], axis=0)), np.linalg.pinv(Gabs[:, m].max(0)) - 1
            isi = np.dot(isi, np.linalg.pinv(2*N*(N-1)))
            isiGrp = np.nan
            success = np.nan
        elif np.ndim(W) == 3 and np.ndim(A) == 3:
            [K,N,M] = np.shape(W)
            if M != N: #CHANGE TO ERROR MESSAGE !!!!!!!!!!!!!!!
                print('This more general case has not been considered here.')
            L = M

            isi = 0
            GabsTotal = np.zeros([N, M])
            G = np.zeros([K, N, M])
            for k in range(0, K):
                if (not W is None or (not W is None and not A is None)) and s is None:
                    Gk = np.matmul(W[k,:,:], A[k,:,:])
                    Gabs = np.abs(Gk)
                    if gen_perm_inv_flag:
                        max_G = Gabs.max(1)
                        Gabs = np.multiply(np.tile(np.divide(1, max_G), np.shape(Gabs)[1]), Gabs)
                else:
                    yk = np.matmul(np.matmul(W[k,:,:], A[k,:,:]), s[k,:,:])
                    Dk = np.diag(np.divide(1, np.std(s[k,:,:], 1, ddof=1)))
                    Uk = np.diag(np.divide(1, np.std(yk, 1, ddof=1)))
                    Gk = np.dot(np.matmul(np.matmul(Uk, W[k,:,:]), A[k,:,:]), np.linalg.pinv(Dk))
                    Gabs = abs(Gk)

                G[k,:,:] = Gk

                if not Nuse is None:
                    Np = Nuse
                    Mp = Nuse
                    Lp = Nuse
                else:
                    Np = N
                    Mp = M
                    Lp = L

                if k == 0:
                    colMaxG = Gabs.max(1)[1]
                    if np.max(np.shape(np.unique(colMaxG))) != Np:
                        success = False
                else:
                    colMaxG_k = Gabs.max(1)[1]
                    if not np.any(colMaxG_k == colMaxG, 0):
                        success = False

                GabsTotal = GabsTotal + Gabs

                for n in range(0, Np):
                    isi = isi + np.dot(np.sum(Gabs[n, :], axis=0), np.linalg.pinv([[Gabs[n, :].max(0)]])) - 1
                for m in range (0, Mp):
                    isi = isi + np.dot(np.sum(Gabs[:, m], axis=0), np.linalg.pinv([[Gabs[:, m].max(0)]])) - 1

            isi = np.dot(isi, np.linalg.pinv([[2*Np*(Np-1)*K]]))

            Gabs = GabsTotal
            if gen_perm_inv_flag:
                max_G = Gabs.max(1)
                Gabs = np.multiply(np.tile(np.divide(1, max_G), np.shape(Gabs)[1]), Gabs)
            isiGrp = 0

            for n in range(0, Np):
                isiGrp = isiGrp + np.dot(np.sum(Gabs[n,:], axis=0), np.linalg.pinv([[Gabs[n,:].max(0)]])) - 1
            for m in range(0, Mp):
                isiGrp = isiGrp + np.dot(np.sum(Gabs[:,m], axis=0), np.linalg.pinv([[Gabs[:,m].max(0)]])) - 1

            isiGrp = np.dot(isiGrp, np.linalg.pinv([[2*Lp*(Lp-1)]]))
        else:

            print('Need inputs to all be of either dimension 2 or 3')

    try:
      success
    except:
        try:
          isiGrp
        except:
          return [isi, None, None, None]
        else:
          return [isi, isiGrp, None, None]
    else:
      return [isi,isiGrp,success,G]

#--------------------------------------------------------------------
def sub2ind(array_shape, selected_rows, selected_cols):
    """ This method cannot take int parameters. Must be arrays of same size """

    result = []
    rows = array_shape[0]

    for i in range(0, len(selected_rows)):
        result = np.append(result, (selected_cols[i] + 1)*rows  - (rows - selected_rows[i]))

    return result


#--------------------------------------------------------------------
def randmv_laplace(d=None, T=None, *argv):
    global Exprnd_available
    global Randraw_available

    if Exprnd_available:
        if os.path.isfile('exprnd.m'):
            Exprnd_available = False

    if not Exprnd_available and Randraw_available is None:
        if os.path.isfile('randraw.m'):
            Randraw_available = True
        else:
            try:
                raise ValueError('The exprnd function is unavailable, download randraw function via Matlab file exchange to proceed without the statistics toolbox.')
            except ValueError as err:
                print(err.args)

    if d is None:
        test_randmv_laplace
        return

    Params = {'lambda': 1, 'mu' : np.zeros([int(d),1]), 'Gamma' : np.eye(int(d))}

    Params = getopt(Params, argv)
    if Params['lambda'] < 0 or np.imag(Params['lambda']) != 0:
        try:
            raise ValueError('Rate should be real-valued and greater than 0.')
        except ValueError as err:
            print(err.args)

    Params['mu'] = np.transpose(Params['mu']).flatten().reshape(np.size(Params['mu']), 1)
    if len(Params['mu']) != int(d) or np.any(np.imag(Params['lambda'] != 0)):
        try:
            raise ValueError('Mean vector should be real-valued and correct dimension (d x 1).')
        except ValueError as err:
            print(err.args)

    if Params['Gamma'].shape[0] != int(d) or Params['Gamma'].shape[1] != int(d) or np.abs(
            np.linalg.det(Params['Gamma']) - 1) > 0.0001 or np.any(
                    np.imag(np.transpose(np.ndarray.flatten(
                            Params['Gamma'].T.flatten().reshape(Params['Gamma'].size, 1)))) != 0):
        try:
            raise ValueError('Internal covariance structure needs to be real-valued square matrix with a determinant of one.')
        except ValueError as err:
            print(err.args)

    X = np.random.randn(int(d),int(T))

    if Exprnd_available:
        Z = np.sqrt(np.random.exponential(1/Params['lambda'], [1, int(T)]))

    elif Randraw_available:
        print('RANDRAW is unavailable')
        """Z = sqrt(randraw('exp', Params.lambda, [1 T])); NEEEEEEEEEEEEEEEEEEEEEEEDS FIXING, No python alt """

    Y = Params['mu'] + np.multiply(Z, np.matmul(np.sqrt(Params['Gamma']), X))

    return Y

#--------------------------------------------------------------------
def getopt(properties, *argv):

    if len(argv) != 0 and isinstance(argv[0], tuple):
        argv = argv[0]
    # Process the properties (optional input arguments)

    prop_names = np.array([])
    for k, v in properties.items():
        prop_names = np.append(prop_names, [[k]])

    TargetField = []

    for ii in range(0, len(argv)):
        arg = argv[ii]
        if not TargetField:
            """ ERROR CHECKING UNNECESARY
            try:
                if (str(arg).isalpha):
                    raise ValueError('Property names must be character strings')
            except ValueError as err:
                print(err.args)

            comparisons = np.array(prop_names) == arg
            try:
                if np.where(comparisons == True)[0].size == 0:
                    raise ValueError('invalid property ' + arg + '; must be one of: ' + prop_names)
            except ValueError as err:
                print(err.args)
            """
            TargetField = arg
        else:
            properties[TargetField] = arg
            TargetField = ''
    try:
        if len(TargetField) != 0:
            raise ValueError('Property names and values must be specified in pairs.')
    except ValueError as err:
        print(err.args)

    return properties
