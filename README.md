[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.python.org/pypi/pyiva)

# PyIVA
*pyiva* is an implementation of the independent vector analysis (IVA) algorithm using a multivariate Laplace prior. It uses the same score function as found in the original IVA paper. IVA is an extension of independent component analysis (ICA) to multiple statistically-dependent datasets, and can be used for dimensionality reduction and data fusion.

Primary author: Austin Kim (austinkim1004 at gmail.edu)
Code maintainer: Zois Boukouvalas (zoisb at umd.edu)
Edited and packaged by Daniel C. Elton

This code is based on the Matlab code iva_laplace.m by Matthew Anderson (matt.anderson at umbc.edu)

## Installation
to install, run `python setup.py install`

## Example usage
```python
    from pyiva.iva_laplace import iva_laplace

    W = iva_laplace(X)
```

## Documentation on *iva_laplace()*
Required arguments:

* *X* : numpy array of shape (N, K, T) containing data observations from K data sets. Here X{k}=A{k}S{k}, where A{k} is an N x N unknown invertible mixing matrix and S{k} is N x T matrix with the nth row corresponding to T samples of the nth source in the kth dataset. For IVA it is assumed that each source is statistically independent of all the sources within a dataset and exactly dependent on at most one source in each of the other datasets. The data, X, is a 3-dimensional matrix of dimensions N x K x T. The latter enforces the assumption of an equal number of samples in each dataset.

Optional keyword arguments:

*   *A* : [], true mixing matrices A, automatically sets verbose
*   *whiten* : Boolean, default = True
*   *verbose* : Boolean, default = False : enables print statements
*   *W_init* : [], ... % initial estimates for demixing matrices in W
*   *maxIter* : 2*512, ... % max number of iterations
*   *terminationCriterion* : string, default = 'ChangeInCost' : criterion for terminating iterations, either 'ChangeInCost' or 'ChangeInW'
*   *termThreshold* : float, default = 1e-6, : termination threshold
*   *alpha0* : float, default = 0.1 : initial step size scaling

Output:
*  *W* : the estimated demixing matrices so that ideally W{k}A{k} = P*D{k} where P is any arbitrary permutation matrix and D{k} is any diagonal invertible (scaling) matrix.  Note P is common to all datasets; this is to indicate that the local permutation ambiguity between dependent sources across datasets should ideally be resolved by IVA.

During runtime the following are reported:

* *cost* - the cost for each iteration
* *isi* - joint inter-symbol-interference is available if user supplies true
mixing matrices for computing a performance metric

## Citation
If you use pyIVA, please cite

@misc{PyIVA,  
author = {Zois Boukouvalas and Austin Kim and Daniel C. Elton},  
title = {{PyIVA}},  
howpublished = {\url{https://github.com/zoisboukouvalas/pyiva}}  
}

@ARTICLE{2018arXiv181100628B,    
   author = {{Boukouvalas}, Z. and {Elton}, D.~C. and {Chung}, P.~W. and
	{Fuge}, M.~D.},  
    title = "{Independent Vector Analysis for Data Fusion Prior to Molecular Property Prediction with Machine Learning}",  
  journal = {ArXiv e-prints},  
archivePrefix = "arXiv",  
   eprint = {1811.00628},  
 primaryClass = "stat.ML",  
 keywords = {Statistics - Machine Learning, Condensed Matter - Materials Science, Computer Science - Machine Learning},  
     year = 2018,  
    month = nov,  
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv181100628B},  
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}  
}

## References
[1] T. Kim, I. Lee, & T.-W. Lee, "Independent Vector Analysis: Definition and Algorithms," *Proc. of 40th Asilomar Conference on Signals, Systems, and Computers*, 2006, 1393-1396

[2] T. Kim, T. Eltoft, & T.-W. Lee, "Independent Vector Analysis: an extension of ICA to multivariate components," *Lecture Notes in Computer Science: Independent Component Analysis and Blind Signal Separation, Independent Component Analysis and Blind Signal Separation*, Springer Berlin / Heidelberg, 2006, **3889**, 165-172

[3] T. Kim, H. T. Attias, S.-Y. Lee, & T.-W. Lee, "Blind Source Separation Exploiting Higher-Order Frequency Dependencies," *IEEE Trans. Audio Speech Lang. Process.*, 2007, **15**, 70-79
