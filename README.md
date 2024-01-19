# Support Vector Clustering

## Gabriel Buffet, Nils Cazemier & Meilame Tayebjee
### ENSAE 2023-2024

This project has been realised for the Advanced Machine Learning course of the MEng @ ENSAE (Paris, France) by Prof. Austin Stromme.

It is based on the paper _Support Vector Clustering_ by Asa Ben-Hur, David Horn, Hava T. Siegelmann and Vladimir Vapnik in 2001 (article available in the repo: Paper.pdf).

We reimplement the SVC algorithm in _SVC.py_.

_Tests.ipynb_ contains tests on synthetic data, with careful parameter tuning for SVC and comparison with classic algorithms, namely: K-Means, Agglomerative Clustering and DBSCAN.

 In _WineSVC.ipynb_, we run and fine tune SVC over a real dataset (Wine dataset: https://archive.ics.uci.edu/dataset/109/wine - no need to download, the dataset is directly imported into the notebook) to compare it with classic algorithms.

 These notebooks are directly runnable - provided libraries such as _numba_, _cvxpy_ and _networkx_  have been installed (using  $\texttt{pip install ...}$ if not) - and reproduce the results presented in the report.

 In _Report.pdf_, we remind key highlights of each algorithm with a special focus on SVC, and we detail and comment our results.
 
Feel free to reach out to us at meilame.tayebjee@polytechnique.edu / gabriel.buffet@polytechnique.edu / nils.cazemier@polytechnique.edu for any questions or recommendations.
