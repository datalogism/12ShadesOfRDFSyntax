# EVALUATION DETAILS

As presented in the paper designed three meta-metrics: 

(1) the learning velocity $V$ is the number of epochs needed to reach the first saturation ($> 0.9$) of a given metric, e.g.  $V_{F^+_1}$ is the number of epochs needed to reach the first saturation when $F^+_1>0.9$ ; 

(2) the stability of the learning process is defined as the ratio of epochs during which a metric gets worse after the first saturation, e.g. for  $F^+_1$ we note the stability $S_{F^{+}_1}$; 

(3) the final divergence of the learning process is defined as the number of folds for which there is a final divergence, e.g. the divergence $D_{F^+_1}$ is the number of folds for which the final $F^+_1$ is lower than the value of its first saturation.

First, F1+ metrics **never** saturate (Issue.1) on the 5 folds in the case of our experiment design; So we prefer to focus on F1-. 
Nevertheless, $R_{TP}$ and $F_-$ still have chances to be face to the following issues: 
- [Issue.1] When the saturation doesn't happen on a folds we do not know how to aggregate velocity metric ($V_M$)     
- [Issue.2] When models never get worse on every fold we do not know how to compute the saturation metric ($S_M$)
