# EVALUATION DETAILS

## The training behaviours meta-metrics

As presented in the paper designed three meta-metrics: 

(1) the learning velocity $V$ is the number of epochs needed to reach the first saturation ($> 0.9$) of a given metric, e.g.  $V_{F^+_1}$ is the number of epochs needed to reach the first saturation when $F^+_1>0.9$ ; 

(2) the stability of the learning process is defined as the ratio of epochs during which a metric gets worse after the first saturation, e.g. for  $F^+_1$ we note the stability $S_{F^{+}_1}$ ; 

(3) the final divergence of the learning process is defined as the number of folds for which there is a final divergence, e.g. the divergence $D_{F^+_1}$ is the number of folds for which the final $F^+_1$ is lower than the value of its first saturation.

## Why we do need to better formalise our metrics?

First, $F_1^+$ metrics \textbf{never} saturate on the 5 folds in the case of our experimental design; So we prefer to focus on $F_1^-$. 

Nevertheless, $R_{TP}$ and $F_1^-$  can still face to the following issues: 
* [Issue.1] When the saturation doesn't happen on a fold we do not know how to aggregate velocity metric ($V_M$) 
* [Issue.2] When models never get worse on every fold we do not know how to compute the saturation metric ($S_M$)

Let's take a look at a concrete example, where $nb_{fold}=4$ on the computation of the velocity ( $V_{F^-_1}$). \\ The $f_2$, the first fold is saturating at the epoch number 3 and seems to diverge.  


## TABLE CONSTRUCTION
The metrics computed in Table 1 of our paper are obtained with the helps of the [ExtractResults.py](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/eval/ExtractResults.py) that first extract metrics from the server and connect existing artefact to wandb runs available at [https://wandb.ai/celian-ringwald/12ShadesOfRDF](https://wandb.ai/celian-ringwald/12ShadesOfRDF). This script produced the json that retrieve for each folds tokenizer path, and the checkpoints path, and the carbon cost.

We combine both this json file with the wandb data to compute as we describe bellow the three sets of aggregated data : 
* [results_12ShadesSyntax_MEAN_NAIVE.csv](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/eval/results_12ShadesSyntax_MEAN_NAIVE.csv)
* [results_12ShadesSyntax_MEAN_WISE.csv](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/eval/results_12ShadesSyntax_MEAN_WISE.csv)
* [results_12ShadesSyntax_STRICT.csv](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/eval/results_12ShadesSyntax_STRICT.csv)

The one used for building the table is  [results_12ShadesSyntax_STRICT.csv](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/eval/results_12ShadesSyntax_STRICT.csv). 
