# EVALUATION DETAILS

## The training behaviours meta-metrics

As presented in the paper designed three meta-metrics: 

(1) the learning velocity $V$ is the number of epochs needed to reach the first saturation ($> 0.9$) of a given metric, e.g.  $V_{F^+_1}$ is the number of epochs needed to reach the first saturation when $F^+_1>0.9$ ; 

(2) the stability of the learning process is defined as the ratio of epochs during which a metric gets worse after the first saturation, e.g. for  $`F^+_1`$ we note the stability $S_{F^{+}_1}$ ; 

(3) the final divergence of the learning process is defined as the number of folds for which there is a final divergence, e.g. the divergence $D_{F^+_1}$ is the number of folds for which the final $F^+_1$ is lower than the value of its first saturation.

## Why we do need to better formalise our metrics?

First, $F_1^+$ metrics **never** saturate on the 5 folds in the case of our experimental design; So we prefer to focus on $F_1^-$. 

Nevertheless, $R_{TP}$ and $F_1^-$  can still face to the following issues: 
* [Issue.1] When the saturation doesn't happen on a fold we do not know how to aggregate velocity metric ($V_M$) 
* [Issue.2] When models never get worse on every fold we do not know how to compute the saturation metric ($S_M$)

Let's take a look at a concrete example, where $nb_{fold}=4$ on the computation of the velocity ( $V_{F^-_1}$). 
The $f_2$, the first fold is saturating at the epoch number 3 and seems to diverge : 

![Image Example](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/eval/MetricsExample.png)

**We can see here the illustration of Issue.1 on the forth fold and the Issue.2 at the second fold**

### Three ways to compute the aggregation

* **STRICT** : $`\overline{V_{F^-_1}}=\emptyset`$  because one of the folds doesn't saturate or $`\overline{V_{F^-_1}}=+\infty`$
* **Naive mean** :  $`\widehat{V_{F^-_1}}=(3+6+1)/4`$, we add together the different $`V_{F^-_1}`$ and divide by the number of folds
* **Wise mean**:  $`\widetilde{V_{F^-_1}}=(3+6+1)/3`$, we add together the different $`V_{F^-_1}`$ and divide by the number of folds where the saturation happens
## FORMALISATION

Let there be a $model$ we wanna evaluate on $n_f$ folds. This $model$ is trained on a period of $1$ to $n_{steps}$. During the evaluation process, we compute at step $i$ the metric $M_{i}$ we want to study, where  $0 \leqslant M_{f,i} \leqslant 1$. 


### The Velocity

We define the **Velocity** of a given fold f on M as follows : 
```math
V_{M_f}=min(\{ i | i \in  [ 1; n_{epoch} ]; \exists M_{f,i} > 0.9\})
```
The **strict** computation of this metric will be :
```math
\overline{V_{M}}=\left\{ 
  \begin{array}{ c l }
    \frac{1}{n_{f}}\sum{V_{M_f}} & \quad \textrm{if } \forall f \exists M_{f,i}>0.9 \ \\
    \varnothing       & \quad \textrm{otherwise} 
  \end{array}
\right.
```
The **wise** computation of this metric  with $n_F$ the number of folds where saturation happens as follows :
```math
\widetilde{V_{M}}=\left\{ 
  \begin{array}{ c l }
    \frac{1}{n_{F}}\sum{V_{M_f}} & \quad \textrm{if } \exists f \textrm{where } \exists M_{f,i}>0.9 \ \\
    \varnothing       & \quad \textrm{otherwise} 
  \end{array}
\right.
```

The **naive** computation of this metric with $n_f$ the number total of folds, as follows: 
```math
\widehat{V_{M}}=\left\{ 
  \begin{array}{ c l }
    \frac{1}{n_{f}}\sum{V_{M_f}} & \quad \textrm{if } \exists f \textrm{where } \exists M_{f,i}>0.9 \ \\
    \varnothing       & \quad \textrm{otherwise} 
  \end{array}
\right.
```

### The Broken steps set

To be able to define the **Stability** we need first to define what is the set of the $`broken\_steps`$ for a given fold $f$, this set gathers all the index of the steps where the metric of interest was smaller than the value of this same metric at the first saturation  :
```math
broken\_steps_{f}= \{j | j \in ]V_{M_f}, n_{epoch}]; M_{f,j} <0.9 \}
```

### The Stability

The  **Stability** metric is computed for a given fold $f$ as follows,:

```math
S_{M_f}= \frac{\|broken\_steps_{f}\|}{n_{epoch}}$$}
```
 where $`\|broken\_steps_{f}\|`$ the cardinality of the broken steps set.

* The **strict** computation of this metric will be computed as:
 ```math
\overline{S_{M_f}}=\left\{ 
  \begin{array}{ c l }
    \frac{1}{n_{f}}\sum{S_{M_f}} & \quad \textrm{if} \forall f \exists \|broken\_steps_{f}\| \ne  \varnothing \wedge \exists V_{M_f} \\
    \varnothing       & \quad \textrm{otherwise} 
  \end{array}
\right.
```
* The **wise** computation of this metric will be computed with $n_F$ the number of folds where saturation happens as follows :
 ```math
\widetilde{S_{M_f}}=\left\{ 
  \begin{array}{ c l }
    \frac{1}{n_{F}}\sum{V_{M_f}} & \quad \textrm{if} \exists f \textrm{where}\exists \|broken\_steps_{f}\| \ne  \varnothing \wedge \exists V_{M_f} \ \ \\
    \varnothing       & \quad \textrm{otherwise} 
  \end{array}
\right.
```

* The **naive** computation of this metric will be computed with $n_f$ the number total of folds, as follows:
 ```math
\widehat{S_{M_f}}=\left\{ 
  \begin{array}{ c l }
    \frac{1}{n_{f}}\sum{V_{M_f}} & \quad \textrm{if} \exists f \textrm{where} \exists \|broken\_steps_{f}\| \ne  \varnothing \wedge \exists V_{M_f} \ \ \\
    \varnothing       & \quad \textrm{otherwise} 
  \end{array}
\right.
```


### The Divergence

The  **Divergence** metric is a binary value given for a particular fold $f$,
where $`M_{f_{n_{epoch}}}`$ the value of $M_f$ obtained at the last epoch ($n_{epoch}$)  as follows:

 ```math
D_{M_f}=\left\{ 
  \begin{array}{ c l }
    1 & \quad \textrm{if} \exists V_{M_f} \wedge M_{V_{M_f}} > M_{f_{n_{epoch}}} \\
    0       & \quad \textrm{otherwise} 
  \end{array}
\right.
```


## TABLE CONSTRUCTION
The metrics computed in Table 1 of our paper are obtained with the helps of the [ExtractResults.py](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/eval/ExtractResults.py) that first extract metrics from the server and connect existing artefact to wandb runs available at [https://wandb.ai/celian-ringwald/12ShadesOfRDF](https://wandb.ai/celian-ringwald/12ShadesOfRDF). This script produced the json that retrieve for each folds tokenizer path, and the checkpoints path, and the carbon cost.

We combine both this json file with the wandb data to compute as we describe bellow the three sets of aggregated data : 
* [results_12ShadesSyntax_MEAN_NAIVE.csv](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/eval/results_12ShadesSyntax_MEAN_NAIVE.csv)
* [results_12ShadesSyntax_MEAN_WISE.csv](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/eval/results_12ShadesSyntax_MEAN_WISE.csv)
* [results_12ShadesSyntax_STRICT.csv](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/eval/results_12ShadesSyntax_STRICT.csv)

The one used for building the table is  [results_12ShadesSyntax_STRICT.csv](https://github.com/datalogism/12ShadesOfRDFSyntax/blob/main/eval/results_12ShadesSyntax_STRICT.csv). 
