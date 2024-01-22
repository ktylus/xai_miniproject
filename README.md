## Reproduction of Exploring Multi-Task Learning for Explainability paper

The goal of this repository is to reproduce results from Exploring Multi-Task Learning for Explainability paper.

### Data preprocessing

### Parameter tuning
We used Ray Tune package for hyperparameter tuning.\
The code can be found in parameter_tuning.py.\
Results can be found in tuning_results.txt.

### Global Explainability evaluation
We trained the models as described in the paper and used our implementation of the described metrics.\
The general tendency is that our results for the base model are better, but the results for the surrogate model (explainability metrics) are worse.\
The results can be seen here: ![results](https://github.com/ktylus/xai_miniproject/assets/30349386/9c30f0e8-c822-4a07-81f7-a0130ea235cb)

### Local Explainability evaluation

### Difficulties
We have not found enough information in the paper in order to precisely reproduce the results.\
Therefore, we had to improvise in some places.\
Here, we briefly describe our difficulties during the reproduction of the results.\
The data processing steps are not described in the paper, we tried to process the datasets to the best of our abilities, but we have no knowledge of the steps taken by the authors, that produced the results shown in the paper.\
The MSE metric in the results has the value between 0 and 1, however the targets in regression datasets are of different scale. We assumed that the target was normalized, however this is not clear.\
The authors mentioned that the hyperparameters of the base model, such as number of layers or their size (also probably the batch size), have been found by tuning, however their values are not shown.
