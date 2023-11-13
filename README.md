# Causal_Inference_CallUserRetention

This repository provides a guideline to perform causal inference on the customer's retention based on the telecom churn data publicly available from Kaggle. The method used here mainly referred to Guelman and Guillén (2014). This project explored the causal effect of retention calls on customers' retention, including building lease models for each treatment (i.e., retention calls or not), training a propensity model for each user to receive a treatment (i.e., given we have only one treatment-comparison pairs, we just need to train one propensity model), matching customer pairs so that each pair includes one customer from treatment group and the other one from the comparison group, and measuring the causal effect of treatment on customer's retention.

## Building lease models for each treatment
This project employed 'RetentionCalls' variable to define the treatment of each customer, i.e., customers having received retention calls were denoted as treatment, while those without referred to comparison group. With that, data was divided into treatment and comparison groups, and a lapse model was trained for each group as illustrated in globalLapseModel.py script.

## Training a propensity model
This step utilized treatment as the target variable to train a model measuring how likely a customer to receive certain treatment, i.e., retention calls or not. This training will not use the churn result as a feature to train the model. As in the lapse model, 'RetentionCalls' was edited to denote the treatment variable, and was further used as target variable to train the propensity model. One thing to note here is that the goal to train the propensity model is to obtain estimates of the propensity score that statistically balance the covariates between each treatment dichotomy (e.g., received retention calls and not), rather than train a model as accurate as possible to reflect each customer’s probability to receive a treatment (Guelman and Guillén 2014). To achieve this goal, this project used the average standardized absolute mean difference in the covariates (ASAM) to evaluate the propensity model (McCaffrey et al. 2004). The match_causalEffectMeasure.py script covers the propensity score model and the computation of ASAM, but the retraining of propensity score model to minimize the ASAM was not included. Users need to perform several iterations to train their propensity score model to minimize their ASAM by needs.

## Matching
According to Guelman and Guillén (2014), this project used the Mahalanobis distance to evaluate the similarity among customers, including the propensity score as an additional covariate and propensity score calipers. There are many other available matching options as summarized in Guelman and Guillén (2014) and Mahalanobis distance is an example among them used for this demonstration. You can see the section ‘Calculate the Mahalanobis distance matrix’ from the match_causalEffectMeasure.py script.

## Measuring causal treatment effect
With paired customers from treatment and comparison groups, this step stared with extracting their lapse/churn probabilities from the Lapse Model in step 1. Then, computed the mean of the difference between treatment and comparison groups to measure the causal effect of treatment, retention calls, on customer’s retention. 

## Data 
Public data from Kaggle: https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom?select=cell2celltrain.csv

## Requirements
The requirements.txt file specifies the package versions used in the scripts.

## References
1. Guelman, L., & Guillén, M. (2014). A causal inference approach to measure price elasticity in automobile insurance. Expert Systems with Applications, 41(2), 387-396.. 
2. McCaffrey, D. F., Ridgeway, G., & Morral, A. R. (2004). Propensity score estimation with boosted regression for evaluating causal effects in observational studies. Psychological methods, 9(4), 403.
