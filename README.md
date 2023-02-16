# practicum_final_project
Final graduation project - Yandex Practicum DataScience Bootcamp
# REPORT
## Predicting customer churn


**Brief overview**
In this project, we are building a model for a telecom operator, Niedinogorazryva.com, who wants to predict customer churn. Since attracting a new client is expensive, if it turns out that the user plans to leave, he will be offered promotional codes and special discounts.

We have personal data about some clients, information about their tariffs and contracts.

**Project goal**: Build a model with a predictive power in terms of the ROC-AUC metric of at least 0.85

We have personal data about some clients, information about their tariffs and contracts.

**Overview of the services**

The operator provides two main types of services:

1. Fixed telephone connection. It is possible to connect a telephone set to several lines at the same time.
2. Internet. The connection can be of two types: via a telephone line (DSL *, * from the English * digital subscriber line *, "digital subscriber line") or fiber optic cable (* Fiber optic *). 

The following services are also available:

- Internet security: antivirus (*DeviceProtection*) and blocking of unsafe sites (*OnlineSecurity*);
- Dedicated technical support line (*TechSupport*);
- Cloud file storage for data backup (*OnlineBackup*);
- Streaming TV (*StreamingTV*) and movie catalog (*StreamingMovies*).

Clients can pay for services every month or sign a contract for 1-2 years. Various payment methods and the possibility of receiving an electronic check are available


**What do we do?**

1. Open the data and do the primary research (EDA)
     - Fill in the gaps, check the correctness of types, remove duplicates
     - We study the data, do the initial analytics, build graphs of the distribution of individual variables and the correlation matrix
3. Divide into test and training samples (0.25 / 0.75)
4. Create additional features, make a preprocessor
     - Analyze for multicollinearity, feature selection
4. Train multiple models
     - Logistic regression, random forest, gradient boosting
     - We carry out a selection of hyperparameters using cross-validation
6. We select models according to the ROC-AUC metric (at least 0.85)
7. Feature Importance Analysis
8. Final conclusions
     - Recommend the model to be used by the client
     - Making other recommendations on observations from the data
    
## EDA

* We have 7043 customer data samples. At the same time, not all clients use all services, since there are 6361 data on telephone contracts, and 5517 Internet
* We do not see any obvious gaps in the data
* For analysis, we need a target feature that we create - whether the client terminated the contract or not:
     * 1 yes, terminated
     * 0 no, not terminated


### Merging tables, filling gaps, changing data types
* The data structure is quite linear - everything is merged by the customer_id key, so we merge tables by this key
   
- We filled in the gaps:

     * From the data, information was lost which contract each client has - Internet or phone.
     * By internet, by the connection type column - we will add "no_internet" instead of empty values, so we will recover information - who has internet and who doesn't.
     * Do the same on the phone (no_telephone)
     * For services, fill in No in case of empty cells - this means that the service is not provided
    
**Data type change**

* We also noticed the wrong type for total_charges because it contains empty strings as a space ' '
* Since there are only 11 such lines, we suggest simply deleting them

### Data analysis

* We have added a feature for how long the contract was opened - `client_years`
* We see that the distribution is similar to bimodal - many new clients and many old clients who are about 6 years old

![image.png](attachment:image.png)

### Target correlation with variables:
* Constructed a correlation matrix of target phik with other values.
* We also indicate the stat significance. Stat significance for fi correlation works differently, it shows the number of standard deviations and small values indicate insignificance, and high values vice versa:
     - Only gender, the presence of a telephone or lines did not become significant. (compare with 5 standard deviations)
    
**We observe a linear dependence of the average force (0.3-0.5)**:
- `paperless_billing`, `payment_method` Those who use electronic billing leave the company more often, perhaps these people use the Internet more often to compare prices, for example, with competitors
- `client_years`, `total_charges` - older clients leave less often, so they accumulate more total_charges
- `monthly_charges` - shows the presence of a dependency, it is more difficult to guess with a dependency sign
    
Other coefficients are too low or not significant

### Correlation matrix

* We also explored linear dependencies between predicates using the correlation matrix phik
     * `monthly_charges` - linearly dependent on additional services such as online_backup, streaming services. High dependence on the type of Internet
     * `total_charges` - also dependent on additional services, there is a dependence on how long they have been clients
     * `internet_serivice` - high dependency on phone and lines
     * `device_protection`, `tech_support` and other services have a linear relationship of medium strength
* All dependencies above stat are significant

![image-2.png](attachment:image-2.png)


**Target depends on the type of phone or Internet service**:
- We see that, on average, the target does not change much from the presence of a phone or several lines (about 0.25)
- **However, the fiber target is much higher than the 0.42 dataset**, this may indicate that the quality of the fiber is not the best, and companies should pay attention to this

![image-3.png](attachment:image-3.png)
![image-4.png](attachment:image-4.png)

### Portrait of an churned client

**Portrait of an exiting client from initial data analysis helped focus further research and facilitate feature selection**

* They pay a higher monthly fee
* They have less total payouts by more than twice the median
*Use optical fiber
* Have been a customer for less than 1 year

## Dealing with class imbalance

- In our case, there is an imbalance of classes but not pronounced (1 to 3). In addition, the metric is ROC-AUC, this metric is sensitive to class imbalance in the sense that guessing the correct values of a rarer class has a large weight, in contrast to accuracy, where the weight is the same. Therefore, we consider the choice of metric to be successful in this exercise.
- We will also use built-in balancing methods in algorithms using class weights

## Feature engineering and selection


### Feature engineering
* In order to improve the quality of the model, we have added several features that, in our opinion, could help with the explanation of the target
     1. `monthly_total_ratio` How much the client pays per month / average per month of the total amount
         * Clients for whom this value is lower buy additional services and most likely close contracts less often
     2. `total_per_person` how much was spent / an estimate of the number of family (1 - one, 2 - partner, 3 - partner and dependent)
         * This estimate better shows the real costs per person under the contract. The entire household uses the Internet and telephone
     3. `monthly_per_person` payment per month / estimated number of family (1 - one, 2 - partner, 3 - partner and dependent)
         * Same logic as above, but for a month
     4. `last_year_payment` How much the client paid in the last 12 months based on full payment
         * Those who paid more in the last 12 months used the company's services more actively. This statistic helps distinguish between old customers (active in the distant past or now)

### VIF selection

**Check for multicollinearity**:

* We use the Variance Inflation Factor method, we calculate the VIF for each numerical predictor in the model.

We use the following conventional interpretation:

  - VIF = 1: No correlation between predictor and other traits
  - VIF between 1 and 5: there is little correlation between predictor and other traits
  - VIF > 5: Strong correlation between predictor and other traits

**Multicollinearity problem**
  * We note the problem of data multicollinearity in the dataset, first of all, it is due to the fact that in the original dataset there are only 2 numeric variables (full payment and per month), which are obviously related to each other.
* We created several features, but they were all correlated, after removing the feature, we are left with two that show a low degree of correlation.
* Final characteristics are the time (years) from the beginning of the contract and the ratio of the monthly payment to the final one, `client_years`, `monthly_total_ratio`.

![image-6.png](attachment:image-6.png)

## Encoding variables and creating a preprocessing pipeline

* We create preprocessing, it is built from
     1. For numerical features:
         1. Creation of features
         2. Standard scaler
     2. For categorical features:
          1. For feature coding we will use One-hot as we don't have many categories of each categorical variable (max 3)
         
## Training and selection of models

* We have chosen three classification models:
     1. Logistic regression
     2. Random Forest
     3. Gradient boosting (katboost)
* We work with class balance using built-in methods inside models for balancing
* The evaluation metric was ROC-AUC, this metric is sensitive to class imbalance in the sense that guessing the correct values of a rarer class is heavily weighted, unlike accuracy, where the weight is the same. Therefore, we consider the choice of metric to be successful in this exercise.
* Calibration of hyperparameters is done using gridsearch and cross-validation on 5 folds

### Best hyperparameters for models

1. Logistic regression:
- `C`: 3.5
- `class_weight`: balanced
- `solver`: lbfgs

2. Random Forest
- `class_weight`: None
- `max_depth`: 10
- `n_estimators`: 400

3 Catboost
- `class_weights`: SqrtBalanced
- `depth`: 4
- `iterations`: 700
- `learning_rate`: 0.07

### Model selection
* Gradient boosting performed best of 0.96 ROC-AUC on cross-validation, which is above the required thrashhold of 0.85
![image-7.png](attachment:image-7.png)

## Feature importances

* We analyzed the significance of features using the built-in catboost methods, which work on the basis of premutation and track the change in the loss function.

The top 2 features are numeric features:
- `monthly_total_ratio` the ratio of the regular monthly payment to the total monthly payment. Here we believe that if the client gets more additional services, then he has less motivation to close the contract.
- `client_years` how long ago the contract was made. As we saw earlier - the longer the client keeps the contract, the less often he leaves it. Since we have data on exits from the contract for the last 4 months, we believe that this value is approximately equal to how many times the client has been a client



<h4><center>Top 10</center></h4>


![image-10.png](attachment:image-10.png)

## Additional score metrics

* For clarity, we have built an error matrix and additional metrics

* According to the error matrix and additional metrics, we see that the model does a good job with both precision and recall, while the precision is higher, which means the model is not particularly mistaken in determining single classes

* Our recommendation for further work is to select a model that optimizes the recall metric - if we want to find exactly all customers who are going to terminate the contract

![image-12.png](attachment:image-12.png)


* `Accuracy score`: 0.92
* `Precision score`: 0.86
* `Recall score`: 0.82
* `F1 score`: 0.84


<h4><center>ROC-AUC</center></h4>


![image-11.png](attachment:image-11.png)


## Test set predictions

* On the test sample, the result is 0.96, which is higher than the prog metric and coincides with the validation. We consider the model successful.

## General conclusions

**In this work, we analyzed the data and developed a model for No Gap.com that predicts which customers are most likely to leave a contract and allows the marketing department to focus on retaining those customers**


**It was done according to plan, but there were difficulties**
* The work was done according to the original plan, but we encountered difficulties - the main problem was the multicollinearity of numerical variables and their small number (only two). It can resolve and improve the model if additional data is provided

**Which customers should the marketing department focus on**
*On those provided by the model, but analysis has shown that clients who have been clients for less than a year are more likely to terminate contracts. As well as customers who pay higher monthly commissions.

**Further work and data recommendations:**
- Data on terminations of contracts only for the last 4 months. Check that there is no error and expand the sample for a large number of months
- We also recommend expanding the data on clients and adding at least the age of the client, perhaps his address, because if the client uses a phone, the connection in his area may not be so good and people leave a certain area
* Our recommendation for further work is to select a model that optimizes the recall metric - if we want to find exactly all customers who are going to terminate the contract

**Total, we have developed a successful model for the company and important insights from data analysis that will help retain customers and earn money. Further research and improvement of the model is possible upon receipt of additional data indicated above**
