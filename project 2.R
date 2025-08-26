################################################################################
# load the packages:
library(data.table)
library(caret)
library(glmnet)
library(splines)
library(ggeffects)
library(R.utils)

################################################################################

# Part 1:Classifying Political Tweets (High-dimensional Text Data)
# 高维、稀疏文本数据 → 逻辑回归 + 正则化 (Ridge/Lasso)

## Objective: Predict whether a tweet is authored by Donald Trump or Bernie Sanders, 
## and identify which words are most discriminative.
## 1.load data:
data_trump <- fread(file = "C:/Users/wjh/Desktop/ML/lab1/trumpbernie.csv")
dim(data_trump) # high-dimensional --> ordinary logistic regression may overfit.

## 2.implement Logistic regression model with glm:
glm_trump <- glm(trump_tweet ~ ., data = data_trump, family = "binomial")
coef(glm_trump)[1010:1050]

## 3.Trainning accuracy:
comparison_df <- data.frame(train_predictions=glm_trump$fitted.values,
                            observed=glm_trump$y)

comparison_df$train_predictions<-ifelse(comparison_df$train_predictions>=0.5,
                                        yes = 1,
                                        no = 0)
nrow(comparison_df[comparison_df$train_predictions==comparison_df$observed,]) /
  nrow(comparison_df)

## 4.Cross-validation with caret
tc <- caret::trainControl(method = 'cv', number = 3)
data_trump$trump_tweet <- as.factor(data_trump$trump_tweet)

set.seed(12345)
estimate_glm <- caret::train(trump_tweet ~ ., 
                             data = data_trump, 
                             method = "glm",
                             trControl = tc)
estimate_glm

## 5.Ridge regression with glmnet:
x <- as.matrix(subset(data_trump, select = -trump_tweet))
y <- data_trump$trump_tweet 

set.seed(12345)
cvglmnet_trump <- cv.glmnet(x = x, 
                            y = y,
                            nfolds = 5,              
                            standardize = TRUE,     
                            family='binomial',       
                            alpha=0,                
                            type.measure = 'class')
cvglmnet_trump  # reduce variance, determine the optimal λ parameter
                # compare the accuracy of ridge regression versus logistic regression.
    
## 6.plot it:
plot(cvglmnet_trump, sign.lambda = 1) 

## 7.interpret coeffients:
best_coefs <- coef(cvglmnet_trump, s = "lambda.min")
coef_df <- as.matrix(best_coefs) 
coef_df <- data.frame(word = rownames(coef_df),
                      coefficient = coef_df[,1])  # Examine which terms lean more towards Trump/Bernie
coef_df <- subset(coef_df, word != "(Intercept)") # Examine which terms lean more towards Trump/Bernie

################################################################################

# Part 2: Predicting Ad Purchases (Low-dimensional Tabular Data)
# 低维、可能非线性 → GAM model

## Objective: Predict whether an individual purchases a product given Age, Gender, Salary, 
## and explore non-linear effects.
## 1.Load dataset"
social_data <- fread(file = "C:/Users/wjh/Desktop/ML/lab1/Kaggle_Social_Network_Ads.csv")
social_data$Purchased <- as.factor(social_data$Purchased)

## 2.Standard logistic regression with 5-fold CV
tc_2 <- trainControl(method = 'cv', number = 5)

set.seed(12345)
estimate_glm_2 <- caret::train(Purchased ~ .,
                               data = social_data,
                               method = "glm",
                               family = "binomial",
                               trControl = tc_2)
estimate_glm_2

## 3.Generalized Additive Models (GAMs) with natural splines
set.seed(12345)
GAM_2 <- caret::train(Purchased ~ ns(Age, df = 2) + ns(Salary, df = 2) + Gender,
                      data = social_data,
                      method = "glm",
                      family = "binomial",
                      trControl = tc_2)  # ns(): natural spline

set.seed(12345)
GAM_3 <- caret::train(Purchased ~ ns(Age, df = 3) + ns(Salary, df = 3) + Gender,
                      data = social_data,
                      method = "glm",
                      family = "binomial",
                      trControl = tc_2)

set.seed(12345)
GAM_4 <- caret::train(Purchased ~ ns(Age, df = 4) + ns(Salary, df = 4) + Gender,
                      data = social_data,
                      method = "glm",
                      family = "binomial",
                      trControl = tc_2)

summary(caret::resamples(x = list(estimate_glm_2,
                                  GAM_2,
                                  GAM_3,
                                  GAM_4)))

## 4.Examine predictive relationships (ggeffects)
final_model <- glm(Purchased ~ ns(Age, df = 3) +  # df=3 is the best GAM specification
                     ns(Salary, df = 3) + 
                     Gender,
                   data = social_data,
                   family = "binomial")

ggpreds <- ggpredict(final_model)
plot(ggpreds)

## 5.summarization:
## GAM is suitable for low-dimensional scenarios (exhibit non-linearity);
## Ridge/Lasso is more appropriate for high-dimensional settings.

################################################################################

