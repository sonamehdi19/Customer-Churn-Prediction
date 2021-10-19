library(tidyverse)
library(scorecard)
library(skimr)
library(caret)
library(inspectdf)
library(glue)
library(highcharter)
library(h2o)

raw<-read_delim('bank-full (1).csv', delim=';')

raw %>% glimpse()

raw$y <- raw$y %>% 
  factor(levels = c('yes','no'),
         labels = c(1,0))

sapply(raw, class)

#checking the target datatype and its values 
raw %>% glimpse()   
raw$y%>% unique() %>% length()     

#target balance 
raw$y %>% table() %>% prop.table()

# ----------------------------- Data Preprocessing -----------------------------
raw %>% inspect_na()
#no null value in the dataset

df.num <- raw %>% select_if(is.numeric) 

df.chr <- raw %>%
  mutate_if(is.character,as.factor) %>% 
  select_if(is.factor) %>% 
  select(y,everything())

#Firstly, we start with processing numerical values 
# Outliers
num_vars <- df.num %>% names()

for_vars <- c()
for (b in 1:length(num_vars)) {
  OutVals <- boxplot(df.num[[num_vars[b]]], plot=F)$out
  if(length(OutVals)>0){
    for_vars[b] <- num_vars[b]
  }
}

for_vars <- for_vars %>% as.data.frame() %>% drop_na() %>% pull(.) %>% as.character()
for_vars %>% length()

for (o in for_vars) {
  OutVals <- boxplot(df.num[[o]], plot=F)$out
  mean <- mean(df.num[[o]],na.rm=T)
  
  o3 <- ifelse(OutVals>mean,OutVals,NA) %>% na.omit() %>% as.matrix() %>% .[,1]
  o1 <- ifelse(OutVals<mean,OutVals,NA) %>% na.omit() %>% as.matrix() %>% .[,1]
  
  val3 <- quantile(df.num[[o]],0.75,na.rm = T) + 1.5*IQR(df.num[[o]],na.rm = T)
  df.num[which(df.num[[o]] %in% o3),o] <- val3
  
  val1 <- quantile(df.num[[o]],0.25,na.rm = T) - 1.5*IQR(df.num[[o]],na.rm = T)
  df.num[which(df.num[[o]] %in% o1),o] <- val1
}

# One Hote Encoding for categorical variables 
ohe <- dummyVars(" ~ .", data = df.chr[,-1]) %>% 
  predict(newdata = df.chr[,-1]) %>% 
  as.data.frame()

df <- cbind(df.chr[1],ohe,df.num) 

#correcting the format of column names 
names(df) <- names(df) %>% 
  str_replace_all(" ","_") %>%
  str_replace_all("-","_") %>%
  str_replace_all("\\(","_") %>% 
  str_replace_all("\\)","") %>%
  str_replace_all("\\/","_") %>% 
  str_replace_all("\\:","_") %>% 
  str_replace_all("\\.","_") %>% 
  str_replace_all("\\,","_")

# --------------------------------- Modeling ---------------------------------

# Weight Of Evidence ----

# IV (information values) 
iv <- df %>% 
  iv(y = 'y') %>% as_tibble() %>%
  mutate(info_value = round(info_value, 3)) %>%
  arrange(desc(info_value))

# Exclude not important variables 
ivars <- iv %>% 
  filter(info_value>0.02) %>% 
  select(variable) %>% .[[1]] 

df.iv <- df %>% select(y,ivars)

df.iv %>% dim()        #checking final dimension of dataf

# woe binning 
bins <- df.iv %>% woebin("y")   #dividing duration variable into bins 

bins$duration%>% as_tibble()
bins$duration%>% woebin_plot()

# breaking data into train and test & converting into woe values
dt_list <- df.iv %>% 
  split_df("y", ratio = 0.8, seed = 123)

train_woe <- dt_list$train %>% woebin_ply(bins) 
test_woe <- dt_list$test %>% woebin_ply(bins)

names <- train_woe %>% names() %>% gsub("_woe","",.)                   
names(train_woe) <- names; 
names(test_woe) <- names;
train_woe %>% inspect_na() %>% tail(2) ; 
test_woe %>% inspect_na() %>% tail(2);
#as there was no missing values in the dataset, the results are zero for all

# Multicollinearity ----

# coef_na
#determining the target and feature variables 
target <- 'y'
features <- train_woe %>% select(-y) %>% names()

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")
glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features %in% coef_na]
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")

# VIF (Variance Inflation Factor) 
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")

while(glm %>% vif() %>% arrange(desc(gvif)) %>% .[1,2] >= 1.5){
  afterVIF <- glm %>% vif() %>% arrange(desc(gvif)) %>% .[-1,"variable"]
  afterVIF <- afterVIF$variable
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = train_woe, family = "binomial")
}

glm %>% vif() %>% arrange(desc(gvif)) %>% pull(variable) -> features 

# Modeling with GLM ----
h2o.init()

train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
test_h2o <- test_woe %>% select(target,features) %>% as.h2o()

model <- h2o.glm(
  x = features, y = target, family = "binomial", 
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T)

#elimination of features based on p-value
while(model@model$coefficients_table %>%
      as.data.frame() %>%
      select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] >= 0.05){
  model@model$coefficients_table %>%
    as.data.frame() %>%
    select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%   #we should not consider features that have null p-value, bec the calc were not carried on
    .[-1,] %>%                     #on them to make decision
    arrange(desc(p_value)) %>%
    .[1,1] -> v       #v-ye set olundu 
  features <- features[features!=v]      #p-valuesi 0.05 den yuxari olan featureleri atiriq 
  
  
  
  train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
  test_h2o <- test_woe %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target, family = "binomial", 
    training_frame = train_h2o, validation_frame = test_h2o,
    nfolds = 10, seed = 123, remove_collinear_columns = T,
    balance_classes = T, lambda = 0, compute_p_values = T)
}

model@model$coefficients_table %>%
  as.data.frame() %>%
  select(names,p_value) %>%
  mutate(p_value = round(p_value,3))

model@model$coefficients %>%
  as.data.frame() %>%
  mutate(names = rownames(model@model$coefficients %>% as.data.frame())) %>%
  `colnames<-`(c('coefficients','names')) %>%
  select(names,coefficients)

#orange pie for noticing visually the importance of features on predicting the target variable 
h2o.varimp(model) %>% as.data.frame() %>% .[.$percentage != 0,] %>%
  select(variable, percentage) %>%
  hchart("pie", hcaes(x = variable, y = percentage)) %>%
  hc_colors(colors = 'orange') %>%
  hc_xAxis(visible=T) %>%
  hc_yAxis(visible=T)

# ---------------------------- Evaluation Metrices ----------------------------

# Prediction & Confusion Matrix
pred <- model %>% h2o.predict(newdata = test_h2o) %>% 
  as.data.frame() %>% select(p1,predict)

#F1 score 
model %>% h2o.performance(newdata = test_h2o) %>%
  h2o.find_threshold_by_max_metric('f1')

#plotting ROC curve 
eva <- perf_eva(
  pred = pred %>% pull(p1),
  label = dt_list$test$y %>% as.character() %>% as.numeric(),
  binomial_metric = c("auc","gini"),
  show_plot = "roc")

# Check overfitting ----
#auc and gini scores for both train and test data 
model %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)

#As the results for the test and train data are similar, it shows that there is no overfitting problem.
