#load library
library(dplyr)
library(ggplot2)
library(heatmaply)
library(e1071)
library(caret)
library(pROC)
library(kernlab)


# load data
titanic <- read.csv("titanic.csv", header=TRUE)
head(titanic)

# Check for missing values
summary(is.na(titanic))

# Replace missing values
missing_index <- which(is.na(titanic$Age))
length(missing_index)
titanic$Age[missing_index] <- mean(titanic$Age, na.rm=TRUE)

# Basic data analysis
# Calculate the count of survived and deceased by gender
survival_count <- table(titanic$Sex, titanic$Survived)
survival_count <- as.data.frame(survival_count)
colnames(survival_count) <- c("sex", "survived", "count")

# Plot the graph
ggplot(survival_count, aes(x=sex, y=count, fill=factor(survived))) +
  geom_bar(stat="identity", position="dodge")+
  labs(title="Survival count by gender",
       x="Sex",
       y="Count",
       fill="Survived")+
  theme_minimal()

# Comparison of survival by class
class_count <- table(titanic$Pclass, titanic$Survived)
class_count <- as.data.frame(class_count)
colnames(class_count) <- c("class", "survived", "count")

ggplot(class_count, aes(x=class, y=count, fill=factor(survived)))+
  geom_bar(stat="identity", position="dodge")+
  labs(title="Survived count by class",
       x="Class",
       y="Count",
       fill="Survived")+
  theme_minimal()

# Comparison of survival by age group
age_count <- table(titanic$Age, titanic$Survived)
age_count <- as.data.frame(age_count)
colnames(age_count) <- c("age", "survived", "count")

age_numeric <- as.numeric(age_count$age)

age_group <- cut(age_numeric, breaks = c(0, 9, 19, 29, 39, 49, 59, 69, 79, Inf),
                 labels = c("under 10", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"))

age_count$age_group <- age_group

ggplot(age_count, aes(x=age_group, y=count, fill=factor(survived)))+
  geom_bar(stat="identity")+
  labs(title="Survived counts by age group",
       x="Age Group",
       y="Count")+
  theme_minimal()

# Comparison of survival by parch
parch_count <- table(titanic$Parch, titanic$Survived)
parch_count <- as.data.frame(parch_count)
colnames(parch_count) <- c("parch", "survived", "count")

ggplot(parch_count, aes(x=parch, y=count, fill=factor(survived)))+
  geom_bar(stat="identity")+
  labs(title="Survived count by parch",
       x="Parch",
       y="Count",
       fill="Survived")+
  theme_minimal()


# Comparison of survival by fare
fare_count <- table(titanic$Fare, titanic$Survived)
fare_count <- as.data.frame(fare_count)
colnames(fare_count) <- c("fare", 'survived', 'count')

fare_numeric <- as.numeric(fare_count$fare)

fare_group <- cut(fare_numeric, breaks = c(0, 8, 15, 32, 100, Inf),
                  labels = c("under 8", "8-15", "15-32", "32-100", "100+"))

fare_count$fare_group <- fare_group

ggplot(fare_count, aes(x=fare_group, y=count, fill=factor(survived)))+
  geom_bar(stat="identity")+
  labs(title="Survived by fare",
       x="Fare Group",
       y="Counts",
       fill="Survived")+
  theme_minimal()


# Checking correlation between independent variables
numeric_vars <- titanic[,sapply(titanic, is.numeric)]

# Calculating correlation matrix
correlation_matrix <- cor(numeric_vars)
print(correlation_matrix)

# Visualizing correlation matrix as a heatmap
heatmaply(correlation_matrix, labels=TRUE)

# Converting categorical data to numeric using ifelse
Gender <- ifelse(titanic$Sex=='female',1,0)
titanic$Gender <- Gender


### The reason for data visualization and basic statistical analysis
# 1. Understanding the correlation between independent variables
# 2. Insight into the data

### Applying models
# 1. Support Vector Machine
# Classification: Very useful when the dependent variable is categorical
# Examples: Predicting survival on the Titanic (0/1), Predicting acceptance to graduate school (0/1)

# Step 1: Load data
titanic

# Step 2: Data preprocessing
# One-hot encoding: Transforming each category into a binary vector
# Used when categories have no inherent order and are not related
# Useful for datasets with many categories or where labels are not ranked
# Embarked has missing values
space_index <- which(titanic$Embarked=="")
titanic$Embarked[space_index] <- "S"

# Extract columns for one-hot encoding
embark_levels <- unique(titanic$Embarked)
encoded_embark <- matrix(0, nrow = nrow(titanic), ncol = length(embark_levels))

# One-hot encoding for each column
for (i in 1:length(embark_levels)) {
  encoded_embark[, i] <- ifelse(titanic$Embarked == embark_levels[i], 1, 0)
}

# Add encoded columns to dataframe
encoded_embark_df <- as.data.frame(encoded_embark)
names(encoded_embark_df) <- paste("encoded_embark", embark_levels, sep = "_")
titanic$Embarked <- encoded_embark_df

# Select only numeric values
numeric_vars <- titanic %>%
  select(-1) %>%
  select_if(function(col) is.numeric(col)) 
  

# Step 3: Data splitting
set.seed(206)
train_index <- sample(1:nrow(titanic), 0.7*nrow(titanic))
train_data <- numeric_vars[train_index,]
test_data <- numeric_vars[-train_index,]



# Step 4: Building SVM model
# Searching for the best kernel using grid search

# Hyperparameter grid for grid search
hypergrid <- expand.grid(
  kernel = c("linear", "polynomial", "radial", "sigmoid"),
  cost = c(0.01, 0.1, 1, 10)
)

# Control parameters for cross-validation
ctrl <- trainControl(method = "cv", number = 5)

# Grid search and model training
svm_model <- train(
  Survived ~ ., 
  data = train_data,
  method
