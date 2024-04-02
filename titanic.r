#load library
library(dplyr)
library(ggplot2)
library(heatmaply)
library(e1071)
library(caret)
library(pROC)
library(kernlab)


# data load
titanic <- read.csv("titanic.csv", header=TRUE)
head(titanic)

#결측치 확인
summary(is.na(titanic))

# 결측치 대체 하기
missing_index <- which(is.na(titanic$Age))

# 개수 확인하는거
length(missing_index)

# 평균으로 대체하기
titanic$Age[missing_index] <- mean(titanic$Age, na.rm=TRUE)

#기본 데이터 분석
# 성별에 따른 생존자와 사망자 수 계산
survival_count <- table(titanic$Sex, titanic$Survived)
#데이터 프레임: ggplot2 사용에 유리함, 시각화 쉬움움
survival_count <- as.data.frame(survival_count)
colnames(survival_count) <- c("sex", "survived", "count")

#그래프 그리기
# 1. ggplot() 함수를 사용해 새로운 그래프 객체 생성. 레이어 설정
ggplot(survival_count, aes(x=sex, y=count, fill=factor(survived))) +
# 2. geom_bar: 막대 그래프 생성. stat="identity" 입력 데이터를 그대로 사용. 막대 그린다
# position dodge: 막대를 나란히 그린다 
     geom_bar(stat="identity", position="dodge")+
  labs(title="survival count by gender",
       x="sex",
       y="count",
       # fill: 범례의 제목을 설정함 
       fill="survived")+
# theme_minimal(): 그래프의 테마 설정. 최소한의 테마 선택한다 의미0
  theme_minimal()

# 클래스 별 사망자, 생존자 비교 
class_count <- table(titanic$Pclass, titanic$Survived)
class_count <- as.data.frame(class_count)
colnames(class_count) <- c("class", "survived", "count")

ggplot(class_count, aes(x=class, y=count, fill=factor(survived)))+
  geom_bar(stat="identity", position="dodge")+
  labs(title="survived count by class",
       x="class",
       y="count",
       fill="survived")+
  theme_minimal()

# 계속 오류남 > 해결, cut 할때 사용하는 데이터를 수치형 데이터 쓰기
# 나잇대별 생존자, 사망자 비교
age_count <- table(titanic$Age, titanic$Survived)
age_count <- as.data.frame(age_count)
colnames(age_count) <- c("age", "survived", "count")

# age_count$age를 수치형으로 변환
age_numeric <- as.numeric(age_count$age)


# 나이 그룹 설정
age_group <- cut(age_numeric, breaks = c(0, 9, 19, 29, 39, 49, 59, 69, 79, Inf),
                 labels = c("under 10", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"))


# age_group 열 추가
age_count$age_group <- age_group

# 그래프 그리기 
ggplot(age_count, aes(x=age_group, y=count, fill=factor(survived)))+
  geom_bar(stat="identity")+
  labs(title="survived counts by age group",
       x="age_group",
       y="count")+
  theme_minimal()

# parch 별 생존율율
parch_count <- table(titanic$Parch, titanic$Survived)
parch_count <- as.data.frame(parch_count)
colnames(parch_count) <- c("parch", "survived", "count")

ggplot(parch_count, aes(x=parch, y=count, fill=factor(survived)))+
  geom_bar(stat="identity")+
  labs(title="survived count by parch",
       x="parch",
       y="count",
       fill="survived")+
  theme_minimal()


# 표 요금 별 생존율
fare_count <- table(titanic$Fare, titanic$Survived)
fare_count <- as.data.frame(fare_count)
colnames(fare_count) <- c("fare", 'survived', 'count')

fare_numeric <- as.numeric(fare_count$fare)

# 항시 구간 설정할 때 맨 앞에 0 넣어주기 
# 구간별로 값 지정 (0~8, 8~15, 15~32 이렇게 가는거라서 숫자가 1개 더 많음 )
fare_group <- cut(fare_numeric, breaks = c(0, 8, 15, 32, 100, Inf),
                  labels = c("under 8", "8-15", "15-32", "32-100", "100+"))

# 그룹 추가 
fare_count$fare_group <- fare_group

# 그래프 그리기
ggplot(fare_count, aes(x=fare_group, y=count, fill=factor(survived)))+
  geom_bar(stat="identity")+
  labs(title="survived by fare",
       x="fare group",
       y="counts",
       fill="survived")+
  theme_minimal()


# 독립변수들 끼리 상관성 correlation 보기
# corr() 적용하려면 numeric 형식이여야 한다.
# numeric_vars: titanic 데이터에서 숫자형 변수만 선택하는 것 
# sapply: lappy랑 비슷한데 벡터나 데이터 프레임의 각 요소에 함수 적용하는 것
# 결과를 벡터나 데이터 프레임으로 반환. 
## titanic 데이터 프레임의 각 열에 is.numeric() 함수를 적용해 숫자형 확인
# 이후 그 결과를 벡터로 반환함,. 
numeric_vars <- titanic[,sapply(titanic, is.numeric)]

# 상관계수 행렬 계산
correlation_matrix <- cor(numeric_vars)
print(correlation_matrix)

# 상관계수 행렬을 히트맵으로 시각화
heatmaply(correlation_matrix, labels=TRUE)

# 범주형 데이터 수치형으로 바꾸기 > ifelse 사용함 
Gender <- ifelse(titanic$Sex=='female',1,0)
titanic$Gender <- Gender



### 데이터 시각화랑 기초통계 분석 하는 이유
# 1. 독립변수 간 상관관계가 어떻게 되어있는지 파악
# 2. 데이터 인사이트 파악 가능




### 모델 적용해보기
# 1. 서포트 벡터 머신 
# 분류: 종속변수가 범주형 데이터인 경우 적용하기 매우 좋음
# 예: 타이타닉 생존 여부(0/1), 대학원 합격 예측 (0/1)

# 1단계: 데이터 불러오기
titanic

# 2단계: 데이터 전처리
## 범주형을 수치형으로 바꾸기: 원 핫 인코딩
# 각 범주를 이진 형태의 벡터로 변환하는 것. 각 범주에 해당하면 1, 아니면 0
# 범주형 데이터가 각 범주간에 상호 연관성이 없을 때 사용. 
# 범주가 많거나 레이블이 순서에 따라 순위가 매겨지지 않을 때 유용

# Embarked 결측 확인하기

#결측치 확인

# 빈칸 대체 하기
space_index <- which(titanic$Embarked=="")
titanic$Embarked[space_index] <- "S"

## 원-핫 인코딩 할 방법
# 1. 원 핫 인코딩 열 추출
##방법: 1. a_levels <- unique(data$a)
##방법: 2. encoded_a <- matrix(0, nrow=(data), ncol=length(a_levels))
## ** 여기서 0은 encoded_a 행렬의 모든 요소를 초기화 하는데 사용함. 

# 2. 각 열에 대한 인코딩 실행
#방법: for(i in 1:length(a_levels)) {
#        encoded_a[,1] <- ifelse(data$a == a_levels[i],1,0)
#       }

# 3. 데이터 정리
# 방법: a_df <- as.data.frame(encoded_a)
#       colnames(encoded_a) <- c("a","b")
#       data$a <- a_df  t  # 새로운 수치형 데이터 입력 완료 


# 원-핫 인코딩할 열 추출
embark_levels <- unique(titanic$Embarked)
encoded_embark <- matrix(0, nrow = nrow(titanic), ncol = length(embark_levels))

# 각 열에 대해 원-핫 인코딩 수행
for (i in 1:length(embark_levels)) {
  encoded_embark[, i] <- ifelse(titanic$Embarked == embark_levels[i], 1, 0)
}

# 생성된 원-핫 인코딩 열을 데이터프레임에 추가
# embarked 원 핫 인코딩
encoded_embark_df <- as.data.frame(encoded_embark)
names(encoded_embark_df) <- paste("encoded_embark", embark_levels, sep = "_")
titanic$Embarked <- encoded_embark_df


# 수치형인 값들만 뽑아내기
numeric_vars <- titanic %>%
  select(-1) %>%
  select_if(function(col) is.numeric(col)) 
  


# 3단계: 데이터 분할
set.seed(206) # 재현성 계산을 위해 시드 설정함
# train index로 행 70% 뽑아내고, 그 행에 대한 데이터 뽑기 
train_index <- sample(1:nrow(titanic), 0.7*nrow(titanic))
train_data <- numeric_vars[train_index,]
test_data <- numeric_vars[-train_index,]



# 4단계: SVM 모델 구축
# SVM 커널에서 뭐가 제일 좋은지 찾아가는 방법 

# 그리드 탐색을 위한 하이퍼파라미터 그리드 설정
hypergrid <- expand.grid(
  kernel = c("linear", "polynomial", "radial", "sigmoid"),
  cost = c(0.01, 0.1, 1, 10)
)

# 교차 검증을 위한 제어 매개변수 설정
ctrl <- trainControl(method = "cv", number = 5)

# 그리드 탐색 및 모델 훈련
svm_model <- train(
  Survived ~ ., 
  data = train_data,
  method = "svm",
  trControl = ctrl,
  tuneGrid = hypergrid
)

# 최적의 커널과 하이퍼파라미터 출력
# 가장 많이 쓰이는 커널: 선형/방사형 커널 
print(svm_model$bestTune)


# scale=false: 상수의 값. 모든 샘플에 대해 동일한 값을 갖는다 
# 이거는 선형 커널 사용. 
svm_model <- svm(Survived ~ .,
                 data=train_data, kernel="linear", scale=FALSE)

# 예측
svm_pred <- predict(svm_model, test_data)
# 이거 가동할 때 오류가 뜨는데 이거는 매우 일반적인 경고니까 그냥 무시하기.
# 그리고 예측값이 0/1이 아니라 소숫점으로 나오는데 한번 더 변환이 필요함

# 소숫점 데이터 0/1로 변환하기 
svm_pred_binary <- ifelse(svm_pred >= 0.5, 1, 0)


# 모델 평가
# 정확도(accuracy): 모든 샘플 중 올바르게 분류된 샘플의 비율
# 정밀도(precision): +로 예측한 샘플 중 실제 +의 비율, >>> + 예측의 정확도
# 재현율(recall): 실제 +인 값들 중 모델이 실제로 +로 예측한 비율 >> + 샘플을 식별하는 능력
# F1점수(F1-score): 정밀도와 재현율의 조화 평균, 불균형한 데이터셋에서 모델 평가에 유용

accuracy <- mean(svm_pred_binary == test_data$Survived)

predicted <- svm_pred_binary
actual <- test_data$Survived
results <- table(actual, predicted)
print(results)

# 혼동행렬(confusion matrix): 모델의 성능을 평가하기 위해 사용되는 표 
# 모델의 예측 결과와 실제 값 사이의 관계를 요약해서 보여줌

#                      Predicted Negative | Predicted Positive |          
#  -----------------------------------------------------
#  Actual Negative   |        TN          |         FP         |
#  -----------------------------------------------------
#  Actual Positive   |        FN          |         TP         |
#  -----------------------------------------------------

# True Positive(TP): 모델 + , 실제 + 
# False Positive (FP): 모델 +, 실제 - 
# True Negative (TN): 모델 -, 실제 -
# False Negative (FN): 모델 -, 실제 +


# 혼동 행렬 및 평가 지표 계산
cm <- confusionMatrix(factor(predicted), factor(actual))
print(cm)

# ROC 곡선 시각화
plot(roc(actual, predicted), col = "blue")


# 예측 값의 빈도 구하기
prediction_counts <- table(svm_pred_binary)

# 시각화 해서 결과 보기 
barplot(prediction_counts)



