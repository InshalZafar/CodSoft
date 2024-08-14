# Load required libraries
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(xgboost)
library(corrplot)
library(gridExtra)
library(cluster)
library(factoextra)

# Load the Iris dataset from the specified path
file_path <- 'C:/Users/PMLS/Desktop/python/Internship tasks/codsoft/task2/IRIS.csv'
iris_df <- read.csv(file_path)

# Initial Data Inspection
cat("First 5 rows of the dataset:\n")
print(head(iris_df))
cat("\nLast 5 rows of the dataset:\n")
print(tail(iris_df))
cat("\nDimensions of the dataset:\n")
print(dim(iris_df))
cat("\nSummary statistics:\n")
print(summary(iris_df))

# Check for outliers using boxplots
p1 <- ggplot(iris_df, aes(x = species, y = sepal_length)) + geom_boxplot() + ggtitle('Sepal Length by Species')
p2 <- ggplot(iris_df, aes(x = species, y = sepal_width)) + geom_boxplot() + ggtitle('Sepal Width by Species')
p3 <- ggplot(iris_df, aes(x = species, y = petal_length)) + geom_boxplot() + ggtitle('Petal Length by Species')
p4 <- ggplot(iris_df, aes(x = species, y = petal_width)) + geom_boxplot() + ggtitle('Petal Width by Species')
grid.arrange(p1, p2, p3, p4, ncol = 2, top = "Boxplots of Sepal and Petal Dimensions by Species")

# Data Cleaning
# Remove duplicate rows to ensure data integrity
iris_df <- iris_df %>% distinct()

# Strip any leading/trailing whitespace from the species column
iris_df$species <- trimws(iris_df$species)

# Check for missing or invalid values
cat("\nMissing values in each column:\n")
print(sapply(iris_df, function(x) sum(is.na(x))))

# Data Exploration
# Convert species to colors
species_colors <- as.numeric(factor(iris_df$species))
# Pairplot to visualize relationships between features
pairs(iris_df[1:4], col = species_colors, main = "Pairplot of Iris Dataset")

# Heatmap to visualize correlation between features
cor_matrix <- cor(iris_df[1:4])
corrplot(cor_matrix, method = "color", addCoef.col = "magenta", tl.col = "lightgreen", number.cex = 0.8, title = "Correlation Heatmap of Iris Dataset", mar=c(0,0,1,0))

# Encode the target variable (species) into numeric values
iris_df$species <- as.factor(iris_df$species)

# Split the data into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(iris_df$species, p = .8, 
                                  list = FALSE, 
                                  times = 1)
irisTrain <- iris_df[ trainIndex,]
irisTest  <- iris_df[-trainIndex,]

# Model Training and Evaluation
# Define the models with important hyperparameters
models <- list(
  "Random Forest" = randomForest(species ~ ., data = irisTrain, ntree = 100, mtry = 2, importance = TRUE),
  "Gradient Boosting" = xgboost(data = as.matrix(irisTrain[,-5]), label = as.numeric(irisTrain$species)-1, 
                                nrounds = 100, objective = "multi:softprob", num_class = 3, eta = 0.1, max_depth = 3)
)

# Train and evaluate each model
for (name in names(models)) {
  model <- models[[name]]
  if (name == "Random Forest") {
    pred <- predict(model, irisTest)
    accuracy <- sum(pred == irisTest$species) / nrow(irisTest)
    cat("\nModel:", name, "\n")
    cat("Accuracy:", accuracy, "\n")
    print(confusionMatrix(as.factor(pred), irisTest$species))
  } else {
    pred_prob <- predict(model, as.matrix(irisTest[,-5]))
    pred <- max.col(matrix(pred_prob, ncol=3, byrow=TRUE)) - 1
    accuracy <- sum(pred == as.numeric(irisTest$species)-1) / nrow(irisTest)
    cat("\nModel:", name, "\n")
    cat("Accuracy:", accuracy, "\n")
    print(confusionMatrix(as.factor(pred), as.factor(as.numeric(irisTest$species)-1)))
  }
}

# Hyperparameter Tuning for the Best Model (Random Forest)
# Define the parameter grid for hyperparameter tuning
control <- trainControl(method = "cv", number = 5)
tunegrid <- expand.grid(.mtry = c(1:4))
rf_gridsearch <- train(species ~ ., data = irisTrain, method = "rf", 
                       tuneGrid = tunegrid, trControl = control)
best_model <- rf_gridsearch$finalModel

# Evaluate the best model on the test set
pred <- predict(best_model, irisTest)
accuracy <- sum(pred == irisTest$species) / nrow(irisTest)
cat("\nBest Model after Grid Search:\n")
print(best_model)
cat("Accuracy:", accuracy, "\n")
print(confusionMatrix(pred, irisTest$species))

# Feature Importance
# Visualize feature importances for the best model
importance <- varImp(best_model, scale = FALSE)
plot(importance, main = "Feature Importances of Best Model")

# Clustering with KMeans
set.seed(42)
kmeans_model <- kmeans(iris_df[1:4], centers = 3, nstart = 25)
iris_df$cluster <- as.factor(kmeans_model$cluster)

# Plotting the clusters
fviz_cluster(kmeans_model, data = iris_df[1:4], 
             geom = "point",
             stand = FALSE,
             ellipse.type = "convex",
             ggtheme = theme_minimal())
# Decision Boundary Visualization
plot_decision_boundaries <- function(data, model, feature1, feature2, x_label, y_label) {
  grid <- expand.grid(
    x = seq(min(data[[feature1]]) - 1, max(data[[feature1]]) + 1, length.out = 100),
    y = seq(min(data[[feature2]]) - 1, max(data[[feature2]]) + 1, length.out = 100)
  )
  names(grid) <- c(feature1, feature2)
  grid$pred <- as.factor(predict(model, grid, type = "class"))
  
  ggplot() +
    geom_point(data = data, aes_string(x = feature1, y = feature2, color = "species"), size = 3) +
    geom_tile(data = grid, aes_string(x = feature1, y = feature2, fill = "pred"), alpha = 0.3) +
    scale_fill_manual(values = c("pink", "skyblue", "grey")) +
    labs(x = x_label, y = y_label, title = "Decision Boundary Visualization") +
    theme_minimal()
}

# Train a simple Random Forest model for visualization
rf_vis_model <- randomForest(species ~ sepal_length + sepal_width, data = iris_df, ntree = 100)

# Plot decision boundaries for Random Forest
plot_decision_boundaries(iris_df, rf_vis_model, "sepal_length", "sepal_width", "Sepal Length", "Sepal Width")

# Comprehensive Report
report <- paste("Final Model Report:\n",
                "Best Model:", best_model, "\n",
                "Accuracy:", accuracy, "\n",
                "Confusion Matrix:\n", capture.output(confusionMatrix(pred, irisTest$species)), "\n")
cat(report)
