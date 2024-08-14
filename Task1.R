# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(ggplot2)
library(corrplot)
library(tidyverse)
library(dplyr)

# Load the dataset
file_path <- file.path("C:", "Users", "PMLS", "Desktop", "python", "Internship tasks", "codsoft", "task1", "Titanic-Dataset.csv")
titanic_data <- read.csv(file_path)

# Initial Data Exploration
cat("Initial Data Overview\n")
cat("====================\n")
head(titanic_data)      # Preview the first few rows
tail(titanic_data)      # Preview the last few rows
summary(titanic_data)   # Summary statistics for all columns
str(titanic_data)       # Structure of the dataset
cat("\nDimensions of the dataset: ", dim(titanic_data), "\n") # Dimensions of the dataset

# Check for missing values
cat("\nMissing Values Per Column\n")
cat("==========================\n")
missing_values <- sapply(titanic_data, function(x) sum(is.na(x)))
print(missing_values)

# Visualize missing values using ggplot2
missing_df <- data.frame(Feature = names(missing_values), MissingValues = missing_values)
ggplot(data = missing_df, aes(x = reorder(Feature, MissingValues), y = MissingValues, fill = MissingValues)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_gradient(low = "darkred", high = "lightblue") +
  theme_minimal() +
  ggtitle("Missing Values by Feature") +
  xlab("Features") + ylab("Number of Missing Values")

# Data Cleaning and Imputation
# Fill missing 'Age' values with the median (less sensitive to outliers)
titanic_data$Age[is.na(titanic_data$Age)] <- median(titanic_data$Age, na.rm = TRUE)

# Define a mode function for imputation
mode_function <- function(x) {
  return(names(sort(table(x), decreasing = TRUE))[1])
}

# Fill missing 'Embarked' values with mode
titanic_data$Embarked[is.na(titanic_data$Embarked)] <- mode_function(titanic_data$Embarked)

# Drop rows with missing 'Fare' or any other remaining NA values
titanic_data <- na.omit(titanic_data)

# Convert categorical variables to factors
titanic_data$Sex <- as.factor(titanic_data$Sex)
titanic_data$Embarked <- as.factor(titanic_data$Embarked)
titanic_data$Pclass <- as.factor(titanic_data$Pclass)

# Data Exploration after Cleaning
cat("\nData Overview After Cleaning\n")
cat("============================\n")
summary(titanic_data)   # Updated summary statistics
str(titanic_data)       # Updated structure of the dataset

# Outlier Detection: Boxplot for numeric variables
ggplot(titanic_data, aes(x = Pclass, y = Age, fill = Pclass)) +
  geom_boxplot(outlier.color = "orange", outlier.shape = 16, outlier.size = 2) +
  scale_fill_manual(values = c("lightgreen", "blue", "lightpink")) +
  theme_minimal() +
  ggtitle("Outlier Detection: Age by Passenger Class") +
  xlab("Passenger Class") + ylab("Age")

ggplot(titanic_data, aes(x = factor(Survived), y = Fare, fill = factor(Survived))) +
  geom_boxplot(outlier.color = "red", outlier.shape = 16, outlier.size = 2) +
  scale_fill_manual(values = c("yellow", "purple")) +
  theme_minimal() +
  ggtitle("Outlier Detection: Fare by Survival Status") +
  xlab("Survived") + ylab("Fare")


# Visualization 1: Survival Rate by Passenger Class
ggplot(titanic_data, aes(x = Pclass, fill = factor(Survived))) +
  geom_bar(position = "fill", color = "black") +
  scale_fill_manual(values = c("lightpink", "darkred")) +
  theme_minimal() +
  ggtitle("Survival Rate by Passenger Class") +
  xlab("Passenger Class") + ylab("Proportion")

# Visualization 2: Survival Rate by Gender with Background Color
ggplot(titanic_data, aes(x = Sex, fill = factor(Survived))) +
  geom_bar(position = "fill", color = "black") +
  scale_fill_manual(values = c("lightgreen", "red")) +
  theme_minimal(base_family = "Arial", base_size = 8) +
  theme(panel.background = element_rect(fill = "grey")) +
  ggtitle("Survival Rate by Gender") +
  xlab("Gender") + ylab("Proportion")

# Visualization 3: Age Distribution with Survival Overlay (Scatter Plot)
ggplot(titanic_data, aes(x = Age, y = ..density.., color = factor(Survived))) +
  geom_point(stat = "density", position = "identity", alpha = 0.3) +
  scale_color_manual(values = c("darkgreen", "darkblue")) +
  theme_minimal(base_family = "Arial", base_size = 10) +
  theme(panel.background = element_rect(fill = "grey")) +
  ggtitle("Age Distribution with Survival Overlay (Scatter Plot)") +
  xlab("Age") + ylab("Density")

# Visualization 4: Fare Distribution by Passenger Class (Violin Plot with Larger Size)
ggplot(titanic_data, aes(x = Pclass, y = Fare, fill = Pclass)) +
  geom_violin(trim = FALSE, scale = "width", adjust = 1.5) +  # Adjusting the width and scaling for larger appearance
  scale_fill_manual(values = c("lightyellow", "pink", "purple")) +
  theme_minimal(base_size = 16) +  # Increase the base size for larger text elements
  theme(
    axis.text = element_text(size = 14),   # Larger axis text
    axis.title = element_text(size = 16),  # Larger axis titles
    plot.title = element_text(size = 20, face = "bold"),  # Larger, bold plot title
    legend.title = element_text(size = 16),  # Larger legend title
    legend.text = element_text(size = 14),   # Larger legend text
    panel.grid = element_line(size = 1)  # Thicker grid lines
  ) +
  ggtitle("Fare Distribution by Passenger Class (Violin Plot with Larger Size)") +
  xlab("Passenger Class") + ylab("Fare")




# Ensure the FamilySize feature is created
if (!"FamilySize" %in% colnames(titanic_data)) {
  titanic_data$FamilySize <- titanic_data$SibSp + titanic_data$Parch + 1
}
# Calculate the counts for each FamilySize
family_size_counts <- titanic_data %>%
  group_by(FamilySize) %>%
  summarise(Count = n())

# Visualization 5: Family Size Distribution (Scatter Plot)
ggplot(family_size_counts, aes(x = factor(FamilySize), y = Count)) +
  geom_point(size = 3, color = "darkblue", fill = "lightblue", shape = 21, stroke = 1) +
  theme_minimal() +
  ggtitle("Family Size Distribution (Scatter Plot)") +
  xlab("Family Size") + ylab("Count") +
  theme(
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 12),
    plot.title = element_text(size = 12)
  )


# Visualization 6: Survival Rate by Family Size
ggplot(titanic_data, aes(x = FamilySize, fill = factor(Survived))) +
  geom_bar(position = "fill", color = "black") +
  scale_fill_manual(values = c("lightcoral", "darkred")) +
  theme_minimal() +
  ggtitle("Survival Rate by Family Size") +
  xlab("Family Size") + ylab("Proportion")



# Visualization 8: Correlation Matrix for Numeric Features
numeric_features <- titanic_data %>% select(Age, Fare, FamilySize)
cor_matrix <- cor(numeric_features)
cat("\nCorrelation Matrix\n")
cat("===================\n")
print(cor_matrix)
corrplot(cor_matrix, method = "ellipse", col = colorRampPalette(c("darkorange", "darkblue"))(100))

# Visualization 9: Boxplot of Age by Passenger Class (Outlier Detection)
ggplot(titanic_data, aes(x = Pclass, y = Age, fill = Pclass)) +
  geom_boxplot(outlier.color = "purple", outlier.shape = 16, outlier.size = 2) +
  scale_fill_manual(values = c("lightgreen", "lightblue", "lightpink")) +
  theme_minimal() +
  ggtitle("Age Distribution by Passenger Class") +
  xlab("Passenger Class") + ylab("Age")

# Visualization 10: Boxplot of Fare by Survival Status (Outlier Detection)
ggplot(titanic_data, aes(x = factor(Survived), y = Fare, fill = factor(Survived))) +
  geom_boxplot(outlier.color = "maroon", outlier.shape = 15, outlier.size = 2) +
  scale_fill_manual(values = c("yellow", "darkgrey")) +
  theme_minimal() +
  ggtitle("Fare Distribution by Survival Status") +
  xlab("Survived") + ylab("Fare")

# Feature Engineering
# Create 'FamilySize' as a new feature
titanic_data$FamilySize <- titanic_data$SibSp + titanic_data$Parch + 1

# Create 'IsAlone' as a new feature
titanic_data$IsAlone <- ifelse(titanic_data$FamilySize == 1, 1, 0)

# Drop unnecessary columns
titanic_data <- titanic_data %>% select(-PassengerId, -Name, -Ticket, -Cabin)

# Modeling
set.seed(42)
# Split the data into training and testing sets
train_index <- createDataPartition(titanic_data$Survived, p = 0.8, list = FALSE)
train_data <- titanic_data[train_index, ]
test_data <- titanic_data[-train_index, ]

# Model 1: Logistic Regression
logreg_model <- glm(Survived ~ Pclass + Sex + Age + Fare + Embarked + FamilySize + IsAlone, 
                    data = train_data, family = binomial)
logreg_pred <- predict(logreg_model, test_data, type = "response")
logreg_pred_class <- ifelse(logreg_pred > 0.5, 1, 0)
logreg_accuracy <- mean(logreg_pred_class == test_data$Survived)
cat("\nLogistic Regression Accuracy: ", logreg_accuracy, "\n")

# Model 2: Random Forest
rf_model <- randomForest(Survived ~ Pclass + Sex + Age + Fare + Embarked + FamilySize + IsAlone, 
                         data = train_data, ntree = 100)
rf_pred <- predict(rf_model, test_data)
rf_accuracy <- mean(rf_pred == test_data$Survived)
cat("Random Forest Accuracy: ", rf_accuracy, "\n")

# Model Performance Comparison
model_performance <- data.frame(
  Model = c("Logistic Regression", "Random Forest"),
  Accuracy = c(logreg_accuracy, rf_accuracy)
)

ggplot(model_performance, aes(x = Model, y = Accuracy, fill = Model)) + 
  geom_bar(stat = "identity", color = "black") + 
  scale_fill_manual(values = c("navy", "pink")) +
  theme_minimal() +
  ggtitle("Model Performance Comparison") +
  ylim(0, 1)


# Generate Confusion Matrices
logreg_cm <- table(Predicted = logreg_pred_class, Actual = test_data$Survived)
rf_cm <- table(Predicted = rf_pred, Actual = test_data$Survived)

# Function to create a confusion matrix heatmap
plot_confusion_matrix <- function(cm, title) {
  cm_df <- as.data.frame(cm)
  
  ggplot(cm_df, aes(x = Predicted, y = Actual)) +
    geom_tile(aes(fill = Freq), color = "white") +
    geom_text(aes(label = Freq), vjust = 0.5, size = 6) +
    scale_fill_gradient(low = "lightblue", high = "darkblue") +
    theme_minimal(base_size = 15) +
    ggtitle(title) +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      axis.title.x = element_text(size = 12),
      axis.title.y = element_text(size = 12),
      axis.text = element_text(size = 12, face = "bold"),
      panel.grid = element_blank(),
      legend.position = "none"
    )
}

# Plot Confusion Matrices
p1 <- plot_confusion_matrix(logreg_cm, "Confusion Matrix: Logistic Regression")
p2 <- plot_confusion_matrix(rf_cm, "Confusion Matrix: Random Forest")

# Display the plots together using gridExtra
library(gridExtra)
grid.arrange(p1, p2, ncol = 2)






