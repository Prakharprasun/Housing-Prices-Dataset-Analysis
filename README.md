# Housing Price Analysis Using Support Vector Machine (SVM)
This project focuses on housing data analysis and classification using a Support Vector Machine (SVM) with a linear kernel.
## Overview:
This project leverages machine learning techniques to analyze housing data, perform feature engineering, and predict whether a house is located in a preferred area (prefarea) using a Support Vector Machine (SVM) with a linear kernel. The dataset used contains various housing features, including price, area, number of rooms, and other attributes, which are preprocessed and transformed for model training and evaluation.
## Installation and Dependencies:
To run this project, the following Python libraries must be installed:

pandas: Data manipulation and analysis.

numpy: Numerical computing.

scikit-learn: Machine learning algorithms and preprocessing utilities.

matplotlib: Visualization and plotting.

seaborn: Enhanced data visualization.

Google Colab: For running the code in a cloud-based environment, with integration for Google Drive to access the dataset.

## Install the dependencies via pip:
pip install pandas numpy scikit-learn matplotlib seaborn
## Key Components:
### Data Loading: 
The dataset is loaded from Google Drive and read using Pandas. The initial data structure is displayed to provide insight into the features and their respective data types.

### Preprocessing:
One-Hot Encoding: Categorical variables are encoded into numerical format using one-hot encoding, which converts non-numeric columns into binary indicators.
Boolean Conversion: Boolean columns are transformed into integers (0 and 1) for model compatibility.

Feature Engineering: New features are created:

1. PriceperArea: Ratio of the house price to its area.

2. TotalRooms: Total number of rooms, including bedrooms, bathrooms, parking spaces, guest rooms, and basement.

Scaling: Numerical columns are standardized using the StandardScaler to ensure they have zero mean and unit variance, optimizing the performance of the SVM algorithm.

One-hot encoding is an essential technique in data preprocessing for several reasons. It transforms categorical data into a format that machine learning models can easily understand and use. This transformation allows each category to be treated independently without implying any false relationships between them. Additionally, many data processing and machine learning libraries support one-hot encoding. It fits smoothly into the data preprocessing workflow, making it easier to prepare datasets for various machine learning algorithms.

Label encoding is another method to convert categorical data into numerical values by assigning each category a unique number. However, this approach can create problems because it might suggest an order or ranking among categories that doesn't actually exist. For example, assigning 1 to Red, 2 to Green, and 3 to Blue could make the model think that Green is greater than Red and Blue is greater than both. This misunderstanding can negatively affect the model's performance.

One-hot encoding solves this problem by creating a separate binary column for each category. This way, the model can see that each category is distinct and unrelated to the others. Label encoding is useful when the categorical data has an inherent ordinal relationship, meaning the categories have a meaningful order or ranking. In such cases, the numerical values assigned by label encoding can effectively represent this order, making it a suitable choice.

Feature engineering is the process of transforming raw data into features that are suitable for machine learning models. In other words, it is the process of selecting, extracting, and transforming the most relevant features from the available data to build more accurate and efficient machine learning models.

The success of machine learning models heavily depends on the quality of the features used to train them. Feature engineering involves a set of techniques that enable us to create new features by combining or transforming the existing ones. These techniques help to highlight the most important patterns and relationships in the data, which in turn helps the machine learning model to learn from the data more effectively.

Scaling your data in machine learning (ML)is important because many algorithms use the Euclidean distance between two data points in their computations/derivations, which is sensitive to the scale of the variables. If one variable is on a much larger scale than another, that variable will dominate the distance calculation, and the algorithm will be affected by that variable more than the other. Scaling the data can help to balance the impact of all variables on the distance calculation and can help to improve the performance of the algorithm. In particular, several ML techniques, such as neural networks, require that the input data be normalised for it to work well.

There are several libraries in Python that can be used to scale data:

1. Standardisation: The mean of each feature becomes 0 and the standard deviation becomes 1.

2. Normalisation: The values of each feature are between 0 and 1.

3. Min-Max Scaling: The minimum value of each feature becomes 0 and the maximum value becomes 1.

### Data Preparation: 
The data is split into three subsets:

Training Set (60%)

Validation Set (20%)

Test Set (20%)

The target variable, prefarea, is encoded as binary (0 for "no" and 1 for "yes") and used as the label for classification.

### Model Training: 
A Support Vector Machine (SVM) with a linear kernel is trained on the training data. SVM is chosen for its capability to classify data by finding the optimal hyperplane (a hyperplane is a decision boundary that divides the input space into two or more regions, each corresponding to a different class or output label. In a 2D space, a hyperplane is a straight line that divides the space into two halves. In a 3D space, however, a hyperplane is a plane that divides the space into two halves. Meanwhile in higher-dimensional spaces, a hyperplane is a subspace of one dimension less than the input space.) that separates different classes.

Support Vector Machine (SVM) is a supervised machine learning algorithm used for both classification and regression. Though we say regression problems as well it’s best suited for classification. The main objective of the SVM algorithm is to find the optimal hyperplane in an N-dimensional space that can separate the data points in different classes in the feature space. The hyperplane tries that the margin between the closest points of different classes should be as maximum as possible. The dimension of the hyperplane depends upon the number of features. If the number of input features is two, then the hyperplane is just a line. If the number of input features is three, then the hyperplane becomes a 2-D plane. It becomes difficult to imagine when the number of features exceeds three. 

### Model Evaluation: 
The model is evaluated using accuracy scores on the training, validation, and test sets. This allows for assessing the model's generalization performance across different data subsets.
### Visualization:
Feature vs. Price Scatter Plots: Visualizes the relationship between each feature and housing prices. This helps in identifying potential correlations between variables and price.
SVM Decision Boundary: For each feature, the decision boundary learned by the SVM model is plotted. This boundary illustrates how the model separates data points into different classes (preferred vs. non-preferred areas).

A scatter plot (aka scatter chart, scatter graph) uses dots to represent values for two different numeric variables. The position of each dot on the horizontal and vertical axis indicates values for an individual data point. Scatter plots are used to observe relationships between variables.

SVCs aim to find a hyperplane that effectively separates the classes in their training data by maximizing the margin between the outermost data points of each class. This is achieved by finding the best weight vector that defines the decision boundary hyperplane and minimizes the sum of hinge losses for misclassified samples, as measured by the hinge_loss function. By default, regularization is applied with the parameter C=1, which allows for a certain degree of misclassification tolerance.

If the data is not linearly separable in the original feature space, a non-linear kernel parameter can be set. Depending on the kernel, the process involves adding new features or transforming existing features to enrich and potentially add meaning to the data. When a kernel other than "linear" is set, the SVC applies the kernel trick, which computes the similarity between pairs of data points using the kernel function without explicitly transforming the entire dataset. The kernel trick surpasses the otherwise necessary matrix transformation of the whole dataset by only considering the relations between all pairs of data points. The kernel function maps two vectors (each pair of observations) to their similarity using their dot product.

## Instructions for Running the Program:
Mount Google Drive: Ensure that the dataset is stored in Google Drive. Modify the data_path in the script to reflect the correct location of the dataset in your Google Drive.

Load and Preprocess the Data: The load_data() function loads the dataset into a Pandas DataFrame, and preprocess_data() applies one-hot encoding, feature engineering, and scaling.

Prepare the Data for Modeling: The prepare_data() function splits the data into training, validation, and test sets. It also defines the target variable, prefarea.

Train the SVM Model: Call train_svm() to fit the model to the training data.

## Evaluate Model Performance: 
The evaluate_model() function computes accuracy scores on the training, validation, and test sets.
## Visualize Features and Decision Boundaries: 
The plot_features_vs_price() function generates scatter plots of various features against price, while plot_svm_decision_boundary() visualizes the SVM decision boundaries.
## Notes on Model Selection:
A linear kernel SVM was chosen for its simplicity and interpretability in this binary classification task. Despite potential trade-offs in accuracy for more complex datasets, a linear SVM provides a clear decision boundary that is easy to visualize and analyze. Alternative models, such as non-linear SVMs or other classification algorithms (e.g., Random Forest, Logistic Regression), may also be explored for further optimization.
## References:
Sklearn

Atlassian

GeeksforGeeks
