# Human Activity Recognition & Model Evaluation with Blockchain

This repository contains two projects:

1. **Human Activity Recognition with Smartphones**
2. **Machine Learning Model Evaluation with Blockchain Integration**

---

## Project 1: Human Activity Recognition with Smartphones

### Overview
This project builds a model to predict human activities such as **Walking**, **Walking Upstairs**, **Walking Downstairs**, **Sitting**, **Standing**, or **Laying** using data from smartphones.

### Dataset
- **UCI HAR Dataset**: Data collected from 30 participants performing daily activities while carrying a smartphone.

### Steps Involved
1. **Importing Libraries**: Libraries like NumPy, Pandas, Matplotlib, Seaborn, and Plotly are used for data handling and visualization.
2. **Data Preparation**:
   - Loading features from `features.txt`.
   - Importing training and test data from text files.
3. **Model Building**: Applying machine learning techniques for activity classification.
4. **Evaluation**: Assessing the model’s performance using appropriate metrics.

### How to Run
1. Clone the repository.
2. Ensure the **UCI HAR Dataset** is in the correct directory structure.
3. Run the notebook `RPY.ipynb`.

---

## Project 2: Machine Learning Model Evaluation with Blockchain

### Overview
This project implements various machine learning models and deep learning techniques for data analysis. Additionally, it integrates a **blockchain** mechanism to securely store model evaluation metrics.

### Models Implemented
1. **Machine Learning**:
   - Logistic Regression
   - Support Vector Classifier (SVC)
   - Random Forest Classifier
2. **Deep Learning**:
   - Neural Network built using TensorFlow and Keras.

### Blockchain Integration
- A custom blockchain class is used to store model performance metrics such as accuracy, precision, recall, and F1-score.

### Key Functions
- **`deep_learning_model()`**: Builds and trains the neural network.
- **`plot_training_history()`**: Visualizes model training accuracy and loss.
- **`evaluate_and_add_to_blockchain()`**: Evaluates the model and adds the results to the blockchain.

### How to Run
1. Clone the repository.
2. Install required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook `presentation 2.ipynb`.

---

## License
This project is licensed under the MIT License.

## Acknowledgements
- **UCI HAR Dataset** for providing the data for the Human Activity Recognition project.
- TensorFlow, Scikit-learn, and other open-source libraries used in these projects.

---

Feel free to contribute, raise issues, or suggest improvements!

