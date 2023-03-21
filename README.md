# Income Category Prediction using Census Data

## Model Details

### Intended Use

This model is intended for the prediction of salary classes (above or below a certain threshold, e.g., 50K) based on various demographic and job-related features collected from the U.S. Census data.

### Selected Model

- Model: RandomForestClassifier
- Training Library: scikit-learn
- Random State: 42

## Training Data

The training data is sourced from the U.S. Census data, specifically the census.csv file. It contains demographic and job-related features such as workclass, education, marital status, occupation, relationship, race, sex, and native country.

## Evaluation Data

The evaluation data is also sourced from the U.S. Census data, with an 80-20 train-test split performed on the dataset. The test set is used to evaluate the model's performance on unseen data.

## Metrics

The primary metrics used to evaluate the model's performance are weighted precision, recall, and F1 score. The model's performance on these metrics after training and evaluation is as follows:

- Precision: 0.7655
- Recall: 0.6378
- F1 Score: 0.6958

## Ethical Considerations

It is important to consider the potential ethical implications of using this model for decision-making processes. The model is trained on data that includes sensitive demographic information such as race, sex, and native country, which may lead to biased predictions. These biases could perpetuate existing inequalities and discrimination in income distribution and job opportunities.

## Caveats and Recommendations

The model's performance might be affected by class imbalance in the dataset, where the number of instances in one class is significantly larger than the other class. It is recommended to apply techniques such as oversampling or undersampling to address this issue. The model is trained on the U.S. Census data, and its generalizability to other countries or regions may be limited. It is recommended to fine-tune the model with additional data that is more representative of the target population. Periodically retrain the model with updated data to ensure that it stays current with evolving socio-economic trends. Users of the model should be cautious of potential biases and should not solely rely on the model's predictions for making important decisions. The model's output should be combined with other sources of information and expert opinions to make well-rounded decisions.

## Project Structure

<pre>
project_name/
│
├── data/ # Data folder
│ └── census.csv # CSV file containing the dataset
│
├── src/ # Main source code folder
│ ├── app/ # Main application folder
│ │ ├── api/ # API endpoints and routes
│ │ │ ├── **init**.py
│ │ │ └── endpoints.py
│ │ ├── models/ # ML model-related code and files
│ │ │ ├── **init**.py
│ │ │ ├── data.py
│ │ │ ├── train_model.py
│ │ │ └── model.pkl # Serialized model file
│ │ ├── utils/ # Utility functions and classes
│ │ │ ├── **init**.py
│ │ │ └── utils.py
│ │ ├── **init**.py
│ │ ├── config.py # Configuration file
│ │ └── main.py # FastAPI app initialization and configuration
│ └── **init**.py
│
├── tests/ # Test cases and files
│ ├── **init**.py
│ └── test_app.py
│
├── Dockerfile # Docker configuration file
├── requirements.txt # Project dependencies
├── setup.py # Package and distribution setup
└── README.md # Project documentation
</pre>

The data folder contains the dataset used by the model, while the src folder contains the main source code for the project. The app subfolder contains all the code related to the API, including the API endpoints and routes in the api subfolder, the machine learning models in the models subfolder, and utility functions in the utils subfolder. The config.py file contains configuration settings for the project, and the main.py file initializes and configures the FastAPI application.

The tests folder contains test cases for the application.

The Dockerfile is used to create a Docker image for the application, while the requirements.txt file lists the project dependencies. The setup.py file is used to package and distribute the project. Finally, the README.md file provides documentation for the project.
