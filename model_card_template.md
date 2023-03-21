# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Model: RandomForestClassifier
Training Library: scikit-learn
Random State: 42
Intended Use
This model is intended for the prediction of salary classes (above or below a certain threshold, e.g., 50K) based on various demographic and job-related features collected from the U.S. Census data.

## Training Data

The training data is sourced from the U.S. Census data, specifically the census.csv file. It contains demographic and job-related features such as workclass, education, marital status, occupation, relationship, race, sex, and native country.

## Evaluation Data

The evaluation data is also sourced from the U.S. Census data, with an 80-20 train-test split performed on the dataset. The test set is used to evaluate the model's performance on unseen data.

## Metrics

The primary metrics used to evaluate the model's performance are weighted precision, recall, and F1 score. The model's performance on these metrics after training and evaluation is as follows:

Precision: 0.7655
Recall: 0.6378
F1 Score: 0.6958

## Ethical Considerations

It is important to consider the potential ethical implications of using this model for decision-making processes. The model is trained on data that includes sensitive demographic information such as race, sex, and native country, which may lead to biased predictions. These biases could perpetuate existing inequalities and discrimination in income distribution and job opportunities.

## Caveats and Recommendations

The model's performance might be affected by class imbalance in the dataset, where the number of instances in one class is significantly larger than the other class. It is recommended to apply techniques such as oversampling or undersampling to address this issue.
The model is trained on the U.S. Census data, and its generalizability to other countries or regions may be limited. It is recommended to fine-tune the model with additional data that is more representative of the target population.
Periodically retrain the model with updated data to ensure that it stays current with evolving socio-economic trends.
Users of the model should be cautious of potential biases and should not solely rely on the model's predictions for making important decisions. The model's output should be combined with other sources of information and expert opinions to make well-rounded decisions.
