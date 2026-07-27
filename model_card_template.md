# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a RandomForestClassifier from scikit-learn, trained to predict
whether an individual's annual income exceeds $50K based on U.S. Census
demographic data. It was developed as part of a portfolio project demonstrating
a scalable ML pipeline deployed via FastAPI and trained with scikit-learn's default hyperparameters.
## Intended Use
This model is intended for educational and demonstration purposes, showing
an end-to-end ML pipeline (data processing, training, evaluation, and
deployment via API). It predicts a binary income category (<=50K or >50K)
from demographic features for a given individual record.

This model is NOT intended for use in real-world decision-making that affects
individuals, such as lending, hiring, insurance underwriting, or benefits
eligibility.
## Training Data
The data is the "Census Income" dataset from the UCI Machine Learning Repository,
containing demographic information (age, workclass, education, marital status,
occupation, relationship, race, sex, native country, etc.) along with a binary
salary label (<=50K or >50K). The dataset was split 80/20 into train and test
sets using a fixed random_state for reproducibility. Categorical features were
one-hot encoded and the label was binarized using sklearn's LabelBinarizer,
both fit on the training set only.

## Evaluation Data
The evaluation (test) set is the held-out 20% split from the same Census Income
dataset, processed using the encoder and label binarizer fit on the training
data (no re-fitting on test data, to avoid data leakage).
## Metrics
The model was evaluated using precision, recall, and F1 (fbeta) score.

Model overall performance on the test set:
- Precision: 0.7470
- Recall: 0.6334
- F1: 0.6855

Performance was also computed on slices of the data for each categorical
feature (see slice_output.txt), to check for performance disparities across
subgroups such as sex, race, and native-country.These slices showed that the model does NOT perform well across all demographics.
## Ethical Considerations
This dataset includes sensitive demographic attributes such as race, sex, and
native country. Slice-based evaluation (see slice_output.txt) revealed
performance differences across subgroups within these features, meaning the
model does not perform equally well for all demographic groups.

## Caveats and Recommendations
- This model is intended as a learning/demonstration exercise for building
and deploying an ML pipeline, not for production use in decisions that
affect people's lives (loan decisions, benefits decisions, etc).
- The dataset is from a specific historical time period and U.S. context, so it is not intended to be utilized for current day predictions or other pouplations.
- Some categorical slices (e.g. rare native-country values) have very small
  sample sizes, making their per-slice metrics unstable or unreliable.