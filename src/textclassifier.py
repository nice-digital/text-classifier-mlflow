import mlflow
import mlflow.sklearn
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer, fbeta_score
import pandas as pd
import numpy as np
import argparse

class ModelHyperparameterTuner:
    def __init__(self, dataset_name, dataset_training_path, dataset_testing_path):
        """
        Initialize the tuner with dataset name and paths for training and testing datasets.

        Parameters:
        dataset_name (str): Name of the dataset.
        dataset_training_path (str): File path to the training dataset.
        dataset_testing_path (str): File path to the testing dataset.
        """
        self.dataset_name = dataset_name
        self.dataset_training_path = dataset_training_path
        self.dataset_testing_path = dataset_testing_path

        self.X_train, self.y_train, self.X_test, self.y_test = self.load_dataset(dataset_training_path, dataset_testing_path)

    def load_dataset(self, dataset_training_path, dataset_testing_path):
        """
        Load the training and testing datasets from CSV files, preprocess them by removing duplicates,
        renaming columns, and combining titles and abstracts.
        Parameters:
        dataset_training_path (str): File path to the training dataset.
            Training dataset consists of 'title' and 'abstract' of scientific publications and a target field
            which specifies whether each row was included (1) or excluded(0) in the study
        dataset_testing_path (str): File path to the testing dataset.
            Testing dataset has the same format as training dataset
        Returns:
        X_train, y_train, X_test, y_test: Processed training and testing features and labels.
        """

        # Load dataset from the provided path
        training_data = pd.read_csv(dataset_training_path)
        testing_data = pd.read_csv(dataset_testing_path)

        # Rename columns for downstream processing
        rename_map = {'Title': 'title', 'Abstract': 'abstract'}
        training_data.rename(columns=rename_map, inplace=True)
        testing_data.rename(columns=rename_map, inplace=True)

        try:
            training_data['title_orig'] = training_data['title']
            testing_data['title_orig'] = testing_data['title']
        except Exception as e:
            print(e)
            print("Error- No title detected! Title is needed!")
            raise

        # Drop any duplicates based on 'title'
        training_data.drop_duplicates(subset=['title'], inplace=True)
        testing_data.drop_duplicates(subset=['title'], inplace=True)

        # Output number of records after removing duplicates
        print("Number of studies in the training dataset after de-dupe: " + str(training_data.shape[0]))
        print("Number of studies in the testing dataset after de-dupe: " + str(testing_data.shape[0]))

        training_data['titleabstract'] = training_data['title'] + " " + training_data['abstract']
        training_data['titleabstract'] = training_data['titleabstract'].str.lower()

        # Combine 'title' and 'abstract' into a single column and convert to lowercase
        testing_data['titleabstract'] = testing_data['title'] + " " + testing_data['abstract']
        testing_data['titleabstract'] = testing_data['titleabstract'].str.lower()

        # Keep only the 'titleabstract' and 'target' columns, drop missing values
        training_data = training_data[['titleabstract', 'target']].dropna()
        testing_data = testing_data[['titleabstract', 'target']].dropna()

        # Separate features (X) and labels (y)
        X_train = training_data['titleabstract']
        y_train = training_data['target']

        X_test = testing_data['titleabstract']
        y_test = testing_data['target']

        # Log dataset details to MLflow for tracking
        run_name = "Dataset Details"
        with mlflow.start_run(run_name= run_name):
            mlflow.log_param('dataset', self.dataset_name)
            mlflow.log_param('num_training_samples', training_data.shape[0])
            mlflow.log_param('num_testing_samples', testing_data.shape[0])
            #mlflow.log_param('num_features', data.shape[1] - 1)
            #mlflow.log_param('feature_names', list(data.columns[:-1]))

        # Return the training and testing features and labels
        return X_train, y_train, X_test, y_test

    def define_hyperparameter_space(self, model_type):
        """
        Define the hyperparameter search space for different models.
        Parameters:
        model_type (str): The type of model ('logistic_regression' or 'random_forest').
        Returns:
        dict: A dictionary defining the search space for the specified model's hyperparameters.
              The values are instances of hyperopt functions (hp.loguniform, hp.quniform, and hp.choice),
              which define how each hyperparameter should be sampled.
        """
        if model_type == 'logistic_regression':
            return {
                # parameter C stands for inverse regularisation strength. Smaller C indicates stronger regularisation
                # penalty l1 is chosen as it's a very high dimensional dataset, and will shrink weights to zero
                # only liblinear works with l1 and small-medium datasets.
                #'C': hp.loguniform('C', np.log(0.001), np.log(100)),
                #'C': hp.choice('C', [100, 10, 3.0, 0.1, 0.01]),
                'C': hp.quniform('C', 0.1, 10, 0.1),
                'solver': hp.choice('solver', ['liblinear']),
                'penalty': hp.choice('penalty', ['l1']),
                'max_iter': scope.int(hp.quniform('max_iter', 50, 500, 50))
            }
        elif model_type == 'random_forest':
            return {
                'n_estimators': scope.int(hp.quniform('n_estimators', 10, 100, 10)),  # Range: 10-100, step of 10
                'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),           # Range: 3-10, step of 1
                'min_samples_split': scope.int(hp.quniform('min_samples_split', 5, 10, 1)),  # Range: 5-10, step of 1
                'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 2, 5, 1))      # Range: 2-5, step of 1
            }

    def objective_function(self, params, model_type):
        """
        Objective function for hyperparameter optimization. This function directs to the appropriate
        model-specific objective function based on the model type.
        Parameters:
        params (dict): Hyperparameters to evaluate.
        model_type (str): Type of model ('logistic_regression' or 'random_forest').
        Returns:
        dict: Result containing the loss and status of the evaluation.
        """
        if model_type == 'logistic_regression':
            return self.logistic_regression_objective(params)
        elif model_type == 'random_forest':
            return self.random_forest_objective(params)

    def logistic_regression_objective(self, params):
        """
        Objective function for tuning hyperparameters of a Logistic Regression model using Hyperopt.
        This function is used in a hyperparameter optimization process. It initializes a logistic regression
        model with the given hyperparameters, performs cross-validation, and logs the model performance metrics
        and parameters using MLflow. The goal is to maximize the F2-score (emphasizing recall).
        Parameters:
        params : dict
        A dictionary containing the hyperparameters for Logistic Regression, such as 'C', 'solver',
        and 'max_iter'. Hyperopt supplies these parameters during the optimization process.
        Returns:
        dict
        A dictionary containing the negative F2-score as the loss (since Hyperopt minimizes the objective),
        the status of the optimization, and the trained classifier model.

        """
        # Convert hyperparameters to proper types if needed
        print(params)
        run_name = f"Logistic2_C={params['C']:.2f}_max_iter={params['max_iter']}_solver={params['solver']}"

        with mlflow.start_run(run_name=run_name, nested=True):

            model = LogisticRegression(**params, class_weight = 'balanced', random_state=42)
            # Create pipeline with TF-IDF vectorizer and logistic regression model
            text_clf = Pipeline([
                ('tfidfvect', TfidfVectorizer(ngram_range=(3, 3), stop_words='english')),
                ('clf', model)
            ])

            # Train the classifier on training data
            classifier = text_clf.fit(self.X_train, self.y_train)
            # F-beta score: F1 score with a focus on recall (beta > 1 emphasizes recall)
            # using the below scorer which is a weighted F1-score, with beta > 1, recall is prioritised
            # more than precision. The score ranges from 0-1, with 1 being the perfect score.
            f2_scorer = make_scorer(fbeta_score, beta=2)
            f2_score = cross_val_score(classifier, self.X_train, self.y_train, cv=5, scoring=f2_scorer).mean()

            mlflow.log_params(params)
            mlflow.log_metric('cross val f2 score', f2_score)
            # example of setting tags; certain tags such as parameters, user, source code etc are auto-tagged
            mlflow.set_tags(
                tags={
                    "project": "Text classifier",
                    "optimizer_engine": "hyperopt",
                    "model_family": "logistic regression",
                    "feature_set_version": 1,
                }
            )
            # Return the negative score because Hyperopt minimizes the objective
            return {'loss': -f2_score, 'status': STATUS_OK, 'model': classifier}

    def test_model(self, model):
        """
        Tests a trained model on the test dataset and evaluates its performance across different classification thresholds.
        Parameters:
        model : object
        The trained model (with a `predict` and `predict_proba` method) to be tested on the test dataset.
        Returns:
        None
        """
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        best_recall = 0
        best_threshold = 0
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        # Loop over the list of custom thresholds
        for threshold in thresholds:
            run_name = f"Best Model, threshold={threshold:.2f}"
            with mlflow.start_run(run_name=run_name, nested=True):
                # Apply the threshold to convert probabilities into class predictions
                y_pred_custom = (y_pred_proba >= threshold).astype(int)

                # Calculate performance metrics based on the current threshold
                recall = recall_score(self.y_test, y_pred_custom)
                precision = precision_score(self.y_test, y_pred_custom)
                accuracy = accuracy_score(self.y_test, y_pred_custom)
                f1 = f1_score(self.y_test, y_pred_custom)

                # Log the current threshold and the corresponding metrics to MLflow
                mlflow.log_metric(f'recall_threshold_{threshold}', recall)
                mlflow.log_metric(f'precision_threshold_{threshold}', precision)
                mlflow.log_metric(f'accuracy_threshold_{threshold}', accuracy)
                mlflow.log_metric(f'f1_score_threshold_{threshold}', f1)
                mlflow.log_metric(f'threshold', threshold)
                print("***threshold,recall: ", threshold, recall)
                if recall >= best_recall:
                    print('update recall, threshold:', recall, threshold)
                    best_recall = recall
                    best_threshold = threshold
        print(best_recall, best_threshold)



    def random_forest_objective(self, params):
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])

        run_name = f"RF2_n_estimators={params['n_estimators']}_max_depth={params['max_depth']}_min_samples_split={params['min_samples_split']}_min_samples_leaf={params['min_samples_leaf']}"

        with mlflow.start_run(run_name=run_name, nested=True):
            model = RandomForestClassifier(**params, class_weight='balanced', random_state=42)

            # Create pipeline with TF-IDF vectorizer and logistic regression model
            text_clf = Pipeline([
                ('tfidfvect', TfidfVectorizer(ngram_range=(1, 1), stop_words='english')), #considering only unigrams as trigrams is too high dimensional for RF
                ('clf', model)
            ])
            # Train the classifier on training data
            classifier = text_clf.fit(self.X_train, self.y_train)

            f2_scorer = make_scorer(fbeta_score, beta=2)
            f2_score = cross_val_score(classifier, self.X_train, self.y_train, cv=5, scoring=f2_scorer).mean()

            run_id = mlflow.active_run().info.run_id
            mlflow.log_param("Run ID", run_id)
            mlflow.log_params(params)
            mlflow.log_metric('cross val f2 score', f2_score)
            # Return the negative score because Hyperopt minimizes the objective
            return {'loss': -f2_score, 'status': STATUS_OK, 'model': classifier}

    def optimize_model(self, model_type, max_evals=50):
        space = self.define_hyperparameter_space(model_type)
        trials = Trials()

        parent_run_name = f"{model_type.replace('_', ' ').title()} Hyperparameter Tuning"
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            mlflow.log_param("model_type", model_type)
            best_params =  fmin(
                fn=lambda params: self.objective_function(params, model_type),
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials
            )
            best_trial = min(trials.trials, key=lambda x: x['result']['loss'])
            #best_params = best_trial['result']['params']
            best_model = best_trial['result'].get('model')

            print(best_params)
            with mlflow.start_run(run_name="Best_Model", nested = True) as run:
                # Find the trial with the lowest loss
                # Log the best parameters found
                best_model_run_id = run.info.run_id
                mlflow.log_param("Best Run ID", best_model_run_id)
                for param_name, param_value in best_params.items():
                    # note that hp.choice index of the best parameters are returned, and not the list values.
                    mlflow.log_param(param_name, param_value)
                best_score = -best_trial['result']['loss']
                mlflow.log_metric("best_cross_valid_f2score", best_score)

            with mlflow.start_run(run_name="Best Model on Test dataset", nested = True):
                self.test_model(best_model)

        return


def main(experiment_name, dataset_name, dataset_training_path, dataset_testing_path,  model_type, max_evals):
    """
    Main function to orchestrate hyperparameter tuning using MLflow and Hyperopt.
    This function sets up the MLflow tracking environment, initializes the model hyperparameter tuner,
    and starts the optimization process for the specified model.
    Parameters:
    experiment_name : str
        Name of the MLflow experiment where results will be logged.
    dataset_name : str
        Name of the dataset being used for training and testing the model.
    dataset_training_path : str
        Path to the training dataset (CSV file).
    dataset_testing_path : str
        Path to the testing dataset (CSV file).
    model_type : str
        The type of model to be optimized (e.g., 'logistic_regression', 'random_forest').
    max_evals : int
        Maximum number of evaluations to perform during hyperparameter optimization.
        This controls how many combinations of hyperparameters are tried during the optimization process.
    Returns:
        None
    """

    # Initialize the tuner class
    # The logs are created in mlruns folder within the project folder. By setting the following line, the logs
    # will be directed to the local mlflow tracking ui running on port 5000. Replace 5000 with 8080 if a mlflow
    # tracking server is started. In the below line of code, it connects to a mlflow UI started with 'mlflow ui'
    # from the command line within the project folder
    mlflow.set_tracking_uri("http://localhost:5000")

    # Set or create an MLflow experiment where the results will be logged.
    # If the experiment with the given name doesn't exist, MLflow will create it.
    mlflow.set_experiment(experiment_name)

    # Initialize the ModelHyperparameterTuner class with the dataset details.
    tuner = ModelHyperparameterTuner(dataset_name, dataset_training_path, dataset_testing_path)

    # Start the hyperparameter optimization process.
    # This function will tune the `model_type` (e.g., logistic regression or random forest)
    # using the specified number of evaluations (`max_evals`).
    tuner.optimize_model(model_type, max_evals)


# This block of code is executed when the script is run directly (not imported as a module).
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Hyperparameter tuning with MLflow and Hyperopt.')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment.', required=True)
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset.', required=True)
    parser.add_argument('--dataset_training_path', type=str, help='Path to the training dataset CSV file.', required=True)
    parser.add_argument('--dataset_testing_path', type=str, help='Path to the testing dataset CSV.', required=True)
    parser.add_argument('--model_type', type=str, choices=['logistic_regression', 'random_forest'],
                        help='Type of model to tune.', required=True)
    parser.add_argument('--max_evals', type=int, default=50,
                        help='Maximum number of evaluations for hyperparameter optimization.')

    args = parser.parse_args()

    # Call the `main()` function, passing the parsed arguments for experiment name, dataset paths,
    # model type, and maximum evaluations. These are used for model training.
    main(args.experiment_name, args.dataset_name,args.dataset_training_path, args.dataset_testing_path, args.model_type, args.max_evals)