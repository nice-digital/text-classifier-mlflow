import mlflow
import mlflow.sklearn
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import argparse


class ModelHyperparameterTuner:
    def __init__(self, dataset_name, dataset_training_path, dataset_testing_path):
        self.dataset_name = dataset_name
        self.dataset_training_path = dataset_training_path
        self.dataset_testing_path = dataset_testing_path

        self.X_train, self.y_train, self.X_test, self.y_test = self.load_dataset(dataset_training_path, dataset_testing_path)

    def load_dataset(self, dataset_training_path, dataset_testing_path):
        # Load dataset from the provided path
        training_data = pd.read_csv(dataset_training_path)
        testing_data = pd.read_csv(dataset_testing_path)

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

        # drop any duplicates based on 'title'
        training_data.drop_duplicates(subset=['title'], inplace=True)
        testing_data.drop_duplicates(subset=['title'], inplace=True)
        print("Number of studies in the training dataset after de-dupe: " + str(training_data.shape[0]))
        print("Number of studies in the testing dataset after de-dupe: " + str(testing_data.shape[0]))

        training_data['titleabstract'] = training_data['title'] + " " + training_data['abstract']
        training_data['titleabstract'] = training_data['titleabstract'].str.lower()

        testing_data['titleabstract'] = testing_data['title'] + " " + testing_data['abstract']
        testing_data['titleabstract'] = testing_data['titleabstract'].str.lower()

        training_data = training_data[['titleabstract', 'target']]
        testing_data = testing_data[['titleabstract', 'target']]

        training_data = training_data.dropna()
        testing_data = testing_data.dropna()

        # Assuming the last column is the target variable
        X_train = training_data['titleabstract']
        y_train = training_data['target']

        X_test = testing_data['titleabstract']
        y_test = testing_data['target']
        run_name = "Dataset Details"
        # Log dataset information in MLflow
        with mlflow.start_run(run_name= run_name):
            mlflow.log_param('dataset', self.dataset_name)
            mlflow.log_param('num_training_samples', training_data.shape[0])
            mlflow.log_param('num_testing_samples', testing_data.shape[0])
            #mlflow.log_param('num_features', data.shape[1] - 1)
            #mlflow.log_param('feature_names', list(data.columns[:-1]))

        return X_train, y_train, X_test, y_test

    def define_hyperparameter_space(self, model_type):
        if model_type == 'logistic_regression':
            return {
                # parameter C stands for inverse regularisation strength. Smaller C indicates stronger regularisation
                # 
                'C': hp.loguniform('C', np.log(0.001), np.log(100)),
                'max_iter': scope.int(hp.quniform('max_iter', 50, 500, 50)),
                'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
            }
        elif model_type == 'random_forest':
            return {
                'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 50)),
                'max_depth': scope.int(hp.quniform('max_depth', 5, 50, 5)),
                'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
                'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 20, 1))
            }

    def objective_function(self, params, model_type):
        if model_type == 'logistic_regression':
            return self.logistic_regression_objective(params)
        elif model_type == 'random_forest':
            return self.random_forest_objective(params)

    def logistic_regression_objective(self, params, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
        # Convert hyperparameters to proper types if needed
        params['max_iter'] = int(params['max_iter'])
        solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        if isinstance(params['solver'], int):
            params['solver'] = solvers[params['solver']]

        run_name = f"Logistic2_C={params['C']:.2f}_max_iter={params['max_iter']}_solver={params['solver']}"

        with mlflow.start_run(run_name=run_name, nested=True):
            # Initialize the logistic regression model
            model = LogisticRegression(**params, random_state=42)

            # Create pipeline with TF-IDF vectorizer and logistic regression model
            text_clf = Pipeline([
                ('tfidfvect', TfidfVectorizer(ngram_range=(3, 3), stop_words='english')),
                ('clf', model)
            ])

            # Train the classifier on training data
            classifier = text_clf.fit(self.X_train, self.y_train)

            # Get predicted probabilities for the positive class (class 1)
            y_pred_proba = classifier.predict_proba(self.X_test)[:, 1]

            # Initialize a variable to track the best recall for comparison
            best_recall = 0
            best_threshold = 0.5

            # Loop over the list of custom thresholds
            for threshold in thresholds:
                # Apply the threshold to convert probabilities into class predictions
                y_pred_custom = (y_pred_proba >= threshold).astype(int)

                # Calculate performance metrics based on the current threshold
                recall = recall_score(self.y_test, y_pred_custom, average='weighted')
                precision = precision_score(self.y_test, y_pred_custom, average='weighted')
                accuracy = accuracy_score(self.y_test, y_pred_custom)
                f1 = f1_score(self.y_test, y_pred_custom, average='weighted')

                # Log the current threshold and the corresponding metrics to MLflow
                mlflow.log_metric(f'recall_threshold_{threshold}', recall)
                mlflow.log_metric(f'precision_threshold_{threshold}', precision)
                mlflow.log_metric(f'accuracy_threshold_{threshold}', accuracy)
                mlflow.log_metric(f'f1_score_threshold_{threshold}', f1)
                mlflow.log_metric(f'threshold', threshold)

                # Keep track of the best recall and corresponding threshold
                if recall >= best_recall:
                    best_recall = recall
                    best_threshold = threshold

            # Introduce a penalty for low precision, if desired
            precision_threshold = 0.4
            penalty = 0.0
            if precision < precision_threshold:
                penalty = 1.0 - precision / precision_threshold

            # Optimize for recall, but with a penalty if precision is too low
            loss = -best_recall + penalty

            # Log the best threshold and its recall to MLflow
            mlflow.log_metric('best_recall', best_recall)
            mlflow.log_metric('best_threshold', best_threshold)

            return {'loss': loss, 'status': STATUS_OK}

    def random_forest_objective(self, params):
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])

        run_name = f"RF2_n_estimators={params['n_estimators']}_max_depth={params['max_depth']}_min_samples_split={params['min_samples_split']}_min_samples_leaf={params['min_samples_leaf']}"

        with mlflow.start_run(run_name=run_name, nested=True):
            model = RandomForestClassifier(**params, random_state=42)

            # Create pipeline with TF-IDF vectorizer and logistic regression model
            text_clf = Pipeline([
                ('tfidfvect', TfidfVectorizer(ngram_range=(3, 3), stop_words='english')),
                ('clf', model)
            ])

            # Train the classifier on training data
            classifier = text_clf.fit(self.X_train, self.y_train)

            y_pred = classifier.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')

            mlflow.log_params(params)
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1_score', f1)

            return {'loss': -accuracy, 'status': STATUS_OK}

    def optimize_model(self, model_type, max_evals=50):
        space = self.define_hyperparameter_space(model_type)
        trials = Trials()

        parent_run_name = f"{model_type.replace('_', ' ').title()} Hyperparameter Tuning"
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            mlflow.log_param("model_type", model_type)
            best_params = fmin(
                fn=lambda params: self.objective_function(params, model_type),
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials
            )

        print(f"Best hyperparameters for {model_type}: {best_params}")
        return best_params


def main(dataset_name, dataset_training_path, dataset_testing_path,  model_type, max_evals):
    # Initialize the tuner class
    # The logs are created in mlruns folder within the project folder. By setting the following line, the logs
    # will be directed to the local mlflow tracking ui running on port 5000. Replace 5000 with 8080 if a mlflow
    # tracking server is started.
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Breast Cancer Models")

    tuner = ModelHyperparameterTuner(dataset_name, dataset_training_path, dataset_testing_path)

    # Optimize the model
    tuner.optimize_model(model_type, max_evals)


if __name__ == "__main__":
    #mlflow.delete_experiment(experiment_id='937001756573940241')

    parser = argparse.ArgumentParser(description='Hyperparameter tuning with MLflow and Hyperopt.')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset.', required=True)
    parser.add_argument('--dataset_training_path', type=str, help='Path to the training dataset CSV file.', required=True)
    parser.add_argument('--dataset_testing_path', type=str, help='Path to the testing dataset CSV.', required=True)
    parser.add_argument('--model_type', type=str, choices=['logistic_regression', 'random_forest'],
                        help='Type of model to tune.', required=True)
    parser.add_argument('--max_evals', type=int, default=2,
                        help='Maximum number of evaluations for hyperparameter optimization.')

    args = parser.parse_args()

    main(args.dataset_name,args.dataset_training_path, args.dataset_testing_path, args.model_type, args.max_evals)