import os
import xgboost as xgb
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import HashingVectorizer


class BatchVectorizerXGBoostConcatCrossVal:
    """Custom vectorizer and classifier class to process data in
    batches.  Useful when there are RAM and CPU limitations.

    Args:
        BaseEstimator (Estimator): base class for estimator
    """

    def __init__(
        self,
        objective,
        max_depth,
        gamma,
        reg_alpha,
        reg_lambda,
        colsample_bytree,
        min_child_weight,
        n_estimators,
        seed,
        use_label_encoder=False,
        base_score=None,
        booster=None,
        colsample_bylevel=None,
        colsample_bynode=None,
        enable_categorical=None,
        gpu_id=None,
        importance_type=None,
        interaction_constraints=None,
        learning_rate=None,
        max_delta_step=None,
        *args,
        **kwargs
    ):
        """intialization of class

        Args:
            vectorizer (Vectorizer): vectorizer to turn text into vectors
            batch_size (int): size of each batch used for training
        """

        self.vectorizer = HashingVectorizer(ngram_range=(1, 4), stop_words="english", n_features=1000)
        self.batch_size = 100000
        self.xgbclass = xgb.XGBClassifier(
            objective=objective,
            max_depth=max_depth,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            n_estimators=n_estimators,
            seed=seed,
            base_score=base_score,
            booster=booster,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            enable_categorical=enable_categorical,
            importance_type=importance_type,
            interaction_constraints=interaction_constraints,
            learning_rate=learning_rate,
            max_delta_step=max_delta_step
        )
        # self._estimator_type = "classifier"

    def fit(self, X_train, y):
        """fit the vectorizer and the classifier

        Args:
            X_texts (pd.Series): input texts
            X_node_embeds (pd.Series): input node embeddings
            y (pd.Series): labels
        """
        
        X_texts = X_train["texts"]
        X_node_embeds = X_train["node_embeds"]

        # check whether vectorize and modelr has been fitted
        vec_fitted = False
        model_fitted = False

        # number of batches to be processed
        if len(y) >= self.batch_size:
            num_batches = len(y) // self.batch_size
        else:
            num_batches = 1
            
        for X_text_chunk, X_node_embeds_chunk, y_chunk in zip(
            np.array_split(X_texts, num_batches), np.array_split(X_node_embeds, num_batches), 
            np.array_split(y, num_batches)
        ):

            # convert text chunk to unicode
            X_text_chunk = X_text_chunk.values.astype("U")

            if not vec_fitted:
                X_text_trans = self.vectorizer.fit_transform(X_text_chunk)
                vec_fitted = True
            else:
                X_text_trans = self.vectorizer.transform(X_text_chunk)
            del X_text_chunk
            
            X_text_trans = X_text_trans.todense()
            X_concat = np.hstack([np.vstack(X_text_trans), np.vstack(X_node_embeds_chunk)])
            # xg_chunk = xgb.DMatrix(X_trans, label=y_chunk)
            y_chunk = np.array(y_chunk).reshape(-1)
            if model_fitted:
                self.xgbclass.fit(
                    X_concat, y_chunk, xgb_model=self.xgbclass.get_booster()
                )
            else:
                self.xgbclass.fit(X_concat, y_chunk)
                model_fitted = True
            # if os.path.exists("model.model"):
            #     self.model = xgb.train(params, xg_chunk, 30, xgb_model="model.model")
            # else:
            #     self.model = xgb.train(params, xg_chunk, 30)
            # self.model.save_model("model.model")

        return self
    
    def get_params(self, deep):
        return self.xgbclass.get_params(deep=deep)

    def set_params(self, **parameters):
        return self.xgbclass.set_params(**parameters)
    
    def __sklearn_is_fitted__(self):
        return True

    def predict(self, X_train):
        
        X_texts = X_train["texts"]
        X_node_embeds = X_train["node_embeds"]

        X_unicode = X_texts.values.astype("U")
        X_text_trans = self.vectorizer.transform(X_unicode).todense()
        X_concat = np.hstack([np.vstack(X_text_trans), np.vstack(X_node_embeds)])
        # xg_test = xgb.DMatrix(X_trans)
        predictions = np.array(list(map(int, np.round(self.xgbclass.predict(X_concat)))))
        return predictions

        # predictions = self.clf.predict(X_trans)
        # return predictions


class BatchVectorizerClassifierCitationsNonTexts:
    """Custom vectorizer and classifier class to process data in
    batches.  Useful when there are RAM and CPU limitations.

    Args:
        BaseEstimator (Estimator): base class for estimator
    """

    def __init__(
        self,
        objective,
        max_depth,
        gamma,
        reg_alpha,
        reg_lambda,
        colsample_bytree,
        min_child_weight,
        n_estimators,
        seed=0,
        *args,
        **kwargs
    ):
        """intialization of class

        Args:
            clf (Estimator): classifier for training
            vectorizer (Vectorizer): vectorizer to turn text into vectors
            batch_size (int): size of each batch used for training
        """

        self.clf = xgb.XGBClassifier(
            objective=objective,
            max_depth=max_depth,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            n_estimators=180,
            seed=0,
        )
        self.vectorizer = HashingVectorizer(ngram_range=(1, 4), stop_words="english")
        self.batch_size = 1000
        # self._estimator_type = "classifier"

    def fit(self, X, y):
        """fit the vectorizer and the classifier

        Args:
            X (pd.Series(str)): inputs
            y (np.array(int)): labels
        """

        # check whether vectorizer has been fitted
        vec_fitted = False

        # number of batches to be processed
        if len(y) >= self.batch_size:
            num_batches = len(y) // self.batch_size
        else:
            num_batches = 1

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # for X_chunk, y_chunk in tqdm(
        #     zip(np.array_split(X, num_batches), np.array_split(y, num_batches),),
        #     total=num_batches,
        # ):

        for X_chunk, y_chunk in zip(
            np.array_split(X, num_batches), np.array_split(y, num_batches)
        ):
            self.clf.fit(X_chunk, y_chunk)

        return self

    def predict(self, X):

#         X_unicode = X.values.astype("U")
#         X_trans = self.vectorizer.transform(X_unicode)

        predictions = self.clf.predict(X)
        return predictions

    def get_params(self, deep):
        return self.clf.get_params(deep=deep)

    def set_params(self, **parameters):
        return self.clf.set_params(**parameters)

class BatchVectorizerXGBoostConcat:
    """Custom vectorizer and classifier class to process data in
    batches.  Useful when there are RAM and CPU limitations.

    Args:
        BaseEstimator (Estimator): base class for estimator
    """

    def __init__(
        self,
        objective,
        max_depth,
        gamma,
        reg_alpha,
        reg_lambda,
        colsample_bytree,
        min_child_weight,
        n_estimators,
        seed,
        use_label_encoder=False,
        base_score=None,
        booster=None,
        colsample_bylevel=None,
        colsample_bynode=None,
        enable_categorical=None,
        gpu_id=None,
        importance_type=None,
        interaction_constraints=None,
        learning_rate=None,
        max_delta_step=None,
        *args,
        **kwargs
    ):
        """intialization of class

        Args:
            vectorizer (Vectorizer): vectorizer to turn text into vectors
            batch_size (int): size of each batch used for training
        """

        self.vectorizer = HashingVectorizer(ngram_range=(1, 4), stop_words="english", n_features=1000)
        self.batch_size = 100000
        self.xgbclass = xgb.XGBClassifier(
            objective=objective,
            max_depth=max_depth,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            n_estimators=n_estimators,
            seed=seed,
            base_score=base_score,
            booster=booster,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            enable_categorical=enable_categorical,
            importance_type=importance_type,
            interaction_constraints=interaction_constraints,
            learning_rate=learning_rate,
            max_delta_step=max_delta_step
        )
        # self._estimator_type = "classifier"

    def fit(self, X_texts, X_node_embeds, y):
        """fit the vectorizer and the classifier

        Args:
            X_texts (pd.Series): input texts
            X_node_embeds (pd.Series): input node embeddings
            y (pd.Series): labels
        """

        # check whether vectorize and modelr has been fitted
        vec_fitted = False
        model_fitted = False

        # number of batches to be processed
        if len(y) >= self.batch_size:
            num_batches = len(y) // self.batch_size
        else:
            num_batches = 1
            
        for X_text_chunk, X_node_embeds_chunk, y_chunk in zip(
            np.array_split(X_texts, num_batches), np.array_split(X_node_embeds, num_batches), 
            np.array_split(y, num_batches)
        ):

            # convert text chunk to unicode
            X_text_chunk = X_text_chunk.values.astype("U")

            if not vec_fitted:
                X_text_trans = self.vectorizer.fit_transform(X_text_chunk)
                vec_fitted = True
            else:
                X_text_trans = self.vectorizer.transform(X_text_chunk)
            del X_text_chunk
            
            X_text_trans = X_text_trans.todense()
            X_concat = np.hstack([np.vstack(X_text_trans), np.vstack(X_node_embeds_chunk)])
            # xg_chunk = xgb.DMatrix(X_trans, label=y_chunk)
            y_chunk = np.array(y_chunk).reshape(-1)
            if model_fitted:
                self.xgbclass.fit(
                    X_concat, y_chunk, xgb_model=self.xgbclass.get_booster()
                )
            else:
                self.xgbclass.fit(X_concat, y_chunk)
                model_fitted = True
            # if os.path.exists("model.model"):
            #     self.model = xgb.train(params, xg_chunk, 30, xgb_model="model.model")
            # else:
            #     self.model = xgb.train(params, xg_chunk, 30)
            # self.model.save_model("model.model")

        return self
    
    def predict_proba(self,X_texts, X_node_embeds):
    
        X_unicode = X_texts.values.astype("U")
        X_text_trans = self.vectorizer.transform(X_unicode).todense()
        X_concat = np.hstack([np.vstack(X_text_trans), np.vstack(X_node_embeds)])
        # xg_test = xgb.DMatrix(X_trans)
        predictions_proba = self.xgbclass.predict_proba(X_concat)
        return predictions_proba
    
    def get_params(self, deep):
        return self.xgbclass.get_params(deep=deep)

    def set_params(self, **parameters):
        return self.xgbclass.set_params(**parameters)

    def predict(self, X_texts, X_node_embeds):

        X_unicode = X_texts.values.astype("U")
        X_text_trans = self.vectorizer.transform(X_unicode).todense()
        X_concat = np.hstack([np.vstack(X_text_trans), np.vstack(X_node_embeds)])
        # xg_test = xgb.DMatrix(X_trans)
        predictions = np.array(list(map(int, np.round(self.xgbclass.predict(X_concat)))))
        return predictions

        # predictions = self.clf.predict(X_trans)
        # return predictions


class BatchVectorizerClassifierCitationsNonTexts:
    """Custom vectorizer and classifier class to process data in
    batches.  Useful when there are RAM and CPU limitations.

    Args:
        BaseEstimator (Estimator): base class for estimator
    """

    def __init__(
        self,
        objective,
        max_depth,
        gamma,
        reg_alpha,
        reg_lambda,
        colsample_bytree,
        min_child_weight,
        n_estimators,
        seed=0,
        *args,
        **kwargs
    ):
        """intialization of class

        Args:
            clf (Estimator): classifier for training
            vectorizer (Vectorizer): vectorizer to turn text into vectors
            batch_size (int): size of each batch used for training
        """

        self.clf = xgb.XGBClassifier(
            objective=objective,
            max_depth=max_depth,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            n_estimators=180,
            seed=0,
        )
        self.vectorizer = HashingVectorizer(ngram_range=(1, 4), stop_words="english")
        self.batch_size = 1000
        # self._estimator_type = "classifier"

    def fit(self, X, y):
        """fit the vectorizer and the classifier

        Args:
            X (pd.Series(str)): inputs
            y (np.array(int)): labels
        """

        # check whether vectorizer has been fitted
        vec_fitted = False

        # number of batches to be processed
        if len(y) >= self.batch_size:
            num_batches = len(y) // self.batch_size
        else:
            num_batches = 1

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # for X_chunk, y_chunk in tqdm(
        #     zip(np.array_split(X, num_batches), np.array_split(y, num_batches),),
        #     total=num_batches,
        # ):

        for X_chunk, y_chunk in zip(
            np.array_split(X, num_batches), np.array_split(y, num_batches)
        ):
            self.clf.fit(X_chunk, y_chunk)

        return self

    def predict(self, X):

#         X_unicode = X.values.astype("U")
#         X_trans = self.vectorizer.transform(X_unicode)

        predictions = self.clf.predict(X)
        return predictions

    def get_params(self, deep):
        return self.clf.get_params(deep=deep)

    def set_params(self, **parameters):
        return self.clf.set_params(**parameters)


class BatchVectorizerXGBoost:
    """Custom vectorizer and classifier class to process data in
    batches.  Useful when there are RAM and CPU limitations.

    Args:
        BaseEstimator (Estimator): base class for estimator
    """

    def __init__(
        self,
        objective,
        max_depth,
        gamma,
        reg_alpha,
        reg_lambda,
        colsample_bytree,
        min_child_weight,
        n_estimators,
        seed,
        use_label_encoder=False,
        base_score=None,
        booster=None,
        colsample_bylevel=None,
        colsample_bynode=None,
        enable_categorical=None,
        gpu_id=None,
        importance_type=None,
        interaction_constraints=None,
        learning_rate=None,
        max_delta_step=None,
        *args,
        **kwargs
    ):
        """intialization of class

        Args:
            vectorizer (Vectorizer): vectorizer to turn text into vectors
            batch_size (int): size of each batch used for training
        """

        self.vectorizer = HashingVectorizer(ngram_range=(1, 4), stop_words="english", n_features=1000)
        self.batch_size = 100000
        self.xgbclass = xgb.XGBClassifier(
            objective=objective,
            max_depth=max_depth,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            n_estimators=n_estimators,
            seed=seed,
            base_score=base_score,
            booster=booster,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            enable_categorical=enable_categorical,
            importance_type=importance_type,
            interaction_constraints=interaction_constraints,
            learning_rate=learning_rate,
            max_delta_step=max_delta_step
        )
        # self._estimator_type = "classifier"

    def fit(self, X, y):
        """fit the vectorizer and the classifier

        Args:
            X (pd.Series(str)): inputs
            y (np.array(int)): labels
        """

        # check whether vectorize and modelr has been fitted
        vec_fitted = False
        model_fitted = False

        # number of batches to be processed
        if len(y) >= self.batch_size:
            num_batches = len(y) // self.batch_size
        else:
            num_batches = 1

        # for X_chunk, y_chunk in tqdm(
        #     zip(np.array_split(X, num_batches), np.array_split(y, num_batches),),
        #     total=num_batches,
        # ):

        for X_chunk, y_chunk in zip(
            np.array_split(X, num_batches), np.array_split(y, num_batches)
        ):

            # convert text chunk to unicode
            X_chunk = X_chunk.values.astype("U")

            if not vec_fitted:
                X_trans = self.vectorizer.fit_transform(X_chunk)
                vec_fitted = True
            else:
                X_trans = self.vectorizer.transform(X_chunk)
            del X_chunk

            # xg_chunk = xgb.DMatrix(X_trans, label=y_chunk)
            y_chunk = np.array(y_chunk).reshape(-1)
            if model_fitted:
                self.xgbclass.fit(
                    X_trans, y_chunk, xgb_model=self.xgbclass.get_booster()
                )
            else:
                self.xgbclass.fit(X_trans, y_chunk)
                model_fitted = True
            # if os.path.exists("model.model"):
            #     self.model = xgb.train(params, xg_chunk, 30, xgb_model="model.model")
            # else:
            #     self.model = xgb.train(params, xg_chunk, 30)
            # self.model.save_model("model.model")

        return self
    
    def get_params(self, deep):
        return self.xgbclass.get_params(deep=deep)

    def set_params(self, **parameters):
        return self.xgbclass.set_params(**parameters)
    
    def __sklearn_is_fitted__(self):
        return True
    
    def predict_proba(self, X):
        X_unicode = X.values.astype("U")
        X_trans = self.vectorizer.transform(X_unicode)
        # xg_test = xgb.DMatrix(X_trans)
        return self.xgbclass.predict_proba(X_trans)
    
    def decision_function(self, X):

        X_unicode = X.values.astype("U")
        X_trans = self.vectorizer.transform(X_unicode)

        predictions = self.xgbclass.decision_function(X_trans)
        return predictions
       
    
    def predict(self, X):

        X_unicode = X.values.astype("U")
        X_trans = self.vectorizer.transform(X_unicode)
        # xg_test = xgb.DMatrix(X_trans)
        predictions = np.array(list(map(int, np.round(self.xgbclass.predict(X_trans)))))
        return predictions

        # predictions = self.clf.predict(X_trans)
        # return predictions


class BatchVectorizerClassifier:
    """Custom vectorizer and classifier class to process data in
    batches.  Useful when there are RAM and CPU limitations.

    Args:
        BaseEstimator (Estimator): base class for estimator
    """

    def __init__(
        self,
        loss="hinge",
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=0.001,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
    ):
        """intialization of class

        Args:
            clf (Estimator): classifier for training
            vectorizer (Vectorizer): vectorizer to turn text into vectors
            batch_size (int): size of each batch used for training
        """

        self.clf = SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            n_jobs=n_jobs,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            warm_start=warm_start,
            average=average,
        )
        self.vectorizer = HashingVectorizer(ngram_range=(1, 4), stop_words="english", n_features=1000)
        self.batch_size = 100000
        # self._estimator_type = "classifier"

    def fit(self, X, y):
        """fit the vectorizer and the classifier

        Args:
            X (pd.Series(str)): inputs
            y (np.array(int)): labels
        """

        # check whether vectorizer has been fitted
        vec_fitted = False

        # number of batches to be processed
        if len(y) >= self.batch_size:
            num_batches = len(y) // self.batch_size
        else:
            num_batches = 1

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # for X_chunk, y_chunk in tqdm(
        #     zip(np.array_split(X, num_batches), np.array_split(y, num_batches),),
        #     total=num_batches,
        # ):

        for X_chunk, y_chunk in zip(
            np.array_split(X, num_batches), np.array_split(y, num_batches)
        ):

            # convert text chunk to unicode
            X_chunk = X_chunk.values.astype("U")

            if not vec_fitted:
                X_trans = self.vectorizer.fit_transform(X_chunk)
                vec_fitted = True
            else:
                X_trans = self.vectorizer.transform(X_chunk)
            del X_chunk

#             self.clf.fit(X_trans, y_chunk, classes=(self.classes_))
            self.clf.fit(X_trans, y_chunk)

        return self

    def predict(self, X):

        X_unicode = X.values.astype("U")
        X_trans = self.vectorizer.transform(X_unicode)

        predictions = self.clf.predict(X_trans)
        return predictions
    
    def decision_function(self, X):

        X_unicode = X.values.astype("U")
        X_trans = self.vectorizer.transform(X_unicode)

        predictions = self.clf.decision_function(X_trans)
        return predictions

    def predict_proba(self, X):
        # Check if fit has been called
#         check_is_fitted(self)

        # Input validation
#         X = check_array(X)
        X_unicode = X.values.astype("U")
        X_trans = self.vectorizer.transform(X_unicode)

        proba = self.clf.predict_proba(X_trans)
        return proba

    def get_params(self, deep):
        return self.clf.get_params(deep=deep)

    def set_params(self, **parameters):
        return self.clf.set_params(**parameters)
    
    
class BatchVectorizerClassifierCitations:
    """Custom vectorizer and classifier class to process data in
    batches.  Useful when there are RAM and CPU limitations.

    Args:
        BaseEstimator (Estimator): base class for estimator
    """

    def __init__(
        self,
        loss="hinge",
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=0.001,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
    ):
        """intialization of class

        Args:
            clf (Estimator): classifier for training
            vectorizer (Vectorizer): vectorizer to turn text into vectors
            batch_size (int): size of each batch used for training
        """

        self.clf = SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            n_jobs=n_jobs,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            warm_start=warm_start,
            average=average,
        )
        self.vectorizer = HashingVectorizer(ngram_range=(1, 4), stop_words="english", n_features=1000)
        self.batch_size = 100
        # self._estimator_type = "classifier"

    def fit(self, X, y):
        """fit the vectorizer and the classifier

        Args:
            X (pd.Series(str)): inputs
            y (np.array(int)): labels
        """

        # check whether vectorizer has been fitted
        vec_fitted = False

        # number of batches to be processed
        if len(y) >= self.batch_size:
            num_batches = len(y) // self.batch_size
        else:
            num_batches = 1

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # for X_chunk, y_chunk in tqdm(
        #     zip(np.array_split(X, num_batches), np.array_split(y, num_batches),),
        #     total=num_batches,
        # ):

        for X_chunk, y_chunk in zip(
            np.array_split(X, num_batches), np.array_split(y, num_batches)
        ):

            # convert text chunk to unicode
#             X_chunk = X_chunk.values.astype("U")

#             if not vec_fitted:
#                 X_trans = self.vectorizer.fit_transform(X_chunk)
#                 vec_fitted = True
#             else:
#                 X_trans = self.vectorizer.transform(X_chunk)
#             del X_chunk
            print('X_chunk', X_chunk)
            print('y_chunk', y_chunk)
            self.clf.fit(X_chunk, y_chunk)

        return self

    def predict(self, X):

#         X_unicode = X.values.astype("U")
#         X_trans = self.vectorizer.transform(X_unicode)

        predictions = self.clf.predict(X)
        return predictions

    # def predict_proba(self, X):
    #     # Check if fit has been called
    #     check_is_fitted(self)

    #     # Input validation
    #     X = check_array(X)
    #     X_unicode = X.values.astype("U")
    #     X_trans = self.vectorizer.transform(X_unicode)

    #     proba = self.clf.predict_proba(X_trans)
    #     return proba

    def get_params(self, deep):
        return self.clf.get_params(deep=deep)

    def set_params(self, **parameters):
        return self.clf.set_params(**parameters)

