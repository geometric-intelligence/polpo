from sklearn.compose import TransformedTargetRegressor
from sklearn.utils.validation import check_is_fitted


class ObjectBasedTransformedTargetRegressor(TransformedTargetRegressor):
    def fit(self, X, y, **fit_params):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        **fit_params : dict
            - If `enable_metadata_routing=False` (default): Parameters directly passed
              to the `fit` method of the underlying regressor.

            - If `enable_metadata_routing=True`: Parameters safely routed to the `fit`
              method of the underlying regressor.

            .. versionchanged:: 1.6
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if y is None:
            raise ValueError(
                f"This {self.__class__.__name__} estimator "
                "requires y to be passed, but the target y is None."
            )
        self._fit_transformer(y)

        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y)

        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)

        self.regressor_ = self._get_regressor(get_clone=True)

        self.regressor_.fit(X, y_trans, **fit_params)

        if hasattr(self.regressor_, "feature_names_in_"):
            self.feature_names_in_ = self.regressor_.feature_names_in_

        return self

    def predict(self, X, **predict_params):
        """Predict using the base regressor, applying inverse.

        The regressor is used to predict and the `inverse_func` or
        `inverse_transform` is applied before returning the prediction.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        **predict_params : dict of str -> object
            Parameters passed to the `predict` method of the underlying
            regressor.

        Returns
        -------
        y_hat : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self)
        pred = self.regressor_.predict(X, **predict_params)
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(pred.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(pred)

        return pred_trans
