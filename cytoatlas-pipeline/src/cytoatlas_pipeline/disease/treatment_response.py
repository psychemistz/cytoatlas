"""
Treatment response prediction using activity signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ResponsePredictionResult:
    """Result of treatment response prediction."""

    auc_cv: float
    feature_importance: pd.Series
    top_predictors: list[str]
    n_responders: int
    n_non_responders: int
    model_type: str


class TreatmentResponsePredictor:
    """Predicts treatment response from activity signatures."""

    def __init__(
        self,
        model_type: str = "ensemble",
        cv_folds: int = 5,
    ):
        self.model_type = model_type
        self.cv_folds = cv_folds

    def predict(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        response_col: str = "response",
        responder_label: str = "responder",
    ) -> ResponsePredictionResult:
        """Train and evaluate response predictor."""
        # Align
        common = list(set(activity.columns) & set(metadata.index))
        activity = activity[common]
        metadata = metadata.loc[common]

        # Prepare data
        X = activity.T.values
        y = (metadata[response_col] == responder_label).astype(int).values

        n_responders = y.sum()
        n_non_responders = len(y) - n_responders

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Models
        if self.model_type == "logistic":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == "rf":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            # Ensemble: average of both
            lr = LogisticRegression(max_iter=1000, random_state=42)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)

            lr_scores = cross_val_score(lr, X_scaled, y, cv=self.cv_folds, scoring="roc_auc")
            rf_scores = cross_val_score(rf, X_scaled, y, cv=self.cv_folds, scoring="roc_auc")

            auc_cv = (lr_scores.mean() + rf_scores.mean()) / 2

            # Fit for feature importance
            lr.fit(X_scaled, y)
            rf.fit(X_scaled, y)

            importance = (np.abs(lr.coef_[0]) + rf.feature_importances_) / 2
            feature_importance = pd.Series(importance, index=activity.index)
            top_predictors = feature_importance.nlargest(10).index.tolist()

            return ResponsePredictionResult(
                auc_cv=auc_cv,
                feature_importance=feature_importance,
                top_predictors=top_predictors,
                n_responders=n_responders,
                n_non_responders=n_non_responders,
                model_type="ensemble",
            )

        # Single model
        scores = cross_val_score(model, X_scaled, y, cv=self.cv_folds, scoring="roc_auc")
        model.fit(X_scaled, y)

        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        else:
            importance = model.feature_importances_

        feature_importance = pd.Series(importance, index=activity.index)
        top_predictors = feature_importance.nlargest(10).index.tolist()

        return ResponsePredictionResult(
            auc_cv=scores.mean(),
            feature_importance=feature_importance,
            top_predictors=top_predictors,
            n_responders=n_responders,
            n_non_responders=n_non_responders,
            model_type=self.model_type,
        )


def predict_treatment_response(
    activity: pd.DataFrame,
    metadata: pd.DataFrame,
    response_col: str = "response",
) -> ResponsePredictionResult:
    """Convenience function for response prediction."""
    predictor = TreatmentResponsePredictor()
    return predictor.predict(activity, metadata, response_col)
