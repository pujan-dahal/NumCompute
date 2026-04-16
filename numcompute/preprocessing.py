"""
numcompute/preprocessing.py
============================
Data preprocessing transformers for NumCompute.
 
Every class follows the same three-method contract:
 
    transformer.fit(X)            # learn parameters from training data
    transformer.transform(X)      # apply the learned transformation
    transformer.fit_transform(X)  # both at once (convenience shortcut)
 
All hot paths use NumPy vectorised operations.  No Python loops over
rows or columns anywhere in the production code.
 
 
Author : (Anshu Shrestha)
"""
 
import numpy as np
from typing import Optional, Tuple, List
 

# Internal base class — shared boilerplate

 
class _BaseScaler:
    """Shared boilerplate for all transformers.
 
    Provides ``fit_transform`` so subclasses only need to implement
    ``fit`` and ``transform``.  Not part of the public API.
    """
 
    def fit(self, X: np.ndarray) -> "_BaseScaler":
        raise NotImplementedError
 
    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
 
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to *X*, transform *X*, and return the result.
 
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
 
        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)
 
    @staticmethod
    def _validate(
        X: np.ndarray,
        fitted_attr: Optional[str] = None,
        obj=None,
    ) -> np.ndarray:
        """Coerce *X* to a 2-D float64 array; optionally check fit state.
 
        Parameters
        ----------
        X : array-like
            Will be coerced to ``np.float64``.
        fitted_attr : str, optional
            Name of an attribute that exists only after fitting.
        obj : object, optional
            The transformer instance to inspect.
 
        Returns
        -------
        np.ndarray, shape (n_samples, n_features), dtype float64
 
        Raises
        ------
        ValueError
            If *X* cannot be represented as a 2-D numeric array.
        RuntimeError
            If the transformer has not been fitted yet.
        """
        try:
            X = np.array(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"X must be convertible to a numeric NumPy array. "
                f"Original error: {exc}"
            )
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples, n_features), got shape {X.shape}."
            )
        if fitted_attr is not None and obj is not None:
            if not hasattr(obj, fitted_attr):
                raise RuntimeError(
                    "This transformer has not been fitted yet. "
                    "Call fit(X) before transform(X)."
                )
        return X
 