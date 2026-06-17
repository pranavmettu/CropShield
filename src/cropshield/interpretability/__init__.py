"""
Interpretability modules for CropShield.

Submodules
----------
feature_importance
    Tree model feature importance extraction and plotting.
shap_analysis
    SHAP value computation and summary visualisation.

Important note
--------------
Feature importance and SHAP values describe correlations within the
training data. They do not prove causal relationships between weather,
drought, and yield outcomes. Interpret results with this limitation
clearly in mind.
"""
