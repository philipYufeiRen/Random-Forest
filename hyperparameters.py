
# Use Validation Set to Tune hyperparameters for the Amazon dataset
# Use Optimal Parameters to get good accuracy on Test Set
AMAZON_HYPERPARAMETERS = {
    # NOTE: 10000 takes quite a while to run. Suppose if we ever want to modify the code again,
    #       we could parallelize the code such that each tree is constructed in parallel (why not right?)
    "num_trees": 600,
    "features_percent": 0.006,
    "data_percent": 0.6,
    "max_depth": 90,
    "min_leaf_data": 20,
    "min_entropy": 0.1,
    "num_split_retries": 6
}
# Use Validation Set to Tune hyperparameters for the Titanic dataset
# Use Optimal Parameters to get good accuracy on Test Set
TITANIC_HYPERPARAMETERS = {
    "num_trees": 10,
    "features_percent": 0.8,
    "data_percent": 0.8,
    "max_depth": 20,
    "min_leaf_data": 0,
    "min_entropy": 0,
    "num_split_retries": 10
}
