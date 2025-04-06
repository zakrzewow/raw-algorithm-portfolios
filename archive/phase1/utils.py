import numpy as np


def get_n_splits(df, n, instance_number, solver_number, random_state=0):
    """
    Generate indices to split data into training and test sets.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing solver evaluation data
    n : int
        Number of splits
    instance_number : int
        Number of instances to select
    solver_number : int
        Number of solver configurations to select
    random_state : int, default=0
        Random state for reproducibility

    Yields:
    -------
    train_idx : numpy.array
        Indices of training set
    test_idx : numpy.array
        Indices of test set

    Notes:
    ------
    1. Randomly selects solver_number solvers and instance_number instances from df
    2. Ensures that all evaluations from one solver are either in train or test set
    """
    rng = np.random.default_rng(random_state)

    if solver_number % n != 0:
        raise ValueError(
            f"solver_number ({solver_number}) must be divisible by n ({n})"
        )

    all_solver_ids = df["solver_id"].unique()
    all_instance_ids = df["instance_id"].unique()

    if len(all_solver_ids) < solver_number:
        raise ValueError(
            f"Not enough solvers in df ({len(all_solver_ids)}) to select {solver_number}"
        )

    if len(all_instance_ids) < instance_number:
        raise ValueError(
            f"Not enough instances in df ({len(all_instance_ids)}) to select {instance_number}"
        )

    selected_solver_ids = rng.choice(all_solver_ids, size=solver_number, replace=False)
    selected_instance_ids = rng.choice(
        all_instance_ids, size=instance_number, replace=False
    )

    subset_df = df[
        df["solver_id"].isin(selected_solver_ids)
        & df["instance_id"].isin(selected_instance_ids)
    ]

    expected_rows = solver_number * instance_number
    if len(subset_df) != expected_rows:
        raise ValueError(
            f"Incomplete data: Found {len(subset_df)} rows instead of expected {expected_rows}"
        )

    solvers_per_fold = solver_number // n
    shuffled_solver_ids = rng.permutation(selected_solver_ids)

    for i in range(n):
        test_solvers = shuffled_solver_ids[
            i * solvers_per_fold : (i + 1) * solvers_per_fold
        ]

        test_idx = subset_df[subset_df["solver_id"].isin(test_solvers)].index.values
        train_idx = subset_df[~subset_df["solver_id"].isin(test_solvers)].index.values

        total_samples = len(train_idx) + len(test_idx)
        if total_samples != expected_rows:
            raise ValueError(
                f"Total samples ({total_samples}) doesn't match expected number ({expected_rows})"
            )

        yield train_idx, test_idx
