import pandas as pd

from src.splits import split_xy_by_issue_date, time_split_masks


def test_time_split_masks_use_fixed_issue_date_windows():
    issue_d = pd.to_datetime(
        [
            "2015-12-31",
            "2016-01-01",
            "2016-12-31",
            "2017-01-01",
        ]
    )

    train_mask, val_mask, test_mask = time_split_masks(pd.Series(issue_d))

    assert train_mask.tolist() == [True, False, False, False]
    assert val_mask.tolist() == [False, True, True, False]
    assert test_mask.tolist() == [False, False, False, True]


def test_split_xy_by_issue_date_keeps_issue_d_out_of_features():
    X = pd.DataFrame({"loan_amnt": [1, 2, 3], "grade_enc": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    issue_d = pd.Series(pd.to_datetime(["2015-06-01", "2016-06-01", "2017-06-01"]))

    X_train, X_val, X_test, y_train, y_val, y_test = split_xy_by_issue_date(X, y, issue_d)

    assert X_train["loan_amnt"].tolist() == [1]
    assert X_val["loan_amnt"].tolist() == [2]
    assert X_test["loan_amnt"].tolist() == [3]
    assert y_train.tolist() == [0]
    assert y_val.tolist() == [1]
    assert y_test.tolist() == [0]
