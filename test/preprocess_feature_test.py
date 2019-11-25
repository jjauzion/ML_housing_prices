import pandas as pd

from src import preprocess_feature


def test_transform_feature_add():
    df = pd.DataFrame({"a": [0, 1, 2, 3], "b": [10, 11, 12, 13]})
    ans = df["a"] + df["b"]
    dic = {
        "a+b": {
            "from": ["a", "b"],
            "operator": "+"
        }
    }
    to_del = preprocess_feature.transform_feature(df, dic, delete_original=True)
    assert len(to_del) == len(["a", "b"])
    for key in to_del:
        assert key in ["a", "b"]
    assert df["a+b"].equals(ans)
    assert "a" not in df.columns
    assert "b" not in df.columns


def test_transform_feature_add_same_col():
    df = pd.DataFrame({"a": [0, 1, 2, 3], "b": [10, 11, 12, 13]})
    ans = df["a"] + df["a"]
    dic = {
        "a+a": {
            "from": ["a", "a"],
            "operator": "+",
            "delete_original": "True"
        }
    }
    to_del = preprocess_feature.transform_feature(df, dic, delete_original=True)
    assert len(to_del) == len(["a"])
    for key in to_del:
        assert key in ["a"]
    assert df["a+a"].equals(ans)
    assert "a" not in df.columns
    assert "b" in df.columns


def test_transform_feature_keep_original():
    df = pd.DataFrame({"a": [0, 1, 2, 3], "b": [10, 11, 12, 13]})
    ans = df["a"] + df["b"]
    dic = {
        "a+b": {
            "from": ["a", "b"],
            "operator": "+",
            "delete_original": "False"
        }
    }
    to_del = preprocess_feature.transform_feature(df, dic, delete_original=True)
    assert len(to_del) == 0
    for key in to_del:
        assert key in []
    assert df["a+b"].equals(ans)
    assert "a" in df.columns
    assert "b" in df.columns


def test_transform_feature_keep_original_2():
    df = pd.DataFrame({"a": [0, 1, 2, 3], "b": [10, 11, 12, 13]})
    ans = df["a"] + df["b"]
    dic = {
        "a+b": {
            "from": ["a", "b"],
            "operator": "+",
            "delete_original": "True"
        }
    }
    to_del = preprocess_feature.transform_feature(df, dic, delete_original=False)
    assert len(to_del) == len(["a", "b"])
    for key in to_del:
        assert key in ["a", "b"]
    assert df["a+b"].equals(ans)
    assert "a" in df.columns
    assert "b" in df.columns


def test_transform_feature_sub():
    df = pd.DataFrame({"a": [0, 1, 2, 3], "b": [10, 11, 12, 13]})
    ans = df["a"] - df["b"]
    dic = {
        "a-b": {
            "from": ["a", "b"],
            "operator": "-"
        }
    }
    to_del = preprocess_feature.transform_feature(df, dic, delete_original=True)
    assert len(to_del) == len(["a", "b"])
    for key in to_del:
        assert key in ["a", "b"]
    assert df["a-b"].equals(ans)
    assert "a" not in df.columns
    assert "b" not in df.columns


def test_transform_feature_multi_trans():
    df = pd.DataFrame({"a": [0, 1, 2, 3], "b": [10, 11, 12, 13]})
    ans1 = df["a"] + df["b"]
    ans2 = df["a"] + df["a"]
    ans3 = df["a"] - df["b"]
    dic = {
        "a+b": {
            "from": ["a", "b"],
            "operator": "+",
            "delete_original": "False"
        },
        "a-b": {
            "from": ["a", "b"],
            "operator": "-",
            "delete_original": "False"
        },
        "a+a": {
            "from": ["a", "a"],
            "operator": "+",
            "delete_original": "True"
        }
    }
    to_del = preprocess_feature.transform_feature(df, dic, delete_original=True)
    assert len(to_del) == len(["a"])
    for key in to_del:
        assert key in ["a"]
    assert df["a+b"].equals(ans1)
    assert df["a+a"].equals(ans2)
    assert df["a-b"].equals(ans3)
    assert "a" not in df.columns
    assert "b" in df.columns
