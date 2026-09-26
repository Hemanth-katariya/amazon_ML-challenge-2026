"""Checks the local scorer against the problem statement. Run: python test_metric.py"""
import math

from metric import f05, macro_f05, score_report


def close(a, b, tol=1e-3):
    return math.isclose(a, b, abs_tol=tol)


def test_problem_statement_example():
    # PS: predict [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812] -> 0.714
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    assert close(f05(pred, true), 0.714), f05(pred, true)


def test_singletons():
    assert f05(set(), set()) == 1.0            # correctly predicted no match
    assert f05({"S2-1"}, set()) == 0.0         # false merge on a singleton


def test_misses_and_perfect():
    assert f05(set(), {"S2-1"}) == 0.0         # predicted empty, had matches
    assert f05({"S3-9"}, {"S2-1"}) == 0.0      # all wrong
    assert f05({"S2-1", "S3-2"}, {"S2-1", "S3-2"}) == 1.0


def test_precision_weighted_more_than_recall():
    true = {"a", "b", "c", "d"}
    under = f05({"a", "b"}, true)              # P=1,  R=0.5
    over = f05(true | {"x", "y", "z", "w"}, true)  # P=0.5, R=1
    assert under > over, (under, over)


def test_macro_average_and_missing_rows():
    truth = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S3-1", "S3-2"}}
    pred = {"S1-1": {"S2-1"}, "S1-3": {"S3-1"}}  # S1-2 missing -> empty -> 1.0
    expected = (1.0 + 1.0 + f05({"S3-1"}, {"S3-1", "S3-2"})) / 3
    assert close(macro_f05(pred, truth), expected)
    rep = score_report(pred, truth)
    assert close(rep["f05"], expected)
    assert close(rep["singleton_share"], 1 / 3)
    assert rep["singleton_f05"] == 1.0
    assert close(rep["matched_recall"], 0.75)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"all {len(tests)} tests passed")
