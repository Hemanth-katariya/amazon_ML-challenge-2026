"""Reference points on the validation split, so later scores have context.

  * empty  : predict no match for everyone.
  * oracle : perfect predictions (sanity check for the scorer: must be 1.0).
"""
import config
from metric import load_ground_truth, score_report
from split import random_split, s1_frame


def fmt(rep):
    return "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v:,}"
                     for k, v in rep.items())


def main():
    truth = load_ground_truth(config.ground_truth_parquet())
    _, val = random_split(s1_frame())
    print("empty :", fmt(score_report({}, truth, val)))
    print("oracle:", fmt(score_report(truth, truth, val)))


if __name__ == "__main__":
    main()
