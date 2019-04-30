import svm_train
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process Command line variables")
    parser.add_argument("--data_dir", nargs=1)
    parser.add_argument("--results_dir", nargs=1)
    args = parser.parse_args()
    data_dir = args.data_dir[0]
    results_dir = args.results_dir[0]

    svm_train.train_linear_svm(data_dir, results_dir)
    svm_train.train_gauss_svm(data_dir, results_dir)
    return 0


if __name__ == "__main__":
    main()
