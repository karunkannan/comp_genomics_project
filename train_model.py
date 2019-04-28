import svm_train


def main():
    svm_train.train_linear_svm("./data", "./plots")
    svm_train.train_gauss_svm("./data", "./plots")
    return 0


if __name__ == "__main__":
    main()
