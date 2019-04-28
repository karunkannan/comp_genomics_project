import svm_train


def main():
    svm_train.predict_linear_svm("./data", "./plots")
    svm_train.predict_gauss_svm("./data", "./plots")
    return 0


if __name__ == "__main__":
    main()
