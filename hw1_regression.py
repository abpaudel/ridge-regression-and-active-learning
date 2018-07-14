import numpy as np
import sys

def r_regression(X_train, y_train, lamda, variance):
    wRR = (np.linalg.inv(lamda*np.eye(X_train.shape[1]) + (X_train.T).dot(X_train))).dot((X_train.T).dot(y_train))
    return wRR

def a_learning(lamda, variance, X_train, X_test):
    covariance = np.linalg.inv(lamda * np.eye(X_train.shape[1]) + (1 / variance) * (X_train.T).dot(X_train))
    indices = list(range(X_test.shape[0]))
    active = []
    for i in range(0,10):
        variance_matrix = X_test.dot(covariance).dot(X_test.T)
        result = np.argmax(variance_matrix.diagonal())
        actual_row = indices[result]
        active.append(actual_row)
        np.concatenate((X_train, X_test[result].reshape(1, X_test[result].shape[0])))
        X_test = np.delete(X_test, (result), axis=0)
        indices.pop(result)
        covariance = np.linalg.inv(lamda * np.eye(X_train.shape[1]) + (1 / variance) * (X_train.T).dot(X_train))
    active = [j+1 for j in active]
    return active

def main():
    lambda_input = int(sys.argv[1])
    sigma2_input = float(sys.argv[2])
    X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
    y_train = np.genfromtxt(sys.argv[4])
    X_test = np.genfromtxt(sys.argv[5], delimiter = ",")
    wRR = r_regression(X_train, y_train, lambda_input, sigma2_input)
    np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n")
    active = a_learning(lambda_input, sigma2_input, X_train, X_test)
    np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=";;")


if __name__ == "__main__":
    main()