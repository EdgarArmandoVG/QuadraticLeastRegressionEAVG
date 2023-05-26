#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

class Dataset {
private:
    Eigen::VectorXd vardnty;
    Eigen::VectorXd varintx;

public:
    // Constructor
    Dataset(const Eigen::VectorXd& datos1, const Eigen::VectorXd& datos2) : vardnty(datos1), varintx(datos2) {}

    Eigen::VectorXd getVardnty() const {
        return vardnty;
    }

    Eigen::VectorXd getVarintx() const {
        return varintx;
    }
};

class QuadraticLeastRegression {
private:
    double beta0;
    double beta1;
    double beta2;
    double rsquared;

public:
    void train(const Dataset& dataset) {
        Eigen::VectorXd X = dataset.getVarintx();
        Eigen::VectorXd Y = dataset.getVardnty();

        Eigen::MatrixXd A(X.size(), 3);
        A << Eigen::VectorXd::Ones(X.size()), X, X.array().square();

        Eigen::VectorXd coefficients = (A.transpose() * A).ldlt().solve(A.transpose() * Y);

        beta0 = coefficients[0];
        beta1 = coefficients[1];
        beta2 = coefficients[2];

        double sum_y = Y.sum();
        double mean_y = sum_y / Y.size();
        double ssr = 0.0;
        double sst = 0.0;

        for (int i = 0; i < Y.size(); i++) {
            double yhat = beta0 + beta1 * X[i] + beta2 * (X[i] * X[i]);
            ssr += (yhat - yhat) * (yhat - yhat);
            sst += (Y[i] - mean_y) * (Y[i] - mean_y);
        }

        rsquared = 1 - (ssr / sst);
    }

    double predict(double varx) const {
        return beta0 + (beta1 * varx) + (beta2 * (varx * varx));
    }

    double getBeta0() const {
        return beta0;
    }

    double getBeta1() const {
        return beta1;
    }

    double getBeta2() const {
        return beta2;
    }

    double getR_squared() const {
        return rsquared;
    }
};

int main() {
    std::vector<double> varsx = {-3, -2, -1, 0, 1, 2, 3};
    std::vector<double> varsy = {7.5, 3, 0.5, 1, 3, 6, 14};
    Eigen::VectorXd X(varsx.size());
    Eigen::VectorXd Y(varsy.size());

    for (int i = 0; i < varsx.size(); i++) {
        X(i) = varsx[i];
        Y(i) = varsy[i];
    }

    Dataset datasetvals(Y, X);

    QuadraticLeastRegression quadraticRegression;
    quadraticRegression.train(datasetvals);

    double varxpredict;
    std::cout << "Introduzca X para predecir Y: ";
    std::cin >> varxpredict;

    double varypredict = quadraticRegression.predict(varxpredict);
    double rsquared = quadraticRegression.getR_squared();

    std::cout << "Y = B0 + B1*X + B2*X^2" << std::endl;
    std::cout << "Y = (" << quadraticRegression.getBeta0() << ") + (" << quadraticRegression.getBeta1() << ")*X + (" << quadraticRegression.getBeta2() << ")*X^2" << std::endl;
    std::cout << "Y = " << varypredict << std::endl;
    std::cout << "R-squared = " << rsquared << std::endl;

    return 0;
}
