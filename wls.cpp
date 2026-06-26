#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <fstream>
#include <Eigen/Dense>
#include <random>

using namespace std;
using namespace Eigen;

int main() {

// polynomial deg
int n = 1;

// standard dev and correlation btwn adjacent errors
double sigma = 15.0;
double rho = 0.5;

// data
int m = 50;

VectorXd x(m), y_clean(m);
for (int i = 0; i < m; i++) x(i) = i;
for (int i = 0; i < m; i++) y_clean(i) = 10 + 3*x(i);

// noise
default_random_engine gen(42);
normal_distribution<double> dist(0.0,1.0);

VectorXd eps(m);
for (int i = 0; i < m; i++){
    eps(i) = dist(gen);
}

VectorXd noise(m);
noise(0) = eps(0);
for (int i = 1; i < m; i++){
    noise(i) = rho * noise(i-1) + sqrt(1-pow(rho,2)) * eps(i);
}

VectorXd y = y_clean + noise;

    // covarience matrix
    MatrixXd V(m,m);

        for (int i = 0; i < m; i++){
            for (int j = 0; j < m; j++){
                V(i,j)= pow(sigma, 2) * pow(rho, abs(i-j));
            }
        }
    // A
    MatrixXd A(m, (n+1));
    for (int i = 0; i < m; i++){
        for (int j = 0; j < (n+1); j++){
            A(i,j) = pow(x(i),j);
        }
    }
 
    // set up system
    VectorXd b = y;
    MatrixXd V_inv = V.inverse();
    MatrixXd A_T = A.transpose();
    MatrixXd lhs = A_T * V_inv * A;
    VectorXd rhs = A_T * V_inv * b;

    VectorXd wls = lhs.colPivHouseholderQr().solve(rhs);
    VectorXd yhat = A * wls;

    // calculate root mean square error (rmse)
    double rmse = sqrt((yhat - y).squaredNorm() / m);

    cout << "\nWLS:\n";
    for (int i = 0; i < (n+1); i++){
        cout << " coeff[" << i << "]: " << wls(i) << "\n";
    }
    cout << "rmse_wls: " << rmse << "\n";

    VectorXd ols_coeffs = A.colPivHouseholderQr().solve(y);
    VectorXd yhat_ols = A * ols_coeffs;
    double rmse_ols = sqrt((yhat_ols - y).squaredNorm() / m);

    cout << "\nOLS:\n";
    for (int i = 0; i < n+1; i++)
        cout << "coeff[" << i << "]: " << ols_coeffs(i) << "\n";
    cout << "rmse: " << rmse_ols << "\n";

    // prep python plot
    ofstream file("wls_cpp_output.csv");
    file << "x,y,yhat\n";
    for (int i = 0; i < m; i++)
        file << x(i) << "," << y(i) << "," << yhat(i) << "\n";
    file.close();

    // comparison

    // simulation
    int n_sims = 1000;
    MatrixXd wls_coeffs_all(n_sims, n+1);
    MatrixXd ols_coeffs_all(n_sims, n+1);

    for (int sim = 0; sim < n_sims; sim++) {
        // new noise each iteration
        VectorXd eps_s(m);
        for (int i = 0; i < m; i++) eps_s(i) = dist(gen);

        VectorXd noise_s(m);
        noise_s(0) = eps_s(0);
        for (int i = 1; i < m; i++)
            noise_s(i) = rho * noise_s(i-1) + sqrt(1-pow(rho,2)) * eps_s(i);

        VectorXd y_noisy = y_clean + noise_s;

        // wls
        VectorXd wls_s = lhs.colPivHouseholderQr().solve(A_T * V_inv * y_noisy);
        wls_coeffs_all.row(sim) = wls_s;

        // ols
        VectorXd ols_s = A.colPivHouseholderQr().solve(y_noisy);
        ols_coeffs_all.row(sim) = ols_s;
    }

    // mean
    VectorXd wls_mean = wls_coeffs_all.colwise().mean();
    VectorXd ols_mean = ols_coeffs_all.colwise().mean();

    // variance
    VectorXd wls_var(n+1), ols_var(n+1);
    for (int j = 0; j < n+1; j++) {
        wls_var(j) = (wls_coeffs_all.col(j).array() - wls_mean(j)).square().mean();
        ols_var(j) = (ols_coeffs_all.col(j).array() - ols_mean(j)).square().mean();
    }

    cout << "wls coeff variance:\n" << wls_var << "\n";
    cout << "ols coeff variance:\n" << ols_var << "\n";

}