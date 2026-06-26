#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <fstream>

using namespace std;

int main() {
  // set up system
vector<double> x = {1, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 20, 21, 22, 24, 25, 27, 28};
vector<double> y = {3.2, 7.1, 6.4, 9.8, 8.3, 12.5, 11.2, 14.7, 13.1, 17.4, 15.9, 18.2, 20.6, 19.3, 22.8, 21.4, 24.1, 26.7, 25.3, 28.9};

int n = x.size();

    vector<vector<double>> A(n, vector<double>(2, 0.0));

        for (int i = 0; i < n; i++){
            for (int j = 0; j < 2; j++){
                A[i][j] = pow(x[i], j);
            }
        }
    
    vector<vector<double>> A_T(2, vector<double>(n,0.0));
    
    for (int i = 0; i < n; i++){
        for (int j = 0; j < 2; j++){
            A_T[j][i] = A[i][j];
        }
    }

    vector<vector<double>> matrix_mult(2, vector<double>(2,0.0));

    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j ++){
            for (int k = 0; k < n; k++){
                matrix_mult[i][j] += A_T[i][k] * A[k][j];
             }
        }
    }

    double a = matrix_mult[0][0];
    double b = matrix_mult[0][1];
    double c = matrix_mult[1][0];
    double d = matrix_mult[1][1];

    double det = (a*d) - (b*c);

    vector<vector<double>> matrix_mult_inv(2, vector<double>(2,0.0));

    if (abs(det) < numeric_limits<double>::epsilon()){
        cout << "Matrix cannot be inverted. \n";
    }
    else {
        double inv_det = 1.0 / det;
        matrix_mult_inv[0][0] = d*inv_det;
        matrix_mult_inv[0][1] = -b*inv_det;
        matrix_mult_inv[1][0] = -c*inv_det;
        matrix_mult_inv[1][1] = a*inv_det;
    }

    vector<double> A_T_b(2,0.0);
    
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < n; j++){
            A_T_b[i] += A_T[i][j] * y[j];
        }
    }

    vector<double> ols(2,0.0);

    for(int i = 0; i < 2; i ++){
        for (int j = 0; j < 2; j++){
            ols[i] += matrix_mult_inv[i][j] * A_T_b[j];
        }
    }

    double rmse = 0.0;

    for (int i = 0; i < n; i++){
        double y_hat = ols[0] + ols[1] * x[i];
        rmse += pow(y_hat - y[i], 2);
    }

    rmse = sqrt(rmse/n);

    cout << "intercept: " << ols[0] << "\n";
    cout << "slope: " << ols[1] << "\n";
    cout << "rmse: " << rmse << "\n";

    ofstream file("ols_cpp_output.csv");
    file << "x,y,y_hat\n";
    for(int i = 0; i < n; i++){
        double y_hat = 0.0;
        for(int j = 0; j < 2; j++){
            y_hat += A[i][j] * ols[j];
        }
        file << x[i] << "," << y[i] << "," << y_hat << "\n";
    }
    file.close();

}           