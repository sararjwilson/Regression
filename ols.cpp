#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <fstream>

using namespace std;

int main() {

// polynomial degree
int n = 3;

vector<double> x = {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6};

// y = 1 + 2x + 3x^2 + 4x^3
vector<double> y = {
    4*(-27) + 3*9  + 2*(-3) + 1,   // -98
    4*(-8)  + 3*4  + 2*(-2) + 1,   // -23
    4*(-1)  + 3*1  + 2*(-1) + 1,   //   0
    4*(0)   + 3*0  + 2*(0)  + 1,   //   1
    4*(1)   + 3*1  + 2*(1)  + 1,   //  10
    4*(8)   + 3*4  + 2*(2)  + 1,   //  41
    4*(27)  + 3*9  + 2*(3)  + 1,   // 112
    4*(64)  + 3*16 + 2*(4)  + 1,   // 233
    4*(125) + 3*25 + 2*(5)  + 1,   // 416
    4*(216) + 3*36 + 2*(6)  + 1    // 673
};


int m = x.size();
// A
    vector<vector<double>> A(m, vector<double>(n+1, 0.0));

        for (int i = 0; i < m; i++){
            for (int j = 0; j < (n+1); j++){
                A[i][j] = pow(x[i], j);
            }
        }
    // A transpose
    vector<vector<double>> A_T((n+1), vector<double>(m,0.0));
    
    for (int i = 0; i < m; i++){
        for (int j = 0; j < (n+1); j++){
            A_T[j][i] = A[i][j];
        }
    }
    // matrix multiplication
    vector<vector<double>> matrix_mult((n+1), vector<double>((n+1),0.0));

    for (int i = 0; i < (n+1); i++){
        for (int j = 0; j < (n+1); j ++){
            for (int k = 0; k < m; k++){
                matrix_mult[i][j] += A_T[i][k] * A[k][j];
             }
        }
    }
    // determinant

    vector<double> A_T_b((n+1), 0.0);

    // A transpose times b
    for ( int i = 0; i < (n+1); i++){
        for( int j = 0; j < m; j++) {
            A_T_b[i] += A_T[i][j] * y[j];
        }
    }

    // calculate inverse

    // build augmented matrix, then gaussian elim
    vector<vector<double>> aug( (n+1), vector<double>((n+2), 0.0));
        for (int i = 0; i < (n+1); i++) {
            for (int j = 0; j < (n+1); j++) {
                aug[i][j] = matrix_mult[i][j];
            }
            aug[i][n+1] = A_T_b[i];
        }

    // forward elim
    for ( int j = 0; j < (n+1); j++) {
        int pivot = -1;
        double best = 0.0;
        for ( int i = j; i < (n+1); i++) {
            if (abs(aug[i][j]) > best) {
                best = abs(aug[i][j]);
                pivot = i;
            } 
        }
        if (pivot == -1) {
            cout << "Matrix is singular and cannot be inverted. \n";
        }
        swap(aug[j],aug[pivot]);
        for (int i = (j+1); i< (n+1); i++) {
            double mult = aug[i][j] / aug[j][j];
            for(int k = j; k < (n+2); k++){
                aug[i][k] -= mult * aug[j][k];
            }
        }  
    }

    // back sub
    vector<double> ols((n+1),0.0);
    for( int i = n; i >= 0; i--) {
        ols[i] = aug[i][n+1];
        for (int j = (i+1); j < (n+1); j++) {
            ols [ i] -= aug[i][j] * ols[j];
        }
        ols[i] /= aug[i][i];
    }



    // calculate root mean square error (rmse)
    double rmse = 0.0;

    for (int i = 0; i < m; i++){
        double yhat = 0.0;
        for (int j = 0; j < (n+1); j++) {
            yhat += ols[j] * pow(x[i], j);
        }
        rmse += pow(yhat - y[i], 2);
    }

    rmse = sqrt(rmse/m);

    for (int i = 0; i < (n+1); i++){
        cout << " coeff[" << i << "]: " << ols[i] << "\n";
    }
    cout << "rmse: " << rmse << "\n";
    
    // prep python plot
    ofstream file("ols_cpp_output.csv");
    file << "x,y,yhat\n";
    for(int i = 0; i < m; i++){
        double yhat = 0.0;
        for(int j = 0; j < (n+1); j++){
            yhat += A[i][j] * ols[j];
        }
        file << x[i] << "," << y[i] << "," << yhat << "\n";
    }
    file.close();

}           