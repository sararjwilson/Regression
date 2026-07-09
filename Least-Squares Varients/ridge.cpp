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
int n = 3;

// data
int m = 100;

VectorXd x(m), y_clean(m);
for (int i = 0; i < m; i++) x(i) = i / 10.0;
for (int i = 0; i < m; i++) y_clean(i) = 10 + 3*x(i);

// noise 
double sigma = 15.0; // noise std dev

default_random_engine gen(42);
normal_distribution<double> dist(0.0, sigma);

VectorXd noise(m);
for (int i = 0; i < m; i++) noise(i) = dist(gen);

VectorXd y = y_clean + noise;

MatrixXd I = MatrixXd::Identity((n+1),(n+1));


    // X
    MatrixXd X(m, (n+1));
    for (int i = 0; i < m; i++){
        for (int j = 0; j < (n+1); j++){
            X(i,j) = pow(x(i),j);
        }
    }

     // k-fold cross validation
     int k = 5;
     int fold_size = m / k; 

     vector<double> lambdas = {0.001, 0.01, 0.1, 1, 10, 100, 1000};

     double best_lambda = lambdas[0];
     double best_rmse = numeric_limits<double>::max();

     for( double lambda : lambdas) {

        double total_rmse = 0.0;

        for ( int fold = 0; fold < k; fold ++) {
            int start_val = fold * fold_size;
            int end_val = start_val + fold_size;

            vector<int> train_idx, val_idx;
            for (int i = 0; i < m; i++) {
                if ( i >= start_val && i < end_val) val_idx.push_back(i);
                else train_idx.push_back(i);
            }

            int m_train = train_idx.size();
            int m_val = val_idx.size();

            MatrixXd X_train(m_train, (n+1));
            VectorXd y_train(m_train);

            for(int i = 0; i < m_train; i++){
                X_train.row(i) = X.row(train_idx[i]);
                y_train(i) = y(train_idx[i]);
            }

            
            MatrixXd X_val(m_val, (n+1));
            VectorXd y_val(m_val);
            
            for( int i =0; i < m_val; i++){
                X_val.row(i) = X.row(val_idx[i]);
                y_val(i) = y(val_idx[i]);
            }

            // set up system
            MatrixXd X_train_T = X_train.transpose();
            MatrixXd lhs = X_train_T * X_train + lambda * I;
            VectorXd rhs = X_train_T  * y_train;

            VectorXd ridge = lhs.colPivHouseholderQr().solve(rhs);
            VectorXd yhat_val = X_val * ridge;

            double val_rmse = sqrt((yhat_val - y_val).squaredNorm() / m_val);
            total_rmse += val_rmse;
        }

        double avg_rmse = total_rmse / k;

        cout << "lambda: " << lambda << " average rmse: "<< avg_rmse << "\n";

        if (avg_rmse < best_rmse) {
            best_rmse = avg_rmse;
            best_lambda = lambda;
        }
     }

    cout << "\nbest lambda: " << best_lambda << " with rmse: " << best_rmse << "\n";

    // set up system
    MatrixXd X_T = X.transpose();
    MatrixXd lhs = X_T * X + best_lambda * I;
    VectorXd rhs = X_T * y;

    VectorXd ridge = lhs.colPivHouseholderQr().solve(rhs);
    VectorXd yhat = X * ridge;

      cout << "\nRidge:\n";
    for (int i = 0; i < (n+1); i++){
        cout << " coeff[" << i << "]: " << ridge(i) << "\n";
    }

    // prep python plot
    ofstream file("ridge_cpp_output.csv");
    file << "x,y,yhat\n";
    for (int i = 0; i < m; i++)
        file << x(i) << "," << y(i) << "," << yhat(i) << "\n";
    file.close();

}