#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <fstream>

using namespace std;
using namespace Eigen;

//   g++ -I /opt/homebrew/opt/eigen/include/eigen3 sindy_lorenz.cpp -o sindy 

// set up parameters 
const double sigma = 10;
const double rho = 28;
const double beta = 8.0/3.0;
double tol = numeric_limits<double>::epsilon();

Vector3d f(const Vector3d& state) {
// system dynamics:
// ẋ = σ(y − x)
// ẏ = x(ρ − z) − y
// ż = xy − βz

    double x = state(0);
    double y = state(1);
    double z = state(2);

    double xdot = sigma * (y - x);
    double ydot = x * (rho - z) - y;
    double zdot = x * y - beta * z;

    return Vector3d(xdot, ydot, zdot);
}

// RK4
Vector3d rk4step(const Vector3d& state, double dt) {

    Vector3d k1 = f(state);
    Vector3d k2 = f(state + k1 * (dt/2));
    Vector3d k3 = f(state + k2 * (dt/2));
    Vector3d k4 = f(state + k3 * dt);

    Vector3d state_next = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    return state_next;
}

// central difference
vector<Vector3d> fd(const vector<Vector3d>& states, double dt) {
    int n = states.size();
    vector<Vector3d> fd_est;

    for (int i = 1; i < (n-1); i++){
       Vector3d fd_steps = (states[i+1] - states[i-1]) / (2*dt);
       fd_est.push_back(fd_steps);
    }
    return fd_est;
}

// noise
vector<Vector3d> add_noise(const vector<Vector3d>& states, double noise_lvl){
    default_random_engine generator;
    normal_distribution<double> distribution(0.0, noise_lvl);
    vector<Vector3d> states_noise(states.size());

    for(int i = 0; i < states.size(); i++){
        double x_noise = distribution(generator);
        double y_noise = distribution(generator);
        double z_noise = distribution(generator);

        states_noise[i] = Vector3d(states[i](0) + x_noise, states[i](1) + y_noise, states[i](2) + z_noise);
    }
    return states_noise;
}

// SINDy set-up
RowVectorXd build_library_row(const Vector3d& state) {
    double x = state(0);
    double y = state(1);
    double z = state(2);

    RowVectorXd row(10);
    row << 1, x, y, z, x*x, y*y, z*z, x*y, x*z, y*z;
    return row;
}

MatrixXd build_library(const vector<Vector3d>& states){
    int n = states.size();
    int num_terms = 10; 
    
    MatrixXd Theta(n, num_terms);

    for(int i = 0; i < n; i++){
        Theta.row(i) = build_library_row(states[i]);
    }
    return Theta;
}

VectorXd build_target(const vector<Vector3d>& states, int component){
    int n = states.size();
    VectorXd target(n); 

    for (int i = 0; i < n; i++){
        Vector3d derivative = f(states[i]);
        target(i) = derivative(component);
    }
    return target;
}

VectorXd stlsq(const MatrixXd& Theta, const VectorXd& target, double lambda){
    vector<int> indices(Theta.cols());
    for (int i = 0; i < Theta.cols(); i++){
        indices[i] = i;
    }

    bool convergence = false;
    while(!convergence){
        int prior_size = indices.size();

        MatrixXd Theta_slimmed = Theta(Eigen::indexing::all, indices);
        VectorXd xi = Theta_slimmed.householderQr().solve(target);

        vector<int> new_indices;
        for(int i = 0; i < indices.size(); i++){
            if(abs(xi(i)) >= lambda){
                new_indices.push_back(indices[i]);
            }
        }
        indices = new_indices;

        if(indices.size() == prior_size){
            convergence = true;
        }
    }
    MatrixXd Theta_final = Theta(Eigen::indexing::all, indices);
    VectorXd xi_final_slim = Theta_final.householderQr().solve(target);

    VectorXd xi_full = VectorXd::Zero(Theta.cols());

    for (int i = 0; i < indices.size(); i++) {
        xi_full(indices[i]) = xi_final_slim(i);
    }

    return xi_full;
}

// recover ode
void print_equation(const VectorXd& xi, const string& state_name) {
    vector<string> term_names = { "1", "x" , "y", "z", "x^2", "y^2", "z^2", "xy", "xz", "yz"};
    cout << "d" << state_name << "/dt = ";

    bool first = true;
    for (int i = 0; i < xi.size(); i++){
        if(abs(xi(i)) > tol){
            if(!first){
                cout << " + ";
            }
            cout << xi(i) << "*" << term_names[i];
            first = false;
        }
    }
    cout << "\n";
}

int main() {
    // set-up
    double dt = 0.01;
    double t_end = 20.0;
    int n = static_cast<int>(t_end/dt);

    // initial conditions for x,y,z
    Vector3d state(1.0,0,0);

    vector<Vector3d> states;
    vector<double> times;

    double t = 0.0;
    for (int i = 0; i < n; i++){
        states.push_back(state);
        times.push_back(t);

        state = rk4step(state, dt);
        t += dt;
    }

    double noise_lvl = 0.05;
    vector<Vector3d> states_noise = add_noise(states, noise_lvl);

    vector<Vector3d> fd_est = fd(states_noise, dt);
    int m = fd_est.size();
    VectorXd xdot_target_noise(m);
    VectorXd ydot_target_noise(m);
    VectorXd zdot_target_noise(m);

    for (int i = 0; i < m; i++) {
        xdot_target_noise(i) = fd_est[i](0);
        ydot_target_noise(i) = fd_est[i](1);
        zdot_target_noise(i) = fd_est[i](2);
    }


    // SINDy
    MatrixXd Theta = build_library(states); // candidate library
    vector<Vector3d> states_noise_matched(states_noise.begin() + 1, states_noise.end() - 1);
    MatrixXd Theta_noise = build_library(states_noise_matched);

    VectorXd xdot_target = build_target(states,0);
    VectorXd ydot_target = build_target(states,1);
    VectorXd zdot_target = build_target(states,2);

    // initial ols
    VectorXd xi_x = Theta.householderQr().solve(xdot_target);
    VectorXd xi_y = Theta.householderQr().solve(ydot_target);
    VectorXd xi_z = Theta.householderQr().solve(zdot_target);

    vector<int> indices(10);

    for (int i = 0; i < 10; i++){
        indices[i] = i;
    }

    // stlsq
    VectorXd xi_x_final = stlsq(Theta_noise, xdot_target_noise, 0.5);
    VectorXd xi_y_final = stlsq(Theta_noise, ydot_target_noise, 0.5);
    VectorXd xi_z_final = stlsq(Theta_noise, zdot_target_noise, 0.5);

    print_equation(xi_x_final, "x");
    print_equation(xi_y_final, "y");
    print_equation(xi_z_final, "z");
    
    // export to plot
    ofstream outfile("lorenz_data.csv");
    outfile << "t,x,y,z\n";
    for (int i = 0; i < n; i++) {
        outfile << times[i] << "," << states[i](0) << "," << states[i](1) << "," << states[i](2) << "\n";
    }
    outfile.close();

    // python3 sindy_lorenz_plot.py

}