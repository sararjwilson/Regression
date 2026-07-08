#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <fstream>

using namespace std;
using namespace Eigen;

//   g++ -I /opt/homebrew/opt/eigen/include/eigen3 sindy_sho.cpp -o sindysho

// set up parameters 
const double omega = 1.0;
double tol = numeric_limits<double>::epsilon();

// direct ode computation
Vector2d ode_rhs(const Vector2d& state) {
// system dynamics: simple harmonic oscillator
// d^2x/dt^2 = - omega_0^2 x

    double x = state(0);
    double v = state(1);
    double a = -omega * x;

    return Vector2d(v,a);
}

// RK4 for second order ode to generate data 
Vector2d rk4step(const Vector2d& state, double dt) {

    Vector2d k1 = ode_rhs(state);
    Vector2d k2 = ode_rhs(state + k1 * (dt/2));
    Vector2d k3 = ode_rhs(state + k2 * (dt/2));
    Vector2d k4 = ode_rhs(state + k3 * dt);

    Vector2d state_next = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    return state_next;
}

// central difference for derivative terms in candidate library
vector<Vector2d> fd(const vector<Vector2d>& states, double dt) {
    int n = states.size();
    vector<Vector2d> fd_est;

    for (int i = 1; i < (n-1); i++){
       Vector2d fd_steps = (states[i+1] - states[i-1]) / (2*dt);
       fd_est.push_back(fd_steps);
    }
    return fd_est;
}

// noise generation
vector<Vector2d> add_noise(const vector<Vector2d>& states, double noise_lvl){
    static random_device rd;
    static default_random_engine generator(rd());
    normal_distribution<double> distribution(0.0, noise_lvl);
    vector<Vector2d> states_noise(states.size());

    for(int i = 0; i < states.size(); i++){
        double x_noise = distribution(generator); // noise gen should just be on x? ask about this
        double v_noise = distribution(generator);

        states_noise[i] = Vector2d(states[i](0) + x_noise, states[i](1) + v_noise);
    }
    return states_noise;
}

// SINDy set-up
RowVectorXd build_library_row(const Vector2d& state) {
    double x = state(0);
    double v = state(1);

    RowVectorXd row(6);
    row << 1, x, v, x*x, v*v, x*v;
    return row;
}

MatrixXd build_library(const vector<Vector2d>& states){
    int n = states.size();
    int num_terms = 6; 
    
    MatrixXd Theta(n, num_terms);

    for(int i = 0; i < n; i++){
        Theta.row(i) = build_library_row(states[i]);
    }
    return Theta;
}

// collection of derivatives evaluated at each time step
VectorXd ground_truth_derivatives(const vector<Vector2d>& states, int component){
    int n = states.size();
    VectorXd target(n); 

    for (int i = 0; i < n; i++){
        Vector2d derivative = ode_rhs(states[i]);
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
    vector<string> term_names = { "1", "x" , "v", "x^2", "v^2", "xv"};
    cout << "d" << state_name << "/dt = " << "\n";

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

// TO DO: residuals 

int main() {
    // set-up
    double dt = 0.01;
    double t_end = 20.0;
    int n = static_cast<int>(t_end/dt);

    // initial conditions for position and velocity
    Vector2d state(1,0);

    vector<Vector2d> states;
    vector<double> times;

    double t = 0.0;
    for (int i = 0; i < n; i++){
        states.push_back(state);
        times.push_back(t);

        state = rk4step(state, dt);
        t += dt;
    }

    double noise_lvl = 0.0;
    vector<Vector2d> states_noise = add_noise(states, noise_lvl);

    vector<Vector2d> fd_est = fd(states_noise, dt);
    int m = fd_est.size();
    VectorXd v_target_noise(m);
    VectorXd a_target_noise(m);

    for (int i = 0; i < m; i++) {
        v_target_noise(i) = fd_est[i](0);
        a_target_noise(i) = fd_est[i](1);
    }


    // SINDy
    MatrixXd Theta = build_library(states); // candidate library
    vector<Vector2d> states_noise_matched(states_noise.begin() + 1, states_noise.end() - 1);
    MatrixXd Theta_noise = build_library(states_noise_matched);

    VectorXd v_target = ground_truth_derivatives(states,0);
    VectorXd a_target = ground_truth_derivatives(states,1);

    // stlsq
    VectorXd xi_v_final = stlsq(Theta_noise, v_target_noise, 0.05);
    VectorXd xi_a_final = stlsq(Theta_noise, a_target_noise, 0.05);

    print_equation(xi_v_final, "v");
    print_equation(xi_a_final, "a");

    // export to plot
    ofstream outfile("sho_data.csv");
    outfile << "t,x,v\n";
    for (int i = 0; i < n; i++) {
        outfile << times[i] << "," << states[i](0) << "," << states[i](1) << "\n";
    }
    outfile.close();

    // python3 sindy_lorenz_plot.py

}