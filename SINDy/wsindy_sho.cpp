#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <fstream>

using namespace std;
using namespace Eigen;

// clang++ -I /opt/homebrew/opt/eigen/include/eigen3 wsindy_sho.cpp -o wsindysho

// set up parameters 
const double omega = 1.0; // natural angular frequency
double tol = numeric_limits<double>::epsilon(); // machine epsilon 

Vector2d ode_rhs(const Vector2d& traj) {
// system dynamics: simple harmonic oscillator
// d^2x/dt^2 = - omega_0^2 x

    double x = traj(0);
    double xd = traj(1);
    double xdd = - pow(omega,2) * x;

    return Vector2d(xd, xdd); // return velocity and acceleration together
}

// construct weak formulation:
// test fn phi compactly supported on [a,b]
// wf: int_a^b (xdot*phidot)dt = omega_0^2 int_a^b (x*phi)dt

const double a = 0;
const double b = 1;
const double p = 2;
const double q = p;

// simulation settings
const double dt = 0.01;
const double t_end = 20.0;
const double t0 = 0.0;
const int N = static_cast<int>(t_end/dt);

// test function window settings
const double window_width = 1.0;
const double window_spacing = 0.3;

// test function constant
double get_C(double ak, double bk, double p, double q) {
    return 1.0/(pow(p,p)*pow(q,q)) * pow((p+q)/(bk-ak), p+q);
}

VectorXd get_phi_window(double ak, double bk, int global_size){
    double Ck = get_C(ak,bk,p,q);
    // initilize phi with zeros to account for compact support. min size is N+1, use next power of 2 for computational efficiency
    VectorXd phi = VectorXd::Zero(global_size) ;

    int i_start = static_cast<int>(round((ak - t0)/dt));
    int i_end   = static_cast<int>(round((bk - t0)/dt));

    for(int i = i_start; i <= i_end; i++){
        double t = t0 + i*dt;
        double first = max(0.0, t-ak);
        double second = max(0.0, bk-t);

        double phi_at_t = Ck*(pow(first,p))*(pow(second,q));

        phi(i) = phi_at_t;
    }
    return phi;
}

// analytic derivative of phi
VectorXd get_phid_window(double ak, double bk, int global_size) {
    double Ck = get_C(ak,bk,p,q);

    int i_start = static_cast<int>(round((ak-t0)/dt));
    int i_end = static_cast<int>(round((bk-t0)/dt));

    VectorXd phid = VectorXd::Zero(global_size);

    for(int i = i_start; i <= i_end; i++){
        double t = t0 + i*dt;
        double first = max(0.0, t-ak);
        double second = max(0.0, bk-t);

        double term1;
        if (first > 0.0) {
            term1 = p * pow(first, p-1) * pow(second, q);
        } else {
            term1 = 0.0;
        }

        double term2;
        if (second > 0.0) {
            term2 = q * pow(first, p) * pow(second, q-1);
        } else {
            term2 = 0.0;
        }

        phid(i) = Ck * (term1 - term2);
    }
    return phid;
}

void gen_testfns(vector<VectorXd>& phis, vector<VectorXd>& phids, int global_size) {
    phis.clear();
    phids.clear();
    for(double ak = t0; ak + window_width <= t_end + tol; ak += window_spacing){
        double bk = ak + window_width;
        phis.push_back(get_phi_window(ak,bk,global_size));
        phids.push_back(get_phid_window(ak,bk,global_size));
    }

}

// RK4 to get x, v at timesteps
Vector2d rk4step(const Vector2d& traj, double dt) {

    Vector2d k1 = ode_rhs(traj);
    Vector2d k2 = ode_rhs(traj + k1 * (dt/2));
    Vector2d k3 = ode_rhs(traj + k2 * (dt/2));
    Vector2d k4 = ode_rhs(traj + k3 * dt);

    Vector2d traj_next = traj + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    return traj_next;
}

double integrate(const VectorXd& I, double dt) {
    int N = I.size() - 1;
    double integral = (I(0) + I(N)) / 2;
    for(int k = 1; k < N; k++){
        integral += I(k);
    }
    integral *= dt;
    return integral;
}

// noise generation
vector<Vector2d> add_noise(const vector<Vector2d>& trajs, double noise_lvl){
    static random_device rd;
    static default_random_engine generator(rd());
    normal_distribution<double> distribution(0.0, noise_lvl);
    vector<Vector2d> trajs_noise(trajs.size());

    for(size_t i = 0; i < trajs.size(); i++){
        double x_noise = distribution(generator);
        double v_noise = distribution(generator);

        trajs_noise[i] = Vector2d(trajs[i](0) + x_noise, trajs[i](1) + v_noise);
    }
    return trajs_noise;
}

// collection of phi
VectorXd build_target(const VectorXd& xd, const vector<VectorXd>& phids, double dt) {
    int n = phids.size();
    VectorXd target(n);
    for (int i = 0; i < n; i++){
        target(i) = integrate(xd.cwiseProduct(phids[i]),dt);
    }
    return target;
}

// WSINDy
RowVectorXd build_library_row(const VectorXd& x, const VectorXd& v, const VectorXd& phi, double dt) {
    RowVectorXd row(6);
    
    row(0) = integrate(phi, dt);
    row(1) = integrate(x.cwiseProduct(phi), dt); // component-wise product;
    row(2) = integrate(v.cwiseProduct(phi), dt);
    row(3) = integrate(x.cwiseProduct(x).cwiseProduct(phi), dt);
    row(4) = integrate(v.cwiseProduct(v).cwiseProduct(phi), dt);
    row(5) = integrate(x.cwiseProduct(v).cwiseProduct(phi), dt);

    return row;
}

MatrixXd build_library(const VectorXd& x, const VectorXd& v, const vector<VectorXd>& phis, double dt) {
    int n = phis.size();
    int num_terms = 6;

    MatrixXd Theta(n, num_terms);

    for(int i = 0; i < n; i++){
        Theta.row(i) = build_library_row(x, v, phis[i], dt);
    }
    return Theta;
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

        if(indices.empty() || indices.size() == prior_size){
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
void print_equation(const VectorXd& xi, const string& traj_name) {
    vector<string> term_names = { "1", "x" , "v", "x^2", "v^2", "xv"};
    cout << "d" << traj_name << "/dt = " << "\n";

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

int main(){

    // initial conditions for position and velocity
    Vector2d traj(1.0,0.0);
    vector<Vector2d> traj_clean;

    for (int i = 0; i<= N; i++){
        traj_clean.push_back(traj);
        traj = rk4step(traj,dt);
    }

    double noise_lvl = 0.1;
    vector<Vector2d> traj_noise = add_noise(traj_clean, noise_lvl);

    VectorXd x(traj_noise.size());
    VectorXd xd(traj_noise.size());
    for (size_t i = 0; i < traj_noise.size(); i++) {
        x(i) = traj_noise[i](0);
        xd(i) = traj_noise[i](1);
    }

    int global_size = x.size();

    vector<VectorXd> phis, phids;
    gen_testfns(phis,phids,global_size);

    MatrixXd Theta = build_library(x,xd,phis,dt);
    VectorXd target = build_target(xd,phids,dt);

    double lambda = 0.5;
    VectorXd xi = stlsq(Theta, target, lambda);

    print_equation(xi, "v");

    // export to plot
    ofstream outfile("wsindy_sho_data.csv");
    outfile << "t,x,v\n";
    for (size_t i = 0; i < traj_noise.size(); i++) {
        double t = t0 + i*dt;
        outfile << t << "," << traj_noise[i](0) << "," << traj_noise[i](1) << "\n";
    }
    outfile.close();

    // python3 wsindy_sho_plot.py
}
