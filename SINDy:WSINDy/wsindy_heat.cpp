#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <random>
#include <fstream>

using namespace std;
using namespace Eigen;

// clang++ -I /opt/homebrew/opt/eigen/include/eigen3 wsindy_heat.cpp -o wsindyheat

// set up parameters 
constexpr double kappa = 0.1; // thermal diffusivity
double tol = numeric_limits<double>::epsilon(); // machine epsilon 

// spatial domain
constexpr double Lx = 1.0;
constexpr int nx = 100;
constexpr double dx = Lx/(nx-1);
constexpr double x0 = 0.0;

// time domain
constexpr double dt = 0.001;
constexpr int nt = 1001;
constexpr double t0 = 0.0;
constexpr double t_end = (nt-1)*dt;


// approx laplacian with finite difference, build matrix
MatrixXd laplacian(int nx, double dx) {
// system dynamics: heat equation
// ∂u/∂t = κ∇²u
    MatrixXd L = MatrixXd::Zero(nx, nx); // for homogenous Dirichlet boundary conditions
    double fd = 1/(pow(dx,2));

    for (int i = 1; i < (nx-1); i++){
        L(i, i-1) = fd;
        L(i, i) = -2*fd;
        L(i, i+1) = fd;
    }
    return L;
}

// test function parameters
constexpr double a = 0;
constexpr double b = 1;
constexpr int p = 4;
constexpr int q = 4;


// test function window settings
constexpr double window_width_x = 0.40;
constexpr double window_spacing_x = 0.04;
constexpr double window_width_t = 0.10; //* t_end;
constexpr double window_spacing_t = 0.02; //* t_end;

// test function constant
double get_C(double ak, double bk, double p, double q) {
    return 1.0/(pow(p,p)*pow(q,q)) * pow((p+q)/(bk-ak), p+q);
}

VectorXd get_phi_t_window(double ak, double bk, int global_size){
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

VectorXd get_phid_t_window(double ak, double bk, int global_size) {
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

VectorXd get_phi_x_window(double ak, double bk, int global_size){
    double Ck = get_C(ak,bk,p,q);
    VectorXd phi = VectorXd::Zero(global_size) ;

    int i_start = static_cast<int>(round((ak - x0)/dx));
    int i_end   = static_cast<int>(round((bk - x0)/dx));

    for(int i = i_start; i <= i_end; i++){
        double x = x0 + i*dx;
        double first = max(0.0, x-ak);
        double second = max(0.0, bk-x);

        double phi_at_x = Ck*(pow(first,p))*(pow(second,q));

        phi(i) = phi_at_x;
    }
    return phi;
}

VectorXd get_phid_x_window(double ak, double bk, int global_size) {
    double Ck = get_C(ak,bk,p,q);

    int i_start = static_cast<int>(round((ak-x0)/dx));
    int i_end = static_cast<int>(round((bk-x0)/dx));

    VectorXd phid = VectorXd::Zero(global_size);

    for(int i = i_start; i <= i_end; i++){
        double x = x0 + i*dx;
        double first = max(0.0, x-ak);
        double second = max(0.0, bk-x);

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

VectorXd get_phidd_x_window(double ak, double bk, int global_size) {
    double Ck = get_C(ak,bk,p,q);

    int i_start = static_cast<int>(round((ak-x0)/dx));
    int i_end = static_cast<int>(round((bk-x0)/dx));

    VectorXd phidd = VectorXd::Zero(global_size);

    for(int i = i_start; i <= i_end; i++){
        double x = x0 + i*dx;
        double first = max(0.0, x-ak);
        double second = max(0.0, bk-x);

        double term1; // p(p-1)*first^(p-2)*second^q
        if (first > 0.0) {
            term1 = p*(p-1) * pow(first, p-2) * pow(second, q);
        } else {
            term1 = 0.0;
        }

        double term2; // 2pq*first^(p-1)*second^(q-1)
        if (first > 0.0 && second > 0.0) {
            term2 = 2*p*q * pow(first, p-1) * pow(second, q-1);
        } else {
            term2 = 0.0;
        }

        double term3; // q(q-1)*first^p*second^(q-2)
        if (second > 0.0) {
            term3 = q*(q-1) * pow(first, p) * pow(second, q-2);
        } else {
            term3 = 0.0;
        }

        phidd(i) = Ck * (term1 - term2 + term3);
    }
    return phidd;
}

void gen_testfns_t(vector<VectorXd>& phis_t, vector<VectorXd>& phids_t) {
    phis_t.clear();
    phids_t.clear();
    for(double ak = t0; ak + window_width_t <= t_end + tol; ak += window_spacing_t){
        double bk = ak + window_width_t;
        phis_t.push_back(get_phi_t_window(ak, bk, nt));
        phids_t.push_back(get_phid_t_window(ak, bk, nt));
    }
}

void gen_testfns_x(vector<VectorXd>& phis_x, vector<VectorXd>& phidds_x) {
    phis_x.clear();
    phidds_x.clear();
    for(double ak = x0; ak + window_width_x <= Lx + tol; ak += window_spacing_x){
        double bk = ak + window_width_x;
        phis_x.push_back(get_phi_x_window(ak, bk, nx));
        phidds_x.push_back(get_phidd_x_window(ak, bk, nx));
    }
}

// weak formulation
double integrate(const VectorXd& I, double h) {
    int N = I.size() - 1;
    double integral = (I(0) + I(N)) / 2;
    for(int k = 1; k < N; k++){
        integral += I(k);
    }
    integral *= h;
    return integral;
}

// contraction
double contract2d(const MatrixXd& U, const VectorXd& wx, const VectorXd& wt) {
    VectorXd temp(nx);
    // integrate over t for each fixed x row to collapse (nx,nt) to nx
    for( int i = 0; i < nx; i++){
        temp(i) = integrate(U.row(i).transpose().cwiseProduct(wt), dt);
    }
    // integrate resulting vector over x to get scalar
    return integrate(temp.cwiseProduct(wx), dx);
}

// WSINDy with MSTLS
pair<VectorXd, VectorXd> threshold_bounds(const MatrixXd& Theta, const VectorXd& target, double lambda) {
    VectorXd L(Theta.cols());
    VectorXd U(Theta.cols());
    double r = 0;
    for (int j = 0; j < Theta.cols(); j++){
        r = target.norm()/Theta.col(j).norm();
        L(j) = lambda * max(1.0,r);
        U(j) = (1.0/lambda) * min(1.0,r);
    }
    return {L, U};
}

VectorXd mstls(const MatrixXd& Theta, const VectorXd& target, double lambda){
    vector<int> indices(Theta.cols());
    for (int i = 0; i < Theta.cols(); i++){
        indices[i] = i;
    }
    auto [L, U] = threshold_bounds(Theta,target,lambda);

    bool convergence = false;
    while(!convergence){
        int prior_size = indices.size();

        MatrixXd Theta_slimmed = Theta(Eigen::indexing::all, indices);
        VectorXd xi = Theta_slimmed.householderQr().solve(target);

        vector<int> new_indices;
        for(int i = 0; i < indices.size(); i++){
            if(abs(xi(i)) >= L(indices[i]) && abs(xi(i)) <= U(indices[i])){
                new_indices.push_back(indices[i]);
            }
        }
        indices = new_indices;

        if(indices.empty() || indices.size() == prior_size){
            convergence = true;
        }
    }

        if (indices.empty()) {
            return VectorXd::Zero(Theta.cols());
        }

    MatrixXd Theta_final = Theta(Eigen::indexing::all, indices);
    VectorXd xi_final_slim = Theta_final.householderQr().solve(target);

    VectorXd xi_full = VectorXd::Zero(Theta.cols());

    for (int i = 0; i < indices.size(); i++) {
        xi_full(indices[i]) = xi_final_slim(i);
    }
    return xi_full;
}


// noise
MatrixXd add_noise(const MatrixXd& u_data, double noise_lvl){
    static random_device rd;
    static default_random_engine generator(rd());
    normal_distribution<double> distribution(0.0, noise_lvl);

    MatrixXd u_noisy = u_data;
    for (int i = 0; i < u_data.rows(); i++){
        for (int j = 0; j < u_data.cols(); j++){
            u_noisy(i,j) += distribution(generator);
        }
    }
    return u_noisy;
}

// library construction
VectorXd build_target(const MatrixXd& u_data, const vector<VectorXd>& phis_x, const vector<VectorXd>& phids_t) {
    int nx_w = phis_x.size();
    int nt_w = phids_t.size();
    VectorXd target(nx_w * nt_w);
    int row = 0;
    for (int i = 0; i < nx_w; i++) {
        for (int j = 0; j < nt_w; j++) {
            target(row) = -contract2d(u_data, phis_x[i], phids_t[j]);
            row++;
        }
    }
    return target;
}

MatrixXd build_library(const MatrixXd& u_data, const vector<VectorXd>& phis_x, const vector<VectorXd>& phidds_x, const vector<VectorXd>& phis_t) {
    int nx_w = phis_x.size();
    int nt_w = phis_t.size();
    MatrixXd Theta(nx_w * nt_w, 4);
    MatrixXd u2 = u_data.array().square();
    MatrixXd u3 = u_data.array().cube();
    int row = 0;
    for (int i = 0; i < nx_w; i++) {
        for (int j = 0; j < nt_w; j++) {
            Theta(row, 0) = contract2d(u_data, phis_x[i],  phis_t[j]);
            Theta(row, 1) = contract2d(u2,     phis_x[i],  phis_t[j]);
            Theta(row, 2) = contract2d(u3,     phis_x[i],  phis_t[j]);
            Theta(row, 3) = contract2d(u_data, phidds_x[i], phis_t[j]);
            row++;
        }
    }
    return Theta;
}

// recover pde
void print_equation(const VectorXd& xi, const string& term) {
    vector<string> term_names = {"u", "u^2", "u^3", "u_xx"}; 
    cout << "d" << term << "/dt = " << "\n";

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
    // initial condition: sin(pi*x)
    VectorXd u0(nx);
    for (int i = 0; i < nx; i++){
        double x = i*dx;
        u0(i) =
      1.0*sin(M_PI*x) + 0.7*sin(2*M_PI*x) + 0.5*sin(3*M_PI*x) + 0.4*sin(5*M_PI*x) + 0.25*sin(8*M_PI*x); 
    }

    MatrixXd L = laplacian(nx, dx);
    MatrixXd I = MatrixXd::Identity(nx, nx);

    // CN to avoid CFL issues
    MatrixXd A_lhs = (I - ((kappa * dt)/2) * L);
    MatrixXd A_rhs = (I + ((kappa * dt)/2) * L);

    PartialPivLU<MatrixXd> lhs(A_lhs);
    
    MatrixXd u_data(nx, nt); // keep ic
    u_data.col(0) = u0;
    VectorXd u = u0;

    for (int i = 1; i < nt; i++){
        VectorXd rhs = A_rhs * u;
        u = lhs.solve(rhs);
        u_data.col(i) = u;
    }


    double noise_lvl = 0.0;
    MatrixXd u_noise = add_noise(u_data, noise_lvl);

    vector<VectorXd> phis_t, phids_t;
    gen_testfns_t(phis_t, phids_t);

    vector<VectorXd> phis_x, phidds_x;
    gen_testfns_x(phis_x, phidds_x);

    double lambda = 0.001; // threshold for mstls
    MatrixXd Theta = build_library(u_noise, phis_x, phidds_x, phis_t);
    VectorXd target = build_target(u_noise, phis_x, phids_t);

    VectorXd xi = mstls(Theta, target, lambda);

    print_equation(xi, "u");

    // export to plot
    ofstream outfile("wsindy_heat_data.csv");
        outfile << "t,x,u\n";
        for (int i = 0; i < nx; i++){
            double x = x0 + i*dx;
            for (int step = 0; step < nt; step++){
                double t = t0 + step*dt;
                outfile << t << "," << x << "," << u_data(i, step) << "\n";
            }
        }
        outfile.close();

    // python3 wsindy_heat_plot.py
}