#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <omp.h>
#include <chrono>


using namespace std;

// ------------------------------------------------------------
// Global parameters
// ------------------------------------------------------------
const double gamma_val = 1.4;   // Ratio of specific heats
const double CFL = 0.5;         // CFL number

// ------------------------------------------------------------
// Compute pressure from the conservative variables
// ------------------------------------------------------------
double pressure(double rho, double rhou, double rhov, double E) {
    double u = rhou / rho;
    double v = rhov / rho;
    double kinetic = 0.5 * rho * (u * u + v * v);
    return (gamma_val - 1.0) * (E - kinetic);
}

// ------------------------------------------------------------
// Compute flux in the x-direction
// ------------------------------------------------------------
void fluxX(double rho, double rhou, double rhov, double E, 
           double& frho, double& frhou, double& frhov, double& fE) {
    double u = rhou / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhou;
    frhou = rhou * u + p;
    frhov = rhov * u;
    fE = (E + p) * u;
}

// ------------------------------------------------------------
// Compute flux in the y-direction
// ------------------------------------------------------------
void fluxY(double rho, double rhou, double rhov, double E,
           double& frho, double& frhou, double& frhov, double& fE) {
    double v = rhov / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhov;
    frhou = rhou * v;
    frhov = rhov * v + p;
    fE = (E + p) * v;
}

// ------------------------------------------------------------
// Main simulation routine
// ------------------------------------------------------------
int main(){
    // ----- Grid and domain parameters -----
    const int Nx = 200;         // Number of cells in x (excluding ghost cells)
    const int Ny = 100;         // Number of cells in y
    const double Lx = 2.0;      // Domain length in x
    const double Ly = 1.0;      // Domain length in y
    const double dx = Lx / Nx;
    const double dy = Ly / Ny;

    // Create flat arrays (with ghost cells)
    const int total_size = (Nx + 2) * (Ny + 2);
    
    vector<double> rho(total_size);
    vector<double> rhou(total_size);
    vector<double> rhov(total_size);
    vector<double> E(total_size);
    
    vector<double> rho_new(total_size);
    vector<double> rhou_new(total_size);
    vector<double> rhov_new(total_size);
    vector<double> E_new(total_size);
    
    // A mask to mark solid cells (inside the cylinder)
    vector<bool> solid(total_size, false);

    // ----- Obstacle (cylinder) parameters -----
    const double cx = 0.5;      // Cylinder center x
    const double cy = 0.5;      // Cylinder center y
    const double radius = 0.1;  // Cylinder radius

    // ----- Free-stream initial conditions (inflow) -----
    const double rho0 = 1.0;
    const double u0 = 1.0;
    const double v0 = 0.0;
    const double p0 = 1.0;
    const double E0 = p0/(gamma_val - 1.0) + 0.5*rho0*(u0*u0 + v0*v0);

    // ----- Initialize grid and obstacle mask -----
    for (int i = 0; i < Nx+2; i++){
        for (int j = 0; j < Ny+2; j++){
            // Compute cell center coordinates
            double x = (i - 0.5) * dx;
            double y = (j - 0.5) * dy;
            // Mark cell as solid if inside the cylinder
            if ((x - cx)*(x - cx) + (y - cy)*(y - cy) <= radius * radius) {
                solid[i*(Ny+2)+j] = true;
                // For a wall, we set zero velocity
                rho[i*(Ny+2)+j] = rho0;
                rhou[i*(Ny+2)+j] = 0.0;
                rhov[i*(Ny+2)+j] = 0.0;
                E[i*(Ny+2)+j] = p0/(gamma_val - 1.0);
            } else {
                solid[i*(Ny+2)+j] = false;
                rho[i*(Ny+2)+j] = rho0;
                rhou[i*(Ny+2)+j] = rho0 * u0;
                rhov[i*(Ny+2)+j] = rho0 * v0;
                E[i*(Ny+2)+j] = E0;
            }
        }
    }

    // ----- Determine time step from CFL condition -----
    double c0 = sqrt(gamma_val * p0 / rho0);
    double dt = CFL * min(dx, dy) / (fabs(u0) + c0)/2.0;

    // ----- Time stepping parameters -----
    const int nSteps = 2000;

    // ----- Main time-stepping loop -----
    int update_count = 0;
    int copy_count = 0;
    int kinetic_count = 0;
    vector<int> lrbt_count(4, 0); // left, right, bottom, top boundary condition counts
    std::chrono::duration<double> update_time(0);
    std::chrono::duration<double> copy_time(0);
    std::chrono::duration<double> kinetic_time(0);
    vector<std::chrono::duration<double>> boundary_times(4, std::chrono::duration<double>(0));
    
    auto t1 = std::chrono::high_resolution_clock::now();
    //#pragma omp parallel
    for (int n = 0; n < nSteps; n++){
        // --- Apply boundary conditions on ghost cells ---
        // Left boundary (inflow): fixed free-stream state
        auto lb1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < Ny+2; j++){
            rho[0*(Ny+2)+j] = rho0;
            rhou[0*(Ny+2)+j] = rho0*u0;
            rhov[0*(Ny+2)+j] = rho0*v0;
            E[0*(Ny+2)+j] = E0;
        }
        auto lb2 = std::chrono::high_resolution_clock::now();
        boundary_times[0] += lb2 - lb1;
        lrbt_count[0]++;

        // Right boundary (outflow): copy from the interior
        auto rb1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < Ny+2; j++){
            rho[(Nx+1)*(Ny+2)+j] = rho[Nx*(Ny+2)+j];
            rhou[(Nx+1)*(Ny+2)+j] = rhou[Nx*(Ny+2)+j];
            rhov[(Nx+1)*(Ny+2)+j] = rhov[Nx*(Ny+2)+j];
            E[(Nx+1)*(Ny+2)+j] = E[Nx*(Ny+2)+j];
        }
        auto rb2 = std::chrono::high_resolution_clock::now();
        boundary_times[1] += rb2 - rb1;
        lrbt_count[1]++;
        
        // Bottom boundary: reflective
        auto bb1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < Nx+2; i++){
            rho[i*(Ny+2)+0] = rho[i*(Ny+2)+1];
            rhou[i*(Ny+2)+0] = rhou[i*(Ny+2)+1];
            rhov[i*(Ny+2)+0] = -rhov[i*(Ny+2)+1];
            E[i*(Ny+2)+0] = E[i*(Ny+2)+1];
        }
        auto bb2 = std::chrono::high_resolution_clock::now();
        boundary_times[2] += bb2 - bb1;
        lrbt_count[2]++;

        // Top boundary: reflective
        auto tb1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < Nx+2; i++){
            rho[i*(Ny+2)+(Ny+1)] = rho[i*(Ny+2)+Ny];
            rhou[i*(Ny+2)+(Ny+1)] = rhou[i*(Ny+2)+Ny];
            rhov[i*(Ny+2)+(Ny+1)] = -rhov[i*(Ny+2)+Ny];
            E[i*(Ny+2)+(Ny+1)] = E[i*(Ny+2)+Ny];
        }
        auto tb2 = std::chrono::high_resolution_clock::now();
        boundary_times[3] += tb2 - tb1;
        lrbt_count[3]++;

        // --- Update interior cells using a Lax-Friedrichs scheme ---
        auto lu1 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= Nx; i++){
            for (int j = 1; j <= Ny; j++){
                // If the cell is inside the solid obstacle, do not update it
                if (solid[i*(Ny+2)+j]) {
                    rho_new[i*(Ny+2)+j] = rho[i*(Ny+2)+j];
                    rhou_new[i*(Ny+2)+j] = rhou[i*(Ny+2)+j];
                    rhov_new[i*(Ny+2)+j] = rhov[i*(Ny+2)+j];
                    E_new[i*(Ny+2)+j] = E[i*(Ny+2)+j];
                    continue;
                }

                // Compute a Lax averaging of the four neighboring cells
                rho_new[i*(Ny+2)+j] = 0.25 * (rho[(i+1)*(Ny+2)+j] + rho[(i-1)*(Ny+2)+j] + 
                                            rho[i*(Ny+2)+(j+1)] + rho[i*(Ny+2)+(j-1)]);
                rhou_new[i*(Ny+2)+j] = 0.25 * (rhou[(i+1)*(Ny+2)+j] + rhou[(i-1)*(Ny+2)+j] + 
                                            rhou[i*(Ny+2)+(j+1)] + rhou[i*(Ny+2)+(j-1)]);
                rhov_new[i*(Ny+2)+j] = 0.25 * (rhov[(i+1)*(Ny+2)+j] + rhov[(i-1)*(Ny+2)+j] + 
                                            rhov[i*(Ny+2)+(j+1)] + rhov[i*(Ny+2)+(j-1)]);
                E_new[i*(Ny+2)+j] = 0.25 * (E[(i+1)*(Ny+2)+j] + E[(i-1)*(Ny+2)+j] + 
                                        E[i*(Ny+2)+(j+1)] + E[i*(Ny+2)+(j-1)]);

                // Compute fluxes
                double fx_rho1, fx_rhou1, fx_rhov1, fx_E1;
                double fx_rho2, fx_rhou2, fx_rhov2, fx_E2;
                double fy_rho1, fy_rhou1, fy_rhov1, fy_E1;
                double fy_rho2, fy_rhou2, fy_rhov2, fy_E2;

                fluxX(rho[(i+1)*(Ny+2)+j], rhou[(i+1)*(Ny+2)+j], rhov[(i+1)*(Ny+2)+j], E[(i+1)*(Ny+2)+j],
                    fx_rho1, fx_rhou1, fx_rhov1, fx_E1);
                fluxX(rho[(i-1)*(Ny+2)+j], rhou[(i-1)*(Ny+2)+j], rhov[(i-1)*(Ny+2)+j], E[(i-1)*(Ny+2)+j],
                    fx_rho2, fx_rhou2, fx_rhov2, fx_E2);
                fluxY(rho[i*(Ny+2)+(j+1)], rhou[i*(Ny+2)+(j+1)], rhov[i*(Ny+2)+(j+1)], E[i*(Ny+2)+(j+1)],
                    fy_rho1, fy_rhou1, fy_rhov1, fy_E1);
                fluxY(rho[i*(Ny+2)+(j-1)], rhou[i*(Ny+2)+(j-1)], rhov[i*(Ny+2)+(j-1)], E[i*(Ny+2)+(j-1)],
                    fy_rho2, fy_rhou2, fy_rhov2, fy_E2);

                // Apply flux differences
                double dtdx = dt / (2 * dx);
                double dtdy = dt / (2 * dy);
                
                rho_new[i*(Ny+2)+j] -= dtdx * (fx_rho1 - fx_rho2) + dtdy * (fy_rho1 - fy_rho2);
                rhou_new[i*(Ny+2)+j] -= dtdx * (fx_rhou1 - fx_rhou2) + dtdy * (fy_rhou1 - fy_rhou2);
                rhov_new[i*(Ny+2)+j] -= dtdx * (fx_rhov1 - fx_rhov2) + dtdy * (fy_rhov1 - fy_rhov2);
                E_new[i*(Ny+2)+j] -= dtdx * (fx_E1 - fx_E2) + dtdy * (fy_E1 - fy_E2);
            }
        }
        auto lu2 = std::chrono::high_resolution_clock::now();
        update_time += lu2 - lu1;
        update_count++;

        // Copy updated values back
        auto lc1 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= Nx; i++){
            for (int j = 1; j <= Ny; j++){
                rho[i*(Ny+2)+j] = rho_new[i*(Ny+2)+j];
                rhou[i*(Ny+2)+j] = rhou_new[i*(Ny+2)+j];
                rhov[i*(Ny+2)+j] = rhov_new[i*(Ny+2)+j];
                E[i*(Ny+2)+j] = E_new[i*(Ny+2)+j];
            }
        }
        auto lc2 = std::chrono::high_resolution_clock::now();
        copy_time += lc2 - lc1;
        copy_count++;

        // Calculate total kinetic energy
        auto lk1 = std::chrono::high_resolution_clock::now();
        double total_kinetic = 0.0;
        #pragma omp parallel for reduction(+:total_kinetic)
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                double u = rhou[i*(Ny+2)+j] / rho[i*(Ny+2)+j];
                double v = rhov[i*(Ny+2)+j] / rho[i*(Ny+2)+j];
                total_kinetic += 0.5 * rho[i*(Ny+2)+j] * (u * u + v * v);
            }
        }
        auto lk2 = std::chrono::high_resolution_clock::now();
        kinetic_time += lk2 - lk1;
        kinetic_count++;

        if (n % 50 == 0) {
            cout << "Step " << n << " completed, total kinetic energy: " << total_kinetic << endl;
        }
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "main loop took "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()
              << " microseconds\n\n";
    
    std::cout << "Boundary conditions executed:  " << lrbt_count[0] << ", " << lrbt_count[1] << ", " << lrbt_count[2] << ", " << lrbt_count[3] << " times\n";
    std::cout << "Total boundary condition time: " << std::chrono::duration_cast<std::chrono::microseconds>(boundary_times[0]).count() << ", "
              << std::chrono::duration_cast<std::chrono::microseconds>(boundary_times[1]).count() << ", "
              << std::chrono::duration_cast<std::chrono::microseconds>(boundary_times[2]).count() << ", "
              << std::chrono::duration_cast<std::chrono::microseconds>(boundary_times[3]).count() << " microseconds\n\n";

    std::cout << "Update loop executed:  " << update_count << " times\n";
    std::cout << "Total update time:     " << std::chrono::duration_cast<std::chrono::microseconds>(update_time).count() << " microseconds\n\n";

    std::cout << "Copy loop executed:    " << copy_count << " times\n";
    std::cout << "Total copy time:       " << std::chrono::duration_cast<std::chrono::microseconds>(copy_time).count() << " microseconds\n\n";

    std::cout << "Kinetic loop executed: " << kinetic_count << " times\n";
    std::cout << "Total kinetic time:    " << std::chrono::duration_cast<std::chrono::microseconds>(kinetic_time).count() << " microseconds\n";

    double cells = Nx * Ny;
    double total_updates = update_count;
    double total_copies = copy_count;
    double total_kinetic = kinetic_count;

    // ---- Bytes moved ----
    double bytes_update = cells * 160.0 * total_updates;
    double bytes_copy = cells * 64.0 * total_copies;
    double bytes_kin = cells * 24.0 * total_kinetic;

    // ---- Convert time to seconds ----
    double update_sec = update_time.count();
    double copy_sec = copy_time.count();
    double kin_sec = kinetic_time.count();

    // ---- Compute bandwidth in GB/s ----
    double bw_update = bytes_update / update_sec / 1e9;
    double bw_copy = bytes_copy / copy_sec / 1e9;
    double bw_kinetic = bytes_kin / kin_sec / 1e9;

    std::cout << "\nEstimated Achieved Bandwidth:\n";
    std::cout << "Update loop:  " << bw_update << " GB/s\n";
    std::cout << "Copy loop:    " << bw_copy   << " GB/s\n";
    std::cout << "Kinetic loop: " << bw_kinetic << " GB/s\n" << std::endl;

    std::cout << "Name" << '\t' << "Count" << '\t' << "Time(s)" << '\t' << "GB/s" << "\n";
    std::cout << "update" << '\t' << update_count << '\t' << update_sec << '\t' << bw_update << "\n";
    std::cout << "copy" << '\t' << copy_count << '\t' << copy_sec << '\t' << bw_copy << "\n";
    std::cout << "kinetic" << '\t' << kinetic_count << '\t' << kin_sec << '\t' << bw_kinetic << "\n";
    return 0;
}

