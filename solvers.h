void gen_Z_topography(double * xs, double * Z, int nx, double k, double theta);
void gen_Z_x_topography(double * xs, double * Z_x, int nx, double k, double theta, double gamma_of_k);
void solve(double k, double theta, int nx, double * xs, double * Z_x, double * h, double dx, double q0_sq, double gamma_theta);