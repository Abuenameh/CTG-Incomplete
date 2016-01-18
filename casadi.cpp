#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

using namespace boost;
using namespace boost::filesystem;

#include "casadi.hpp"

namespace casadi {

    inline bool isnan(SX& sx) {
        return sx.at(0).isNan();
    }

    inline bool isinf(SX sx) {
        return sx.at(0).isInf();
    }
}

//double GroundStateProblem::energy2(vector<double>& fin, vector<double>& J, double U0, vector<double>& dU, double mu, double theta) {
//
//    double E = 0;
//    for (int i = 0; i < L; i++) {
//        for (int n = 0; n <= nmax; n++) {
//        E += energy2(i, n, fin, J, U0, dU, mu, theta);
//        }
//    }
//    return E;
//}

SX energy(SX& fin, SX& J, SX& U0, SX& dU, SX& mu, SX& theta) {

    SX E = 0;
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
        E += energy(i, n, fin, J, U0, dU, mu, theta);
        }
    }
    return E;
}

//double GroundStateProblem::energy2(int i, int n, vector<double>& fin, vector<double>& J, double U0, vector<double>& dU, double mu, double theta) {
//
//    complex<double> expth = complex<double>(cos(theta), sin(theta));
//    complex<double> expmth = ~expth;
//    complex<double> exp2th = expth*expth;
//    complex<double> expm2th = ~exp2th;
//
//    vector<complex<double>* > f(L);
//    vector<double> norm2(L, 0);
//    for (int j = 0; j < L; j++) {
//        f[j] = reinterpret_cast<complex<double>*> (&fin[2 * j * dim]);
//        for (int m = 0; m <= nmax; m++) {
//            norm2[j] += f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
//        }
//    }
//
//    complex<double> E = complex<double>(0, 0);
//
//    complex<double> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;
//
//    //    for (int i = 0; i < L; i++) {
//
//    int k1 = mod(i - 2);
//    int j1 = mod(i - 1);
//    int j2 = mod(i + 1);
//    int k2 = mod(i + 2);
//
//    Ej1 = complex<double>(0, 0);
//    Ej2 = complex<double>(0, 0);
//
//            for (int m = 1; m <= nmax; m++) {
//                if (n != m - 1) {
//                    Ej1 += J[j1] * expth * g2(n, m) * (eps(dU, i, j1, n, m) / eps(U0, n, m))
//                            * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n] * f[j1][m];
//                    Ej2 += J[i] * expmth * g2(n, m) * (eps(dU, i, j2, n, m) / eps(U0, n, m))
//                            * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n] * f[j2][m];
//                }
//            }
//    
//    Ej1 /= norm2[i] * norm2[j1];
//    Ej2 /= norm2[i] * norm2[j2];
//
//    E += Ej1;
//    E += Ej2;
//    
//    return E.real();
//}

SX energy(int i, int n, SX& fin, SX& J, SX& U0, SX& dU, SX& mu, SX& theta) {

    complex<SX> expth = complex<SX>(cos(theta), sin(theta));
    complex<SX> expmth = ~expth;
    complex<SX> exp2th = expth*expth;
    complex<SX> expm2th = ~exp2th;

    vector<vector<complex < SX>>> f(L, vector<complex < SX >> (dim, complex<SX>(0, 0)));
    vector<SX> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        for (int k = 0; k < dim; k++) {
            int l = 2 * (j * dim + k);
            f[j][k] = complex<SX>(fin[l], fin[l + 1]);
        }
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    complex<SX> E = complex<SX>(0, 0);

    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;

    //    for (int i = 0; i < L; i++) {

    int k1 = mod(i - 2);
    int j1 = mod(i - 1);
    int j2 = mod(i + 1);
    int k2 = mod(i + 2);

    Ei = complex<SX>(0, 0);
    Ej1 = complex<SX>(0, 0);
    Ej2 = complex<SX>(0, 0);
    Ej1j2 = complex<SX>(0, 0);
    Ej1k1 = complex<SX>(0, 0);
    Ej2k2 = complex<SX>(0, 0);

//    for (int n = 0; n <= nmax; n++) {
        Ei += (0.5 * (U0 + dU[i]) * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

        if (n < nmax) {
            Ej1 += -J[j1] * expth * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                    * f[i][n] * f[j1][n + 1];
            Ej2 += -J[i] * expmth * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
                    * f[j2][n + 1];

            if (n > 0) {
                Ej1 += 0.5 * J[j1] * J[j1] * exp2th * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                        * ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1] * f[j1][n + 1];
                Ej2 += 0.5 * J[i] * J[i] * expm2th * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                        * ~f[i][n + 1] * ~f[j2][n - 1] * f[i][n - 1] * f[j2][n + 1];
            }
            if (n < nmax - 1) {
                Ej1 -= 0.5 * J[j1] * J[j1] * exp2th * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                        * ~f[i][n + 2] * ~f[j1][n] * f[i][n] * f[j1][n + 2];
                Ej2 -= 0.5 * J[i] * J[i] * expm2th * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                        * ~f[i][n + 2] * ~f[j2][n] * f[i][n] * f[j2][n + 2];
            }

            if (n > 1) {
                Ej1 += -J[j1] * J[j1] * exp2th * g(n, n - 1) * g(n - 1, n)
                        * (eps(dU, i, j1, n, n - 1, i, j1, n - 1, n) / (eps(U0, n, n - 1)*(eps(U0, n, n - 1) + eps(U0, n - 1, n))))
                        * ~f[i][n + 1] * ~f[j1][n - 2] * f[i][n - 1] * f[j1][n];
                Ej2 += -J[i] * J[i] * expm2th * g(n, n - 1) * g(n - 1, n)
                        * (eps(dU, i, j2, n, n - 1, i, j2, n - 1, n) / (eps(U0, n, n - 1)*(eps(U0, n, n - 1) + eps(U0, n - 1, n))))
                        * ~f[i][n + 1] * ~f[j2][n - 2] * f[i][n - 1] * f[j2][n];
            }
            if (n < nmax - 2) {
                Ej1 -= -J[j1] * J[j1] * exp2th * g(n, n + 3) * g(n + 1, n + 2)
                        * (eps(dU, i, j1, n, n + 3, i, j1, n + 1, n + 2) / (eps(U0, n, n + 3)*(eps(U0, n, n + 3) + eps(U0, n + 1, n + 2))))
                        * ~f[i][n + 2] * ~f[j1][n + 1] * f[i][n] * f[j1][n + 3];
                Ej2 -= -J[i] * J[i] * expm2th * g(n, n + 3) * g(n + 1, n + 2)
                        * (eps(dU, i, j2, n, n + 3, i, j2, n + 1, n + 2) / (eps(U0, n, n + 3)*(eps(U0, n, n + 3) + eps(U0, n + 1, n + 2))))
                        * ~f[i][n + 2] * ~f[j2][n + 1] * f[i][n] * f[j2][n + 3];
            }

            for (int m = 1; m <= nmax; m++) {
                if (n != m - 1) {
                    Ej1 += 0.5 * J[j1] * J[j1] * g(n, m) * g(m - 1, n + 1) * (1 / eps(U0, n, m))
                            * (~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1] -
                            ~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m]);
                    Ej2 += 0.5 * J[i] * J[i] * g(n, m) * g(m - 1, n + 1) * (1 / eps(U0, n, m))
                            * (~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1] -
                            ~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m]);

                    Ej1 += 1.0 * J[j1] * expth * g(n, m) * (eps(dU, i, j1, n, m) / eps(U0, n, m))
                            * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n] * f[j1][m];
                    Ej2 += 1.0 * J[i] * expmth * g(n, m) * (eps(dU, i, j2, n, m) / eps(U0, n, m))
                            * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n] * f[j2][m];

                    if (n != m - 3 && m > 1 && n < nmax - 1) {
                        Ej1 += -0.5 * J[j1] * J[j1] * exp2th * g(n, m) * g(n + 1, m - 1)
                                * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n + 1, m - 1)))
                                * ~f[i][n + 2] * ~f[j1][m - 2] * f[i][n] * f[j1][m];
                        Ej2 += -0.5 * J[i] * J[i] * expm2th * g(n, m) * g(n + 1, m - 1)
                                * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n + 1, m - 1)))
                                * ~f[i][n + 2] * ~f[j2][m - 2] * f[i][n] * f[j2][m];
                    }
                    if (n != m + 1 && n > 0 && m < nmax) {
                        Ej1 -= -0.5 * J[j1] * J[j1] * exp2th * g(n, m) * g(n - 1, m + 1)
                                * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n - 1, m + 1)))
                                * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n - 1] * f[j1][m + 1];
                        Ej2 -= -0.5 * J[i] * J[i] * expm2th * g(n, m) * g(n - 1, m + 1)
                                * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n - 1, m + 1)))
                                * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n - 1] * f[j2][m + 1];
                    }

                    if (n > 0) {
                        Ej1j2 += -J[j1] * J[i] * g(n, m) * g(n - 1, n)
                                * (eps(dU, i, j1, n, m, i, j2, n - 1, n) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n - 1, n))))
                                * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][n - 1]
                                * f[i][n - 1] * f[j1][m] * f[j2][n];
                        Ej1j2 += -J[i] * J[j1] * g(n, m) * g(n - 1, n)
                                * (eps(dU, i, j2, n, m, i, j1, n - 1, n) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n - 1, n))))
                                * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][n - 1]
                                * f[i][n - 1] * f[j2][m] * f[j1][n];
                    }
                    if (n < nmax - 1) {
                        Ej1j2 -= -J[j1] * J[i] * g(n, m) * g(n + 1, n + 2)
                                * (eps(dU, i, j1, n, m, i, j2, n + 1, n + 2) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n + 1, n + 2))))
                                * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][n + 1]
                                * f[i][n] * f[j1][m] * f[j2][n + 2];
                        Ej1j2 -= -J[i] * J[j1] * g(n, m) * g(n + 1, n + 2)
                                * (eps(dU, i, j2, n, m, i, j1, n + 1, n + 2) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n + 1, n + 2))))
                                * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][n + 1]
                                * f[i][n] * f[j2][m] * f[j1][n + 2];
                    }

                    Ej1 += -0.5 * J[j1] * J[j1] * g(n, m) * g(m - 1, n + 1)
                            * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, n + 1)))
                            * (~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m] -
                            ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1]);
                    Ej2 += -0.5 * J[i] * J[i] * g(n, m) * g(m - 1, n + 1)
                            * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, n + 1)))
                            * (~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m] -
                            ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1]);

                    for (int q = 1; q <= nmax; q++) {
                        if (n < nmax - 1 && n != q - 2) {
                            Ej1j2 += -0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, q)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n + 1, q)))
                                    * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][q - 1]
                                    * f[i][n] * f[j1][m] * f[j2][q];
                            Ej1j2 += -0.5 * J[i] * J[j1] * g(n, m) * g(n + 1, q)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n + 1, q)))
                                    * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][q - 1]
                                    * f[i][n] * f[j2][m] * f[j1][q];
                        }
                        if (n > 0 && n != q) {
                            Ej1j2 -= -0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, q)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n - 1, q)))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][q - 1]
                                    * f[i][n - 1] * f[j1][m] * f[j2][q];
                            Ej1j2 -= -0.5 * J[i] * J[j1] * g(n, m) * g(n - 1, q)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n - 1, q)))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][q - 1]
                                    * f[i][n - 1] * f[j2][m] * f[j1][q];
                        }

                        if (m != q) {
                            Ej1k1 += -0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, q)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                    * ~f[i][n + 1] * ~f[j1][m] * ~f[k1][q - 1]
                                    * f[i][n] * f[j1][m] * f[k1][q];
                            Ej2k2 += -0.5 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, q)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                    * ~f[i][n + 1] * ~f[j2][m] * ~f[k2][q - 1]
                                    * f[i][n] * f[j2][m] * f[k2][q];
                            Ej1k1 -= -0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, q)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][q - 1]
                                    * f[i][n] * f[j1][m - 1] * f[k1][q];
                            Ej2k2 -= -0.5 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, q)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][q - 1]
                                    * f[i][n] * f[j2][m - 1] * f[k2][q];
                        }

                    }

                    for (int p = 0; p < nmax; p++) {

                        if (p != n - 1 && 2 * n - m == p && n > 0) {
                            Ej1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U0, n, m))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][p]
                                    * f[i][n - 1] * f[j1][m] * f[j2][p + 1];
                            Ej1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U0, n, m))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][p]
                                    * f[i][n - 1] * f[j2][m] * f[j1][p + 1];
                        }
                        if (p != n + 1 && 2 * n - m == p - 2 && n < nmax - 1) {
                            Ej1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U0, n, m))
                                    * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][p]
                                    * f[i][n] * f[j1][m] * f[j2][p + 1];
                            Ej1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U0, n, m))
                                    * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][p]
                                    * f[i][n] * f[j2][m] * f[j1][p + 1];
                        }

                        if (p != n - 1 && 2 * n - m != p && n > 0) {
                            Ej1j2 += -0.25 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1)
                                    * (eps(dU, i, j1, n, m, i, j2, p, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, p + 1))))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][p]
                                    * f[i][n - 1] * f[j1][m] * f[j2][p + 1];
                            Ej1j2 += -0.25 * J[i] * J[j1] * g(n, m) * g(n - 1, p + 1)
                                    * (eps(dU, i, j2, n, m, i, j1, p, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, p + 1))))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][p]
                                    * f[i][n - 1] * f[j2][m] * f[j1][p + 1];
                        }
                        if (p != n + 1 && 2 * n - m != p - 2 && n < nmax - 1) {
                            Ej1j2 -= -0.25 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1)
                                    * (eps(dU, i, j1, n, m, i, j2, p, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, p + 1))))
                                    * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][p]
                                    * f[i][n] * f[j1][m] * f[j2][p + 1];
                            Ej1j2 -= -0.25 * J[i] * J[j1] * g(n, m) * g(n + 1, p + 1)
                                    * (eps(dU, i, j2, n, m, i, j1, p, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, p + 1))))
                                    * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][p]
                                    * f[i][n] * f[j2][m] * f[j1][p + 1];
                        }

                        if (p != m - 1 && n != p) {
                            Ej1k1 += -0.25 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, p + 1)
                                    * (eps(dU, i, j1, n, m, j1, k1, p, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, p + 1))))
                                    * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][p] * f[i][n] * f[j1][m - 1] * f[k1][p + 1] -
                                    ~f[i][n + 1] * ~f[j1][m] * ~f[k1][p] * f[i][n] * f[j1][m] * f[k1][p + 1]);
                            Ej2k2 += -0.25 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, p + 1)
                                    * (eps(dU, i, j2, n, m, j2, k2, p, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, p + 1))))
                                    * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][p] * f[i][n] * f[j2][m - 1] * f[k2][p + 1] -
                                    ~f[i][n + 1] * ~f[j2][m] * ~f[k2][p] * f[i][n] * f[j2][m] * f[k2][p + 1]);
                        }
                    }

                    Ej1k1 += 0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                            * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][n]
                            * f[i][n] * f[j1][m - 1] * f[k1][n + 1] -
                            ~f[i][n + 1] * ~f[j1][m] * ~f[k1][n]
                            * f[i][n] * f[j1][m] * f[k1][n + 1]);
                    Ej2k2 += 0.5 * J[j2] * J[i] * expm2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                            * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][n]
                            * f[i][n] * f[j2][m - 1] * f[k2][n + 1] -
                            ~f[i][n + 1] * ~f[j2][m] * ~f[k2][n]
                            * f[i][n] * f[j2][m] * f[k2][n + 1]);

                    Ej1k1 += -J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, m)
                            * (eps(dU, i, j1, n, m, j1, k1, m - 1, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, m))))
                            * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][m - 1] * f[i][n] * f[j1][m - 1] * f[k1][m] -
                            ~f[i][n + 1] * ~f[j1][m] * ~f[k1][m - 1] * f[i][n] * f[j1][m] * f[k1][m]);
                    Ej2k2 += -J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, m)
                            * (eps(dU, i, j2, n, m, j2, k2, m - 1, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, m))))
                            * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][m - 1] * f[i][n] * f[j2][m - 1] * f[k2][m] -
                            ~f[i][n + 1] * ~f[j2][m] * ~f[k2][m - 1] * f[i][n] * f[j2][m] * f[k2][m]);

                    if (m != n - 1 && n != m && m < nmax && n > 0) {
                        Ej1 += -0.25 * J[j1] * J[j1] * exp2th * g(n, m) * g(n - 1, m + 1)
                                * (eps(dU, i, j1, n, m, i, j1, m, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, m + 1))))
                                * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n - 1] * f[j1][m + 1];
                        Ej2 += -0.25 * J[i] * J[i] * expm2th * g(n, m) * g(n - 1, m + 1)
                                * (eps(dU, i, j2, n, m, i, j2, m, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, m + 1))))
                                * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n - 1] * f[j2][m + 1];
                    }
                    if (n != m - 3 && n != m - 2 && n < nmax - 1 && m > 1) {
                        Ej1 -= -0.25 * J[j1] * J[j1] * exp2th * g(n, m) * g(n + 1, m - 1)
                                * (eps(dU, i, j1, n, m, i, j1, m - 2, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, m - 1))))
                                * ~f[i][n + 2] * ~f[j1][m - 2] * f[i][n] * f[j1][m];
                        Ej2 -= -0.25 * J[i] * J[i] * expm2th * g(n, m) * g(n + 1, m - 1)
                                * (eps(dU, i, j2, n, m, i, j2, m - 2, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, m - 1))))
                                * ~f[i][n + 2] * ~f[j2][m - 2] * f[i][n] * f[j2][m];
                    }
                }
            }
        }
//    }

    Ei /= norm2[i];
    Ej1 /= norm2[i] * norm2[j1];
    Ej2 /= norm2[i] * norm2[j2];
    Ej1j2 /= norm2[i] * norm2[j1] * norm2[j2];
    Ej1k1 /= norm2[i] * norm2[j1] * norm2[k1];
    Ej2k2 /= norm2[i] * norm2[j2] * norm2[k2];

    E += Ei;
    E += Ej1;
    E += Ej2;
    E += Ej1j2;
    E += Ej1k1;
    E += Ej2k2;
    //    }

    return E.real();
}