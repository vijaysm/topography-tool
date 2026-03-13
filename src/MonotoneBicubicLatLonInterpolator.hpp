#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

class MonotoneBicubicLatLonInterpolator {
public:
  using Array = Eigen::ArrayXd;

  // ============================
  // Constructor from Eigen matrix
  // ============================
  explicit MonotoneBicubicLatLonInterpolator(const Eigen::MatrixXd &field)
      : data_ptr(field.data()), nlat(field.rows()), nlon(field.cols()),
        dlon(360.0 / nlon), dlat(180.0 / (nlat - 1)) {}

  // ============================
  // Constructor from std::vector
  // ============================
  MonotoneBicubicLatLonInterpolator(const std::vector<double> &data, int nlat,
                                    int nlon)
      : data_ptr(data.data()), nlat(nlat), nlon(nlon), dlon(360.0 / nlon),
        dlat(180.0 / (nlat - 1)) {}

  // ============================
  // Batched evaluation
  // ============================
  void evaluate_batch(const Array &lon, const Array &lat, Array &out) const {
    const int N = lat.size();
    out.resize(N);

#pragma omp parallel for schedule(static)
    for (int k = 0; k < N; ++k) {
      out[k] = evaluate_scalar(lon[k], lat[k]);
    }
  }

  // ============================
  // Scalar kernel
  // ============================
  inline double evaluate_scalar(double lon, double lat) const {
    // Normalize longitude to [0,360)
    lon = std::fmod(lon, 360.0);
    if (lon < 0.0)
      lon += 360.0;

    // Clamp latitude
    lat = std::max(-90.0, std::min(90.0, lat));

    const double x = lon / dlon;
    const double y = (lat + 90.0) / dlat;

    int i = static_cast<int>(std::floor(x));
    int j = static_cast<int>(std::floor(y));

    const double tx = x - i;
    const double ty = y - j;

    // Clamp latitude stencil
    j = clamp(j, 1, nlat - 3);

    // Interpolate in longitude (periodic)
    double row[4];
    for (int n = 0; n < 4; ++n) {
      row[n] =
          steffen_cubic(value(j + n - 1, i - 1), value(j + n - 1, i),
                        value(j + n - 1, i + 1), value(j + n - 1, i + 2), tx);
    }

    return steffen_cubic(row[0], row[1], row[2], row[3], ty);
  }

private:
  const double *data_ptr;
  const int nlat, nlon;
  const double dlon, dlat;

  // ============================
  // Data access (row major)
  // ============================
  inline double value(int j, int i) const {
    return data_ptr[j * nlon + wrap(i)];
    // return data_ptr[wrap(j) * nlat + i];
    // return data_ptr[wrap(i) * nlat + j];
  }

  // ============================
  // Steffen monotone cubic
  // ============================
  static inline double steffen_cubic(double f0, double f1, double f2, double f3,
                                     double t) {
    const double d0 = f1 - f0;
    const double d1 = f2 - f1;
    const double d2 = f3 - f2;

    const double m1 = steffen_slope(d0, d1);
    const double m2 = steffen_slope(d1, d2);

    const double t2 = t * t;
    const double t3 = t2 * t;

    return (2.0 * t3 - 3.0 * t2 + 1.0) * f1 + (t3 - 2.0 * t2 + t) * m1 +
           (-2.0 * t3 + 3.0 * t2) * f2 + (t3 - t2) * m2;
  }

  static inline double steffen_slope(double d0, double d1) {
    if (d0 * d1 <= 0.0)
      return 0.0;
    const double s = d0 + d1;
    return std::copysign(
        std::min({std::abs(d0), std::abs(d1), 0.5 * std::abs(s)}), s);
  }

  static inline int clamp(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
  }

  inline int wrap(int i) const {
    i %= nlon;
    return (i < 0) ? i + nlon : i;
  }
};