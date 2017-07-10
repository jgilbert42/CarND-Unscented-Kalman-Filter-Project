#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  //std_a_ = 30;
  //std_a_ = 3;
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  //std_yawdd_ = 30;
  //std_yawdd_ = 3;
  std_yawdd_ = 0.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  R_lidar_ = MatrixXd(2, 2);
  H_lidar_ = MatrixXd(2, n_x_);

  //measurement covariance matrix - lidar
  R_lidar_ << 0.0225, 0,
              0, 0.0225;

  H_lidar_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  weights_ = VectorXd(2 * n_aug_ + 1);
  double weight_0 = lambda_/(lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    weights_(i) = 0.5/(n_aug_ + lambda_);
  }

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  //cout << "ProcessMeasurement:" << endl << meas_package.raw_measurements << endl;
  if (!use_laser_ && meas_package.is_lidar()) {
    cout << "skipping lidar" << endl;
    return;
  }

  if (!use_radar_ && meas_package.is_radar()) {
    cout << "skipping radar" << endl;
    return;
  }

  if (!is_initialized_) {
    x_.fill(0);
    if (meas_package.is_lidar()) {
      x_(0) = meas_package.raw_measurements[0];
      x_(1) = meas_package.raw_measurements[1];
    } else if (meas_package.is_radar()) {
      float rho = meas_package.raw_measurements[0];
      float theta = meas_package.raw_measurements[1];
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      x_(0) = rho*cos(theta);
      x_(1) = rho*sin(theta);
    }

    time_us_ = meas_package.timestamp;

    is_initialized_ = true;
    return;
  }

  double dt = (meas_package.timestamp - time_us_) / 1000000.0;

  Prediction(dt);

  if (meas_package.is_radar()) {
    UpdateRadar(meas_package);
  } else if (meas_package.is_lidar()) {
    UpdateLidar(meas_package);
  }

  time_us_ = meas_package.timestamp;
}

double normalize_angle(double ang) {
  while (ang > M_PI) ang -= 2.0 * M_PI;
  while (ang < -M_PI) ang += 2.0 * M_PI;
  return ang;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_,n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  MatrixXd L = P_aug.llt().matrixL();

  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  // predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    double px_p, py_p;

    // avoid div by zero
    if (fabs(yawd) > 0.001) {
      px_p = px + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_p = py + v/yawd * (cos(yaw) - cos(yaw+yawd*delta_t));
    } else {
      px_p = px + v*delta_t*cos(yaw);
      py_p = py + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  // predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = normalize_angle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

  //cout << "Xsig_pred_" << Xsig_pred_ << endl;
  //cout << "x_:" << endl << x_ << endl;
  //cout << "P_:" << endl << P_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  VectorXd z = meas_package.raw_measurements;

	VectorXd z_pred = H_lidar_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_lidar_.transpose();
	MatrixXd S = H_lidar_ * P_ * Ht + R_lidar_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_lidar_) * P_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
   VectorXd z = meas_package.raw_measurements;

  int n_z = z.size(); // 3

  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    Zsig(0, i) = sqrt(px * px + py * py); // r
    Zsig(1, i) = atan2(py, px); // phi
    Zsig(2, i) = (px*v1 + py*v2) / sqrt(px*px + py*py); // r_dot
  }

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    z_diff(1) = normalize_angle(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  S = S + R;

  // matrix for cross correlation
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = normalize_angle(z_diff(1));

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = normalize_angle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = z - z_pred;

  z_diff(1) = normalize_angle(z_diff(1));

  x_ = x_ + K * z_diff;
  //x_(3) = normalize_angle(x_(3));

  P_ = P_ - K * S * K.transpose();

  //cout << "new x_:" << endl << x_ << endl;
  //cout << "new P_:" << endl << P_ << endl;
}
