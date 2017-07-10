#ifndef MEASUREMENT_PACKAGE_H_
#define MEASUREMENT_PACKAGE_H_

#include "Eigen/Dense"

class MeasurementPackage {
public:
  long timestamp;

  enum SensorType{
    LASER,
    RADAR
  } sensor_type_;

  Eigen::VectorXd raw_measurements;

  bool is_radar() { return sensor_type_ == RADAR; }
  bool is_lidar() { return sensor_type_ == LASER; }
};

#endif /* MEASUREMENT_PACKAGE_H_ */
