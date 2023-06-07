//
// Created by tomas on 3/17/23.
//

#ifndef MPC_KINEMATIC_MODEL_H
#define MPC_KINEMATIC_MODEL_H

#include "OsqpEigen/OsqpEigen.h"
#include "constants.h"
// reference point - center of rear axle

#define SX 0
#define SY 1
#define YAW 2
#define VX 3

#define ACC 0
#define STEER 1


namespace mpc {

    class KinematicModel {
    public:
        // model parameters
        double max_speed = 5.0; // [m/s]
        double min_speed = 0.0; // [m/s]
        double max_acc = 6.0; // [m/s/s]
        double max_dcc = -10.0; // [m/s/s]
        double max_steer_angle = 0.4; // [rad]
        double min_steer_angle = -0.4; // [rad]
        double lr = 0.17; // [m]
        double lf = 0.158; // [m]
        double predict_horizon;

        Eigen::Matrix<double, NX, 1> current_state;

        explicit KinematicModel(double predictHorizon);

        void setModelConstraints(Eigen::Matrix<double, NX, 1> &xMax, Eigen::Matrix<double, NX, 1> &xMin,
                                 Eigen::Matrix<double, NU, 1> &uMax, Eigen::Matrix<double, NU, 1> &uMin);

        void clipInput(Eigen::Matrix<double, NU, 1> &input);

        void clipOutput(Eigen::Matrix<double, NX, 1> &state);

        Eigen::Matrix<double, NX, 1> getF(Eigen::Matrix<double, NX, 1> &state, Eigen::Matrix<double, NU, 1> &input);

        void linearizeInPoint(Eigen::Matrix<double, NX, NX> &a,
                              Eigen::Matrix<double, NX, NU> &b,
                              Eigen::Matrix<double, NX, 1> &c,
                              Eigen::Matrix<double, NX, 1> &state,
                              Eigen::Matrix<double, NU, 1> &input,
                              double dt);

        void predict_motion(std::vector<Eigen::Matrix<double, NX, 1>> &predicted_trajectory, // NK
                            std::vector<Eigen::Matrix<double, NU, 1>>&inputs, // NK - 1
                            Eigen::Matrix<double, NX, 1> &x0,
                            double dt);

    };

} // mpc

#endif //MPC_KINEMATIC_MODEL_H
