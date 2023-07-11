//
// Created by tomas on 3/17/23.
//

#include "kinematic_model.h"
#include <cmath>
//#include<algorithm>


namespace mpc {

    KinematicModel::KinematicModel(double predictHorizon) : predict_horizon(predictHorizon) {}

    Eigen::Matrix<double, NX, 1> KinematicModel::getF(Eigen::Matrix<double, NX, 1> &state, Eigen::Matrix<double, NU, 1> &input) {
        Eigen::Matrix<double, NX, 1> f;  // x(t+1) = x(t) + f(x(t),u(t)) * dt

        KinematicModel::clipInput(input);

        double wheelbase = this->lr + this->lf;

        f[SX] = state[VX] * cos(state[YAW]);
        f[SY] = state[VX] * sin(state[YAW]);
        f[YAW] = state[VX] / wheelbase * tan(state[STEER]);
        f[VX] = input[ACC];
        f[STEER] = input[STEER_V];

        return f;
    }

    void KinematicModel::setModelConstraints(Eigen::Matrix<double, NX, 1> &xMax, Eigen::Matrix<double, NX, 1> &xMin,
                                             Eigen::Matrix<double, NU, 1> &uMax, Eigen::Matrix<double, NU, 1> &uMin) {

        // input inequality constraints
        uMin << max_dcc, // acceleration
                -max_steer_vel; // steering speed

        uMax << max_acc,
                max_steer_vel;

        // state inequality constraints  [x, y, yaw angle, vx, steer]
        xMin << -OsqpEigen::INFTY, -OsqpEigen::INFTY, -OsqpEigen::INFTY, min_speed, min_steer_angle;

        xMax << OsqpEigen::INFTY, OsqpEigen::INFTY, OsqpEigen::INFTY, max_speed, max_steer_angle;
    }

    void KinematicModel::clipInput(Eigen::Matrix<double, NU, 1> &input) {
        input[STEER_V] = std::max(input[STEER_V], -max_steer_vel);
        input[STEER_V] = std::min(input[STEER_V], max_steer_vel);

        input[ACC] = std::max(input[ACC], max_dcc);
        input[ACC] = std::min(input[ACC], max_acc);
    }

    void KinematicModel::clipOutput(Eigen::Matrix<double, NX, 1> &state) {
        state[VX] = std::max(state[VX], min_speed);
        state[VX] = std::min(state[VX], max_speed);

        state[STEER] = std::max(state[STEER], min_steer_angle);
        state[STEER] = std::min(state[STEER], max_steer_angle);
    }

    void KinematicModel::linearizeInPoint(Eigen::Matrix<double, NX, NX> &a,
                                          Eigen::Matrix<double, NX, NU> &b,
                                          Eigen::Matrix<double, NX, 1> &c,
                                          Eigen::Matrix<double, NX, 1> &state,
                                          Eigen::Matrix<double, NU, 1> &input,
                                          double dt) {

        double wheelbase = lf + lr;

        a << 1.0, 0.0, -dt * state[VX] * sin(state[YAW]), dt * cos(state[YAW]), 0.0,
                0.0, 1.0, dt * state[VX] * cos(state[YAW]), dt * sin(state[YAW]), 0.0,
                0.0, 0.0, 1.0, dt * tan(state[STEER]) / wheelbase, dt * state[VX] / (wheelbase * cos(state[STEER]) * cos(state[STEER])),
                0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0;


        b << 0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                dt, 0.0,
                0.0, dt;

        c << dt * state[VX] * sin(state[YAW]) * state[YAW],
                -dt * state[VX] * cos(state[YAW]) * state[YAW],
                -dt * state[VX] * state[STEER] / (wheelbase * cos(state[STEER]) * cos(state[STEER])),
                0.0,
                0.0;
    }

    void KinematicModel::predict_motion(std::vector <Eigen::Matrix<double, NX, 1>> &predicted_trajectory,
                                        std::vector <Eigen::Matrix<double, NU, 1>> &inputs,
                                        Eigen::Matrix<double, NX, 1> &x0,
                                        double dt) {
        Eigen::Matrix<double, NX, 1> f;
        predicted_trajectory.at(0) = x0;
        for (int i = 0; i < this->predict_horizon - 1; i++) {
            f = KinematicModel::getF(predicted_trajectory.at(i), inputs.at(i));
            predicted_trajectory.at(i + 1) = predicted_trajectory.at(i) + f * dt;
            KinematicModel::clipOutput(predicted_trajectory.at(i + 1));
        }
    }


} // mpc