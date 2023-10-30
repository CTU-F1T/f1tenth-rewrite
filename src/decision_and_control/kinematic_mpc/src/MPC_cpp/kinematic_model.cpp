//
// Created by tomas on 3/17/23.
//

#include "kinematic_model.h"
#include <cmath>
//#include<algorithm>


/* This should be rewritten using base model class so we can easilly add more models. */


namespace mpc {

    KinematicModel::KinematicModel(double predictHorizon) : predict_horizon(predictHorizon) {}

    Eigen::Matrix<double, NX, 1> KinematicModel::getF(Eigen::Matrix<double, NX, 1> &state, Eigen::Matrix<double, NU, 1> &input) {
        Eigen::Matrix<double, NX, 1> f;  // x(t+1) = x(t) + f(x(t),u(t)) * dt

        KinematicModel::clipInput(input);

        double wheelbase = this->lr + this->lf;
//        if (TEMP_MODEL_SWITCH == 1) {
            f[SX] = state[VX] * cos(state[YAW]);
            f[SY] = state[VX] * sin(state[YAW]);
            f[YAW] = state[VX] / wheelbase * tan(state[STEER]);
            f[VX] = input[ACC];
            f[STEER] = input[STEER_V];
//        }else if(TEMP_MODEL_SWITCH == 2){
//            double beta = atan(this->lr / wheelbase * tan(state[STEER]))
//            f[SX] = state[VX] * cos(state[YAW] + beta);
//            f[SY] = state[VX] * sin(state[YAW] + beta);
//            f[YAW] = state[VX] / this->lr * sin(beta);
//            f[VX] = input[ACC];
//            f[STEER] = input[STEER_V];
//        }

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

//      if (TEMP_MODEL_SWITCH == 1) {
        a << 1.0, 0.0, -dt * state[VX] * sin(state[YAW]), dt * cos(state[YAW]), 0.0,
                0.0, 1.0, dt * state[VX] * cos(state[YAW]), dt * sin(state[YAW]), 0.0,
                0.0, 0.0, 1.0, dt * tan(state[STEER]) / wheelbase, dt * state[VX] /
                                                                   (wheelbase * cos(state[STEER]) * cos(state[STEER])),
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
//      }else if (TEMP_MODEL_SWITCH == 2){ // Got lost here so not for now -> this is just kinematic model in different reference frame 
//          double dx_dd_top = (this->lr * state[VX] * wheelbase * sin(state[YAW] + atan(this->lr * tan(state[STEER]) / wheelbase)));
//          double dx_dd_bottom = (this->lr * this->lr * tan(state[STEER]) * tan(state[STEER]) + wheelbase * wheelbase ) * cos(state[STEER]) * cos(state[STEER]);
//          double dx_dd = - dx_dd_top / dx_dd_bottom;
//
//          double dy_dd_top = (this->lr * state[VX] * wheelbase * cos(state[YAW] + atan(this->lr * tan(state[STEER]) / wheelbase)));
//          double dy_dd_bottom = (this->lr * this->lr * tan(state[STEER]) * tan(state[STEER]) + wheelbase * wheelbase ) * cos(state[STEER]) * cos(state[STEER]);
//          double dy_dd = dy_dd_top / dy_dd_bottom;
//
//          double dth_dd = state[VX] / this->lr * ...
//
//      }
    }

    void KinematicModel::predict_motion(std::vector <Eigen::Matrix<double, NX, 1>> &predicted_trajectory,
                                        std::vector <Eigen::Matrix<double, NU, 1>> &inputs,
                                        Eigen::Matrix<double, NX, 1> &x0,
                                        double dt) {
        /* Model integration -> Could we change this to RK ? 
        Would that mess up linearization points since we use forward euler in the MPC? */

        // Not really dependent on the specific model -> put to base class or even mpc.cpp

        Eigen::Matrix<double, NX, 1> f;
        predicted_trajectory.at(0) = x0;
        for (int i = 0; i < this->predict_horizon - 1; i++) {
            f = KinematicModel::getF(predicted_trajectory.at(i), inputs.at(i));
            predicted_trajectory.at(i + 1) = predicted_trajectory.at(i) + f * dt; // x_{k+1} = x_k + f * dt
            KinematicModel::clipOutput(predicted_trajectory.at(i + 1));
        }
    }


} // mpc