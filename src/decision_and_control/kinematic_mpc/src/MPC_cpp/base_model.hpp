//
// Created by marek on 25.5.23.
//

#ifndef MPC_BASE_MODEL_HPP
#define MPC_BASE_MODEL_HPP

#include "OsqpEigen/OsqpEigen.h"
#include "constants.h"

namespace mpc {
    class BaseModel {
    public:
        double predict_horizon;

        Eigen::Matrix<double, NX, 1> current_state;

        virtual void setModelConstraints(Eigen::Matrix<double, NX, 1> &xMax, Eigen::Matrix<double, NX, 1> &xMin,
                                         Eigen::Matrix<double, NU, 1> &uMax, Eigen::Matrix<double, NU, 1> &uMin) {};

        virtual void clipInput(Eigen::Matrix<double, NU, 1> &input) {};

        virtual void clipOutput(Eigen::Matrix<double, NX, 1> &state) {};

        virtual Eigen::Matrix<double, NX, 1>
        getF(Eigen::Matrix<double, NX, 1> &state, Eigen::Matrix<double, NU, 1> &input) {
            return Eigen::Matrix<double, NX, 1>::Zero();
        };

        virtual void linearizeInPoint(Eigen::Matrix<double, NX, NX> &a,
                                      Eigen::Matrix<double, NX, NU> &b,
                                      Eigen::Matrix<double, NX, 1> &c,
                                      Eigen::Matrix<double, NX, 1> &state,
                                      Eigen::Matrix<double, NU, 1> &input,
                                      double dt) {};

        virtual void predict_motion(std::vector<Eigen::Matrix<double, NX, 1>> &predicted_trajectory, // NK
                                    std::vector<Eigen::Matrix<double, NU, 1>> &inputs, // NK - 1
                                    Eigen::Matrix<double, NX, 1> &x0,
                                    double dt) {};

    };
}

#endif //MPC_BASE_MODEL_HPP
