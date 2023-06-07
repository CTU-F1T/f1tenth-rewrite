//
// Created by tomas on 3/17/23.
//

#ifndef MPC_MPC_H
#define MPC_MPC_H

#include <osqp/osqp.h>
#include <Eigen/Eigen>
#include <utility>
#include <OsqpEigen/OsqpEigen.h>
#include <chrono>

#include "kinematic_model.h"
#include "constants.h"

namespace mpc {

    class MPC {
    private:
    public:
        double dt;  // control period [s]
        int prediction_horizon;
        // vehicle model
        KinematicModel model;
        std::vector<Eigen::Matrix<double, NX, 1>> last_mpc_out_state;// = Eigen::Matrix<double, NX, 1>::Zero();
        std::vector<Eigen::Matrix<double, NX, 1>> lin_trajectory;// = Eigen::Matrix<double, NX, NK>::Zero();
        std::vector<Eigen::Matrix<double, NU, 1>> lin_inputs;// = Eigen::Matrix<double, NU, NK - 1>::Zero();
        std::vector<Eigen::Matrix<double, NU, 1>> last_mpc_out_input;// = Eigen::Matrix<double, NU, NK - 1>::Zero();
        std::vector<Eigen::Matrix<double, NX, 1>> patch_ref_trajectory;// = Eigen::Matrix<double, NX, NK>::Zero();

        // definition of matrix sequences A, B, C
        std::vector<Eigen::Matrix<double, NX, NX>> state_matrix_sequence;
        std::vector<Eigen::Matrix<double, NX, NU>> input_matrix_sequence;
        std::vector<Eigen::Matrix<double, NX, 1>> affine_shift_matrix_sequence;
        int last_nearest_point_id;

        // Trajectory definition [s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2] (TUM compatibility)
        std::vector<Eigen::Matrix<double, TUM_TRAJ_PARAMS, 1>> reference_trajectory;

        Eigen::DiagonalMatrix<double, NX> Q;
        Eigen::DiagonalMatrix<double, NX> Qn;
        Eigen::DiagonalMatrix<double, NU> R;

        // debugging variables
        std::chrono::duration<int,std::micro> predict_motion_time;
        std::chrono::duration<int,std::micro> linearization_time;
        std::chrono::duration<int,std::micro> ref_time;
        std::chrono::duration<int,std::micro> cast_time;
        std::chrono::duration<int,std::micro> solver_time;

        OsqpEigen::Solver solver;

        MPC(double dt, int predictionHorizon, KinematicModel model, Eigen::DiagonalMatrix<double, NX> Q, Eigen::DiagonalMatrix<double, NU> R, Eigen::DiagonalMatrix<double, NX> Qn);

        void CastMPCToQPConstraintMatrix(Eigen::SparseMatrix<double> &constraintMatrix);

        void CastMPCToQPHessian(Eigen::SparseMatrix<double> &hessianMatrix);

        void CastMPCToQPGradient(const std::vector<Eigen::Matrix<double, NX, 1>> &xRef, Eigen::VectorXd &gradient);

        void CastMPCToQPConstraintVectors(const Eigen::Matrix<double, NX, 1> &x0,
                                          Eigen::VectorXd &lowerBound, Eigen::VectorXd &upperBound);

        void GetPatchOfRefenceTrajecotry();

        void NearestPoint(const Eigen::Matrix<double, 2, 1> &point,
                          Eigen::Matrix<double, 2, 1> &out_point, double &out_dist, double &out_t, int &out_index, Eigen::MatrixXd &trajectory);

        int plan(std::vector<Eigen::Matrix<double, NX, 1>> &states_plan, // NK
                  std::vector<Eigen::Matrix<double, NU, 1>> &inputs_plan, // NK - 1
                  std::vector<Eigen::Matrix<double, NX, 1>> &predicted_traj,
                  std::vector<Eigen::Matrix<double, NX, 1>> &reference,
                  Eigen::Matrix<double, NX, 1> &x0);

    };
} // mpc

#endif //MPC_MPC_H
