//
// Created by tomas on 3/17/23.
//

#include "mpc.h"

#include <utility>
#include <chrono>

namespace mpc {

    void MPC::CastMPCToQPConstraintMatrix(Eigen::SparseMatrix<double> &constraintMatrix) {

        constraintMatrix.resize(
                NX * this->prediction_horizon + NX * this->prediction_horizon + NU * (this->prediction_horizon - 1),
                NX * this->prediction_horizon + NU * (this->prediction_horizon - 1)
        );

        std::vector<Eigen::Triplet<double>> triplets;

        // populate linear constraint matrix
        for (int i = 0; i < NX * this->prediction_horizon; i++) {
//            constraintMatrix.insert(i, i) = -1;
            triplets.push_back(Eigen::Triplet<double>(i, i, -1));
        }

        for (int i = 0; i < (this->prediction_horizon - 1); i++) {
            for (int j = 0; j < NX; j++) {
                for (int k = 0; k < NX; k++) {
                    double value = state_matrix_sequence[i](j, k);
                    if (value != 0) {
//                        constraintMatrix.insert(NX * (i + 1) + j, NX * i + k) = value;
                        triplets.push_back(Eigen::Triplet<double>(NX * (i + 1) + j, NX * i + k, value));
                    }
                }
            }
        }
        for (int i = 0; i < (this->prediction_horizon - 1); i++) {
            for (int j = 0; j < NX; j++) {
                for (int k = 0; k < NU; k++) {
                    double value = input_matrix_sequence[i](j, k);
                    if (value != 0) {
//                        constraintMatrix.insert(NX * (i + 1) + j, NU * i + k + NX * this->prediction_horizon) = value;
                        triplets.push_back(Eigen::Triplet<double>(NX * (i + 1) + j, NU * i + k + NX * this->prediction_horizon, value));
                    }
                }
            }
        }

        for (int i = 0; i < NX * this->prediction_horizon + NU * (this->prediction_horizon - 1); i++) {
//            constraintMatrix.insert(i + this->prediction_horizon * NX, i) = 1;
            triplets.push_back(Eigen::Triplet<double>(i + this->prediction_horizon * NX, i, 1));
        }
        constraintMatrix.setFromTriplets(triplets.begin(), triplets.end());
    }

    void MPC::CastMPCToQPHessian(Eigen::SparseMatrix<double> &hessianMatrix) {

        hessianMatrix.resize(NX * this->prediction_horizon + NU * (this->prediction_horizon - 1),
                             NX * this->prediction_horizon + NU * (this->prediction_horizon - 1));

        //populate hessian matrix
        for (int i = 0; i < NX * this->prediction_horizon + NU * (this->prediction_horizon - 1); i++) {
            if (i < NX * (this->prediction_horizon - 1)) {
                int posQ = i % NX;
                double value = Q.diagonal()[posQ];
                if (value != 0)
                    hessianMatrix.insert(i, i) = value;
            } else if (i < NX * this->prediction_horizon) {
                int posQ = i % NX;
                double value = Qn.diagonal()[posQ];
                if (value != 0)
                    hessianMatrix.insert(i, i) = value;
            } else {
                int posR = i % NU;
                double value = R.diagonal()[posR];
                if (value != 0)
                    hessianMatrix.insert(i, i) = value;
            }
        }

    }

    void MPC::CastMPCToQPGradient(const std::vector <Eigen::Matrix<double, NX, 1>> &xRef,
                                  Eigen::VectorXd &gradient) {

        std::vector <Eigen::Matrix<double, NX, 1>> Qx_ref(this->prediction_horizon, Eigen::Matrix<double, NX, 1>::Zero());

        for (int i = 0; i < this->prediction_horizon - 1; i++) {
            Qx_ref.at(i) = Q * (-xRef.at(i));
        }
        Qx_ref.at(this->prediction_horizon - 1) = Qn * (-xRef.at(this->prediction_horizon - 1));

        // populate the gradient vector
        gradient = Eigen::VectorXd::Zero(NX * this->prediction_horizon + NU * (this->prediction_horizon - 1), 1);
        for (int i = 0; i < this->prediction_horizon; i++) {
            for (int j = 0; j < NX; j++) {
                double value = Qx_ref.at(i)[j];
                gradient(i * NX + j, 0) = value;
            }
        }
    }

    void MPC::CastMPCToQPConstraintVectors(const Eigen::Matrix<double, NX, 1> &x0,
                                           Eigen::VectorXd &lowerBound, Eigen::VectorXd &upperBound) {

        lowerBound.resize(2 * NX * this->prediction_horizon + NU * (this->prediction_horizon - 1), 1);
        upperBound.resize(2 * NX * this->prediction_horizon + NU * (this->prediction_horizon - 1), 1);

        Eigen::Matrix<double, NX, 1> xMax;
        Eigen::Matrix<double, NX, 1> xMin;
        Eigen::Matrix<double, NU, 1> uMax;
        Eigen::Matrix<double, NU, 1> uMin;
        model.setModelConstraints(xMax, xMin, uMax, uMin);

        // evaluate the lower and the upper inequality vectors
        Eigen::VectorXd lowerInequality = Eigen::MatrixXd::Zero(
                NX * this->prediction_horizon + NU * (this->prediction_horizon - 1), 1);
        Eigen::VectorXd upperInequality = Eigen::MatrixXd::Zero(
                NX * this->prediction_horizon + NU * (this->prediction_horizon - 1), 1);
        for (int i = 0; i < this->prediction_horizon; i++) {
            lowerInequality.block(NX * i, 0, NX, 1) = xMin;
            upperInequality.block(NX * i, 0, NX, 1) = xMax;
        }
        for (int i = 0; i < (this->prediction_horizon - 1); i++) {
            lowerInequality.block(NU * i + NX * this->prediction_horizon, 0, NU, 1) = uMin;
            upperInequality.block(NU * i + NX * this->prediction_horizon, 0, NU, 1) = uMax;
        }

        // evaluate the lower and the upper equality vectors
        Eigen::VectorXd lowerEquality = Eigen::MatrixXd::Zero(NX * this->prediction_horizon, 1);
        Eigen::VectorXd upperEquality = Eigen::MatrixXd::Zero(NX * this->prediction_horizon, 1);


        lowerEquality.block(0, 0, NX, 1) = -x0;
//        std::cout << x0 << std::endl << std::endl;
        for (int i = 0; i < (this->prediction_horizon - 1); i++) {
//            lowerEquality.block((i + 1) * (this->prediction_horizon - 1), 0, NX, 1) = -affine_shift_matrix_sequence.at(i);
            lowerEquality.block((i + 1) * NX, 0, NX, 1) = -affine_shift_matrix_sequence.at(i);
//            std::cout << lowerEquality << std::endl;
        }

        upperEquality = lowerEquality;
        lowerEquality = lowerEquality;

        lowerBound << lowerEquality,
                lowerInequality;

        upperBound << upperEquality,
                upperInequality;
    }

    int MPC::plan(std::vector <Eigen::Matrix<double, NX, 1>> &states_plan,
                  std::vector <Eigen::Matrix<double, NU, 1>> &inputs_plan,
                  std::vector <Eigen::Matrix<double, NX, 1>> &predicted_traj,
                  std::vector <Eigen::Matrix<double, NX, 1>> &reference,
                  Eigen::Matrix<double, NX, 1> &x0) {

        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        std::chrono::time_point<std::chrono::high_resolution_clock> stop;

        start = std::chrono::high_resolution_clock::now();
        // get states around which to linearize (lin_trajectory)
        model.predict_motion(this->lin_trajectory, this->last_mpc_out_input, x0, dt);
//        this->lin_trajectory = states_plan;
        predicted_traj = this->lin_trajectory;

        // get inputs around which to linearize (lin_inputs), currently all zeros is fine   this->prediction_horizon - 1
        lin_inputs = std::vector < Eigen::Matrix < double, NU, 1 >> (this->prediction_horizon - 1, Eigen::Matrix<double, NU, 1>::Zero());
        stop = std::chrono::high_resolution_clock::now();
        this->predict_motion_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        // Linearize model around trajectory
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < this->prediction_horizon - 1; i++) {
            // state_matrix_sequence >> A0, A1, ..., Athis->prediction_horizon-1
            // input_matrix_sequence >> B0, B1, ..., Bthis->prediction_horizon-1
            // affine_shift_matrix_sequence >> C0, C1, ..., Cthis->prediction_horizon-1
            // lin_trajectory.at(i)[STEER_ANGLE] = 0.0;
            model.linearizeInPoint(state_matrix_sequence.at(i),
                                   input_matrix_sequence.at(i),
                                   affine_shift_matrix_sequence.at(i),
                                   lin_trajectory.at(i),
                                   lin_inputs.at(i),  // this->last_mpc_out_input.at(i),
                                   dt);
        }
        // Linearization of the model done
        stop = std::chrono::high_resolution_clock::now();
        this->linearization_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        // Calculate reference
        start = std::chrono::high_resolution_clock::now();
        MPC::GetPatchOfRefenceTrajecotry();

        // Correct reference angle
        for (int i = 0; i < patch_ref_trajectory.size(); i++) {
            if (patch_ref_trajectory[i][2] - lin_trajectory[i][2] < -3.14) patch_ref_trajectory[i][2] += 2.0 * 3.14;
            if (patch_ref_trajectory[i][2] - lin_trajectory[i][2] > 3.14) patch_ref_trajectory[i][2] -= 2.0 * 3.14;
        }

        reference = patch_ref_trajectory;
        stop = std::chrono::high_resolution_clock::now();
        this->ref_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        start = std::chrono::high_resolution_clock::now();
        Eigen::SparseMatrix<double> constraintMatrix;
        this->CastMPCToQPConstraintMatrix(constraintMatrix);

        Eigen::SparseMatrix<double> hessianMatrix;
        this->CastMPCToQPHessian(hessianMatrix);

        Eigen::VectorXd gradient;
        this->CastMPCToQPGradient(patch_ref_trajectory, gradient);

        Eigen::VectorXd upperBound;
        Eigen::VectorXd lowerBound;
        this->CastMPCToQPConstraintVectors(x0, lowerBound, upperBound);
        stop = std::chrono::high_resolution_clock::now();
        this->cast_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);


        start = std::chrono::high_resolution_clock::now();

        solver.data()->clearHessianMatrix();
        solver.data()->clearLinearConstraintsMatrix();
        solver.clearSolver();

        if (!solver.data()->setHessianMatrix(hessianMatrix)) return 1; // Who needs to validate...
        if (!solver.data()->setGradient(gradient)) return 1;
        if (!solver.data()->setLinearConstraintsMatrix(constraintMatrix)) return 1;
        if (!solver.data()->setLowerBound(lowerBound)) return 1;
        if (!solver.data()->setUpperBound(upperBound)) return 1;

        if (!solver.initSolver()) return 1;

        // Calculation of reference trajectory done

        // Create and solve optimization problem
        // state_matrix_sequence >> A_0, A_1, ..., A_{this->prediction_horizon-1}
        // input_matrix_sequence >> B_0, B_1, ..., B_{this->prediction_horizon-1}
        // affine_shift_matrix_sequence >> C_0, C_1, ..., C_{this->prediction_horizon-1}
        Eigen::VectorXd QPSolution;

        if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return 1;
        QPSolution = solver.getSolution();
        for (int i = 0; i < this->prediction_horizon; i++) {
            states_plan.at(i) = QPSolution.block(i * NX, 0, NX, 1);
        }

        int input_offset = NX * this->prediction_horizon;
        for (int i = 0; i < this->prediction_horizon - 1; i++) {
            inputs_plan.at(i) = QPSolution.block(input_offset + i * NU, 0, NU, 1);
        }

        last_mpc_out_input = inputs_plan;
        stop = std::chrono::high_resolution_clock::now();
        this->solver_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        return 0;

    }

    MPC::MPC(double dt, int predictionHorizon, KinematicModel model, Eigen::DiagonalMatrix<double, NX> Q, Eigen::DiagonalMatrix<double, NU> R,
             Eigen::DiagonalMatrix<double, NX> Qn) : dt(dt), prediction_horizon(predictionHorizon),
             model(std::move(model)), Q(Q), R(R), Qn(Qn), last_nearest_point_id(-1) {
        // Initialization of model matrix arrays
        state_matrix_sequence = std::vector < Eigen::Matrix < double, NX, NX >> (this->prediction_horizon - 1, Eigen::Matrix<double, NX, NX>::Zero());
        input_matrix_sequence = std::vector < Eigen::Matrix < double, NX, NU >> (this->prediction_horizon - 1, Eigen::Matrix<double, NX, NU>::Zero());
        affine_shift_matrix_sequence = std::vector < Eigen::Matrix < double, NX, 1
                >> (this->prediction_horizon - 1, Eigen::Matrix<double, NX, 1>::Zero());

        // Initialization of vector for storing the latest result of the optimization
        last_mpc_out_state = std::vector < Eigen::Matrix < double, NX, 1 >> (this->prediction_horizon, Eigen::Matrix<double, NX, 1>::Zero());
        last_mpc_out_input = std::vector < Eigen::Matrix < double, NU, 1 >> (this->prediction_horizon - 1, Eigen::Matrix<double, NU, 1>::Zero());

        // Initialization of vector of states and inputs around which we discretize the model
        lin_trajectory = std::vector < Eigen::Matrix < double, NX, 1 >> (this->prediction_horizon,
                Eigen::Matrix<double, NX, 1>::Zero());
        lin_inputs = std::vector < Eigen::Matrix < double, NU, 1 >> (this->prediction_horizon - 1,
                Eigen::Matrix<double, NU, 1>::Zero());

        // Initialization of vector of reference trajectory patch needed for optimization (this is not for storing the whole trajectory)
        patch_ref_trajectory = std::vector < Eigen::Matrix < double, NX, 1 >> (this->prediction_horizon,
                Eigen::Matrix<double, NX, 1>::Zero());

        solver.settings()->setWarmStart(true);
        solver.settings()->setVerbosity(false);
        solver.settings()->setPolish(true);
        solver.settings()->setAbsoluteTolerance(0.01);
        solver.settings()->setRelativeTolerance(0.01);

        solver.data()->setNumberOfVariables(NX * this->prediction_horizon + NU * (this->prediction_horizon - 1));
        solver.data()->setNumberOfConstraints(2 * NX * this->prediction_horizon + NU * (this->prediction_horizon - 1));
    }

    void MPC::GetPatchOfRefenceTrajecotry() {
        bool first_iter_check = last_nearest_point_id == -1;
        auto curr_size = first_iter_check ? reference_trajectory.size() : (int)(1.4 * this->prediction_horizon);
        Eigen::MatrixXd original_2d_traj(2, curr_size);
        int step_back = 5;

        for (int i = 0; i < curr_size; i++) {
            auto id = first_iter_check ? i : ((last_nearest_point_id - step_back + i) + reference_trajectory.size()) % reference_trajectory.size();
            auto point = reference_trajectory.at(id);
            Eigen::Matrix<double, 2, 1> converted = point.block(1, 0, 2, 1);
            original_2d_traj.col(i) = converted;
        }

        #pragma omp parallel for
        for (int i = 0; i < lin_trajectory.size(); i++) {
            auto point = lin_trajectory.at(i);
            Eigen::Matrix<double, 2, 1> converted = point.block(0, 0, 2, 1);
            Eigen::Matrix<double, 2, 1> out_point;
            double out_dist;
            double out_t;
            int out_index;
            this->NearestPoint(converted, out_point, out_dist, out_t, out_index, original_2d_traj);
            out_index = first_iter_check ? out_index : ((out_index - step_back + last_nearest_point_id) + reference_trajectory.size()) % reference_trajectory.size();
            int next_index = (out_index + 1) % reference_trajectory.size();

            Eigen::Matrix<double, NX, 1> new_traj_point = Eigen::Matrix<double, NX, 1>::Zero();
            new_traj_point.block(0, 0, 2, 1) = out_point;
            Eigen::Matrix<double, 2, 1> vec = reference_trajectory.at(next_index).block(1, 0, 2, 1) - out_point;
            new_traj_point(2) = atan2(vec(1), vec(0));
            new_traj_point(3) = reference_trajectory.at(out_index)(5) * out_t + reference_trajectory.at(next_index)(5) * (1 - out_t);

            patch_ref_trajectory.at(i) = new_traj_point;
            if (i == 0)
            {
                last_nearest_point_id = out_index;
            }
        }
    }

    void MPC::NearestPoint(const Eigen::Matrix<double, 2, 1> &point, Eigen::Matrix<double, 2, 1> &out_point, double &out_dist,
                           double &out_t, int &out_index, Eigen::MatrixXd &trajectory) {
        int first_point;
        Eigen::MatrixXd point_rep = Eigen::MatrixXd::Constant(1, trajectory.cols(), point(0));
        Eigen::VectorXd diff_x = trajectory.row(0) - point_rep;
        point_rep = Eigen::MatrixXd::Constant(1, trajectory.cols(), point(1));
        Eigen::VectorXd diff_y = trajectory.row(1) - point_rep;
        Eigen::VectorXd dists = diff_x.array().square() + diff_y.array().square();

        Eigen::VectorXd::Index index;
        double min_dist_2 = dists.minCoeff(&index);
        first_point = index;

        int second_point;
        if (first_point == 0){
            second_point = 1;
        } else if (first_point == trajectory.cols() - 1){
            second_point = trajectory.cols() - 2;
        } else {
            double dist_back = (trajectory.col(first_point - 1) - point).squaredNorm();
            double dist_forward = (trajectory.col(first_point + 1) - point).squaredNorm();
            second_point = dist_back < dist_forward ? first_point - 1 : first_point + 1;
        }

        if (second_point < first_point){
            second_point++;
            first_point--;
        }

        Eigen::Matrix<double, 2, 1> traj_vec = trajectory.col(second_point) - trajectory.col(first_point);
        double l2 = traj_vec.squaredNorm();
        Eigen::Matrix<double, 2, 1> curr_diff = point - trajectory.col(first_point);
        double dot_prod = curr_diff.dot(traj_vec);
        double t = dot_prod / l2;
        Eigen::Matrix<double, 2, 1> scaled_vec = t * traj_vec;

        out_point = trajectory.col(first_point) + scaled_vec;
        out_dist = 0;
        out_t = t;
        out_index = first_point;
    }
} // mpc
