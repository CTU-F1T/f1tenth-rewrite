#include <iostream>
#include "mpc.h"
#include "kinematic_model.h"

int main() {

    // Definition of parameters
    int prediction_horizon = 15;
    double dt = 0.1;

    // Vehicle model definition
    auto model = mpc::KinematicModel(prediction_horizon);

    // Initialize vehicle state
    Eigen::Matrix<double, NX, 1> x0;
    x0 << 0,0,0,0,0,0,0;

    // Necessary variables
    std::vector<Eigen::Matrix<double, NX, 1>> states_plan;
    std::vector<Eigen::Matrix<double, NU, 1>> inputs_plan;

    states_plan = std::vector<Eigen::Matrix<double, NX, 1>>(prediction_horizon - 1, Eigen::Matrix<double, NX, 1>::Zero());
    inputs_plan = std::vector<Eigen::Matrix<double, NU, 1>>(prediction_horizon - 1, Eigen::Matrix<double, NU, 1>::Zero());
    Eigen::DiagonalMatrix<double, NX> Q;
    Eigen::DiagonalMatrix<double, NU> R;
    Eigen::DiagonalMatrix<double, NX> Qn; // Initialize controller

//    Q.diagonal() << 0, 0, 10., 10., 10., 10., 0, 0, 0, 5., 5., 5.;
//    R.diagonal() << 0.1, 0.1, 0.1, 0.1;

    auto mpc_planner = mpc::MPC(dt, prediction_horizon, model, Q, R, Qn);

    // Temporary debug stuff
    //std::cout << mpc_planner.state_matrix_sequence[0] << std::endl;
    Eigen::Matrix<double, NX, 1> point;
    point << 1,2,0,0,0,0,0;
    std::vector<Eigen::Matrix<double, NX, 1>> traj;
    Eigen::Matrix<double, NX, 1> a;
    a << 0,10,0,0,0,0,0;
    Eigen::Matrix<double, NX, 1> b;
    b << 4,1,0,0,0,0,0;
//    Eigen::Matrix<double, NX, 1> c;
//    c << 1,0,0,0,0,0,0;
    traj.push_back(a);
    traj.push_back(b);
//    traj.push_back(c);
    Eigen::Matrix<double, NX, 1> curr;
    //mpc_planner.GetClosestTrajectoryPoint(traj, point, curr);
    //mpc_planner.NearestPoint(point, traj, curr);
    std::cout << curr << std::endl;

    // Initialization done
    std::cout << "It's lights out and away we go!" << std::endl;

    // This should be in a while loop after we finish debugging
    //mpc_planner.plan(states_plan, inputs_plan, x0);

    return 0;
}
