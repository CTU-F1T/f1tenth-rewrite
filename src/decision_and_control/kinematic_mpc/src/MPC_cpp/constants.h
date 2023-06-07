//
// Created by tomas on 3/21/23.
//

#ifndef MPC_CONSTANTS_H
#define MPC_CONSTANTS_H

// model constants
#define NX 4 // number of states X = [x, y, yaw angle, vx]
#define NU 2 // number of inputs U = [a, steer]

// control constants
// #define NK 15

// trajectory constants
#define TUM_TRAJ_PARAMS 7 //[s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2]

#endif //MPC_CONSTANTS_H
