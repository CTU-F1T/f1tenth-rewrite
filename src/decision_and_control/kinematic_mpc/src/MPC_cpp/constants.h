//
// Created by tomas on 3/21/23.
//

// !!!!! I hate the entire existance of this file and we should not need it. !!!!

#ifndef MPC_CONSTANTS_H
#define MPC_CONSTANTS_H

// This part is dependent on the model that is used
// However we currently need this so sizes of static eigen matrices are corrently compiled
// Maybe rewite them to be dynamic? How would that slow the program down? 
// Or can we define theese constants within each model ?
// I do not know how to properly compile static matrices without theese -> maybe const var can help ? 

// model constants 
#define NX 5 // number of states X = [x, y, yaw angle, vx]
#define NU 2 // number of inputs U = [a, steer]
#define NX_TRAJ 8 // number of parameters of a state in the trajectory // Difference between this and NX ??

// control constants // HUH ? commented out so we probably do not need this ? 
// #define NK 15

// trajectory constants // This is bad TODO get rid of this as soon as possible
#define TUM_TRAJ_PARAMS 7 //[s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2]

#endif //MPC_CONSTANTS_H
