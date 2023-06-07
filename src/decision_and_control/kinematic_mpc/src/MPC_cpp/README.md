# MPC_cpp

- Problem formulation: https://robotology.github.io/osqp-eigen/md_pages_mpc.html
- OSQP solver: https://osqp.org/docs/index.html
- GUROBI solver: https://www.gurobi.com/
- EIGEN3: https://eigen.tuxfamily.org/index.php?title=Main_Page
- OSQP-EIGEN: https://github.com/robotology/osqp-eigen
- GUROBI-EIGEN: https://github.com/jrl-umi3218/eigen-gurobi
- basically what we need to do except in
  python: https://github.com/f1tenth/f1tenth_planning/blob/main/f1tenth_planning/control/kinematic_mpc/kinematic_mpc.py
- ROS: https://docs.ros.org/en/humble/index.html
- f1tenth: https://f1tenth.org/
- LMPC: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8896988
- MPCC: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5717042
- intro to MPC: https://en.wikipedia.org/wiki/Model_predictive_control

## Steps

- Seznamte se s ROSem 2 a projektem F1/10.
- Proveďte rešerši kontrolerů na základě MPC (MPC, MPCC, LMPC, apod.).
- Proveďte rešerši knihoven pro řešení úloh kvadratického programování (QP).
- Vyberte vhodnou knihovnu pro QP a s jejím využitím implementujte MPC.
- Otestujte MPC řízení na formuli.
    * Pro testování bude dostupná již postavená formule (pozn. jiná než pro druhý projekt).
    * Uvažujte standardní dráhu pro závod F1/10, Time-trial: šířka 1.5-2m, očekávaná délka 30m, bez statických či dynamických překážek.
    * Uvažujte rychlosti do 4m/s.
    * Optimalizujte na nejdelší postup tratí (tzv. "progress trati") popsaný středovou čárou.

## Notes
#### General
- matrix A - state matrix (R^{NX} x R^{NX})
- matrix B - input matrix (R^{NX} x R^{UX})
- matrix C - affine shift (R^{NX} x 1)
- matrix Q - state cost (R^{NX} x R^{NX})
- matrix R - input cost (R^{NU} x R^{NU})
#### Kinematic model:
- Number of states of kinematic model: NX = 7;  x = [x, y, yaw angle, vx, vy, yaw rate, steering angle]^T
- Number of inputs of kinematic model: NU = 2;  u = [drive force, steering speed]^T
