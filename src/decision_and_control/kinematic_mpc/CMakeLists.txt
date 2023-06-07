cmake_minimum_required(VERSION 3.8)
project(kinematic_mpc)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(command_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# Include directories
# include_directories("/opt/gurobi1001/linux64/include")
include_directories("/opt/include/osqp")
include_directories("/opt/include/OsqpEigen")
include_directories("/opt/include")
include_directories("/opt/include/eigen3")
include_directories("/home/nvidia/f1tenth/src/decision_and_control/kinematic_mpc/src/MPC_cpp")

# Link directories
# link_directories("/opt/gurobi1001/linux64/lib")
link_directories("/opt/lib")

# Libraries to link
# set(GUROBI_LIBS gurobi_g++5.2 gurobi100)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fopenmp -lstdc++")  # -Ofast -fopenmp -lstdc++ // -msse -msse2 -msse3 -mavx -mfma
#-Ofast -openmp -lstdc++ -O3

add_executable(mpc_node src/mpc_node.cpp src/MPC_cpp/mpc.cpp src/MPC_cpp/mpc.h src/MPC_cpp/kinematic_model.cpp src/MPC_cpp/kinematic_model.h src/MPC_cpp/constants.h)
ament_target_dependencies(mpc_node rclcpp std_msgs sensor_msgs nav_msgs command_msgs tf2 tf2_ros visualization_msgs ackermann_msgs geometry_msgs)

target_include_directories(mpc_node PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_features(mpc_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

# target_link_libraries(mpc_node ${GUROBI_LIBS})
target_link_libraries(mpc_node osqp)
target_link_libraries(mpc_node OsqpEigen)

install(
        DIRECTORY /opt/lib/
        DESTINATION lib
)

install(TARGETS mpc_node
        DESTINATION lib/${PROJECT_NAME})

if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # uncomment the line when a copyright and license is not present in all source files
    #set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # uncomment the line when this package is not in a git repo
    #set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

ament_package()
