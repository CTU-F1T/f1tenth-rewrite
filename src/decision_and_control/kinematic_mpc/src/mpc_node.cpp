#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "mpc.h"
#include "kinematic_model.h"
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cmath>

#include <command_msgs/msg/command.hpp>
#include <command_msgs/msg/command_array_stamped.hpp>
#include <command_msgs/msg/command_parameter.hpp>
#include "rclcpp/rclcpp.hpp"
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

using namespace std::chrono_literals;
using std::placeholders::_1;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class MPC_node : public rclcpp::Node {
    Eigen::Matrix<double, NX, 1> x0;
    int prediction_horizon = 80;
    double dt = 0.025;
    mpc::KinematicModel model = mpc::KinematicModel(this->prediction_horizon);
    Eigen::DiagonalMatrix<double, NX> Q;
    Eigen::DiagonalMatrix<double, NU> R;
    Eigen::DiagonalMatrix<double, NX> Qn; // Initialize controller

    mpc::MPC mpc_planner = mpc::MPC(this->dt, this->prediction_horizon, this->model, Q, R, Qn);
    bool map_loaded = 0;

    // Necessary variables
    std::vector <Eigen::Matrix<double, NX, 1>> states_plan;
    std::vector <Eigen::Matrix<double, NU, 1>> inputs_plan;
    std::vector <Eigen::Matrix<double, NX, 1>> predicted_traj;
    std::vector <Eigen::Matrix<double, NX, 1>> reference;

    std::vector <Eigen::Matrix<double, 6, 1>> log_data;
    int log_every = 5;
    int log_lase = 0;
    int save_every = 300;
    int save_last = 0;

    float drive_speed = 0.0;
    float drive_steering_angle = 0.0;

    std::chrono::time_point <std::chrono::high_resolution_clock> pos_update_prew = std::chrono::high_resolution_clock::now();

public:


    MPC_node() : Node("mcp") {
        // Load and parse reference
        std::string fname = "/home/nvidia/f1tenth/src/perception/recognition/particle_filter/maps/garaz.csv";
        int skip_lines = 1; //1 - Jara

        Eigen::Matrix<double, 7, 1> row_matrix;
        std::string line, word;
        this->mpc_planner.reference_trajectory.clear();
        int skipped_lines = 0;

        std::fstream file(fname, std::ios::in);
        if (file.is_open()) {
            while (getline(file, line)) {
                if (skipped_lines < skip_lines) {
                    skipped_lines++;
                    continue;
                }

                std::stringstream str(line);
                int col = 0;
                int col_ = 0;
                while (getline(str, word, ',')) { // , - Jara
                    if (col == 0 || col == 7 || col == 9 || col == 10) {  // Jara
                        col++;
                        continue;
                    }
                    row_matrix[col_] = std::stod(word);
                    if (col_ == 5) row_matrix[col_] *= 0.8;
//                    if (col_ == 4) row_matrix[col_] += 3.14;
                    col++;
                    col_++;
                }
                row_matrix[4] = row_matrix[4] + 20;
                this->mpc_planner.reference_trajectory.push_back(row_matrix);
            }
            RCLCPP_INFO(this->get_logger(), "Map loaded from file successfully!");
            this->map_loaded = 1;
        } else {
            RCLCPP_INFO(this->get_logger(), "Could not open the file!");
        }

        // Set printing
        std::cout << std::fixed;
        std::cout << std::setprecision(2);


        // Initialize vehicle state
        this->x0 << 0, 0, 0, 0;

        this->mpc_planner.Q.diagonal() << 20.0, 20.0, 12., 5.5;
        this->mpc_planner.Qn.diagonal() << 20.0, 20.0, 40., 5.5;
        this->mpc_planner.R.diagonal() << 0.1, 45.0;

        states_plan = std::vector < Eigen::Matrix < double, NX, 1 >> (prediction_horizon, Eigen::Matrix<double, NX, 1>::Zero());
        inputs_plan = std::vector < Eigen::Matrix < double, NU, 1 >> (prediction_horizon - 1, Eigen::Matrix<double, NU, 1>::Zero());
        predicted_traj = std::vector < Eigen::Matrix < double, NX, 1 >> (prediction_horizon, Eigen::Matrix<double, NX, 1>::Zero());
        reference = std::vector < Eigen::Matrix < double, NX, 1 >> (prediction_horizon, Eigen::Matrix<double, NX, 1>::Zero());

        rclcpp::QoS qos_settings = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
        // Temporary debug stuff
        std::cout << mpc_planner.state_matrix_sequence[0] << std::endl;

        // Initialize publishers for visualization
        pub_vis_whole_trajectory = this->create_publisher<visualization_msgs::msg::MarkerArray>("/vis/raceline", qos_settings);
        pub_vis_predicted_trajectory = this->create_publisher<visualization_msgs::msg::MarkerArray>("/vis/predict", 1);
        pub_vis_mpc_out_trajectory = this->create_publisher<visualization_msgs::msg::MarkerArray>("/vis/mpc_out", 1);
        pub_vis_patch_trajectory = this->create_publisher<visualization_msgs::msg::MarkerArray>("/vis/ref", 1);

        // Initialize publishers for driving
        pub_command = this->create_publisher<command_msgs::msg::CommandArrayStamped>("/command", 1);

        pub_vis_path = this->create_publisher<nav_msgs::msg::Path>("/path1", 1);
        pub_vis_path2 = this->create_publisher<nav_msgs::msg::Path>("/path2", 1);
        pub_vis_path3 = this->create_publisher<nav_msgs::msg::Path>("/path3", 1);

        // Initialize subscribers
        sub_odom = this->create_subscription<nav_msgs::msg::Odometry>("/pf/pose/odom", qos_settings,
                                                                      std::bind(&MPC_node::pose_callback, this, _1));

        // Initialization done
        RCLCPP_INFO(this->get_logger(), "It's lights out and away we go!");


        visualization_msgs::msg::MarkerArray array_m;

        std::cout << "Trajectory size:   " << this->mpc_planner.reference_trajectory.size() << std::endl;

        nav_msgs::msg::Path path_;
        path_.header.frame_id = "/map";
        path_.header.stamp = this->get_clock()->now();

        for (int i = 0; i < this->mpc_planner.reference_trajectory.size(); i++) {
            visualization_msgs::msg::Marker mark;
            mark.pose.position.x = this->mpc_planner.reference_trajectory[i][1];
            mark.pose.position.y = this->mpc_planner.reference_trajectory[i][2];
            mark.pose.position.z = 0;
            mark.scale.x = mark.scale.y = mark.scale.z = 0.04;
            mark.header.stamp = this->get_clock()->now();
            mark.header.frame_id = "/map";
            mark.action = 0;
            mark.ns = "/vis";
            mark.color.r = 200;
            mark.color.g = 0;
            mark.color.b = 0;
            mark.color.a = 200;
            mark.id = i;
            mark.type = mark.SPHERE;
            array_m.markers.push_back(mark);

            geometry_msgs::msg::PoseStamped pp;
            pp.pose.position.x = this->mpc_planner.reference_trajectory[i][1];
            pp.pose.position.y = this->mpc_planner.reference_trajectory[i][2];
            pp.header.frame_id = "/map";
            pp.header.stamp = this->get_clock()->now();
            path_.poses.push_back(pp);
        }
        pub_vis_whole_trajectory->publish(array_m);
        pub_vis_path->publish(path_);
    }

private:
    void pose_callback(nav_msgs::msg::Odometry::SharedPtr msg) {
        if (!this->map_loaded) return;


        nav_msgs::msg::Path path_;
        path_.header.frame_id = "/map";
        path_.header.stamp = this->get_clock()->now();
        for (int i = 0; i < this->mpc_planner.reference_trajectory.size(); i++) {
            geometry_msgs::msg::PoseStamped pp;
            pp.pose.position.x = this->mpc_planner.reference_trajectory[i][1];
            pp.pose.position.y = this->mpc_planner.reference_trajectory[i][2];
            pp.header.frame_id = "/map";
            pp.header.stamp = this->get_clock()->now();
            path_.poses.push_back(pp);
        }
        pub_vis_path->publish(path_);



//        if (this->drive_speed == nan){
//            this->drive_speed;
//        }





        auto pos_update_now = std::chrono::high_resolution_clock::now();
//        auto pos_time_now = std::chrono::duration_cast<std::chrono::microseconds>(pos_update_now);
        auto pos_update_time = std::chrono::duration_cast<std::chrono::microseconds>(pos_update_now - pos_update_prew);
        pos_update_prew = pos_update_now;

//        std::cout << "Pos_update_time  " << pos_update_time.count() << std::endl;

        // Get current vehicle's state[x, y, yaw angle, vx, vy, yaw rate, steering angle]
        tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        // msg->twist.twist.linear.x -> not from this message
        this->x0 << msg->pose.pose.position.x, msg->pose.pose.position.y, yaw, this->drive_speed;
        this->x0(0) = this->x0(0) - cos(yaw) * 0.25;  // 29
        this->x0(1) = this->x0(1) - sin(yaw) * 0.25;

//        std::cout << "X0:" << this->x0 << std::endl;
//        std::cout << "STEER:" << command_param_steer.value << std::endl;

//        this->drive_speed = msg->twist.twist.linear.x;

        // Plan new trajectory
        auto start = std::chrono::high_resolution_clock::now();
        this->mpc_planner.plan(this->states_plan, this->inputs_plan, this->predicted_traj, this->reference, this->x0);
        auto stop = std::chrono::high_resolution_clock::now();
        auto plan_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        // Visualize trajectories in Rviz
//        visualize_position_from_states(this->predicted_traj, this->pub_vis_predicted_trajectory, 0, 100, 0, 0.1);


        nav_msgs::msg::Path path2_;
        path2_.header.frame_id = "/map";
        path2_.header.stamp = this->get_clock()->now();
        for (int i = 0; i < this->states_plan.size(); i++) {
            geometry_msgs::msg::PoseStamped pp;
            pp.pose.position.x = this->states_plan[i][0];
            pp.pose.position.y = this->states_plan[i][1];
            pp.header.frame_id = "/map";
            pp.header.stamp = this->get_clock()->now();
            path2_.poses.push_back(pp);
        }
        pub_vis_path2->publish(path2_);

        nav_msgs::msg::Path path3_;
        path3_.header.frame_id = "/map";
        path3_.header.stamp = this->get_clock()->now();
        for (int i = 0; i < this->states_plan.size(); i++) {
            geometry_msgs::msg::PoseStamped pp;
            pp.pose.position.x = this->reference[i][0];
            pp.pose.position.y = this->reference[i][1];
            pp.header.frame_id = "/map";
            pp.header.stamp = this->get_clock()->now();
            path3_.poses.push_back(pp);
        }
        pub_vis_path3->publish(path3_);

//        visualize_position_from_states(this->reference, this->pub_vis_patch_trajectory, 0, 100, 100, 0.1);

//        visualize_position_from_states(this->states_plan, this->pub_vis_mpc_out_trajectory, 100, 10, 100, 0.1);
//        std::cout << "yaw  " << yaw << "   " << this->reference[0][2] << std::endl;

        this->drive_speed += this->inputs_plan[0][0] * 0.025; // this->dt;

        rclcpp::Time now = this->get_clock()->now();






        // Create steer command
        command_msgs::msg::CommandParameter command_param_steer;
        command_param_steer.parameter = "rad";
        command_param_steer.value = this->inputs_plan[0][1]; // steering angle
        command_msgs::msg::Command command_steer;
        command_steer.command = "steer";
        command_steer.parameters = {command_param_steer};
        // Create drive command
        command_msgs::msg::CommandParameter command_param_drive;
        command_param_drive.parameter = "metric";
        command_param_drive.value = this->drive_speed;
        command_msgs::msg::Command command_drive;
        command_drive.command = "speed";
        command_drive.parameters = {command_param_drive};
        // Create command msg
        command_msgs::msg::CommandArrayStamped command_msg;
        command_msg.header.stamp = now;
        command_msg.commands = {command_drive, command_steer};
        pub_command->publish(command_msg);

//        std::cout << "SPEED:" << command_param_drive.value << std::endl;
//        std::cout << "STEER:" << command_param_steer.value << std::endl;

        if (log_lase == log_every) {

            Eigen::Matrix<double, 6, 1> log_now;
            log_now << this->x0[0], this->x0[1], this->x0[3], this->inputs_plan[0][1],
                    this->reference[0][0] - this->x0[0],  // dx
                    this->reference[0][1] - this->x0[1];  // dy
            this->log_data.push_back(log_now);
            log_lase = 0;
            save_last++;
        }
        log_lase++;

        if (save_last == save_every) {
            // save data
            std::ofstream f;
            f.open("export.csv");
            for (int i = 0; i < this->log_data.size(); i++) {
                f << this->log_data[i][0] << ";" <<
                  this->log_data[i][1] << ";" <<
                  this->log_data[i][2] << ";" <<
                  this->log_data[i][3] << ";" <<
                  this->log_data[i][4] << ";" <<
                  this->log_data[i][5] << "\n";
            }
            save_last = 0;
            std::cout << "LOG SAVED" << std::endl;
        }



//        std_msgs::msg::Float64 drive_msg;
//        drive_msg = this->drive_speed;
//        pub_drive->publish(drive_msg);
//        drive_msg.drive.steering_angle = this->inputs_plan[0][1];


//         std::cout << "MPC_ITERATION_DONE   " << this->inputs_plan[0][0] << "   "  << this->inputs_plan[0][1] << "   " << this->drive_speed << "   "  << this->drive_steering_angle << std::endl;
//        std::cout << "Plan runtime: " << plan_time.count() / 1000.0 << " [ms], " <<
//                  "Predict motion time:  " << this->mpc_planner.predict_motion_time.count() / 1000.0 << " [ms], " <<
//                  "Linearization time:  " << this->mpc_planner.linearization_time.count() / 1000.0 << " [ms], " <<
//                  "Ref calc time:  " << this->mpc_planner.ref_time.count() / 1000.0 << " [ms], " <<
//                  "Cast time:  " << this->mpc_planner.cast_time.count() / 1000.0 << " [ms], " <<
//                  "Solver time:  " << this->mpc_planner.solver_time.count() / 1000.0 << " [ms], " <<
//                  "Time between loops:  " << pos_update_time.count() / 1000.0 << " [ms]" << std::endl;
        // rclcpp::shutdown();
    }

    void visualize_position_from_states(std::vector <Eigen::Matrix<double, NX, 1>> &states,
                                        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_,
                                        float color_r, float color_g, float color_b,
                                        double scale) {
        visualization_msgs::msg::MarkerArray array_m;
        for (int i = 0; i < states.size(); i++) {
            rclcpp::Time now = this->get_clock()->now();
            visualization_msgs::msg::Marker mark;
            mark.pose.position.x = states[i][0];
            mark.pose.position.y = states[i][1];
            mark.pose.position.z = 0;
            mark.scale.x = mark.scale.y = mark.scale.z = scale;
            mark.header.stamp = now;
            mark.header.frame_id = "/map";
            mark.color.r = color_r;
            mark.color.g = color_g;
            mark.color.b = color_b;
            mark.color.a = 200;
            mark.id = i;
            mark.type = mark.SPHERE;
            array_m.markers.push_back(mark);
        }
        publisher_->publish(array_m);
    }

    std::shared_ptr <rclcpp::Subscription<nav_msgs::msg::Odometry>> sub_odom;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_vis_whole_trajectory;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_vis_predicted_trajectory;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_vis_mpc_out_trajectory;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_vis_patch_trajectory;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_vis_path;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_vis_path2;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_vis_path3;

    rclcpp::Publisher<command_msgs::msg::CommandArrayStamped>::SharedPtr pub_command;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPC_node>());
    rclcpp::shutdown();
    return 0;
}
