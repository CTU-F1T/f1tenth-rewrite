/*
 *
 * Main, ROS-enabled layer
 *
 * Authors: Jaroslav Klapálek
 * Copyright (C) 2020 Czech Technical University in Prague
 *
 * This file is a part of follow_the_gap_v0.
 *
 * follow_the_gap_v0 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * follow_the_gap_v0 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with follow_the_gap_v0. If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include "follow_the_gap.h"

#include "rclcpp/rclcpp.hpp"

#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/int32.hpp"

#include "sensor_msgs/msg/laser_scan.hpp"

#include "visualization_msgs/msg/marker.hpp"

#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

#include "obstacle_msgs/msg/obstacles_stamped.hpp"
#include "obstacle_msgs/msg/circle_obstacle.hpp"

// ros::NodeHandle *node_handle_pointer;

int unsigned kPublishMessageBufferSize = 10;
int unsigned kSubscribeMessageBufferSize = 1;

// ros::Publisher publisher_final_heading_angle;
// ros::Publisher publisher_gap_found;
//
// // For visualizing and debugging
// ros::Publisher publisher_visualize_largest_gap;
// ros::Publisher publisher_visualize_final_heading_angle;
// ros::Publisher publisher_visualize_obstacles;


//////////////////////
// Utility functions
//////////////////////

std::vector<FollowTheGap::Obstacle> CreateObstacles(sensor_msgs::msg::LaserScan::ConstSharedPtr const &lidar_data) {

	size_t const data_size = lidar_data->ranges.size();

	float const &angle_increment = lidar_data->angle_increment;
	float const &range_max = lidar_data->range_max;
	float const &range_min = lidar_data->range_min;
	float angle = lidar_data->angle_min;

	std::vector<FollowTheGap::Obstacle> obstacles;

	for (size_t i = 0; i < data_size; ++i, angle += angle_increment) {

		float const &range = lidar_data->ranges[i];
		// Throw away all range values that are NAN
		/* Sometimes LIDARs return NAN / inf values */
		if (std::isnan(range)) {
			continue;
		}
		// Filter data which is outside the lidar fov
		if ((angle < lidar_data->angle_min) || (angle > lidar_data->angle_max)) {
			continue;
		}
		// Filter data which is outside the max range or inside min range of lidar
		if ((range < range_min) || (range > range_max)) {
			continue;
		}
		// Throw away all outliers (point without neighbours)
		// klapajar: TODO: It is possible to use "is someone around me?" but it leads to losing information about space
		//                 It seems to be better using vectors. but TODO.
		float obstacle_radius = FollowTheGap::kCarRadius;

		obstacles.emplace_back(range, angle, obstacle_radius);

	}

	// We reverse the array so that the first obstacles are on the left(largest angle, in accordance with the paper)
	std::reverse(obstacles.begin(), obstacles.end());

	return obstacles;

}

std::vector<FollowTheGap::Obstacle> CreateObstacles(
	obstacle_msgs::msg::ObstaclesStamped::ConstSharedPtr const &obstacles_data
) {

	size_t const data_size = obstacles_data->obstacles.circles.size();

	std::vector<FollowTheGap::Obstacle> obstacles;

	for (size_t i = 0; i < data_size; ++i) {
		obstacles.emplace_back(
			obstacles_data->obstacles.circles[i].center.x,
			obstacles_data->obstacles.circles[i].center.y,
			obstacles_data->obstacles.circles[i].radius,
			FollowTheGap::kCarRadius
		);
	}

	// We reverse the array so that the first obstacles are on the left (largest angle, in accordance with the paper)
	std::sort(obstacles.begin(), obstacles.end());
	// std::reverse(obstacles.begin(), obstacles.end());

	return obstacles;

}

std::vector<FollowTheGap::Obstacle> CreateObstaclesWithAngleFilter(
	sensor_msgs::msg::LaserScan::ConstSharedPtr const &lidar_data
) {
	size_t const data_size = lidar_data->scan_time / lidar_data->time_increment;
	float const &angle_increment = lidar_data->angle_increment;
	float const &range_max = lidar_data->range_max;
	float const &range_min = lidar_data->range_min;
	// We start at the angle corresponding to the right index
	float angle = lidar_data->angle_min + FollowTheGap::AngleFilter::right_index * angle_increment;
	std::vector<FollowTheGap::Obstacle> obstacles;
	std::cerr << "right_index: " << FollowTheGap::AngleFilter::right_index << std::endl;
	std::cerr << "left_index: " << FollowTheGap::AngleFilter::left_index << std::endl;
	for (size_t i = FollowTheGap::AngleFilter::right_index;
		 i <= FollowTheGap::AngleFilter::left_index; ++i, angle += angle_increment) {
		float const &range = lidar_data->ranges[i];
		// Throw away all range values that are NAN
		/* Sometimes LIDARs return NAN / inf values */
		if (std::isnan(range)) {
			continue;
		}
		// Filter data which is outside the lidar fov
		if ((angle < lidar_data->angle_min) || (angle > lidar_data->angle_max)) {
			continue;
		}
		// Filter data which is outside the max range or inside min range of lidar
		if ((range < range_min) || (range > range_max)) {
			continue;
		}
		// Throw away all outliers (point without neighbours)
		// klapajar: TODO: It is possible to use "is someone around me?" but it leads to losing information about space
		//                 It seems to be better using vectors. but TODO.
		float obstacle_radius = FollowTheGap::kCarRadius;
		obstacles.emplace_back(range, angle, obstacle_radius);
	}

	// We reverse the array so that the first obstacles are on the left(largest angle, in accordance with the paper)
	std::reverse(obstacles.begin(), obstacles.end());

	return obstacles;

}


//////////////////////
// Publishers
//////////////////////

void PublishFinalHeadingAngle(float final_heading_angle) {
	std_msgs::msg::Float32 final_heading_angle_message;
	final_heading_angle_message.data = final_heading_angle;
	// TODO: port
	// publisher_final_heading_angle.publish(final_heading_angle_message);
}

void PublishVisualizeFinalHeadingAngle(float final_heading_angle, const std::string &frame_id) {

	geometry_msgs::msg::PoseStamped angle_message;
	angle_message.header.frame_id = frame_id;

	tf2::Quaternion angle_quaternion;
	angle_quaternion.setEuler(0, 0, final_heading_angle);
	angle_quaternion.normalize();
	tf2::convert(angle_quaternion, angle_message.pose.orientation);

	// TODO: port
	// publisher_visualize_final_heading_angle.publish(angle_message);

}

void PublishVisualizeLargestGap(
	FollowTheGap::Obstacle const &gap_left,
	FollowTheGap::Obstacle const &gap_right,
	const std::string &frame_id
) {

	geometry_msgs::msg::PointStamped p0, p1, robot_point;
	robot_point.header.frame_id = frame_id;
	robot_point.point.x = 0;
	robot_point.point.y = 0;
	p0.point.x = gap_left.distance * std::cos(gap_left.angle_right);
	p0.point.y = gap_left.distance * std::sin(gap_left.angle_right);
	p0.header.frame_id = frame_id;
	p1.point.x = gap_right.distance * std::cos(gap_right.angle_left);
	p1.point.y = gap_right.distance * std::sin(gap_right.angle_left);
	p1.header.frame_id = frame_id;

	// TODO: port
	// publisher_visualize_largest_gap.publish(robot_point);
	// publisher_visualize_largest_gap.publish(p0);
	// publisher_visualize_largest_gap.publish(p1);

}

void PublishVisualizeObstacles(
	std::vector<FollowTheGap::Obstacle> const &obstacles,
	const std::string &frame_id
) {
	visualization_msgs::msg::Marker obstacle_points;
	obstacle_points.header.frame_id = frame_id;
	//obstacle_points.header.stamp = ros::Time::now(); // Requires interpolation to the future (sometimes)
	obstacle_points.type = visualization_msgs::msg::Marker::POINTS;

	// obstacle_points.scale.x = kCarRadius;
	// obstacle_points.scale.y = kCarRadius;
	obstacle_points.scale.x = 0.05;
	obstacle_points.scale.y = 0.05;
	obstacle_points.color.r = 0;
	obstacle_points.color.g = 1;
	obstacle_points.color.b = 0;
	obstacle_points.color.a = 1;

	for (auto const &o : obstacles) {
		geometry_msgs::msg::Point p;
		p.x = o.x;
		p.y = o.y;
		obstacle_points.points.push_back(p);
	}

	// TODO: port
	// publisher_visualize_obstacles.publish(obstacle_points);
}

void PublishGapFound(bool gap_found) {
	std_msgs::msg::Bool gap_found_message;
	gap_found_message.data = gap_found;
	// TODO: port
	// publisher_gap_found.publish(gap_found_message);
}


//////////////////////
// Callbacks
//////////////////////

void Callback(sensor_msgs::msg::LaserScan::ConstSharedPtr const &lidar_data) {

	LidarData ld(
		lidar_data->range_min,
		lidar_data->range_max,
		lidar_data->angle_min,
		lidar_data->angle_max,
		lidar_data->angle_increment
	);

	std::vector<FollowTheGap::Obstacle> obstacles = CreateObstacles(lidar_data);

	bool ok;
	float angle;
	std::vector<FollowTheGap::Obstacle> obstacles_out;
	std::vector<FollowTheGap::Obstacle> gap_borders_out;

	std::tie(ok, angle) = FollowTheGap::Callback(obstacles, &ld, obstacles_out, gap_borders_out);

	//std::cout << ok << ", " << angle << std::endl;

	if (ok) {
		PublishFinalHeadingAngle(angle);
		PublishVisualizeFinalHeadingAngle(angle, lidar_data->header.frame_id);
		PublishVisualizeLargestGap(gap_borders_out.at(0), gap_borders_out.at(1), lidar_data->header.frame_id);
	} else {
		PublishGapFound(ok);
	}

	PublishVisualizeObstacles(obstacles_out, lidar_data->header.frame_id);
}

// Note: This method does not yield the same results as the original LiDAR method.It is
// most likely caused by different rounding (Py / C++).
void ObstaclesCallback(
	obstacle_msgs::msg::ObstaclesStamped::ConstSharedPtr const &obstacles_data
) {

	std::vector<FollowTheGap::Obstacle> obstacles = CreateObstacles(obstacles_data);

	bool ok;
	float angle;
	std::vector<FollowTheGap::Obstacle> obstacles_out;
	std::vector<FollowTheGap::Obstacle> gap_borders_out;

	// TODO: We send NULL here as lidar_data are not currently used in the methods inside.
	// This is ok, as we don't have it. But it should be obtained somewhere.
	std::tie(ok, angle) = FollowTheGap::Callback(obstacles, nullptr, obstacles_out, gap_borders_out);

	//std::cout << ok << ", " << angle << std::endl;

	if (ok) {
		PublishFinalHeadingAngle(angle);
		PublishVisualizeFinalHeadingAngle(angle, obstacles_data->header.frame_id);
		PublishVisualizeLargestGap(gap_borders_out.at(0), gap_borders_out.at(1), obstacles_data->header.frame_id);
	} else {
		PublishGapFound(ok);
	}

	PublishVisualizeObstacles(obstacles_out, obstacles_data->header.frame_id);

}

void AngleFilterLeftCallback(std_msgs::msg::Int32::ConstSharedPtr const &message) {
	FollowTheGap::AngleFilter::left_index = message->data;
}

void AngleFilterRightCallback(std_msgs::msg::Int32::ConstSharedPtr const &message) {
	FollowTheGap::AngleFilter::right_index = message->data;
}

void GoalAngleCallback(std_msgs::msg::Float64::ConstSharedPtr const &message) {
	FollowTheGap::g_goal_angle = message->data;
}


//////////////////////
// Main
//////////////////////

int main(int argc, char **argv) {

	// Remind that we have temporarily hidden this function.
	std::cout << "Warning: FilterLoneObstacleGroups is disabled." << std::endl;

	rclcpp::init(argc, argv);
	// ros::init(argc, argv, "follow_the_gap");

	// node_handle_pointer = new ros::NodeHandle;
	std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("follow_the_gap");


/*
    // Subscribers

    // ros::Subscriber lidar_data_subscriber = node_handle_pointer
    //     ->subscribe("/scan", kSubscribeMessageBufferSize,
    //             Callback
    // );

    // ros::Subscriber obstacle_data_subscriber = node_handle_pointer
    //     ->subscribe("/obstacles", kSubscribeMessageBufferSize,
    //             ObstaclesCallback
    // );

    // ros::Subscriber angle_filter_subscriber_right = node_handle_pointer
    //     ->subscribe("/right_constraint_index", kSubscribeMessageBufferSize,
    //             AngleFilterRightCallback
    // );

    // ros::Subscriber angle_filter_subscriber_left = node_handle_pointer
    //     ->subscribe("/left_constraint_index", kSubscribeMessageBufferSize,
    //             AngleFilterLeftCallback
    // );

    // ros::Subscriber goal_angle_subscriber = node_handle_pointer
    //     ->subscribe("/lsr/angle", kSubscribeMessageBufferSize,
    //             GoalAngleCallback
    // );


    // // Publishers
    // publisher_final_heading_angle = node_handle_pointer
    //     ->advertise<std_msgs::msg::Float32>("/final_heading_angle", kPublishMessageBufferSize);

    // publisher_gap_found = node_handle_pointer
    //     ->advertise<std_msgs::msg::Bool>("/gap_found", kPublishMessageBufferSize);

    // publisher_visualize_largest_gap = node_handle_pointer
    //     ->advertise<geometry_msgs::msg::PointStamped>("/visualize_largest_gap", kPublishMessageBufferSize);

    // publisher_visualize_final_heading_angle = node_handle_pointer
    //     ->advertise<geometry_msgs::msg::PoseStamped>("/visualize_final_heading_angle", kPublishMessageBufferSize);

    // publisher_visualize_obstacles = node_handle_pointer
    //     ->advertise<visualization_msgs::Marker>("/visualize_obstacles", kPublishMessageBufferSize);

*/
	rclcpp::spin(node);
	rclcpp::shutdown();

	return 0;

}
