cmake_minimum_required(VERSION 3.5)
project(storage)

# find dependencies
find_package(ament_cmake REQUIRED)

# install files
install(
  DIRECTORY apriltags
  DESTINATION share/${PROJECT_NAME}
)
install(
	DIRECTORY rviz
	DESTINATION share/${PROJECT_NAME}
)
install(
	DIRECTORY stage
	DESTINATION share/${PROJECT_NAME}
)

ament_package()
