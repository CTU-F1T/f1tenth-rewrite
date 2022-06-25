cmake_minimum_required(VERSION 3.5)
project(auto)

# find dependencies
find_package(ament_cmake REQUIRED)

# install config and launch files
install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
