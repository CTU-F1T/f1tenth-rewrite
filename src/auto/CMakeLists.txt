cmake_minimum_required(VERSION 3.5)
project(auto)

# find dependencies
find_package(ament_cmake REQUIRED)

set(AUTO_CLI_HOME "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/auto-cli")
set(AUTO_CLI_VERSION "0.0.1")

# TODO enable autocompletion for Auto CLI
# ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/auto-cli-autocompletion.sh.in")

# install config and launch files
install(
  DIRECTORY launch config scripts/auto-cli
  DESTINATION share/${PROJECT_NAME}
)
# install Auto CLI
# install(
#   PROGRAMS scripts/auto.sh
#   TYPE BIN
#   RENAME auto
# )
configure_file(scripts/auto-cli-wrapper.sh.in "${CMAKE_INSTALL_PREFIX}/bin/auto" @ONLY)

ament_package()
