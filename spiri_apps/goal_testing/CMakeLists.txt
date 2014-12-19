
cmake_minimum_required(VERSION 2.8.3)
project(goal_testing)
find_package(Boost REQUIRED COMPONENTS system python)
find_package(catkin REQUIRED COMPONENTS spiri_api)
find_package(PythonLibs 2.7 REQUIRED)
include_directories(${catkin_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
add_executable(test_goal test_goal.cpp)
target_link_libraries(test_goal ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
