#include <spiri_api/staterobot.h>
#include <iostream>
#include <boost/python.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(libspiri_api_python)
{
    class_<Staterobot,boost::noncopyable>("Staterobot",init<>())
             .def("get_imu",&Staterobot::get_imu_python)
             .def("get_state",&Staterobot::get_state_python)
             .def("get_height_pressure",&Staterobot::get_height_pressure)
             .def("get_gps_data",&Staterobot::get_gps_data_python)
             .def("get_gps_vel",&Staterobot::get_gps_vel_python)
             .def("get_height_altimeter",&Staterobot::get_height_altimeter)
             .def("get_left_image",&Staterobot::get_left_image_python)
             .def("get_right_image",&Staterobot::get_right_image_python)
             .def("get_bottom_image",&Staterobot::get_bottom_image_python)
	     .def("send_goal",&Staterobot::send_goal_python)
	     .def("send_goal_relative",&Staterobot::send_goal_python_relative)
            .def("send_vel",&Staterobot::send_vel_python)
            .def("wait_goal",&Staterobot::wait_goal)

	    .def("stop_traj",&Staterobot::stop_traj)

            .def("land", &Staterobot::land)
            .def("takeoff", &Staterobot::takeoff)

    ;
}
