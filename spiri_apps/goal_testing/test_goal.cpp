#include <spiri_api/staterobot.h>
#include <sstream>
#include <iostream>
int main()
{
    Staterobot robot;
    robot.send_goal(0,0,2,false);
    //sleep(1.0);
    //robot.stop_traj();
    return 0;
}