#include <iostream>
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include "pcl_ros/point_cloud.h"
#include <pcl_ros/transforms.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>
#include <tf/transform_listener.h>
#include <cmath>

class PointCloudCollisionWarningNode
{
    public:
        PointCloudCollisionWarningNode(void);
    
    protected:
        ros::NodeHandle nh_;
        tf::TransformListener tf_listener_;
        std::string point_cloud_topic_name;
        ros::Subscriber point_cloud_sub_;
        double warn_radius_;
        void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &point_cloud_msg);
};

PointCloudCollisionWarningNode::PointCloudCollisionWarningNode(void)
{
    nh_.param<std::string>("point_cloud_topic", point_cloud_topic_name, "/stereo/points2");
    point_cloud_sub_  = nh_.subscribe(point_cloud_topic_name, 1, &PointCloudCollisionWarningNode::pointCloudCallback, this);
    nh_.param<double>("warn_radius", warn_radius_, 1.5);
}


void PointCloudCollisionWarningNode::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &point_cloud_msg)
{      
    // Create a PCL point cloud object, in the frame of reference of the robot,
    // so that any point within warn_radius_ of the origin is a potential danger point
    pcl::PointCloud<pcl::PointXYZRGB> cloud, cloud_baselink_frame;
    pcl::fromROSMsg(*point_cloud_msg, cloud);
    pcl_ros::transformPointCloud ("base_link", ros::Time::now(), cloud, point_cloud_msg->header.frame_id, cloud_baselink_frame, tf_listener_);
    double min_dist = -1;
    geometry_msgs::Point closest_point;
    bool need_to_warn = false;

    // Check for points within warn_radius_ of the origin
    for (size_t i = 0; i < cloud_baselink_frame.points.size (); ++i)
    {
        
        double dist = sqrt(cloud_baselink_frame.points[i].x*cloud_baselink_frame.points[i].x +
                            cloud_baselink_frame.points[i].y*cloud_baselink_frame.points[i].y +
                            cloud_baselink_frame.points[i].z*cloud_baselink_frame.points[i].z );
                            
        if ((dist == dist) && (min_dist < 0 || dist < min_dist)) // dist != dist means dist is nan
        {
            min_dist = dist;
            closest_point.x = cloud_baselink_frame.points[i].x;
            closest_point.y = cloud_baselink_frame.points[i].y;
            closest_point.z = cloud_baselink_frame.points[i].z;
        }
        
        if ( dist < warn_radius_ )
        {
            /*ROS_WARN("There is an obstacle at %f, %f, %f (%s frame)", 
                cloud_baselink_frame.points[i].x, cloud_baselink_frame.points[i].y, cloud_baselink_frame.points[i].z,
                "base_link" );
            break;*/
            need_to_warn = true;
        }
    }
    if (need_to_warn)
        ROS_INFO("Point (%f, %f, %f [%s frame]) @ distance:%f < warn_radius:%f", closest_point.x, closest_point.y, closest_point.z, "base_link", min_dist, warn_radius_);
}
        
int main(int argc, char* argv[])
{
    ros::init(argc, argv, "point_cloud_safety");
    
    PointCloudCollisionWarningNode pccwn;
    ros::spin();
}

