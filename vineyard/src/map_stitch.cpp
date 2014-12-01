#include <iostream>
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace cv;

void printRect(const Rect& rect, std::string label)
{
    int max_x = rect.x + rect.width;
    int max_y = rect.y + rect.height;
    ROS_INFO("%s: %d,%d X %d,%d (%d X %d)", label.c_str(), rect.x, rect.y, max_x, max_y, rect.width, rect.height);
}

class MapStitchNode
{
    public:
        MapStitchNode(void);
    
    protected:
        // ROS bookkeeping
        ros::NodeHandle nh_;
        tf::TransformListener tf_listener_;
        image_transport::ImageTransport it_;
        image_transport::Publisher map_pub_;
        image_transport::Subscriber image_sub_;
        
        // Parameters
        std::string image_topic_name_;
        std::string map_topic_name_;
        int min_matches_;
        double match_distance_factor_;
        int surf_min_hessian_;
        
        // Flow control
        Mat current_map_;
        Rect current_roi_;    
        // Not sure what happens if the callback hangs, so if I set a waitKey(0)
        // this makes sure that any new requests that could have gotten made are dropped    
        bool image_drop_mutex_;

        void stitchMap(const Mat& new_image, const Mat& map_in, Mat& map_out, Rect& map_roi, Rect& next_roi);
        void getProjectedImageBounds(const Mat& image, const Mat& H, Rect& projected);
      
        void layeredCopy(const Mat& top_image, const Rect& top_image_bounds,
                          const Mat& bottom_image, const Rect& bottom_image_bounds,
                          Mat& image_out, const Size& image_out_size);
        
        void imageCallback(const sensor_msgs::ImageConstPtr& msg);  
        bool linkFacingUp(std::string cam_frame);
        
        
        void getValidMatches(const std::vector<KeyPoint>& query_pts, const Mat& query_img, 
                             const std::vector<KeyPoint>& train_pts, const Mat& train_img,
                             const std::vector<DMatch>& matches, std::vector<DMatch>& valid_matches);

};

MapStitchNode::MapStitchNode(void) : it_(nh_)
{
    nh_.param<std::string>("image_topic", image_topic_name_, "/downward_cam/camera/image_rect_color");
    nh_.param<std::string>("map_topic", map_topic_name_, "map");
    nh_.param<int>("min_matches", min_matches_, 12);
    nh_.param<double>("match_distance_factor", match_distance_factor_, 3.0);
    nh_.param<int>("surf_min_hessian", surf_min_hessian_, 400);
    
    image_sub_  = it_.subscribe(image_topic_name_, 1, &MapStitchNode::imageCallback, this);
    map_pub_ = it_.advertise(map_topic_name_, 1);
    
    image_drop_mutex_ = false;
    
    // For debugging
    namedWindow("new image in", CV_WINDOW_NORMAL);
    namedWindow("map", CV_WINDOW_NORMAL);
    namedWindow("projection", CV_WINDOW_NORMAL);
    namedWindow("mask", CV_WINDOW_NORMAL);
    namedWindow("matches", CV_WINDOW_NORMAL);
    
}

void MapStitchNode::stitchMap(const Mat& new_image, const Mat& map_in, Mat& map_out, Rect& map_roi, Rect& next_roi)
{
    //ROS_INFO("--------------------------");
    Mat map_bounded;
    // If there was no ROI (eg. first iteration) search the whole map
    if (map_roi.width == 0 || map_roi.height == 0)
        map_roi = Rect(0, 0, map_in.cols, map_in.rows);

    //ROS_INFO("Map in size: %d,%d", map_in.cols, map_in.rows, map_roi.x, map_roi.y, map_roi.width, map_roi.height);
    //printRect(map_roi, "Map roi in: ");
    map_bounded = Mat(map_in, map_roi);
        
    Mat map_gray, new_image_gray;
    
    // Get single channel versions of the inputs
    cvtColor(map_bounded, map_gray, CV_RGB2GRAY);
    cvtColor(new_image, new_image_gray, CV_RGB2GRAY);
    
    if ( !map_gray.data || ! new_image_gray.data )
    {
        ROS_ERROR("Failed to convert images to grayscale.");
        map_out = map_in;
        next_roi = map_roi;
        return;
    }
    
    imshow("new image in", new_image_gray);
    waitKey(10);
    
    // Use SURF to detect keypoints
    SurfFeatureDetector detector(surf_min_hessian_);
    std::vector< KeyPoint > keypoints_map, keypoints_new_image;

    detector.detect(map_gray, keypoints_map);
    //ROS_INFO("detected map_keypoints");
    detector.detect(new_image_gray, keypoints_new_image);
    //ROS_INFO("detected new image keypoints");
   
    if (keypoints_new_image.size() < 1 || keypoints_map.size() < 1)
    {
        ROS_WARN("One of the images had no keypoints.");
        map_out = map_in;
        next_roi = map_roi;
        return;
    }
    
    // For each keypoint, compute its descriptor
    SurfDescriptorExtractor extractor;

    Mat descriptiors_map, descriptiors_new_image;

    extractor.compute(map_gray, keypoints_map, descriptiors_map);
    //ROS_INFO("extracted map descriptors");
    extractor.compute(new_image_gray, keypoints_new_image, descriptiors_new_image);
    //ROS_INFO("extracted new image descriptors");
    
    if (descriptiors_new_image.rows < 1 || descriptiors_map.rows < 1)
    {
        ROS_WARN("One of the images had no descriptors.");
        map_out = map_in;
        next_roi = map_roi;
        return;
    }
    
    // Use FLANN to get the nearest match for each descriptor
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match(descriptiors_map, descriptiors_new_image, matches);
    //ROS_INFO("Found matches q=map, t=new_image");
    
    // Select only matches that are almost as good as our best match
    // TODO(Arnold): This might be improved somehow
    double min_dist = 1000;
    for( int i = 0; i < descriptiors_new_image.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
    }    
    // If min_dist was really small, we would throw out all the features!
    min_dist = max(min_dist, 0.02);
    
    // Not sure why we are getting invalid matches, but this removes them!
    std::vector< DMatch > valid_matches;
    getValidMatches(keypoints_map, map_gray, keypoints_new_image, new_image_gray, matches, valid_matches);
    
    std::vector< DMatch > good_matches;
    for( int i = 0; i < valid_matches.size(); i++ )
    {
        if( valid_matches[i].distance <= match_distance_factor_*min_dist)
            good_matches.push_back( matches[i] );
    }

    if (good_matches.size() < min_matches_)
    {
        ROS_INFO("Only %lu good matches were found, require %d", good_matches.size(), min_matches_);
        map_out = map_in;
        next_roi = map_roi;
        return;
    }

    // Get the Homography matrix using RANSAC
    std::vector< Point2f > points_map;
    std::vector< Point2f > points_new_image;
    
    // These are used for the drawMatches OpenCV function (debug)
    std::vector< KeyPoint > final_keypoints_map;
    std::vector< KeyPoint > final_keypoints_new_image;

    // Get the pixel locations from the good matches keypoints
    for( int i = 0; i < good_matches.size(); i++ )
    {
        points_map.push_back( keypoints_map[ good_matches[i].queryIdx ].pt );
        points_new_image.push_back( keypoints_new_image[ good_matches[i].trainIdx ].pt );
        
        final_keypoints_map.push_back(keypoints_map[ good_matches[i].queryIdx ]);
        final_keypoints_new_image.push_back(keypoints_new_image[ good_matches[i].trainIdx ]);
    }
    
    Mat H = findHomography(points_new_image, points_map, CV_RANSAC);
    
    // Add an extra translation to the homography matrix so that we are projecting into the 
    // full map instead of the map roi
    H.at<double>(0, 2) = H.at<double>(0, 2) + map_roi.x;
    H.at<double>(1, 2) = H.at<double>(1, 2) + map_roi.y;
    
    Rect projection_bounds;
    getProjectedImageBounds(new_image, H, projection_bounds);
    //printRect(projection_bounds, "Corners projected to:");
    
    // This makes sure we don't accept any obviously wrong projections. Could probably improve performance
    // by being stricter here
    if (projection_bounds.width > 2.0*map_roi.width || 2.0*projection_bounds.width < map_roi.width ||
        projection_bounds.height > 2.0*map_roi.height || 2.0*projection_bounds.height < map_roi.height)
    {
        ROS_INFO("Got a suspicious transformation, aborting");
        map_out = map_in;
        next_roi = map_roi;
        return;
    }
    
    // Offset the map so that we can project the image into positive-only pixel space
    Rect map_translated_bounds = Rect( max(-projection_bounds.x, 0), max(-projection_bounds.y, 0), map_in.cols, map_in.rows);
    //printRect(map_translated_bounds, "Map translated to:");

    Rect pos_projection_bounds = Rect( max(0, projection_bounds.x), max(0, projection_bounds.y), projection_bounds.width, projection_bounds.height);
    //printRect(pos_projection_bounds, "Projection translated to:");
    
    // Get the size of the final output    
    int map_out_width = max( (pos_projection_bounds.x+pos_projection_bounds.width), (map_translated_bounds.x+map_translated_bounds.width) );
    int map_out_height = max( (pos_projection_bounds.y+pos_projection_bounds.height), (map_translated_bounds.y+map_translated_bounds.height) );
    //ROS_INFO("Expanded map size: %d,%d", map_out_width, map_out_height);   
    
    
    // Modify the homography matrix to translate all the projected pixels to be positive
    H.at<double>(0, 2) = H.at<double>(0, 2) + map_translated_bounds.x;
    H.at<double>(1, 2) = H.at<double>(1, 2) + map_translated_bounds.y;
    
    Rect projection_bounds_fullmap;
    getProjectedImageBounds(new_image, H, projection_bounds_fullmap);
    
    // If everyhthing were perfect, this check would be uneccessary.
    // Not sure why this ever catches anything. Usually they are off-by-one type errors.
    // Suspect casting vs rounding in converting int to float
    if (projection_bounds_fullmap.x < 0) {
        projection_bounds_fullmap.x = 0;
        ROS_WARN("Cutting off part of the projection. Rounding errors?");
    }
    if (projection_bounds_fullmap.x+projection_bounds_fullmap.width > map_out_width){
        projection_bounds_fullmap.width = map_out_width - projection_bounds_fullmap.x;
        ROS_WARN("Cutting off part of the projection. Rounding errors?");
    }
    
    if (projection_bounds_fullmap.y < 0) {
        projection_bounds_fullmap.y = 0;
        ROS_WARN("Cutting off part of the projection. Rounding errors?");
    }
    if (projection_bounds_fullmap.y+projection_bounds_fullmap.height > map_out_height){
        projection_bounds_fullmap.height = map_out_height - projection_bounds_fullmap.y;
        ROS_WARN("Cutting off part of the projection. Rounding errors?");
    }

    
    //printRect(projection_bounds_fullmap, "Final projection to:");

    
    // Do the projection
    Mat projection;
    warpPerspective(new_image, projection, H, Size(map_out_width, map_out_height) );
    
    // Layer the new image on top of the translated input map
    layeredCopy(projection, Rect(0, 0, map_out_width, map_out_height),
                  map_in, map_translated_bounds,
                  map_out, Size(map_out_width, map_out_height)
                );
    
    // Set the ROI for the next iteration
    next_roi = projection_bounds_fullmap;
    
    // DEBUG CODE BELOW HERE    
    imshow("projection", projection);
    
    //Mat match_image;
    //ROS_INFO("%d %d %d", static_cast<int>(final_keypoints_map.size()), static_cast<int>(final_keypoints_new_image.size()), static_cast<int>(good_matches.size()));
    
    //drawMatches(map_in, final_keypoints_map, new_image, final_keypoints_new_image, good_matches, match_image);
    //imshow("matches", match_image);
    
    Mat map_dbg = map_out.clone();
    rectangle(map_dbg, next_roi, Scalar(255, 0, 0));
    imshow("map", map_dbg);
    waitKey(5);
}

void MapStitchNode::layeredCopy(const Mat& top_image, const Rect& top_image_bounds,
              const Mat& bottom_image, const Rect& bottom_image_bounds,
              Mat& image_out, const Size& image_out_size)
{
    if (top_image.cols != top_image_bounds.width || top_image.rows != top_image_bounds.height ||
        bottom_image.cols != bottom_image_bounds.width || bottom_image.rows != bottom_image_bounds.height)
    {
        ROS_ERROR("layered copy requires the images to be the same size as their bounds");
        ROS_ERROR("top: %dx%d : %dx%d", top_image.cols, top_image.rows, top_image_bounds.width, top_image_bounds.height);
        ROS_ERROR("bottom: %dx%d : %dx%d", bottom_image.cols, bottom_image.rows, bottom_image_bounds.width, bottom_image_bounds.height);
        return;
    }
    
    // Put the top image into a Mat the size of the output
    Mat top_fullsize = Mat::zeros(image_out_size, CV_8UC3);
    Mat top_fullsize_roi(top_fullsize, top_image_bounds);
    top_image.copyTo(top_fullsize_roi);
    // And convert to a float Mat on [0.,1.]
    top_fullsize.convertTo(top_fullsize, CV_32FC3);
    top_fullsize /= 255.0;
    
    
    // Put the bottom image into a Mat the size of the output
    Mat bottom_fullsize = Mat::zeros(image_out_size, CV_8UC3);
    Mat bottom_fullsize_roi(bottom_fullsize, bottom_image_bounds);
    bottom_image.copyTo(bottom_fullsize_roi);
    // And convert to a float Mat on [0.,1.]
    bottom_fullsize.convertTo(bottom_fullsize, CV_32FC3);
    bottom_fullsize /= 255.0;
    
    // Mask where anywhere that the top image had a non-zero input, there is a 1.0
    Mat top_mask;
    cvtColor(top_fullsize, top_mask, CV_BGR2GRAY);
    threshold(top_mask, top_mask, 0, 1.0, THRESH_BINARY);
    top_mask.convertTo(top_mask, CV_32FC1);
    
    // The blend mask weights the edges of the top image so that the closer to the edge, the less weight
    // the top image has and the more weight the bottom image has
    Mat top_blend_mask = Mat::zeros(image_out_size, CV_32F);
    Mat bottom_blend_mask = Mat::zeros(image_out_size, CV_32F);
    int N = 6;
    float weight = 1.0/((float) (N-1));
    for (int i = 1; i < N; i++)
    {
        Mat top_blend_mask_tmp;
        // Set overall blending width by a ratio of the size of the top image
        int erode_width = ( i*max(image_out_size.width, image_out_size.height) )/( 20.0*(N-1) );
        erode(top_mask, top_blend_mask_tmp, Mat(), Point(-1, -1), erode_width, BORDER_CONSTANT, 0);
        addWeighted(top_blend_mask, 1.0, top_blend_mask_tmp, weight, 0, top_blend_mask);
    }
    // Bottom mask complements top mask so that they sum to 1
    bottom_blend_mask = 1 - top_blend_mask;

    // Apply the masks
    cvtColor(top_blend_mask, top_blend_mask, CV_GRAY2BGR);
    multiply(top_fullsize, top_blend_mask, top_fullsize);
    cvtColor(bottom_blend_mask, bottom_blend_mask, CV_GRAY2BGR);
    multiply(bottom_fullsize, bottom_blend_mask, bottom_fullsize);
    
    // Combine the images, and convert back to a uchar image on [0, 255]
    image_out = (top_fullsize + bottom_fullsize)*255;
    image_out.convertTo(image_out, CV_8UC3);
       
    Mat mask_debug = bottom_blend_mask;
    imshow("mask", mask_debug);
}

void MapStitchNode::getValidMatches(const std::vector<KeyPoint>& query_pts, const Mat& query_img, 
                                       const std::vector<KeyPoint>& train_pts, const Mat& train_img,
                                       const std::vector<DMatch>& matches, std::vector<DMatch>& valid_matches)
{
    for (int i = 0; i < matches.size(); i++) {
        float qx = query_pts[matches[i].queryIdx].pt.x;
        float qy = query_pts[matches[i].queryIdx].pt.y;
        float tx = train_pts[matches[i].trainIdx].pt.x;
        float ty = train_pts[matches[i].trainIdx].pt.y;
        
        // If any training or query point isn't within the bounds of the image, something went wrong
        if ( qx > 0 && qx < query_img.cols && qy > 0 && qy < query_img.rows &&
              tx > 0 && tx < train_img.cols && ty > 0 && ty < train_img.rows )
              
              valid_matches.push_back(matches[i]);
        
        // DEBUG...
        
        if (qx <= 0 || qx >= query_img.cols ||
            qy <= 0 || qy >= query_img.rows)
            ROS_INFO("q %f,%f not in %d,%d", qx, qy, query_img.cols, query_img.rows);
        
        if (tx < 0 || tx >= train_img.cols ||
            ty < 0 || ty >= train_img.rows) 
            ROS_INFO("t %f,%f not in %d,%d", qx, qy, train_img.cols, train_img.rows);
    }        
    //ROS_INFO("Got %lu valid matches", valid_matches.size());
}

void MapStitchNode::getProjectedImageBounds(const Mat& image, const Mat& H, Rect& projected)
{
    // Make a point for each corner of the input image
    std::vector<Point2f> src_corners(4);
    src_corners[0] = cvPoint(0,0);
    src_corners[1] = cvPoint( image.cols, 0 );
    src_corners[2] = cvPoint( image.cols, image.rows );
    src_corners[3] = cvPoint( 0, image.rows );
    
    // By projecting the corners only, figure out where the whole projected image will lie
    std::vector<Point2f> target_corners(4);
    perspectiveTransform(src_corners, target_corners, H);
    
    float min_row = 10000; float min_col = 10000;
    float max_row = 0; float max_col = 0;
    
    for (int i = 0; i < target_corners.size(); i++)
    {
        if (target_corners[i].x > max_col)
            max_col = target_corners[i].x;
        if (target_corners[i].y > max_row)
            max_row = target_corners[i].y;
        if (target_corners[i].x < min_col)
            min_col = target_corners[i].x;
        if (target_corners[i].y < min_row)
            min_row = target_corners[i].y;
    }
    projected = Rect((int) round(min_col), (int) round(min_row), (int) round(max_col-min_col), (int) round(max_row-min_row));
}

bool MapStitchNode::linkFacingUp(std::string frame)
{
    tf::StampedTransform camera_pose;
    try
    {
      tf_listener_.lookupTransform(frame, "world",  
                               ros::Time(0), camera_pose);
    }
    catch (tf::TransformException ex)
    {
      ROS_ERROR("%s",ex.what());
      ros::Duration(1.0).sleep();
      return false;
    }
    
    double roll, pitch, yaw;
    tf::Matrix3x3(camera_pose.getRotation()).getRPY(roll, pitch, yaw);
    if (sqrt(roll*roll + pitch*pitch) < 0.005){
        return true;
    }
    else{
        ROS_WARN("Link %s not facing up", frame.c_str());
        return false;
    }
    
}


void MapStitchNode::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    if (!linkFacingUp("base_link")) return;
    
    if (!image_drop_mutex_) {
        image_drop_mutex_ = true;
        Mat new_image, new_map;
        Rect next_roi;
        try
        {
            new_image = cv_bridge::toCvShare(msg, "bgr8")->image;
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        }
        
        if ( !current_map_.data )
        {
            new_map = new_image;
        }
        else
        {
            stitchMap(new_image, current_map_, new_map, current_roi_, next_roi);
        }
            
        current_map_ = new_map;
        current_roi_ = next_roi;
        
        sensor_msgs::ImagePtr map_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", current_map_).toImageMsg();
        map_pub_.publish(map_msg);
        image_drop_mutex_ = false;
    }
}
        
int main(int argc, char* argv[])
{
    ros::init(argc, argv, "map_stitch");
    
    MapStitchNode pctdin;
    ros::spin();
}

