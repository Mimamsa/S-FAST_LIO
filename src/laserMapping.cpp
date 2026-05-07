#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
//#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include "IMU_Processing.hpp"

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
int add_point_size = 0, kdtree_delete_counter = 0;
bool pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int scan_count = 0, publish_count = 0;
int feats_down_size = 0, NUM_MAX_ITERATIONS = 0, pcd_save_interval = -1, pcd_index = 0;

bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<BoxPointType> cub_needrm;
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());  //畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI()); //畸变纠正后降采样的单帧点云，W系

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

Eigen::Quaterniond Q_world_gravity_aligned = Eigen::Quaterniond::Identity();
M3D R_world_gravity_aligned(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;

esekfom::esekf kf;

state_ikfom state_point;
Eigen::Vector3d pos_lid; //估计的W系下的位置

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());

/**
 * @brief Signal handler for graceful shutdown
 * 
 * Captures system signals (e.g., SIGINT from Ctrl+C) and sets exit flag.
 * Notifies all waiting threads to terminate safely.
 * 
 * @param sig Signal number
 * 
 * @note
 * - Sets global flag `flg_exit`
 * - Wakes up threads waiting on condition variable
 */
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

/**
 * @brief Standard LiDAR callback function (ROS subscriber)
 * 
 * This function receives raw PointCloud2 messages, preprocesses them,
 * and pushes them into a buffer for later synchronization with IMU data.
 * 
 * Thread-safe using mutex and condition variable.
 * 
 * @param msg Incoming ROS PointCloud2 message
 * 
 * @note
 * - Detects timestamp rollback and clears buffer
 * - Uses Preprocess class to convert ROS msg → PCL format
 * - Maintains lidar_buffer and time_buffer
 */
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count++;
    // double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
/*
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}
*/

/**
 * @brief IMU callback function
 * 
 * Receives IMU measurements and stores them in a buffer.
 * Applies optional time synchronization offset between LiDAR and IMU.
 * 
 * @param msg_in Incoming IMU message
 * 
 * @note
 * - Handles timestamp rollback
 * - Supports external or self time synchronization
 * - Buffered for later fusion with LiDAR
 */
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
//把当前要处理的LIDAR和IMU数据打包到meas
/**
 * @brief Synchronize LiDAR and IMU measurements
 * 
 * This function extracts one LiDAR scan and all corresponding IMU data
 * within the scan time interval, forming a MeasureGroup.
 * 
 * @param meas Output measurement group (LiDAR + IMU)
 * @return true if a valid synchronized package is ready
 * @return false if data is insufficient
 * 
 * @note
 * - Uses lidar_buffer and imu_buffer
 * - Estimates scan end time using point timestamp (curvature field)
 * - Ensures IMU covers entire LiDAR scan duration
 */
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 5) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;  //注意curvature中存储的是相对第一个点的时间
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)  //如果最新的imu时间戳都<雷达最终的时间，证明还没有收集足够的imu数据，break
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

/**
 * @brief Transform point from LiDAR (body frame) to world frame
 * 
 * @param pi Input point in LiDAR frame
 * @param po Output point in world frame
 * 
 * Applies:
 *   LiDAR → IMU extrinsic (EKF state)
 *   IMU → World pose (EKF state)
 * 
 * @note
 * Transformation chain:
 *   p_world = R_world_imu * (R_imu_lidar * p_lidar + t_imu_lidar) + t_world_imu
 */
void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(
        state_point.rot.matrix() * \
        (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + \
        state_point.pos
    );

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

/**
 * @brief Transform Eigen vector point from LiDAR frame to world frame
 * 
 * Template version of coordinate transformation for Eigen types.
 * Applies LiDAR→IMU extrinsic and IMU→World pose transformation.
 * 
 * @tparam T Scalar type (float/double)
 * @param pi Input point in LiDAR frame
 * @param po Output point in world frame
 */
template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(
        state_point.rot.matrix() * \
        (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + \
        state_point.pos
    );

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

/**
 * @brief Rotate point according to the origin
 * 
 * @param pi Input point
 * @param po Output point
 * @param r_mtx Rotation matrix
 * 
 * @note
 * Transformation chain:
 *   po = R * pi
 */
void pointRotate(PointType const *const pi, PointType *const po, M3D &r_mtx)
{
    V3D pi_eigen(pi->x, pi->y, pi->z);
    V3D po_eigen(r_mtx * pi_eigen);

    po->x = po_eigen(0);
    po->y = po_eigen(1);
    po->z = po_eigen(2);
}

BoxPointType LocalMap_Points;      // ikd-tree地图立方体的2个角点
bool Localmap_Initialized = false; // 局部地图是否初始化
/**
 * @brief Maintain sliding local map region (FOV-based)
 * 
 * Keeps only a local cube of map points around current LiDAR position.
 * Removes points outside the region to control memory and computation.
 * 
 * @note
 * - Implements moving local map (sliding window)
 * - Based on DET_RANGE and MOV_THRESHOLD
 * - Uses ikdtree.Delete_Point_Boxes()
 */
void lasermap_fov_segment()
{
    cub_needrm.clear(); // 清空需要移除的区域
    kdtree_delete_counter = 0;

    V3D pos_LiD = pos_lid; // W系下位置
    //初始化局部地图范围，以pos_LiD为中心,长宽高均为cube_len
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    //各个方向上pos_LiD与局部地图边界的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 与某个方向上的边界距离（1.5*300m）太小，标记需要移除need_move(FAST-LIO2论文Fig.3)
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || \
            dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            need_move = true;
        }
    }
    if (!need_move)
        return; //如果不需要，直接返回，不更改局部地图

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    //需要移动的距离
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);

    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm); //删除指定范围内的点
}

/**
 * @brief Transform point from LiDAR frame to IMU body frame
 * 
 * Applies only extrinsic calibration between LiDAR and IMU.
 * Does NOT transform to world frame.
 * 
 * @param pi Input point in LiDAR frame
 * @param po Output point in IMU frame
 * 
 * @note
 * Transformation:
 *   p_imu = R_i_l * p_lidar + t_i_l
 */
void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I.matrix() * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

//根据最新估计位姿  增量添加点云到map
/**
 * @brief Incrementally add new points into ikd-tree map
 * 
 * This function:
 * 1. Transforms points into world frame
 * 2. Performs voxel-based filtering
 * 3. Checks nearest neighbors to avoid redundancy
 * 4. Inserts selected points into ikd-tree
 * 
 * @note
 * - Uses Nearest_Points from scan matching
 * - Avoids adding points if similar ones already exist
 * - Maintains map sparsity
 */
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        //转换到世界坐标系
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType mid_point; //点所在体素的中心
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);

            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && \
                fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && \
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]); //如果距离最近的点都在体素外，则该点不需要Downsample
                continue;
            }
            for (int j = 0; j < NUM_MATCH_POINTS; j++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[j], mid_point) < dist) //如果近邻点距离 < 当前点距离，不添加该点
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
/**
 * @brief Publish current LiDAR frame in world coordinate system
 * 
 * Transforms point cloud into world frame and publishes it.
 * Optionally accumulates and saves point clouds to PCD files.
 * 
 * @param pubLaserCloudFull_ ROS publisher for world-frame point cloud
 * 
 * @note
 * - Uses either dense or downsampled point cloud
 * - Frame ID: "camera_init"
 * - Can significantly affect performance if PCD saving is enabled
 */
void publish_frame_world(const ros::Publisher &pubLaserCloudFull_)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
            pointRotate(&laserCloudWorld->points[i], &laserCloudWorld->points[i], R_world_gravity_aligned);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull_.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointBodyToWorld(&feats_undistort->points[i], &laserCloudWorld->points[i]);
            pointRotate(&laserCloudWorld->points[i], &laserCloudWorld->points[i], R_world_gravity_aligned);
        }

        static int scan_wait_num = 0;
        scan_wait_num++;

        if (scan_wait_num % 4 == 0)
            *pcl_wait_save += *laserCloudWorld;

        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

 /**
 * @brief Publish current LiDAR frame in IMU body frame
 * 
 * Transforms undistorted point cloud into IMU frame and publishes it.
 * 
 * @param pubLaserCloudFull_body ROS publisher for body-frame point cloud
 * 
 * @note
 * - Frame ID: "body"
 * - Requires scan_pub_en && scan_body_pub_en enabled
 */
void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

/**
 * @brief Publish global map point cloud
 * 
 * Converts internal map representation into ROS PointCloud2 message.
 * 
 * @param pubLaserCloudMap ROS publisher for map
 * 
 * @note
 * - Uses featsFromMap
 * - Frame ID: "camera_init"
 */
void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

/**
 * @brief Fill pose message with current EKF state
 * 
 * Sets position and orientation (quaternion) from state_point.
 * 
 * @param T ROS message type containing pose field
 * @param out Output pose message
 * 
 * @note
 * - Orientation derived from rotation matrix
 * - Used for Odometry and Path messages
 */
// template <typename T>
// void set_posestamp(T &out)
// {
//     out.pose.position.x = state_point.pos(0);
//     out.pose.position.y = state_point.pos(1);
//     out.pose.position.z = state_point.pos(2);

//     auto q_ = Eigen::Quaterniond(state_point.rot.matrix());
//     //q_ = Q_world_gravity_aligned * q_;
//     out.pose.orientation.x = q_.coeffs()[0];
//     out.pose.orientation.y = q_.coeffs()[1];
//     out.pose.orientation.z = q_.coeffs()[2];
//     out.pose.orientation.w = q_.coeffs()[3];
// }

/**
 * @brief Get current position and rotation from EKF state
 * 
 * @param[in] ekf_state_point EKF state point
 * @param[out] tvec Output position
 * @param[out] quat Output rotation
 * 
 * @note
 * - Used for Odometry and Path messages
 */
void get_ekf_state_point(state_ikfom &ekf_state_point, V3D &tvec, Eigen::Quaterniond &quat)
{
    tvec = Eigen::Vector3d(ekf_state_point.pos);
    quat = Eigen::Quaterniond(state_point.rot.matrix());
}

/**
 * @brief Get current position and rotation from EKF state
 * 
 * @param[in] tvec Input position
 * @param[in] quat Input rotation
 * @param[out] out ROS message type containing pose field
 * 
 * @note
 * - Used for Odometry and Path messages
 */
template <typename T>
void set_stamped_pose(V3D &tvec, Eigen::Quaterniond &quat, T &out)
{
    out.pose.position.x = tvec(0);
    out.pose.position.y = tvec(1);
    out.pose.position.z = tvec(2);

    out.pose.orientation.x = quat.coeffs()[0];
    out.pose.orientation.y = quat.coeffs()[1];
    out.pose.orientation.z = quat.coeffs()[2];
    out.pose.orientation.w = quat.coeffs()[3];
}

/**
 * @brief Publish odometry and broadcast TF transform
 * 
 * Publishes robot pose estimated by EKF as nav_msgs::Odometry.
 * Also broadcasts TF transform from world to body frame.
 * 
 * @param pubOdomAftMapped ROS publisher for odometry
 * 
 * @note
 * - Frame: "camera_init" → "body"
 * - Includes covariance from EKF
 * - TF broadcaster is static inside function
 */
void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    // ----- Publish odometry -----
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    // set_posestamp(odomAftMapped.pose);  // Fill pose message with current EKF state

    // Get EKF state point in Eigen datatype (position and rotation)
    V3D ekf_pos_;
    Eigen::Quaterniond ekf_quat_;
    get_ekf_state_point(state_point, ekf_pos_, ekf_quat_);

    // transform EKF state point
    ekf_pos_ = Q_world_gravity_aligned * ekf_pos_;
    ekf_quat_ = Q_world_gravity_aligned * ekf_quat_;

    // Set transfored EKF state point to message
    set_stamped_pose(ekf_pos_, ekf_quat_, odomAftMapped.pose);

    pubOdomAftMapped.publish(odomAftMapped);

    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    // ----- Broadcast TF transform -----
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));
}

/**
 * @brief Publish trajectory path
 * 
 * Appends current pose to path and publishes it periodically.
 * 
 * @param pubPath ROS publisher for nav_msgs::Path
 * 
 * @note
 * - Publishes every 10 frames to reduce load
 * - Large path may crash RViz if not controlled
 */
void publish_path(const ros::Publisher pubPath)
{
    //set_posestamp(msg_body_pose);

    // Get EKF state point in Eigen datatype (position and rotation)
    V3D ekf_pos_;
    Eigen::Quaterniond ekf_quat_;
    get_ekf_state_point(state_point, ekf_pos_, ekf_quat_);

    // transform EKF state point
    ekf_pos_ = Q_world_gravity_aligned * ekf_pos_;
    ekf_quat_ = Q_world_gravity_aligned * ekf_quat_;

    // Set transfored EKF state point to message
    set_stamped_pose(ekf_pos_, ekf_quat_, msg_body_pose);

    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rviz will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

/**
 * @brief Main SLAM loop
 * 
 * Pipeline:
 * 1. Receive sensor data
 * 2. Synchronize LiDAR + IMU
 * 3. Undistort point cloud using IMU
 * 4. Downsample
 * 5. Perform EKF update (scan-to-map)
 * 6. Update map (ikd-tree)
 * 7. Publish odometry and point clouds
 * 
 * @note
 * - Runs at high frequency (5000 Hz loop)
 * - Core computation triggered only when sync_packages() succeeds
 */
int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);            // 是否发布当前正在扫描的点云的topic
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);          // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true); // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic，需要该变量和上一个变量同时为true才发布
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);                   // 卡尔曼滤波的最大迭代次数
    nh.param<string>("map_file_path", map_file_path, "");                    // 地图保存路径
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");         // 雷达点云topic名称
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");           // IMU的topic名称
    nh.param<bool>("common/time_sync_en", time_sync_en, false);              // 是否需要时间同步，只有当外部未进行时间同步时设为true
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5); // VoxelGrid降采样时的体素大小
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);    // 地图的局部区域的长度（FastLio2论文中有解释）
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f); // 激光雷达的最大探测范围
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);               // IMU陀螺仪的协方差
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);               // IMU加速度计的协方差
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);        // IMU陀螺仪偏置的协方差
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);        // IMU加速度计偏置的协方差
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);        // 最小距离阈值，即过滤掉0～blind范围内的点云
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, MID360); // 激光雷达的类型
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);       // 激光雷达扫描的线数（livox avia为6线）
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);           // 采样间隔，即每隔point_filter_num个点取1个点
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false); // 是否提取特征点（FAST_LIO2默认不进行特征点提取）
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false); // 是否将点云地图保存到PCD文件
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>()); // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>()); // 雷达相对于IMU的外参R

    cout << "Lidar_type: " << p_pre->lidar_type << endl;
    // 初始化path的header（包括时间戳和帧id），path用于保存odemetry的路径
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    /*** ROS subscribe initialization ***/
    //ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_pcl = nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    shared_ptr<ImuProcess> p_imu1(new ImuProcess());
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu1->set_param(
        Lidar_T_wrt_IMU,
        Lidar_R_wrt_IMU,
        V3D(gyr_cov, gyr_cov, gyr_cov),
        V3D(acc_cov, acc_cov, acc_cov),
        V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov),
        V3D(b_acc_cov, b_acc_cov, b_acc_cov)
    );

    signal(SIGINT, SigHandle); //当程序检测到signal信号（例如ctrl+c） 时  执行 SigHandle 函数
    ros::Rate rate(5000);

    while (ros::ok())
    {
        if (flg_exit)
            break;

        ros::spinOnce();

        if (sync_packages(Measures)) //把一次的IMU和LIDAR数据打包到Measures
        {
            double t00 = omp_get_wtime();

            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu1->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            p_imu1->Process(Measures, kf, feats_undistort);

            Q_world_gravity_aligned = p_imu1->Q_world_gravity_aligned;
            R_world_gravity_aligned = p_imu1->Q_world_gravity_aligned.toRotationMatrix();;
            // std::cout << "R_world_gravity_aligned: \n" << p_imu1->R_world_gravity_aligned << std::endl;

            //如果feats_undistort为空 ROS_WARN
            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

            lasermap_fov_segment(); //更新localmap边界，然后降采样当前帧点云

            //点云下采样
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            feats_down_size = feats_down_body->points.size();

            // std::cout << "feats_down_size :" << feats_down_size << std::endl;
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            //初始化ikdtree(ikdtree为空时)
            if (ikdtree.Root_Node == nullptr)
            {
                ikdtree.set_downsample_param(filter_size_map_min);
                feats_down_world->resize(feats_down_size);
                
                // Transform pointcloud from lidar frame to world frame, points by points
                for (int i = 0; i < feats_down_size; i++)
                {
                    pointBodyToWorld(
                        &(feats_down_body->points[i]),
                        &(feats_down_world->points[i])
                    ); // lidar坐标系转到世界坐标系
                }
                
                ikdtree.Build(feats_down_world->points); //根据世界坐标系下的点构建ikdtree
                continue;
            }

            if (0) // If you need to see map point, change to "if(1)"
            {
                PointVector().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
                // std::cout << "ikdtree size: " << featsFromMap->points.size() << std::endl;
            }

            /*** iterated state estimation ***/
            Nearest_Points.resize(feats_down_size); //存储近邻点的vector
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en);

            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            feats_down_world->resize(feats_down_size);
            map_incremental();

            /******* Publish points *******/
            if (path_en)
                publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)
                publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en)
                publish_frame_body(pubLaserCloudFull_body);
            // publish_map(pubLaserCloudMap);

            double t11 = omp_get_wtime();
            std::cout << "feats_down_size: " << feats_down_size << "  Whole mapping time(ms):  " << (t11 - t00) * 1000 << std::endl
                      << std::endl;
        }

        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (size_t i = 1; i <= pcd_index; i++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(i) + string(".pcd"));
            pcl::PCDReader reader;
            reader.read(all_points_dir, *cloud_temp);
            *cloud = *cloud + *cloud_temp;
        }

        string file_name = string("GlobalMap.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *cloud);

        //////////////////////////////////////
        PointVector().swap(ikdtree.PCL_Storage);
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
        featsFromMap->clear();
        featsFromMap->points = ikdtree.PCL_Storage;
        std::cout << "ikdtree size: " << featsFromMap->points.size() << std::endl;
        string file_name1 = string("GlobalMap_ikdtree.pcd");
        pcl::PCDWriter pcd_writer1;
        string all_points_dir1(string(string(ROOT_DIR) + "PCD/") + file_name1);
        cout << "current scan saved to /PCD/" << file_name1 << endl;
        pcd_writer1.writeBinary(all_points_dir1, *featsFromMap);
    }

    return 0;
}
