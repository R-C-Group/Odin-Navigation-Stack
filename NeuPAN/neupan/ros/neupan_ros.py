"""
NeuPAN ROS节点 - 神经网络路径跟随与避障规划器
NeuPAN ROS Node - Neural Network Path Following and Obstacle Avoidance Planner

neupan_core是neupan_ros包的主类，用于在ROS框架中运行NeuPAN算法。
它订阅激光扫描和定位信息，并向机器人发布速度指令。

neupan_core is the main class for the neupan_ros package. It is used to run the 
NeuPAN algorithm in the ROS framework, which subscribes to the laser scan and 
localization information, and publishes the velocity command to the robot.

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han <hanrh@connect.hku.hk>

Modified by Hongle Mo.
Copyright (c) 2025 Manifold Tech Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import os
from math import sin, cos, atan2
import numpy as np
import yaml
from loguru import logger
import time

from neupan import neupan
from neupan.util import get_transform

import rospy
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Empty, Float32
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import LaserScan, PointCloud2
import tf
import sensor_msgs.point_cloud2 as pc2


class neupan_core:
    """
    NeuPAN核心类：端到端神经网络局部规划器
    NeuPAN core class: end-to-end neural network local planner
    
    功能 / Functions:
    - 订阅激光雷达扫描数据 / Subscribe to laser scan data
    - 订阅机器人定位信息 / Subscribe to robot localization
    - 使用神经网络进行路径规划和避障 / Use neural network for path planning and obstacle avoidance
    - 发布速度控制指令 / Publish velocity commands
    """
    
    def __init__(self) -> None:
        """
        初始化NeuPAN ROS节点
        Initialize NeuPAN ROS node
        
        配置步骤 / Configuration steps:
        1. 加载YAML配置文件 / Load YAML configuration
        2. 初始化NeuPAN神经规划器 / Initialize NeuPAN neural planner
        3. 设置ROS发布者和订阅者 / Set up ROS publishers and subscribers
        """

        rospy.init_node("neupan_node", anonymous=True)

        # 加载配置文件 / Load configuration file
        config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        logger.info(f"Configuration:\n {yaml.dump(self.config)}")

        # 激光扫描参数配置 / Laser scan parameters configuration
        self.scan_angle_range = np.array(self.config["scan_angle"], dtype=np.float32)  # 扫描角度范围 / Scan angle range
        self.scan_range = np.array(self.config["scan_range"], dtype=np.float32)  # 扫描距离范围 / Scan distance range

        # 初始化NeuPAN神经网络规划器 / Initialize NeuPAN neural network planner
        pan = {"dune_checkpoint": self.config["dune_checkpoint"]}
        self.neupan_planner = neupan.init_from_yaml(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), self.config["planner_config_file"]
                )
            ),
            pan=pan,
        )

        # 数据存储 / Data storage
        self.obstacle_points = None  # (2, n) 障碍物点云，n为点数 / Obstacle point cloud, n is number of points
        self.robot_state = None  # (3, 1) 机器人状态 [x, y, theta] / Robot state [x, y, theta]
        self.stop = False  # 停止标志 / Stop flag

        # ROS发布者 / ROS publishers
        self.vel_pub = rospy.Publisher(
            self.config["topic"]["cmd_vel"], Twist, queue_size=10
        )  # 发布速度指令 / Publish velocity commands
        self.plan_pub = rospy.Publisher("/neupan_plan", Path, queue_size=10)  # 发布规划路径 / Publish planned path
        self.ref_state_pub = rospy.Publisher(
            "/neupan_ref_state", Path, queue_size=10
        )  # 发布当前参考状态 / Publish current reference state
        self.ref_path_pub = rospy.Publisher(
            "/neupan_initial_path", Path, queue_size=10
        )  # 发布初始路径 / Publish initial path

        # RViz可视化发布者 / RViz visualization publishers
        self.point_markers_pub_dune = rospy.Publisher(
            "/dune_point_markers", MarkerArray, queue_size=10
        )  # DUNE采样点 / DUNE sample points
        self.robot_marker_pub = rospy.Publisher("/robot_marker", Marker, queue_size=10)  # 机器人标记 / Robot marker
        self.point_markers_pub_nrmp = rospy.Publisher(
            "/nrmp_point_markers", MarkerArray, queue_size=10
        )  # NRMP采样点 / NRMP sample points
        self.arrive_pub = rospy.Publisher(self.config['topic']['arrive'], Empty, queue_size=1)  # 到达目标信号 / Arrival signal
        self.last_arrive_flag: bool = False  # 上一次到达标志 / Last arrival flag

        # TF坐标变换监听器 / TF transform listener
        self.listener = tf.TransformListener()

        # ROS订阅者 / ROS subscribers
        rospy.Subscriber(self.config["topic"]["scan"], LaserScan, self.scan_callback)  # 订阅激光扫描 / Subscribe to laser scan

        # 三种初始路径输入方式 / Three types of initial path input:
        # 1. 从给定路径 / From given path
        # 2. 从路点列表 / From waypoints
        # 3. 从目标位置 / From goal position
        rospy.Subscriber(self.config["topic"]["path"], Path, self.path_callback)
        rospy.Subscriber(
            self.config["topic"]["waypoints"], Path, self.waypoints_callback
        )
        rospy.Subscriber(self.config["topic"]["goal"], PoseStamped, self.goal_callback)

        # 动态参数调整订阅者 / Dynamic parameter adjustment subscribers
        rospy.Subscriber("/neupan/q_s", Float32, self.qs_callback)  # 平滑性权重 / Smoothness weight
        rospy.Subscriber("/neupan/p_u", Float32, self.pu_callback)  # 控制权重 / Control weight

    def qs_callback(self, msg):
        logger.info(f"Update q_s to {msg.data}")
        self.neupan_planner.update_adjust_parameters(q_s=float(msg.data))

    def pu_callback(self, msg):
        logger.info(f"Update p_u to {msg.data}")
        self.neupan_planner.update_adjust_parameters(p_u=float(msg.data))

    def run(self):
        """
        主运行循环：执行路径规划和速度控制
        Main run loop: executes path planning and velocity control
        
        流程 / Process:
        1. 获取机器人位姿（通过TF） / Get robot pose (via TF)
        2. 获取障碍物点云 / Get obstacle point cloud
        3. 调用NeuPAN神经规划器 / Call NeuPAN neural planner
        4. 发布速度指令和可视化信息 / Publish velocity commands and visualizations
        """

        r = rospy.Rate(50)  # 50Hz控制频率 / 50Hz control frequency

        while not rospy.is_shutdown():

            try:
                # 通过TF获取机器人在地图坐标系中的位姿 / Get robot pose in map frame via TF
                (trans, rot) = self.listener.lookupTransform(
                    self.config["frame"]["map"],  # 目标坐标系：地图 / Target frame: map
                    self.config["frame"]["base"],  # 源坐标系：机器人底盘 / Source frame: robot base
                    rospy.Time(0),  # 最新的变换 / Latest transform
                )

                yaw = self.quat_to_yaw_list(rot)  # 四元数转欧拉角 / Quaternion to Euler
                x, y = trans[0], trans[1]
                self.robot_state = np.array([x, y, yaw]).reshape(3, 1)

            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ):
                rospy.loginfo_throttle(
                    3,
                    "waiting for tf for the transform from {} to {}".format(
                        self.config["frame"]["base"], self.config["frame"]["map"]
                    ),
                )
                r.sleep()
                continue

            if self.robot_state is None:
                rospy.logwarn_throttle(3, "waiting for robot state")
                r.sleep()
                continue

            rospy.loginfo_once(
                "robot state received {}".format(self.robot_state.tolist())
            )

            # 如果有路点但没有初始路径，从当前位置生成初始路径 / If waypoints exist but no initial path, generate from current position
            if (
                len(self.neupan_planner.waypoints) >= 1
                and self.neupan_planner.initial_path is None
            ):
                self.neupan_planner.set_initial_path_from_state(self.robot_state)

            if self.neupan_planner.initial_path is None:
                rospy.logwarn_throttle(3, "waiting for neupan initial path")
                r.sleep()
                continue

            rospy.loginfo_once("initial Path Received")
            self.ref_path_pub.publish(
                self.generate_path_msg(self.neupan_planner.initial_path)
            )

            if self.obstacle_points is None:
                rospy.logwarn_throttle(
                    1, "No obstacle points, only path tracking task will be performed"
                )

            if not self.last_arrive_flag:
                t_start = time.time()

            # 调用NeuPAN规划器：输入机器人状态和障碍物，输出速度指令 / Call NeuPAN planner: input robot state and obstacles, output velocity command
            action, info = self.neupan_planner(self.robot_state, self.obstacle_points)

            if not self.last_arrive_flag:
                t_end = time.time()
                rospy.loginfo(
                    f"neupan planning time: {(t_end - t_start)*1000:.1f} ms"
                )

            self.stop = info["stop"]  # 碰撞停止标志 / Collision stop flag
            self.arrive = info["arrive"]  # 到达目标标志 / Arrival flag

            # 检测到达目标，发布到达信号 / Detect arrival, publish arrival signal
            if info["arrive"] and not self.last_arrive_flag:
                self.arrive_pub.publish(Empty())
                rospy.loginfo("arrive at the target")

            self.last_arrive_flag = info["arrive"]

            # 发布路径和速度指令 / Publish path and velocity commands
            self.plan_pub.publish(self.generate_path_msg(info["opt_state_list"]))  # 优化后的状态序列 / Optimized state sequence
            self.ref_state_pub.publish(self.generate_path_msg(info["ref_state_list"]))  # 参考状态序列 / Reference state sequence
            if not info["arrive"] or not self.last_arrive_flag:
                self.vel_pub.publish(self.generate_twist_msg(action))  # 发布速度指令 / Publish velocity command

            # 发布可视化标记 / Publish visualization markers
            self.point_markers_pub_dune.publish(self.generate_dune_points_markers_msg())
            self.point_markers_pub_nrmp.publish(self.generate_nrmp_points_markers_msg())
            self.robot_marker_pub.publish(self.generate_robot_marker_msg())

            if info["stop"]:
                rospy.logwarn_throttle(
                    0.5,
                    "neupan stop with the min distance "
                    + str(self.neupan_planner.min_distance.detach().item())
                    + " threshold "
                    + str(self.neupan_planner.collision_threshold),
                )

            r.sleep()

    # scan callback
    def scan_callback(self, scan_msg):

        if self.robot_state is None:
            return None

        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))

        points = []
        # x, y, z, yaw, pitch, roll = self.lidar_offset

        if self.config["flip_angle"]:
            angles = np.flip(angles)

        for i in range(len(ranges)):
            distance = ranges[i]
            angle = angles[i]

            if (
                i % self.config["scan_downsample"] == 0
                and distance >= self.scan_range[0]
                and distance <= self.scan_range[1]
                and angle > self.scan_angle_range[0]
                and angle < self.scan_angle_range[1]
            ):
                point = np.array([[distance * cos(angle)], [distance * sin(angle)]])
                points.append(point)

        if len(points) == 0:
            self.obstacle_points = None
            rospy.loginfo_once("No valid scan points")
            return None

        point_array = np.hstack(points)

        try:
            (trans, rot) = self.listener.lookupTransform(
                self.config["frame"]["map"],
                self.config["frame"]["lidar"],
                rospy.Time(0),
            )

            yaw = self.quat_to_yaw_list(rot)
            x, y = trans[0], trans[1]

            trans_matrix, rot_matrix = get_transform(np.c_[x, y, yaw].reshape(3, 1))
            self.obstacle_points = rot_matrix @ point_array + trans_matrix
            rospy.loginfo_once("Scan obstacle points Received")

            return

        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            rospy.loginfo_throttle(
                3,
                "waiting for tf for the transform from {} to {}".format(
                    self.config["frame"]["lidar"], self.config["frame"]["map"]
                ),
            )
            return

    def path_callback(self, path):

        initial_point_list = []

        for i in range(len(path.poses)):
            p = path.poses[i]
            x = p.pose.position.x
            y = p.pose.position.y

            if self.config["include_initial_path_direction"]:
                theta = self.quat_to_yaw(p.pose.orientation)
            else:
                rospy.loginfo_once(
                    "Using the points gradient as the initial path direction"
                )

                if i + 1 < len(path.poses):
                    p2 = path.poses[i + 1]
                    x2 = p2.pose.position.x
                    y2 = p2.pose.position.y
                    theta = atan2(y2 - y, x2 - x)
                else:
                    theta = initial_point_list[-1][2, 0]

            points = np.array([x, y, theta, 1]).reshape(4, 1)
            initial_point_list.append(points)

        if (
            self.neupan_planner.initial_path is None
            or self.config["refresh_initial_path"]
        ):
            rospy.loginfo_throttle(0.1, "initial path update from given path")
            self.neupan_planner.set_initial_path(initial_point_list)
            self.neupan_planner.reset()

    def waypoints_callback(self, path):
        """
        Utilize multiple waypoints (goals) to set the initial path
        """

        waypoints_list = [self.robot_state]

        for i in range(len(path.poses)):
            p = path.poses[i]
            x = p.pose.position.x
            y = p.pose.position.y

            if self.config["include_initial_path_direction"]:
                theta = self.quat_to_yaw(p.pose.orientation)
            else:
                rospy.loginfo_once(
                    "Using the points gradient as the initial path direction"
                )

                if i + 1 < len(path.poses):
                    p2 = path.poses[i + 1]
                    x2 = p2.pose.position.x
                    y2 = p2.pose.position.y
                    theta = atan2(y2 - y, x2 - x)
                else:
                    theta = waypoints_list[-1][2, 0]

            points = np.array([x, y, theta, 1]).reshape(4, 1)
            waypoints_list.append(points)

        if (
            self.neupan_planner.initial_path is None
            or self.config["refresh_initial_path"]
        ):
            rospy.loginfo_throttle(0.1, "initial path update from waypoints")
            self.neupan_planner.update_initial_path_from_waypoints(waypoints_list)
            self.neupan_planner.reset()

    def goal_callback(self, goal):

        x = goal.pose.position.x
        y = goal.pose.position.y
        theta = self.quat_to_yaw(goal.pose.orientation)

        self.goal = np.array([[x], [y], [theta]])

        rospy.loginfo("set neupan goal: {}".format([x, y, theta]))
        rospy.loginfo_throttle(0.1, "initial path update from goal position")
        self.neupan_planner.update_initial_path_from_goal(self.robot_state, self.goal)
        self.neupan_planner.reset()

    def quat_to_yaw_list(self, quater):

        x = quater[0]
        y = quater[1]
        z = quater[2]
        w = quater[3]

        yaw = atan2(2 * (w * z + x * y), 1 - 2 * (pow(z, 2) + pow(y, 2)))

        return yaw

    # generate ros message
    def generate_path_msg(self, path_list):

        path = Path()
        path.header.frame_id = self.config["frame"]["map"]
        path.header.stamp = rospy.Time.now()
        path.header.seq = 0

        for index, point in enumerate(path_list):
            ps = PoseStamped()
            ps.header.frame_id = self.config["frame"]["map"]
            ps.header.seq = index

            ps.pose.position.x = point[0, 0]
            ps.pose.position.y = point[1, 0]
            ps.pose.orientation = self.yaw_to_quat(point[2, 0])

            path.poses.append(ps)

        return path

    def generate_twist_msg(self, vel):

        if vel is None:
            return Twist()

        speed = vel[0, 0]
        steer = vel[1, 0]

        if self.stop or self.arrive:
            # print('stop flag true')
            return Twist()

        else:
            action = Twist()

            action.linear.x = speed
            action.angular.z = steer

            return action

    def generate_dune_points_markers_msg(self):

        marker_array = MarkerArray()

        if self.neupan_planner.dune_points is None:
            return
        else:
            points = self.neupan_planner.dune_points

            for index, point in enumerate(points.T):

                marker = Marker()
                marker.header.frame_id = self.config["frame"]["map"]
                marker.header.seq = 0
                marker.header.stamp = rospy.get_rostime()

                marker.scale.x = self.config["marker_size"]
                marker.scale.y = self.config["marker_size"]
                marker.scale.z = self.config["marker_size"]
                marker.color.a = 1.0

                marker.color.r = 160 / 255
                marker.color.g = 32 / 255
                marker.color.b = 240 / 255

                marker.id = index
                marker.type = 1
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                marker.pose.position.z = 0.3
                marker.pose.orientation = Quaternion()

                marker_array.markers.append(marker)

            return marker_array

    def generate_nrmp_points_markers_msg(self):

        marker_array = MarkerArray()

        if self.neupan_planner.nrmp_points is None:
            return
        else:
            points = self.neupan_planner.nrmp_points

            for index, point in enumerate(points.T):

                marker = Marker()
                marker.header.frame_id = self.config["frame"]["map"]
                marker.header.seq = 0
                marker.header.stamp = rospy.get_rostime()

                marker.scale.x = self.config["marker_size"]
                marker.scale.y = self.config["marker_size"]
                marker.scale.z = self.config["marker_size"]
                marker.color.a = 1.0

                marker.color.r = 255 / 255
                marker.color.g = 128 / 255
                marker.color.b = 0 / 255

                marker.id = index
                marker.type = 1
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                marker.pose.position.z = 0.3
                marker.pose.orientation = Quaternion()

                marker_array.markers.append(marker)

            return marker_array

    def generate_robot_marker_msg(self):

        marker = Marker()

        marker.header.frame_id = self.config["frame"]["map"]
        marker.header.seq = 0
        marker.header.stamp = rospy.get_rostime()

        marker.color.a = 1.0
        marker.color.r = 0 / 255
        marker.color.g = 255 / 255
        marker.color.b = 0 / 255

        marker.id = 0

        if self.neupan_planner.robot.shape == "rectangle":
            length = self.neupan_planner.robot.length
            width = self.neupan_planner.robot.width
            wheelbase = self.neupan_planner.robot.wheelbase

            marker.scale.x = length
            marker.scale.y = width
            marker.scale.z = self.config["marker_z"]

            marker.type = 1

            x = self.robot_state[0, 0]
            y = self.robot_state[1, 0]
            theta = self.robot_state[2, 0]

            if self.neupan_planner.robot.kinematics == "acker":
                diff_len = (length - wheelbase) / 2
                marker_x = x + diff_len * cos(theta)
                marker_y = y + diff_len * sin(theta)
            else:
                marker_x = x
                marker_y = y

            marker.pose.position.x = marker_x
            marker.pose.position.y = marker_y
            marker.pose.position.z = 0
            marker.pose.orientation = self.yaw_to_quat(self.robot_state[2, 0])

        return marker

    @staticmethod
    def yaw_to_quat(yaw):

        quater = Quaternion()

        quater.x = 0
        quater.y = 0
        quater.z = sin(yaw / 2)
        quater.w = cos(yaw / 2)

        return quater

    @staticmethod
    def quat_to_yaw(quater):

        x = quater.x
        y = quater.y
        z = quater.z
        w = quater.w

        raw = atan2(2 * (w * z + x * y), 1 - 2 * (pow(z, 2) + pow(y, 2)))

        return raw


if __name__ == "__main__":
    try:
        neupan_node = neupan_core()
        neupan_node.run()
    except rospy.ROSInterruptException:
        pass
