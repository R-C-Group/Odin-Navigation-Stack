/*
 * 目标状态机 - 负责管理导航目标和自动重新规划
 * Goal State Machine - Manages navigation goals and automatic replanning
 * 
 * Copyright 2025 Manifold Tech Ltd.(www.manifoldtech.com.co)
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "goal_state_machine.h"

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

/**
 * 目标状态机构造函数
 * Goal State Machine constructor
 * 
 * 功能 / Function:
 * - 订阅目标位置 / Subscribe to goal positions
 * - 订阅到达信号 / Subscribe to arrival signals
 * - 连接路径规划服务 / Connect to path planning service
 * 
 * @param nh ROS节点句柄 / ROS node handle
 * @param private_nh 私有节点句柄 / Private node handle
 */
GoalStateMachine::GoalStateMachine(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh),
      private_nh_(private_nh),
      tf_listener_(tf_buffer_) {
  // 读取参数 / Load parameters
  private_nh_.param("goal_tolerance", goal_tolerance_, goal_tolerance_);  // 目标容差（米） / Goal tolerance (meters)
  private_nh_.param("plan_service", plan_service_name_, plan_service_name_);  // 规划服务名称 / Planning service name
  
  // 订阅到达信号（由NeuPAN发布） / Subscribe to arrival signal (published by NeuPAN)
  arrive_sub_ = nh_.subscribe("/neupan/arrive", 1, &GoalStateMachine::arriveCallback, this);
  
  // 订阅目标点（来自RViz的"2D Nav Goal"） / Subscribe to goal (from RViz "2D Nav Goal")
  goal_sub_ = nh_.subscribe("/move_base_simple/goal", 1, &GoalStateMachine::goalCallback, this);
  
  // 创建路径规划服务客户端 / Create path planning service client
  plan_client_ = nh_.serviceClient<map_planner::PlanPath>(plan_service_name_);
}

/**
 * 目标回调函数：接收新目标并转换到地图坐标系
 * Goal callback: receives new goal and transforms to map frame
 * 
 * @param goal 目标位姿消息 / Goal pose message
 */
void GoalStateMachine::goalCallback(const geometry_msgs::PoseStampedConstPtr& goal) {
  geometry_msgs::PoseStamped map_goal;
  
  // 检查目标坐标系 / Check goal frame
  if (goal->header.frame_id.empty() || goal->header.frame_id == "map") {
    // 已经在地图坐标系中，直接使用 / Already in map frame, use directly
    map_goal = *goal;
    map_goal.header.frame_id = "map";
  } else {
    // 需要坐标变换到地图坐标系 / Need to transform to map frame
    try {
      tf_buffer_.transform(*goal, map_goal, "map", ros::Duration(0.2));
    } catch (const tf2::TransformException& ex) {
      ROS_WARN_THROTTLE(2.0, "Goal transform failed: %s", ex.what());
      return;
    }
  }
  
  // 保存目标并标记为有效 / Save goal and mark as valid
  last_goal_ = map_goal;
  have_goal_ = true;
}

/**
 * 到达回调函数：检测机器人是否真正到达目标，如果还有距离则触发重新规划
 * Arrival callback: checks if robot truly reached goal, triggers replanning if still far away
 * 
 * 工作原理 / Working principle:
 * NeuPAN可能在距离目标还有一段距离时就发出到达信号（因为局部路径结束）
 * NeuPAN may send arrival signal while still far from goal (because local path ended)
 * 此时需要检查实际距离，如果超过容差则调用全局规划器重新规划
 * Need to check actual distance, if exceeds tolerance then call global planner to replan
 * 
 * @param msg 空消息（只是一个触发信号） / Empty message (just a trigger signal)
 */
void GoalStateMachine::arriveCallback(const std_msgs::EmptyConstPtr&) {
  // 检查是否有存储的目标 / Check if there is a stored goal
  if (!have_goal_) {
    ROS_WARN_THROTTLE(2.0, "Arrival received without a stored goal.");
    return;
  }
  
  // 获取机器人当前位置 / Get robot current position
  geometry_msgs::PoseStamped current_pose;
  if (!getRobotPose(current_pose)) return;

  // 计算与目标的距离 / Calculate distance to goal
  const double dx = last_goal_.pose.position.x - current_pose.pose.position.x;
  const double dy = last_goal_.pose.position.y - current_pose.pose.position.y;
  const double distance = std::hypot(dx, dy);  // 欧几里得距离 / Euclidean distance

  // 如果已经在容差范围内，认为到达 / If within tolerance, consider arrived
  if (distance <= goal_tolerance_) {
    ROS_INFO("Robot is within %.2f m of goal.", goal_tolerance_);
    return;
  }

  // 如果还有距离，等待规划服务 / If still far away, wait for planning service
  if (!plan_client_.exists() && !plan_client_.waitForExistence(ros::Duration(0.5))) {
    ROS_WARN("Plan service unavailable.");
    return;
  }

  // 调用全局规划器重新规划 / Call global planner to replan
  map_planner::PlanPath srv;
  srv.request.goal = last_goal_;
  if (plan_client_.call(srv)) {
    ROS_INFO("Requested replanning toward goal (distance %.2f m).", distance);
  } else {
    ROS_WARN("Failed to call plan service.");
  }
}

/**
 * 获取机器人当前位姿（通过TF变换）
 * Get robot current pose (via TF transform)
 * 
 * @param pose 输出参数，存储机器人位姿 / Output parameter, stores robot pose
 * @return 成功返回true，失败返回false / Returns true on success, false on failure
 */
bool GoalStateMachine::getRobotPose(geometry_msgs::PoseStamped& pose) const {
  try {
    // 查找 map -> base_link 的TF变换 / Look up map -> base_link TF transform
    geometry_msgs::TransformStamped tf = tf_buffer_.lookupTransform(
        "map",       // 目标坐标系 / Target frame
        "base_link", // 源坐标系（机器人底盘） / Source frame (robot base)
        ros::Time(0),  // 最新时间戳 / Latest timestamp
        ros::Duration(0.2)  // 超时时间 / Timeout
    );
    
    // 填充位姿信息 / Fill pose information
    pose.header = tf.header;
    pose.pose.position.x = tf.transform.translation.x;
    pose.pose.position.y = tf.transform.translation.y;
    pose.pose.position.z = tf.transform.translation.z;
    pose.pose.orientation = tf.transform.rotation;
    return true;
  } catch (const tf2::TransformException& ex) {
    ROS_WARN_THROTTLE(2.0, "TF lookup failed: %s", ex.what());
    return false;
  }
}

/**
 * 主函数
 * Main function
 */
int main(int argc, char** argv) {
  // 初始化ROS节点 / Initialize ROS node
  ros::init(argc, argv, "goal_state_machine");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  
  // 创建目标状态机实例 / Create goal state machine instance
  GoalStateMachine gsm(nh, private_nh);
  
  // 进入ROS事件循环 / Enter ROS event loop
  ros::spin();
  return 0;
}
