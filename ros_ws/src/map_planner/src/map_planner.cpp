/*
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

#include "map_planner.h"
#include "ros/console.h"

#include <queue>
#include <unordered_map>
#include <vector>
#include <limits>
#include <cmath>

#include <std_msgs/Bool.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace {
// A*算法中的节点结构
// Node structure for A* algorithm
struct Node {
  int index;    // 栅格地图中的索引 / Grid map index
  double g;     // 从起点到当前节点的实际代价 / Actual cost from start to current node
  double f;     // f = g + h，总代价（实际代价 + 启发式代价）/ Total cost (actual + heuristic)
  bool operator>(const Node& other) const { return f > other.f; }
};

// 对角线距离（√2）用于8连通栅格
// Diagonal distance (√2) for 8-connected grid
constexpr double SQRT2 = 1.41421356237;
}  // namespace

/**
 * 地图规划器构造函数
 * MapPlanner constructor
 * 
 * @param nh ROS节点句柄 / ROS node handle
 * @param private_nh 私有节点句柄，用于参数配置 / Private node handle for parameters
 */
MapPlanner::MapPlanner(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh),
      private_nh_(private_nh),
      tf_listener_(tf_buffer_) {
  // 从参数服务器读取配置 / Load configuration from parameter server
  private_nh_.param("inflation_radius", inflation_radius_, inflation_radius_);  // 障碍物膨胀半径(米) / Obstacle inflation radius (meters)
  private_nh_.param("obstacle_threshold", obstacle_threshold_, obstacle_threshold_);  // 障碍物阈值 / Obstacle threshold
  private_nh_.param("publish_path", publish_path_, publish_path_);  // 是否发布路径 / Whether to publish path
  private_nh_.param("service_name", plan_service_name_, plan_service_name_);  // 规划服务名称 / Planning service name
  
  // 初始化发布者 / Initialize publishers
  path_pub_ = nh_.advertise<nav_msgs::Path>("initial_path", 1, true);  // 发布全局路径 / Publish global path
  inflated_map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("inflated_map", 1, true);  // 发布膨胀后的地图 / Publish inflated map
  plan_result_pub_ = nh_.advertise<std_msgs::Bool>("/map_planner/result", 1, true);  // 发布规划结果 / Publish planning result
  
  // 初始化订阅者 / Initialize subscribers
  map_sub_ = nh_.subscribe("map", 1, &MapPlanner::mapCallback, this);  // 订阅栅格地图 / Subscribe to occupancy grid map
  goal_sub_ = nh_.subscribe("/move_base_simple/goal", 1, &MapPlanner::goalCallback, this);  // 订阅目标点 / Subscribe to goal pose
  
  // 初始化服务 / Initialize service
  plan_service_ = private_nh_.advertiseService(plan_service_name_, &MapPlanner::planService, this);  // 路径规划服务 / Path planning service
}

/**
 * 地图回调函数：接收栅格地图并进行障碍物膨胀处理
 * Map callback: receives occupancy grid map and performs obstacle inflation
 * 
 * @param msg 输入的栅格地图消息 / Input occupancy grid message
 */
void MapPlanner::mapCallback(const nav_msgs::OccupancyGridConstPtr& msg) {
  map_ = *msg;
  
  // 验证地图分辨率 / Validate map resolution
  if (map_.info.resolution <= 0.0) {
    ROS_WARN_THROTTLE(5.0, "Map resolution invalid.");
    map_ready_ = false;
    return;
  }
  
  // 计算膨胀半径对应的栅格数量 / Calculate inflation radius in grid cells
  inflation_cells_ = std::max(1, static_cast<int>(std::ceil(inflation_radius_ / map_.info.resolution)));
  
  // 执行障碍物膨胀，增加安全边界 / Perform obstacle inflation to add safety margin
  inflateMap();
  
  map_ready_ = true;
  publishInflatedMap();
  ROS_INFO_ONCE("Inflated map ready for planning.");
}

void MapPlanner::publishInflatedMap() {
  if (!inflated_map_pub_) return;
  nav_msgs::OccupancyGrid inflated = map_;
  inflated.header.stamp = ros::Time::now();
  inflated.data = inflated_data_;
  inflated_map_pub_.publish(inflated);
}

void MapPlanner::publishPlanResult(bool success) {
  if (!plan_result_pub_) return;
  std_msgs::Bool msg;
  msg.data = success;
  plan_result_pub_.publish(msg);
}

void MapPlanner::goalCallback(const geometry_msgs::PoseStampedConstPtr& goal) {
  if (!map_ready_) {
    ROS_WARN_THROTTLE(2.0, "Map not ready for planning.");
    publishPlanResult(false);
    return;
  }
  geometry_msgs::PoseStamped start_pose;
  if (!getRobotPose(start_pose)) {
    ROS_WARN_THROTTLE(2.0, "Unable to get robot pose.");
    publishPlanResult(false);
    return;
  }
  geometry_msgs::PoseStamped goal_in_map = *goal;
  if (goal->header.frame_id != map_.header.frame_id) {
    try {
      // tf_buffer_.transform(*goal, goal_in_map, map_.header.frame_id, ros::Duration(0.1));
      geometry_msgs::TransformStamped tf_stamped =
      tf_buffer_.lookupTransform(map_.header.frame_id,        // target frame
                                goal->header.frame_id,      // source frame
                                ros::Time(0),               // latest available
                                ros::Duration(0.1));        // timeout
      tf2::doTransform(*goal, goal_in_map, tf_stamped);
    } catch (const tf2::TransformException& ex) {
      ROS_WARN_THROTTLE(2.0, "Goal transform failed: %s", ex.what());
      publishPlanResult(false);
      return;
    }
  }
  nav_msgs::Path path;
  const bool success = plan(start_pose, goal_in_map, path);
  publishPlanResult(success);
  if (!success) {
    ROS_WARN("Failed to plan a path.");
    return;
  }
  path_pub_.publish(path);
}

bool MapPlanner::getRobotPose(geometry_msgs::PoseStamped& pose) const {
  try {
    geometry_msgs::TransformStamped tf = tf_buffer_.lookupTransform(map_.header.frame_id, "base_link", ros::Time(0), ros::Duration(0.2));
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
 * A*路径规划核心算法
 * A* path planning core algorithm
 * 
 * @param start 起点位姿（世界坐标系）/ Start pose in world coordinates
 * @param goal 目标位姿（世界坐标系）/ Goal pose in world coordinates
 * @param path 输出的路径 / Output path
 * @return 规划是否成功 / Whether planning succeeded
 */
bool MapPlanner::plan(const geometry_msgs::PoseStamped& start, const geometry_msgs::PoseStamped& goal, nav_msgs::Path& path) {
  // 将世界坐标转换为栅格坐标 / Convert world coordinates to grid coordinates
  int start_x, start_y, goal_x, goal_y;
  if (!worldToMap(start.pose.position, start_x, start_y) || !worldToMap(goal.pose.position, goal_x, goal_y)) {
    ROS_WARN("Start or goal outside the map.");
    return false;
  }
  
  const int width = static_cast<int>(map_.info.width);
  const int height = static_cast<int>(map_.info.height);
  const int start_index = toIndex(start_x, start_y);
  const int goal_index = toIndex(goal_x, goal_y);
  
  // 检查起点和终点是否可通行 / Check if start and goal are free
  if (!isFree(start_index) || !isFree(goal_index)) {
    ROS_WARN("Start or goal is occupied.");
    return false;
  }

  // A*算法数据结构初始化 / A* algorithm data structure initialization
  std::vector<double> g_score(width * height, std::numeric_limits<double>::infinity());  // g(n): 从起点到n的实际代价 / Actual cost from start to n
  std::vector<int> came_from(width * height, -1);  // 记录路径：每个节点的父节点 / Record path: parent of each node
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_set;  // 优先队列：按f值排序的待探索节点 / Priority queue: nodes to explore sorted by f-value

  // 启发式函数：欧几里得距离 / Heuristic function: Euclidean distance
  auto heuristic = [&](int mx, int my) {
    const double dx = static_cast<double>(mx - goal_x);
    const double dy = static_cast<double>(my - goal_y);
    return std::hypot(dx, dy);  // h(n) = sqrt(dx^2 + dy^2)
  };

  // 初始化起点 / Initialize start node
  g_score[start_index] = 0.0;
  open_set.push({start_index, 0.0, heuristic(start_x, start_y)});

  // 8连通邻域：右、左、上、下、右上、右下、左上、左下 / 8-connected neighbors: R, L, U, D, RU, RD, LU, LD
  const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
  const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  const double costs[8] = {1.0, 1.0, 1.0, 1.0, SQRT2, SQRT2, SQRT2, SQRT2};  // 水平/垂直代价=1，对角线代价=√2 / Horizontal/vertical cost=1, diagonal cost=√2

  // A*主循环：探索节点直到找到目标 / A* main loop: explore nodes until goal is found
  while (!open_set.empty()) {
    Node current = open_set.top();  // 取出f值最小的节点 / Get node with smallest f-value
    open_set.pop();
    if (current.index == goal_index) break;  // 到达目标，规划完成 / Reached goal, planning complete

    // 将1D索引转换为2D坐标 / Convert 1D index to 2D coordinates
    int cx = current.index % width;
    int cy = current.index / width;

    // 遍历8个邻居节点 / Iterate through 8 neighbors
    for (int i = 0; i < 8; ++i) {
      const int nx = cx + dx[i];
      const int ny = cy + dy[i];
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;  // 边界检查 / Boundary check
      const int n_index = toIndex(nx, ny);
      if (!isFree(n_index)) continue;  // 跳过障碍物 / Skip obstacles

      // 计算经过当前节点到邻居的代价 / Calculate cost to neighbor through current node
      const double tentative_g = g_score[current.index] + costs[i];
      if (tentative_g < g_score[n_index]) {  // 找到更优路径 / Found better path
        came_from[n_index] = current.index;  // 更新父节点 / Update parent
        g_score[n_index] = tentative_g;  // 更新g值 / Update g-score
        const double f_score = tentative_g + heuristic(nx, ny);  // 计算f = g + h / Calculate f = g + h
        open_set.push({n_index, tentative_g, f_score});  // 加入待探索队列 / Add to open set
      }
    }
  }

  if (came_from[goal_index] == -1 && goal_index != start_index) {
    return false;
  }

  std::vector<int> index_path;
  for (int current = goal_index; current != -1; current = came_from[current]) {
    index_path.push_back(current);
    if (current == start_index) break;
  }
  if (index_path.back() != start_index) return false;
  std::reverse(index_path.begin(), index_path.end());

  path.header.stamp = ros::Time::now();
  path.header.frame_id = map_.header.frame_id;
  path.poses.reserve(index_path.size());
  for (int idx : index_path) {
    const int mx = idx % width;
    const int my = idx / width;
    geometry_msgs::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position = mapToWorld(mx, my);
    pose.pose.orientation.w = 1.0;
    path.poses.push_back(pose);
  }
  return true;
}

/**
 * 障碍物膨胀函数：在障碍物周围创建安全边界
 * Obstacle inflation: creates safety margin around obstacles
 * 
 * 算法：对于每个障碍物，将其周围指定半径内的所有栅格标记为障碍物
 * Algorithm: for each obstacle, mark all cells within specified radius as obstacles
 */
void MapPlanner::inflateMap() {
  inflated_data_ = map_.data;
  if (inflation_cells_ <= 0) return;

  const int width = static_cast<int>(map_.info.width);
  const int height = static_cast<int>(map_.info.height);
  std::vector<int8_t> result = inflated_data_;  // 复制原始地图数据 / Copy original map data

  // 遍历地图中的每个栅格 / Iterate through each cell in the map
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int index = toIndex(x, y);
      // 只处理障碍物栅格 / Only process obstacle cells
      if (map_.data[index] < obstacle_threshold_ || map_.data[index] < 0) continue;
      
      // 在障碍物周围的矩形区域内进行膨胀 / Inflate in rectangular region around obstacle
      for (int dy = -inflation_cells_; dy <= inflation_cells_; ++dy) {
        for (int dx = -inflation_cells_; dx <= inflation_cells_; ++dx) {
          const int nx = x + dx;
          const int ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;  // 边界检查 / Boundary check
          
          // 使用圆形膨胀区域（欧几里得距离） / Use circular inflation region (Euclidean distance)
          if (std::hypot(dx, dy) * map_.info.resolution > inflation_radius_) continue;
          result[toIndex(nx, ny)] = 100;  // 标记为障碍物 / Mark as obstacle
        }
      }
    }
  }
  inflated_data_.swap(result);  // 交换数据，更新膨胀后的地图 / Swap data to update inflated map
}

bool MapPlanner::worldToMap(const geometry_msgs::Point& point, int& mx, int& my) const {
  if (!map_ready_) return false;
  const double origin_x = map_.info.origin.position.x;
  const double origin_y = map_.info.origin.position.y;
  const double resolution = map_.info.resolution;

  mx = static_cast<int>(std::floor((point.x - origin_x) / resolution));
  my = static_cast<int>(std::floor((point.y - origin_y) / resolution));
  return mx >= 0 && my >= 0 && mx < static_cast<int>(map_.info.width) && my < static_cast<int>(map_.info.height);
}

geometry_msgs::Point MapPlanner::mapToWorld(int mx, int my) const {
  geometry_msgs::Point point;
  point.x = map_.info.origin.position.x + (mx + 0.5) * map_.info.resolution;
  point.y = map_.info.origin.position.y + (my + 0.5) * map_.info.resolution;
  point.z = 0.0;
  return point;
}

bool MapPlanner::isFree(int index) const {
  const int8_t value = inflated_data_[index];
  if (value < 0) return false;
  return value < obstacle_threshold_;
}

bool MapPlanner::planService(map_planner::PlanPath::Request& req, map_planner::PlanPath::Response& res) {
  if (!map_ready_) {
    ROS_WARN_THROTTLE(2.0, "Map not ready for planning.");
    publishPlanResult(false);
    return false;
  }
  geometry_msgs::PoseStamped start_pose;
  if (!getRobotPose(start_pose)) {
    ROS_WARN_THROTTLE(2.0, "Unable to get robot pose.");
    publishPlanResult(false);
    return false;
  }
  geometry_msgs::PoseStamped goal = req.goal;
  if (goal.header.frame_id.empty()) {
    goal.header.frame_id = map_.header.frame_id;
  }
  if (goal.header.frame_id != map_.header.frame_id) {
    ROS_WARN("Goal frame (%s) does not match map frame (%s).", goal.header.frame_id.c_str(), map_.header.frame_id.c_str());
    publishPlanResult(false);
    return false;
  }
  nav_msgs::Path path;
  const bool success = plan(start_pose, goal, path);
  publishPlanResult(success);
  if (!success) {
    ROS_WARN("Failed to plan a path.");
    return false;
  }
  res.path = path;
  if (publish_path_) path_pub_.publish(path);
  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "map_planner");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  MapPlanner planner(nh, private_nh);
  ros::spin();
  return 0;
}
