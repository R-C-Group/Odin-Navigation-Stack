/*
 * Copyright 2025 Manifold Tech Ltd.(www.manifoldtech.com.co)
 * 
 * 局部代价地图 (Local Costmap) 实现。
 * 该模块负责管理机器人周边的障碍物信息，支持障碍物记忆衰减、实时扫描更新和障碍物膨胀。
 */

#include "model_planner/local_costmap.h"
#include <cmath>
#include <algorithm>
#include <ros/ros.h>

namespace model_planner {

LocalCostmap::LocalCostmap(int width, int height, float resolution, float origin_x, float origin_y)
    : width_(width), height_(height), resolution_(resolution),
      data_(width * height, 0), decay_factor_(0.95f),
      robot_x_(origin_x), robot_y_(origin_y), robot_yaw_(0.0f) {}

void LocalCostmap::setRobotPose(float robot_x, float robot_y, float robot_yaw) {
    robot_x_ = robot_x;
    robot_y_ = robot_y;
    robot_yaw_ = robot_yaw;
}

void LocalCostmap::updateFromScan(const std::vector<float>& ranges,
                                   float angle_min, float angle_max, float angle_increment,
                                   float range_min, float range_max) {
    // 1. 【核心：时间衰减 (Temporal Decay)】
    // 使得旧的障碍物数据随时间自动“淡出”。这对于处理动态障碍物（如行人）至关重要。
    // 如果不进行衰减，残留在地图上的旧数据会形成“鬼影”，阻塞机器人本可通行的路径。
    applyDecay();
    
    // 该方法执行从原始传感器数据到环境语义地图的完整映射流：
    // A. 传感器坐标系 (极坐标) -> 机器人坐标系 (笛卡尔)
    // B. 机器人坐标系 -> 统一地图坐标系 (World Frame)
    // C. 地图坐标系 -> 离散栅格坐标 (Grid Coordinates)
    
     // 调试输出，正常运行时注释掉
    static int update_count = 0;
    // update_count++;
    // if (update_count % 50 == 0) {
    //     std::cout << "[LocalCostmap] ===== Update #" << update_count << " =====" << std::endl;
    //     std::cout << "[LocalCostmap] Robot Pose: x=" << robot_x_ << ", y=" << robot_y_ 
    //               << ", yaw=" << robot_yaw_ << " rad (" << (robot_yaw_ * 180 / 3.14159) << " deg)" << std::endl;
    //     std::cout << "[LocalCostmap] Map size: " << width_ << " x " << height_ 
    //               << ", resolution: " << resolution_ << " m/cell" << std::endl;
    // }
    
    // 整个流程：
    // 1. 激光扫描是在机器人坐标系中（极坐标）
    // 2. 转换为全局坐标系（笛卡尔坐标）
    // 3. 再转换为栅格坐标
    // 遍历所有激光扫描射线 / Iterate through all laser scan beams
    for (size_t i = 0; i < ranges.size(); ++i) {
        float range = ranges[i];
        
        // 预处理：过滤无效数据 (NaN, Inf) 及超出传感器量程限制的数据
        if (range < range_min || range > range_max || std::isnan(range) || std::isinf(range)) {
            continue;
        }
        
        // 获取当前射线在传感器坐标系中的角度
        float angle = angle_min + i * angle_increment;
        
        // 步骤 A: 极坐标系转换为机器人中心笛卡尔坐标系
        // 公式：x = r * cos(theta), y = r * sin(theta)
        float x_robot = range * std::cos(angle);
        float y_robot = range * std::sin(angle);
        
        // 步骤 B: 机器人坐标系变换到全局世界坐标系
        // 使用当前机器人在世界坐标系下的位置 (robot_x, robot_y) 和朝向 (robot_yaw)
        // 变换阵：[x_g, y_g]^T = R(yaw) * [x_r, y_r]^T + [robot_x, robot_y]^T
        float x_global = x_robot * std::cos(robot_yaw_) - y_robot * std::sin(robot_yaw_) + robot_x_;
        float y_global = x_robot * std::sin(robot_yaw_) + y_robot * std::cos(robot_yaw_) + robot_y_;
        
        // 步骤 C: 将连续的世界坐标映射到离散的栅格索引
        int grid_x, grid_y;
        if (globalToGrid(x_global, y_global, grid_x, grid_y)) {
            // 将击中点（障碍物所在位置）标记为硬障碍物（最高代价值 255）
            setCost(grid_x, grid_y, 255);
            obstacle_count++;

            // 调试输出，正常运行时注释掉
            // if (update_count % 50 == 0 && obstacle_count <= 5) {
            //     std::cout << "[LocalCostmap] Obstacle " << obstacle_count << ": robot(" << x_robot << "," << y_robot 
            //               << ") -> global(" << x_global << "," << y_global << ") -> grid(" << grid_x << "," << grid_y << ")" << std::endl;
            // }
            
            // 在激光束路径上标记为自由空间（Bresenham线算法）
            
            // 2. 【核心：自由空间清理 (Free Space Clearing)】
            // 当激光击中远处的障碍物时，说明该射线经过的所有路径在物理上都应该是“空闲”的。
            // 使用 Bresenham 直线扫描算法，在地图上强行抹除射线路径上的旧障碍物。
            int grid_robot_x, grid_robot_y;
            if (globalToGrid(robot_x_, robot_y_, grid_robot_x, grid_robot_y)) {
                int start_x = grid_robot_x;
                int start_y = grid_robot_y;
                int end_x = grid_x;
                int end_y = grid_y;
            
                // Bresenham 算法实现：一种高效的能在整数栅格中绘制/查找线段的算法
                int dx = std::abs(end_x - start_x);
                int dy = std::abs(end_y - start_y);
                int sx = (end_x > start_x) ? 1 : -1;
                int sy = (end_y > start_y) ? 1 : -1;
                int err = dx - dy;
                
                int x = start_x;
                int y = start_y;
            
                while (true) {
                    // 如果已经追踪到传感器击中的终点（障碍物点），停止清理
                    if (x == end_x && y == end_y) break;

                    // 检查当前栅格点的原代价
                    // 仅当该点不是“确定的强障碍物”(例如代价 < 200)时，将其设为自由空间 (0)
                    if (getCost(x, y) < 200) { 
                        setCost(x, y, 0);
                    }
                    
                    // Bresenham 判定下一步走向
                    int e2 = 2 * err;
                    if (e2 > -dy) {
                        err -= dy;
                        x += sx;
                    }
                    if (e2 < dx) {
                        err += dx;
                        y += sy;
                    }
                }
            }
        }
    }

    // 调试输出，正常运行时注释掉
    // if (update_count % 50 == 0) {
    //     std::cout << "[LocalCostmap] Found " << obstacle_count << " obstacles in this scan" << std::endl;
    // }
}

void LocalCostmap::clearMemory() {
    std::fill(data_.begin(), data_.end(), 0);
}

void LocalCostmap::applyDecay() {
    // 对所有栅格应用衰减因子
    for (auto& cost : data_) {
        if (cost > 0) {
            // 衰减代价值（但保留一定的记忆）
            cost = static_cast<uint8_t>(cost * decay_factor_);
        }
    }
}

uint8_t LocalCostmap::getCost(int x, int y) const {
    if (!isInBounds(x, y)) {
        return 0;
    }
    return data_[y * width_ + x];
}

void LocalCostmap::setCost(int x, int y, uint8_t cost) {
    if (!isInBounds(x, y)) {
        return;
    }
    data_[y * width_ + x] = cost;
}

bool LocalCostmap::globalToGrid(float global_x, float global_y, int& grid_x, int& grid_y) const {
    // 获取地图中心
    int center_x = width_ / 2;
    int center_y = height_ / 2;
    
    // 相对于机器人位置的坐标差
    float dx = global_x - robot_x_;
    float dy = global_y - robot_y_;
    
    // 转换为栅格坐标（以机器人为中心）
    grid_x = center_x + static_cast<int>(dx / resolution_);
    grid_y = center_y + static_cast<int>(dy / resolution_);
    
    bool in_bounds = isInBounds(grid_x, grid_y);
    
    // 调试输出，正常运行时注释掉
    // static int debug_count = 0;
    // if (debug_count++ % 500 == 0) {
    //     std::cout << "[GlobalToGrid] global(" << global_x << "," << global_y 
    //               << ") robot(" << robot_x_ << "," << robot_y_ 
    //               << ") delta(" << dx << "," << dy 
    //               << ") -> grid(" << grid_x << "," << grid_y 
    //               << ") [" << (in_bounds ? "IN" : "OUT") << "]" << std::endl;
    // }
    
    return in_bounds;
}

void LocalCostmap::gridToGlobal(int grid_x, int grid_y, float& global_x, float& global_y) const {
    // 获取地图中心
    int center_x = width_ / 2;
    int center_y = height_ / 2;
    
    // 相对于中心的栅格偏移
    float dx = (grid_x - center_x + 0.5f) * resolution_;
    float dy = (grid_y - center_y + 0.5f) * resolution_;
    
    // 转换为全局坐标系（以机器人为中心）
    global_x = robot_x_ + dx;
    global_y = robot_y_ + dy;
}

bool LocalCostmap::pointToGrid(float robot_x, float robot_y, int& grid_x, int& grid_y) const {
    // 兼容层：将机器人坐标系转换为全局坐标系
    float cos_yaw = std::cos(robot_yaw_);
    float sin_yaw = std::sin(robot_yaw_);
    float global_x = robot_x_ + robot_x * cos_yaw - robot_y * sin_yaw;
    float global_y = robot_y_ + robot_x * sin_yaw + robot_y * cos_yaw;
    
    // 调试输出，正常运行时注释掉
    // static int debug_count = 0;
    // if (debug_count++ % 500 == 0) {
    //     ROS_INFO("[PointToGrid] robot(%.3f,%.3f) + pose(%.3f,%.3f,%.3f) -> global(%.3f,%.3f)",
    //              robot_x, robot_y, robot_x_, robot_y_, robot_yaw_, global_x, global_y);
    // }
    
    // 转换为栅格坐标
    return globalToGrid(global_x, global_y, grid_x, grid_y);
}

void LocalCostmap::gridToPoint(int grid_x, int grid_y, float& robot_x, float& robot_y) const {
    // 兼容层：将栅格坐标转换为全局坐标系，再转换为机器人坐标系
    float global_x, global_y;
    gridToGlobal(grid_x, grid_y, global_x, global_y);
    
    // 转换为机器人坐标系
    float dx = global_x - robot_x_;
    float dy = global_y - robot_y_;
    float cos_yaw = std::cos(robot_yaw_);
    float sin_yaw = std::sin(robot_yaw_);
    robot_x = dx * cos_yaw + dy * sin_yaw;
    robot_y = -dx * sin_yaw + dy * cos_yaw;
}

void LocalCostmap::inflate(float radius) {
    // 障碍物膨胀 (Inflation Algorithm)
    // 目的是为硬障碍物周围创建一层“代价渐变缓冲区”，帮助规划器保持安全距离。
    
    // 膨胀半径转换为栅格数 / Radius in cells
    int inflate_radius = static_cast<int>(std::ceil(radius / resolution_));
    
    // 创建临时地图快照 / Temporary copy for thread-safety or multi-pass avoidance
    std::vector<uint8_t> inflated_data = data_;
    
    // 遍历地图所有栅格 / Iterate through the entire grid
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            // 发现硬障碍物 / Found an original obstacle cell
            if (data_[y * width_ + x] >= 200) {
                // 在定义的半径范围内进行膨胀 / Inflate within the radius
                for (int dy = -inflate_radius; dy <= inflate_radius; ++dy) {
                    for (int dx = -inflate_radius; dx <= inflate_radius; ++dx) {
                        int nx = x + dx;
                        int ny = y + dy;
                        
                        // 检查边界 / Bounds check
                        if (isInBounds(nx, ny)) {
                            // 计算当前点到障碍物中心的欧几里得距离
                            int dist_sq = dx * dx + dy * dy;
                            int radius_sq = inflate_radius * inflate_radius;
                            
                            if (dist_sq <= radius_sq) {
                                // 创建代价梯度：距离越近，代价越高 / Linear cost gradient
                                float dist = std::sqrt(static_cast<float>(dist_sq));
                                // 代价值从 254 (贴近障碍物) 线性降到 0 (半径边缘)
                                uint8_t cost = static_cast<uint8_t>(
                                    254.0f * (1.0f - dist / inflate_radius)
                                );
                                // 保留该位置的最大代价值 / Keep maximum cost seen so far
                                inflated_data[ny * width_ + nx] = 
                                    std::max(inflated_data[ny * width_ + nx], cost);
                            }
                        }
                    }
                }
            }
        }
    }
    
    data_ = inflated_data;
}

bool LocalCostmap::isInBounds(int x, int y) const {
    return x >= 0 && x < width_ && y >= 0 && y < height_;
}

} // namespace model_planner
