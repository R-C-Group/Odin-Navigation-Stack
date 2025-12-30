/*
 * Copyright 2025 Manifold Tech Ltd.(www.manifoldtech.com.co)
 * 
 * 这是一个自定义实现的动态窗口法 (Dynamic Window Approach, DWA) 局部路径规划器。
 * DWA 通过在速度空间采样候选轨迹，并根据避障、路径跟踪、航向指向和速度等指标进行评分，
 * 从而选择最优的控制指令。
 */

#include "model_planner/dwa_planner.h"
#include <ros/ros.h>
#include <cmath>
#include <limits>
#include <algorithm>

namespace model_planner {

static inline double clamp(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

DWAPlanner::DWAPlanner(const LocalCostmap& costmap) : costmap_(costmap) {}

void DWAPlanner::initialize() {
    ROS_INFO("DWA Planner initialized");
}

bool DWAPlanner::plan(const Eigen::Vector3d& current_pose,
                      const Eigen::Vector3d& current_velocity,
                      const std::vector<std::pair<float, float>>& reference_path,
                      Eigen::Vector3d& cmd_vel) {
    reference_path_ = reference_path;
    best_traj_.clear();

    // 1. 【核心：计算动态窗口 (Dynamic Window)】
    // DWA 的核心思想是：机器人由于物理惯性限制，其速度搜索空间应被限制在下一时刻可达的子集内。
    
    // 根据最大加速度(ax_max_)和模拟步长，计算下一时刻线速度的加速度限制范围
    double v_lo = current_velocity(0) - ax_max_ * sim_time_;
    double v_hi = current_velocity(0) + ax_max_ * sim_time_;
    
    // 将线速度搜索区间限制在 [允许的最小速度, 允许的最大速度] 之间
    // allow_backward_ 参数决定是否允许后退（对应线速度负值）
    double v_min = allow_backward_ ? std::max(-vx_max_, v_lo) : std::max(0.0, v_lo);
    double v_max = std::min(vx_max_, v_hi);
    
    // 计算角速度采样范围 [当前角速度 - 最大角加速度*时间, 当前角速度 + 最大角加速度*时间]
    double w_min = current_velocity(2) - alpha_max_ * sim_time_;
    double w_max = current_velocity(2) + alpha_max_ * sim_time_;

    // 最后的安全强制检查：确保速度不超过硬件允许的物理上限(vx_max_/omega_max_)
    if (!allow_backward_) {
        v_min = std::max(0.0, v_min);
    }
    v_max = std::max(v_min, v_max);
    w_min = std::max(-omega_max_, w_min);
    w_max = std::min( omega_max_, w_max);

    // 2. 【核心：采样与暴力搜索 (Sampling & Grid Search)】
    // 在上述计算出的动态窗口速度空间内，均匀采样 Nv 个线速度和 Nw 个角速度组合。
    int Nv = std::max(1, v_samples_);
    int Nw = std::max(1, w_samples_);

    double best_score = -std::numeric_limits<double>::infinity();
    Eigen::Vector2d best_cmd(0, 0); // (v, w)

    // 遍历线速度采样点 / Loop through linear velocity samples
    for (int iv = 0; iv < Nv; ++iv) {
        double v = v_min + (v_max - v_min) * (Nv == 1 ? 0.0 : (double)iv / (Nv - 1));
        // 遍历角速度采样点 / Loop through angular velocity samples
        for (int iw = 0; iw < Nw; ++iw) {
            double w = w_min + (w_max - w_min) * (Nw == 1 ? 0.0 : (double)iw / (Nw - 1));

            // 对每一组候选 (v, w) 进行前向轨迹预测模拟
            // 模拟时间(sim_time_)一般设为 1.5-2.5秒，确保规划具有前瞻性
            std::vector<DWATrajPoint> traj;
            if (!simulateTrajectory(v, w, traj)) {
                // 如果 simulateTrajectory 返回 false，说明该速度指令会导致预测路径内发生碰撞
                continue; 
            }
            
            // 对生成的安全轨迹进行加权评分
            double score = scoreTrajectory(traj);
            
            // 如果得分超过当前最高分，保存为最佳控制指令
            if (score > best_score) {
                best_score = score;
                best_cmd << v, w;
                best_traj_ = std::move(traj);
            }
        }
    }

    // 3. 【异常处理：恢复逻辑 (Recovery Logic)】
    // 如果所有采样速度都会导致碰撞 (best_score 没变)，说明机器人处于受困状态或障碍物极其密集。
    if (best_score == -std::numeric_limits<double>::infinity()) {
        // 进入“旋转恢复”模式：如果开启了此功能，则强制机器人原地旋转指向目标路径的终点方位。
        if (enable_rotate_recovery_ && !reference_path_.empty()) {
            const Eigen::Vector2d goal(reference_path_.back().first, reference_path_.back().second);
            // 计算目标相对于机器人的方位角
            double yaw_to_goal = std::atan2(goal(1), goal(0)); 
            double w = clamp(yaw_to_goal, -omega_max_, omega_max_);
            cmd_vel(0) = 0.0;
            cmd_vel(1) = 0.0;
            // 以 0.6 倍最大角速度原地旋转 / In-place rotation
            cmd_vel(2) = (w >= 0 ? 1.0 : -1.0) * 0.6 * omega_max_;
            best_traj_.clear();
            DWATrajPoint p0(0,0,0,0); best_traj_.push_back(p0);
            return true;
        }
        cmd_vel.setZero(); // 无法找到安全路径，且无法旋转恢复，只能强制停止
        return false;
    }

    // 最终输出：返回评分最高的最优控制指令
    cmd_vel(0) = best_cmd(0);
    cmd_vel(1) = 0.0; // 四足/差速机器人模型，侧向速度置 0
    cmd_vel(2) = best_cmd(1);
    return true;
}

bool DWAPlanner::simulateTrajectory(double v, double w, std::vector<DWATrajPoint>& out) const {
    out.clear();
    // 采用机器人局部坐标系 (Local Base Frame)，当前初始位置始终为 (0, 0, 0)
    double x = 0.0, y = 0.0, th = 0.0; 
    double t = 0.0;
    
    // 计算模拟步数 / Calculate number of steps
    int steps = std::max(1, (int)std::round(sim_time_ / dt_));
    if (no_simulation_) {
        steps = 1; // 极速模式：仅向前预测一步，不进行完整模拟（用于性能优化测试）
    }
    out.reserve(steps + 1);
    out.emplace_back(x, y, th, t);

    // 4. 【核心：运动学积分模拟 (Kinematic Integration)】
    // 使用简单的单轮模型 (Unicycle Model) 进行一阶积分预测未来轨迹
    for (int i = 0; i < steps; ++i) {
        // 此处假设速度在 sim_dt 内恒定
        // 公式：x_{new} = x_{old} + v * cos(th) * dt
        x += v * std::cos(th) * dt_;
        y += v * std::sin(th) * dt_;
        th += w * dt_; // 角速度直接改变航向角
        t += dt_;
        
        DWATrajPoint p(x, y, th, t);
        out.push_back(p);

        // 每一步积分后，立即进行碰撞检查 (Hard Collision Check)
        // 核心：obstacleCost 会根据机器人圆心及外缘栅格代价判断是否安全
        // 代价 >= 1e5 定义为致命区域(Lethal Object)
        if (obstacleCost(p.pose) >= 1e5) {
            return false; // 该轨迹在模拟第 i 步发生碰撞，提前终止
        }
    }
    return true; // 预测周期内全路径安全
}


double DWAPlanner::scoreTrajectory(const std::vector<DWATrajPoint>& traj) const {
    if (traj.empty()) return -std::numeric_limits<double>::infinity();

    // 评分项权重 / Weights
    const double w_clearance = w_clearance_; // 避障权重：远离障碍物
    const double w_path = w_path_;           // 路径一致性权重：靠近参考路径
    const double w_heading = w_heading_;     // 航向权重：朝向目标
    const double w_velocity = w_velocity_;   // 速度权重：追求效率

    // A. 避障评分 (Clearance Score) / min obstacle cost along trajectory
    // 找出整条轨迹中离障碍物最近点的代价。越远离障碍物，分值越高。
    double min_obs_cost = std::numeric_limits<double>::infinity();
    for (const auto& p : traj) {
        double c = obstacleCost(p.pose);
        min_obs_cost = std::min(min_obs_cost, c);
    }
    // 将代价归一化为评分（代价越高，分值越低 / 1 / (1 + cost)）
    double clearance_score = (min_obs_cost >= 1e5) ? -1e3 : (1.0 / (1.0 + min_obs_cost));

    // B. 路径距评分 (Path Score) / average distance to nearest reference point
    // 计算轨迹所有点到参考路径的平均距离。距离越小分值越高。
    double avg_path_dist = 0.0;
    for (const auto& p : traj) {
        avg_path_dist += pathDistCost(p.pose);
    }
    avg_path_dist /= traj.size();
    double path_score = 1.0 / (1.0 + avg_path_dist);

    // C. 航向评分 (Heading Score) / angle to goal from last point
    // 评估轨迹终点处机器人朝向与目标点的角度差。角度越小分值越高。
    double heading_score = 0.0;
    if (!reference_path_.empty()) {
        const auto& last = traj.back();
        Eigen::Vector2d goal(reference_path_.back().first, reference_path_.back().second);
        // 计算目标点相对于最后位置的方位角
        double yaw_to_goal = std::atan2(goal(1) - last.pose(1), goal(0) - last.pose(0));
        // 计算角度差并正则化
        double diff = std::fabs(std::atan2(std::sin(yaw_to_goal - last.theta), std::cos(yaw_to_goal - last.theta)));
        heading_score = 1.0 - clamp(diff / M_PI, 0.0, 1.0); // 偏差越小，得分越接近1
        
        // 如果当前偏差超过对齐阈值，增强该项权重，鼓励机器人原地转向 / Alignment boost
        if (diff > heading_align_thresh_) {
            heading_score *= heading_boost_;
        }
    }

    // D. 速度评分 (Velocity Score) / prefer higher speed magnitude 
    // 效率优先：在安全的前提下，速度越快，得分越高。
    double vel_score = traj.size() >= 2 ? ( (traj[1].pose - traj[0].pose).norm() / dt_ ) : 0.0;
    vel_score = clamp(vel_score / (vx_max_ + 1e-6), 0.0, 1.0);

    // 加权总分 / Final weighted total
    double total = w_clearance * clearance_score + w_path * path_score + w_heading * heading_score + w_velocity * vel_score;
    return total;
}

double DWAPlanner::obstacleCost(const Eigen::Vector2d& pose) const {
    int gx, gy;
    // 将物理位置映射到栅格地图坐标 / Map point to grid
    if (!costmap_.pointToGrid(pose(0), pose(1), gx, gy)) {
        return 1e6; // 超出地图边界，视为最高代价
    }

    // 机器人足迹检查 (Footprint Check) / Approximate robot as a circle
    // 以机器人当前预测位置为中心，检查圆形足迹边缘的所有栅格
    const double r = robot_radius_ + obstacle_margin_;
    const double res = static_cast<double>(costmap_.getResolution());
    const double denom = std::max<double>(res, 1e-3);
    // 计算采样点数，确保圆形周长上每个栅格都被检查到
    const int cells = std::max(8, (int)std::ceil(2 * M_PI * r / denom));
    for (int i = 0; i < cells; ++i) {
        double a = 2 * M_PI * i / cells;
        double x = pose(0) + r * std::cos(a);
        double y = pose(1) + r * std::sin(a);
        int cx, cy;
        if (!costmap_.pointToGrid(x, y, cx, cy)) return 1e6;
        // 如果代价超过100（对应约0.4m内的障碍物），判定为不可行路径
        if (costmap_.getCost(cx, cy) > 100) return 1e6;
    }

    // 返回中心点的归一化代价 / Return normalized cost of center
    uint8_t c = costmap_.getCost(gx, gy);
    double norm = (double)c / 255.0;
    return norm * norm; // 使用平方项增加对障碍物的“敏感度”
}

double DWAPlanner::pathDistCost(const Eigen::Vector2d& p) const {
    if (reference_path_.empty()) return 0.0;
    double best = std::numeric_limits<double>::infinity();
    for (const auto& rp : reference_path_) {
        double dx = p(0) - rp.first;
        double dy = p(1) - rp.second;
        double d2 = dx*dx + dy*dy;
        if (d2 < best) best = d2;
    }
    return std::sqrt(best);
}

} // namespace model_planner
