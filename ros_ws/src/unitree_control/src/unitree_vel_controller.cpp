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

#include <ros/ros.h>
#include <std_srvs/SetBool.h>
#include <geometry_msgs/Twist.h>
#include <string>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <mutex>
#include <csignal>
#include <atomic>

#define NETWORK_INTERFACE "eth0"  // 网络接口名称 / Network interface name
using namespace unitree::common;

/**
 * Unitree Go2 速度控制器类
 * Unitree Go2 velocity controller class
 * 
 * 功能：接收ROS cmd_vel指令并转发给Unitree Go2 SDK
 * Function: receives ROS cmd_vel commands and forwards them to Unitree Go2 SDK
 */
class Custom
{
public:
  /**
   * 构造函数：初始化Unitree Go2机器人控制器
   * Constructor: initializes Unitree Go2 robot controller
   * 
   * @param nh ROS节点句柄 / ROS node handle
   */
  Custom(ros::NodeHandle &nh) : should_exit(false)
  {
    // 初始化Unitree运动客户端 / Initialize Unitree sport client
    sport_client.SetTimeout(10.0f);  // 设置通信超时时间 / Set communication timeout
    sport_client.Init();  // 初始化SDK / Initialize SDK
    sport_client.StandUp();  // 站立机器人 / Stand up robot
    sleep(1);
    sport_client.BalanceStand();  // 进入平衡站立模式 / Enter balance stand mode
    sleep(1);
    // if (!sport_client.ClassicWalk(true)) {
    //   std::cerr << "Failed to switch to Classic Walk mode." << std::endl;
    // }

    // 切换到AI行走模式（自主导航推荐使用） / Switch to AI Walk mode (recommended for autonomous navigation)
    if (sport_client.FreeWalk() == 0) {
      std::cout << "Switched to AI Walk mode." << std::endl;
    } else {
      std::cerr << "Failed to switch to AI Walk mode." << std::endl;
    }
    
    // 禁用Unitree内置的自动避障（使用自定义导航） / Disable Unitree's built-in obstacle avoidance (using custom navigation)
    if (sport_client.FreeAvoid(false) == 0) {
      std::cout << "Disabled obstacle avoidance." << std::endl;
    } else {
      std::cerr << "Failed to disable obstacle avoidance." << std::endl;
    }

    // 注册服务：切换行走模式 / Register service: switch walk mode
    change_walk_mode_srv = nh.advertiseService("unitree/classic_walk_mode", &Custom::ChangeWalkMode, this);

    // 订阅速度指令 / Subscribe to velocity commands
    cmd_vel_sub = nh.subscribe("/cmd_vel", 1, &Custom::CmdVelCallback, this);
  }

  /**
   * 析构函数：安全关闭机器人
   * Destructor: safely shuts down the robot
   */
  ~Custom()
  {
    SafeShutdown();
  }
  
  /**
   * 安全关闭函数：停止运动并蹲下
   * Safe shutdown function: stops movement and sits down
   */
  void SafeShutdown()
  {
    if (!should_exit.exchange(true))  // 原子操作，防止重复关闭 / Atomic operation to prevent duplicate shutdown
    {
      std::cout << "Initiating safe shutdown..." << std::endl;
      sport_client.StopMove();  // 停止运动 / Stop movement
      sport_client.StandDown();  // 蹲下 / Sit down
      std::cout << "Safety procedures completed." << std::endl;
    }
  }

private:
  /**
   * 速度指令回调函数：将ROS cmd_vel转发给Unitree SDK
   * Velocity command callback: forwards ROS cmd_vel to Unitree SDK
   * 
   * @param msg Twist消息，包含线速度(x,y)和角速度(z) / Twist message with linear velocity (x,y) and angular velocity (z)
   */
  void CmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
  {
    // 调用Unitree SDK的Move接口：(vx, vy, vyaw)
    // Call Unitree SDK Move interface: (vx, vy, vyaw)
    // vx: 前后速度 (m/s) / forward-backward velocity
    // vy: 左右速度 (m/s) / left-right velocity
    // vyaw: 旋转角速度 (rad/s) / yaw angular velocity
    sport_client.Move(msg->linear.x, msg->linear.y, msg->angular.z);
  }
  
  /**
   * 行走模式切换服务
   * Walk mode switching service
   * 
   * @param req true=经典模式, false=AI模式 / true=Classic mode, false=AI mode
   * @param res 服务响应，包含成功标志和消息 / Service response with success flag and message
   */
  bool ChangeWalkMode(std_srvs::SetBool::Request &req,
                      std_srvs::SetBool::Response &res) {
    if (req.data) {
      std::cout << "Switching to ClassicWalk mode" << std::endl;
    } else {
      std::cout << "Switching to AI walk mode" << std::endl;
    }
    int result = sport_client.ClassicWalk(req.data);  // 调用SDK切换模式 / Call SDK to switch mode
    res.success = result == 0;  // 0表示成功 / 0 means success
    if (!res.success) {
      std::cerr << "Failed to switch walk mode with error code " << result << std::endl;
    }
    res.message = std::string("Switch walk mode ") + 
      (res.success ? "succeeded." : "failed with error code " + std::to_string(result));
    return true;
  }
  // 成员变量 / Member variables
  unitree::robot::go2::SportClient sport_client;  // Unitree Go2运动控制SDK客户端 / Unitree Go2 sport control SDK client

  ros::ServiceServer change_walk_mode_srv;  // 行走模式切换服务 / Walk mode switching service
  ros::Subscriber cmd_vel_sub;  // 速度指令订阅者 / Velocity command subscriber

  std::atomic<bool> should_exit;  // 退出标志，原子类型保证线程安全 / Exit flag, atomic type ensures thread safety
};

// 全局指针，用于信号处理 / Global pointer for signal handling
Custom* global_custom = nullptr;

/**
 * 信号处理函数：捕获SIGINT/SIGTERM信号，安全关闭机器人
 * Signal handler: catches SIGINT/SIGTERM signals and safely shuts down the robot
 * 
 * @param signum 信号编号 / Signal number
 */
void signalHandler(int signum)
{
  std::cout << "\nInterrupt signal (" << signum << ") received.\n";
  if (global_custom != nullptr) {
    global_custom->SafeShutdown();  // 执行安全关闭 / Execute safe shutdown
  }
  ros::shutdown();  // 关闭ROS节点 / Shutdown ROS node
}

/**
 * 主函数
 * Main function
 */
int main(int argc, char **argv)
{
  // 初始化Unitree通信通道 / Initialize Unitree communication channel
  unitree::robot::ChannelFactory::Instance()->Init(0, NETWORK_INTERFACE);
  
  // 初始化ROS节点 / Initialize ROS node
  ros::init(argc, argv, "unitree_cmd_vel_controller");
  ros::NodeHandle nh;

  // 创建控制器实例 / Create controller instance
  Custom custom(nh);
  global_custom = &custom;  // 设置全局指针供信号处理使用 / Set global pointer for signal handling
 
  // 注册信号处理函数 / Register signal handlers
  signal(SIGINT, signalHandler);   // Ctrl+C
  signal(SIGTERM, signalHandler);  // 终止信号 / Termination signal
  
  // 进入ROS事件循环 / Enter ROS event loop
  ros::spin();
  
  global_custom = nullptr;
  return 0;
}
