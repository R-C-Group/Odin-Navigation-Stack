<p align="center">
  <h2 align="center">Odin Navigation Stack解读</h2>
</p>

[Odin Navigation Stack](https://github.com/ManifoldTechLtd/Odin-Nav-Stack) 是一个基于ROS1 Noetic的四足机器人（Unitree Go2）自主导航系统。系统集成了高精度SLAM、语义目标检测、神经网络规划器和视觉语言模型，提供完整的室内外导航解决方案。


# 核心模块
1. 感知层
* odin_ros_driver: Odin1传感器驱动（提供RGB、深度、IMU、点云）
  * SLAM定位和建图（内部算法，未开源）
  * 实时重定位
  * 多传感器同步
* fish2pinhole: 鱼眼相机到针孔相机的图像转换
  * 功能: 鱼眼相机图像矫正
  * 输入: 鱼眼畸变的RGB和深度图
  * 输出: 针孔相机模型的矫正图像
  * 算法: FishPoly鱼眼投影模型
* fake360: 360度视角生成
* yolo_ros: YOLOv5目标检测与语义定位
  * 功能: 基于YOLOv5的目标检测与3D定位
2. 规划层
* map_planner: 基于栅格地图的A*全局路径规划
  * 算法: A*路径搜索 + 障碍物膨胀
* model_planner: 自定义规划器（可配置全局和局部规划器）
* navigation_planner: ROS标准导航栈封装
* NeuPAN: 端到端神经网络局部规划器（独立Python包）
  * 高效避障: 50Hz实时规划
  * 动态障碍: 适应动态环境
  * 平滑轨迹: 神经网络优化的运动轨迹
3. 控制层
* unitree_control: Unitree Go2速度控制接口
4. 工具层
* pcd2pgm: 点云地图转栅格地图
* pointcloud_saver: 点云地图保存工具
* odin_vlm_terminal: 视觉语言模型终端界面


# 关于人员跟随的梳理
1. 感知层：看到物体 (yolo_detector.py)
  * 检测: 订阅RGB图像，YOLOv5识别物体（如 Person, Chair）。
  * 3D定位: 结合深度图，使用 FishPoly 模型将图像像素坐标 $(u,v)$ 反投影为相机坐标系下的3D坐标 $(x,y,z)$。
  * 发布: 将带有3D位置的检测结果发布到 /yolo_detections_3d。
2. 决策层：理解与计算 (object_query_node.py`此文件的逻辑需要重点关注`)
  * 接收指令: 通过 interactive_mode 接收用户输入（如 "Move to the right of the person"）。
  * 解析 (parse_navigation_command): 提取出动作(Move)、目标(Person)、方位(Right)和索引(第1个)。`【这部分应该是要依赖于自然语言指令解析以及语义导航】`
  * 查找 (find_object): 在检测结果中找到匹配的物体。
  * 坐标计算 (calculate_target_position)：
     1. 获取物体在Map坐标系下的位置（通过TF变换）。
     2. 根据“右边1米”等指令，计算出机器人应该去的最终目标点坐标。
  * 发送目标 (send_navigation_goal): 将计算出的坐标封装为 PoseStamped 消息，发布给导航系统（话题：/move_base_simple/goal）。
3. 调度层：状态管理 (goal_state_machine.cpp)
   * 监听: 监听 /move_base_simple/goal。
   * 触发: 收到新目标后，或者在 arriveCallback 中发现距离目标还有距离。
   * 服务调用: 向 map_planner 发起路径规划请求 (plan_client_.call)。
4. 规划层：生成路径 (map_planner.cpp)
   * 全局规划: A* 算法根据静态栅格地图，计算出一条从当前位置到目标点的无碰撞最优路径。
   * 发布: 将路径发布给局部规划器。
5. 控制层：避障与运动 (neupan_ros.py & unitree_vel_controller.cpp)
   * 局部规划 (NeuPAN): 接收全局路径，结合实时激光雷达/深度图数据，使用神经网络模型进行动态避障，输出速度指令 /cmd_vel。
   * 执行 (Unitree): 接收速度指令，调用机器狗底层 SDK 执行运动。


