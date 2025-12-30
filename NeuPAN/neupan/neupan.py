'''
neupan file is the main class for the NeuPan algorithm. It wraps the PAN class and provides a more user-friendly interface.

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han <hanrh@connect.hku.hk>

NeuPAN planner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

NeuPAN planner is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with NeuPAN planner. If not, see <https://www.gnu.org/licenses/>.
'''

import yaml
import torch
from neupan.robot import robot
from neupan.blocks import InitialPath, PAN
from neupan import configuration
from neupan.util import time_it, file_check, get_transform
import numpy as np
from neupan.configuration import np_to_tensor, tensor_to_np
from math import cos, sin

class neupan(torch.nn.Module):

    """
    NeuPAN 核心类，封装了 PAN 算法并提供用户友好的接口。
    NeuPAN Core Class: Wraps the PAN algorithm and provides a more user-friendly interface.

    Args:
        receding: int, MPC预测时域步数 / Number of steps in the receding horizon.
        step_time: float, MPC时间步长 / Time step in the MPC framework.
        ref_speed: float, 机器人参考速度 / Reference speed of the robot.
        device: str, 运行设备 ('cpu' 或 'cuda') / Device to run the algorithm on.
        robot_kwargs: dict, 机器人参数字典 / Keyword arguments for the robot class.
        ipath_kwargs: dict, 初始路径生成参数 / Keyword arguments for the initial path class.
        pan_kwargs: dict, PAN 算法参数 / Keyword arguments for the PAN class.
        adjust_kwargs: dict, 代价函数权重调整参数 / Keyword arguments for the adjust class.
        train_kwargs: dict, DUNE 网络训练参数 / Keyword arguments for the train class.
        time_print: bool, 是否打印算法运行时间 / Whether to print the forward time of the algorithm.
        collision_threshold: float, 碰撞检测阈值，若最小距离小于此值则停止 / Threshold for collision detection. If collision, the algorithm will stop.
    """

    def __init__(
        self,
        receding: int = 10,
        step_time: float = 0.1,
        ref_speed: float = 4.0,
        device: str = "cpu",
        robot_kwargs: dict = None,
        ipath_kwargs: dict = None,
        pan_kwargs: dict = None,
        adjust_kwargs: dict = None,
        train_kwargs: dict = None,
        **kwargs,
    ) -> None:
        super(neupan, self).__init__()

        # MPC 参数 / MPC parameters
        self.T = receding
        self.dt = step_time
        self.ref_speed = ref_speed

        configuration.device = torch.device(device)
        configuration.time_print = kwargs.get("time_print", False)
        self.collision_threshold = kwargs.get("collision_threshold", 0.1)

        # 初始化 / Initialization
        self.cur_vel_array = np.zeros((2, self.T))  # 当前速度序列缓存 / Cache for current velocity sequence
        
        # 初始化机器人模型 / Initialize robot model
        self.robot = robot(receding, step_time, **robot_kwargs)

        # 初始化参考路径生成器 / Initialize reference path generator
        self.ipath = InitialPath(
            receding, step_time, ref_speed, self.robot, **ipath_kwargs
        )
            
        pan_kwargs["adjust_kwargs"] = adjust_kwargs
        pan_kwargs["train_kwargs"] = train_kwargs
        self.dune_train_kwargs = train_kwargs

        # 初始化 PAN 求解器 / Initialize PAN solver
        self.pan = PAN(receding, step_time, self.robot, **pan_kwargs)

        # 状态信息字典 / State info dictionary
        self.info = {"stop": False, "arrive": False, "collision": False}

    @classmethod
    def init_from_yaml(cls, yaml_file, **kwargs):
        """
        从 YAML 配置文件初始化 NeuPAN 实例
        Initialize NeuPAN instance from a YAML configuration file
        """
        abs_path = file_check(yaml_file)

        with open(abs_path, "r") as f:
            config = yaml.safe_load(f)
            config.update(kwargs)

        # 提取各个模块的配置参数 / Extract configuration parameters for each module
        config["robot_kwargs"] = config.pop("robot", dict())
        config["ipath_kwargs"] = config.pop("ipath", dict())
        config["pan_kwargs"] = config.pop("pan", dict())
        config["adjust_kwargs"] = config.pop("adjust", dict())
        config["train_kwargs"] = config.pop("train", dict())

        return cls(**config)

    @time_it("neupan forward")
    def forward(self, state, points, velocities=None):
        """
        NeuPAN 前向计算：执行路径规划和能够避障的速度生成
        NeuPAN Forward: Executes path planning and generates obstacle-avoiding velocities.

        Args:
            state: 机器人当前状态 [x, y, theta] / Current state of the robot, matrix (3, 1)
            points: 障碍物点坐标 (2, N) / Obstacle point positions
            velocities: 障碍物点速度 (2, N) / Velocity of each obstacle point
        """

        assert state.shape[0] >= 3

        # 1. 检查是否到达终点 / Check if arrived at goal
        if self.ipath.check_arrive(state):
            self.info["arrive"] = True
            return np.zeros((2, 1)), self.info

        # 2. 生成标称参考状态和输入（基于初始路径） / Generate nominal reference state and input (based on initial path)
        nom_input_np = self.ipath.generate_nom_ref_state(
            state, self.cur_vel_array, self.ref_speed
        )

        # 3. 转换为张量 / Convert inputs to tensors
        nom_input_tensor = [np_to_tensor(n) for n in nom_input_np]
        obstacle_points_tensor = np_to_tensor(points) if points is not None else None
        point_velocities_tensor = (
            np_to_tensor(velocities) if velocities is not None else None
        )

        # 4. 运行 PAN 优化算法计算最优状态和控制量 / Run PAN optimization algorithm
        opt_state_tensor, opt_vel_tensor, opt_distance_tensor = self.pan(
            *nom_input_tensor, obstacle_points_tensor, point_velocities_tensor
        )

        # 5. 转换回 NumPy 格式 / Convert results back to NumPy
        opt_state_np, opt_vel_np = tensor_to_np(opt_state_tensor), tensor_to_np(
            opt_vel_tensor
        )

        # 更新当前速度序列缓存 / Update current velocity sequence cache
        self.cur_vel_array = opt_vel_np

        # 存储调试信息 / Store debug info
        self.info["state_tensor"] = opt_state_tensor
        self.info["vel_tensor"] = opt_vel_tensor
        self.info["distance_tensor"] = opt_distance_tensor
        self.info['ref_state_tensor'] = nom_input_tensor[2]
        self.info['ref_speed_tensor'] = nom_input_tensor[3]

        self.info["ref_state_list"] = [
            state[:, np.newaxis] for state in nom_input_np[2].T
        ]
        self.info["opt_state_list"] = [state[:, np.newaxis] for state in opt_state_np.T]

        # 6. 碰撞检测与急停 / Collision check and emergency stop
        if self.check_stop():
            self.info["stop"] = True
            return np.zeros((2, 1)), self.info
        else:
            self.info["stop"] = False

        # 7. 返回当前的控制指令（第一个时间步） / Return current control command (first time step)
        action = opt_vel_np[:, 0:1]

        return action, self.info

    def check_stop(self):
        """
        检查是否需要紧急停止
        Check if emergency stop is required based on min distance to obstacles.
        """
        return self.min_distance < self.collision_threshold
    

    def scan_to_point(
        self,
        state: np.ndarray,
        scan: dict,
        scan_offset: list[float] = [0, 0, 0],
        angle_range: list[float] = [-np.pi, np.pi],
        down_sample: int = 1,
    ) -> np.ndarray | None:
        
        """
        将激光雷达数据转换为世界坐标系下的点云
        Convert lidar scan data to point cloud in world frame.

        Args:
            state: 机器人位姿 [x, y, theta]
            scan: 激光雷达扫描数据字典 / Scan data dict
                ranges: 距离列表 / List of ranges
                angle_min/max: 角度范围 / Angle range
                range_min/max: 距离范围 / Range limits
            scan_offset: 激光雷达相对于机器人的偏移量 [x, y, theta] / Relative position of sensor to robot
            angle_range: 需要保留的角度范围 / Angle range used
            down_sample: 降采样倍率 / Downsample rate

        Return:
            points: (2, N) 障碍物点坐标 / Obstacle points
        """
        point_cloud = []

        ranges = np.array(scan["ranges"])
        angles = np.linspace(scan["angle_min"], scan["angle_max"], len(ranges))

        # 遍历扫描数据，筛选有效点 / Iterate through scan, filter valid points
        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]

            if scan_range < (scan["range_max"] - 0.02) and scan_range > scan["range_min"]:
                if angle > angle_range[0] and angle < angle_range[1]:
                    # 极坐标转笛卡尔坐标（雷达系） / Polar to Cartesian (Sensor frame)
                    point = np.array(
                        [[scan_range * cos(angle)], [scan_range * sin(angle)]]
                    )
                    point_cloud.append(point)

        if len(point_cloud) == 0:
            return None

        point_array = np.hstack(point_cloud)
        
        # 1. 雷达系 -> 机器人系 / Sensor frame -> Robot frame
        s_trans, s_R = get_transform(np.c_[scan_offset])
        temp_points = s_R @ point_array + s_trans

        # 2. 机器人系 -> 世界系 / Robot frame -> World frame
        trans, R = get_transform(state)
        points = (R @ temp_points + trans)[:, ::down_sample]

        return points

    def scan_to_point_velocity(
        self,
        state,
        scan,
        scan_offset=[0, 0, 0],
        angle_range=[-np.pi, np.pi],
        down_sample=1,
    ):
        """
        将激光雷达数据转换为点云和速度（若可用）
        Convert lidar scan data to point cloud and velocity.
        
        Args:
             state: [x, y, theta]
             scan: {}
                 ranges: list[float]
                 angle_min/max: float
                 range_max/min: float
                 velocity: list[float]
             scan_offset: [x, y, theta]
        
        Return:
             points: (2, n)
             velocity: (2, n)
        """
        point_cloud = []
        velocity_points = []

        ranges = np.array(scan["ranges"])
        angles = np.linspace(scan["angle_min"], scan["angle_max"], len(ranges))
        scan_velocity = scan.get("velocity", np.zeros((2, len(ranges))))

        # lidar_state = self.lidar_state_transform(state, np.c_[self.lidar_offset])
        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]

            if scan_range < (scan["range_max"] - 0.02) and scan_range >= scan["range_min"]:
                if angle > angle_range[0] and angle < angle_range[1]:
                    point = np.array(
                        [[scan_range * cos(angle)], [scan_range * sin(angle)]]
                    )
                    point_cloud.append(point)
                    velocity_points.append(scan_velocity[:, i : i + 1])

        if len(point_cloud) == 0:
            return None, None

        point_array = np.hstack(point_cloud)
        s_trans, s_R = get_transform(np.c_[scan_offset])
        temp_points = s_R.T @ (
            point_array - s_trans
        )  # points in the robot state coordinate

        trans, R = get_transform(state)
        points = (R @ temp_points + trans)[:, ::down_sample]

        velocity = np.hstack(velocity_points)[:, ::down_sample]

        return points, velocity


    def train_dune(self):
        """训练 DUNE 网络 / Train DUNE network"""
        self.pan.dune_layer.train_dune(self.dune_train_kwargs)


    def reset(self):
        """重置规划器状态 / Reset planner state"""
        self.ipath.point_index = 0
        self.ipath.curve_index = 0
        self.ipath.arrive_flag = False
        self.info["stop"] = False
        self.info["arrive"] = False
        self.cur_vel_array = np.zeros_like(self.cur_vel_array)

    def set_initial_path(self, path):
        '''
        设置初始路径
        Set the initial path from the given path.
        Args:
            path: list of [x, y, theta, gear] 4x1 vector
        '''
        self.ipath.set_initial_path(path)

    def set_initial_path_from_state(self, state):
        """
        从当前状态初始化路径（通常用于无路点时的原地规划）
        Initialize path from current state.
        Args:
            states: [x, y, theta]
        """
        self.ipath.init_check(state)
    
    def set_reference_speed(self, speed: float):
        """
        设置参考速度
        Args:
            speed: float, 参考速度 / reference speed
        """
        self.ipath.ref_speed = speed
        self.ref_speed = speed
    
    def update_initial_path_from_goal(self, start, goal):
        """
        根据起点和目标点生成初始路径
        Args:
            start: 起点 [x, y, theta]
            goal: 目标点 [x, y, theta]
        """
        self.ipath.update_initial_path_from_goal(start, goal)

    def update_initial_path_from_waypoints(self, waypoints):
        """
        根据一系列路点生成初始路径
        Args:
            waypoints: list of [x, y, theta]
        """
        self.ipath.set_ipath_with_waypoints(waypoints)

    def update_adjust_parameters(self, **kwargs):
        """
        动态更新代价函数权重参数
        update the adjust parameters value: q_s, p_u, eta, d_max, d_min

        Args:
            q_s: float, 状态代价权重 / weight of the state cost
            p_u: float, 速度代价权重 / weight of the speed cost
            eta: float, 避障松弛变量权重 / weight of the collision avoidance cost
            d_max: float, 最大避障安全距离 / maximum safety distance
            d_min: float, 最小避障距离 / minimum distance to the obstacle
        """
        self.pan.nrmp_layer.update_adjust_parameters_value(**kwargs)

    @property
    def min_distance(self):
        return self.pan.min_distance
    
    @property
    def dune_points(self):
        return self.pan.dune_points
    
    @property
    def nrmp_points(self):
        return self.pan.nrmp_points
    
    @property
    def initial_path(self):
        return self.ipath.initial_path
    
    @property
    def adjust_parameters(self):
        return self.pan.nrmp_layer.adjust_parameters
    
    @property
    def waypoints(self):

        '''
        Waypoints for generating the initial path
        '''

        return self.ipath.waypoints
    
    @property
    def opt_trajectory(self):

        '''
        MPC receding horizon trajectory under the velocity sequence
        return a list of state sequence, each state is a 3x1 vector
        '''

        return self.info["opt_state_list"]
    
    @property
    def ref_trajectory(self):

        '''
        Reference trajectory on the initial path
        return a list of state sequence, each state is a 3x1 vector
        '''

        return self.info["ref_state_list"]
