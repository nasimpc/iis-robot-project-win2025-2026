import numpy as np 
import pybullet as p 
from pyswip import Prolog 
import time 


class PIDController :
    

    def __init__ (
    self ,
    kp_lin =0.5 ,ki_lin =0.0 ,kd_lin =0.0 ,
    kp_ang =1.0 ,ki_ang =0.0 ,kd_ang =0.0 ,
    max_lin =0.5 ,max_ang =1.0 ,
    max_integral =5.0 ,
    ):

        self .kp_lin =kp_lin 
        self .ki_lin =ki_lin 
        self .kd_lin =kd_lin 


        self .kp_ang =kp_ang 
        self .ki_ang =ki_ang 
        self .kd_ang =kd_ang 


        self .max_lin =max_lin 
        self .max_ang =max_ang 


        self .max_integral =max_integral 


        self ._integral_lin =0.0 
        self ._integral_ang =0.0 
        self ._prev_error_lin =0.0 
        self ._prev_error_ang =0.0 

    def compute (self ,current_pose ,target_pose ,dt =1.0 /240.0 ):
        
        dx =target_pose [0 ]-current_pose [0 ]
        dy =target_pose [1 ]-current_pose [1 ]
        distance =np .hypot (dx ,dy )


        desired_theta =np .arctan2 (dy ,dx )
        angle_error =desired_theta -current_pose [2 ]
        angle_error =np .arctan2 (np .sin (angle_error ),np .cos (angle_error ))


        self ._integral_lin +=distance *dt 
        self ._integral_lin =np .clip (
        self ._integral_lin ,-self .max_integral ,self .max_integral 
        )
        derivative_lin =(distance -self ._prev_error_lin )/dt if dt >0 else 0.0 
        self ._prev_error_lin =distance 

        v =(
        self .kp_lin *distance 
        +self .ki_lin *self ._integral_lin 
        +self .kd_lin *derivative_lin 
        )


        self ._integral_ang +=angle_error *dt 
        self ._integral_ang =np .clip (
        self ._integral_ang ,-self .max_integral ,self .max_integral 
        )
        derivative_ang =(angle_error -self ._prev_error_ang )/dt if dt >0 else 0.0 
        self ._prev_error_ang =angle_error 

        omega =(
        self .kp_ang *angle_error 
        +self .ki_ang *self ._integral_ang 
        +self .kd_ang *derivative_ang 
        )


        v =np .clip (v ,-self .max_lin ,self .max_lin )
        omega =np .clip (omega ,-self .max_ang ,self .max_ang )

        return v ,omega 

    def reset (self ):
        
        self ._integral_lin =0.0 
        self ._integral_ang =0.0 
        self ._prev_error_lin =0.0 
        self ._prev_error_ang =0.0 


class PathPlanner :
    
    def __init__ (self ,prolog_file ="map.pl"):
        self .prolog =Prolog ()
        self .prolog .consult (prolog_file )

    def plan_path (self ,start ,goal ,obstacles ):
        

        if self ._is_line_free (start ,goal ,obstacles ):
            return [goal ]
        else :



            print ("PathPlanner: Direct path blocked – implement proper planner.")
            return []

    def _is_line_free (self ,a ,b ,obstacles ,clearance =0.5 ):
        
        for obs in obstacles :


            ox ,oy ,size =obs 



            pass 
        return True 


class GraspPlanner :
    
    def __init__ (self ,robot_id ,arm_joint_indices ,end_effector_link_index ):
        self .robot_id =robot_id 
        self .arm_joints =arm_joint_indices 
        self .ee_link =end_effector_link_index 

    def compute_ik (self ,target_position ,target_orientation =None ):
        
        if target_orientation is None :

            target_orientation =p .getQuaternionFromEuler ([np .pi ,0 ,0 ])

        joint_angles =p .calculateInverseKinematics (
        self .robot_id ,self .ee_link ,target_position ,target_orientation ,
        maxNumIterations =100 ,residualThreshold =1e-4 
        )





        return joint_angles [:len (self .arm_joints )]

    def execute_grasp (self ,target_pos ,gripper_joints ):
        

        pre_pos =[target_pos [0 ],target_pos [1 ],target_pos [2 ]+0.1 ]
        angles =self .compute_ik (pre_pos )
        if angles is None :
            return False 

        for i ,joint_idx in enumerate (self .arm_joints ):
            p .setJointMotorControl2 (self .robot_id ,joint_idx ,
            p .POSITION_CONTROL ,
            targetPosition =angles [i ],
            force =100 )

        for _ in range (100 ):
            p .stepSimulation ()
            time .sleep (1. /240. )


        angles =self .compute_ik (target_pos )
        if angles is None :
            return False 
        for i ,joint_idx in enumerate (self .arm_joints ):
            p .setJointMotorControl2 (self .robot_id ,joint_idx ,
            p .POSITION_CONTROL ,
            targetPosition =angles [i ],
            force =100 )
        for _ in range (100 ):
            p .stepSimulation ()
            time .sleep (1. /240. )


        p .setJointMotorControl2 (self .robot_id ,gripper_joints [0 ],
        p .POSITION_CONTROL ,targetPosition =0.0 ,force =20 )
        p .setJointMotorControl2 (self .robot_id ,gripper_joints [1 ],
        p .POSITION_CONTROL ,targetPosition =0.0 ,force =20 )
        for _ in range (50 ):
            p .stepSimulation ()
            time .sleep (1. /240. )


        lift_pos =[target_pos [0 ],target_pos [1 ],target_pos [2 ]+0.15 ]
        angles =self .compute_ik (lift_pos )
        if angles is None :
            return False 
        for i ,joint_idx in enumerate (self .arm_joints ):
            p .setJointMotorControl2 (self .robot_id ,joint_idx ,
            p .POSITION_CONTROL ,
            targetPosition =angles [i ],
            force =100 )
        for _ in range (100 ):
            p .stepSimulation ()
            time .sleep (1. /240. )

        return True 
