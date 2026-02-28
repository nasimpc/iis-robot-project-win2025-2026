import pybullet as p 
import pybullet_data 
import time 
import numpy as np 
import cv2 
import sys 
import os 
import math 
import logging 




sys .path .append (os .path .abspath (os .path .join (os .path .dirname (__file__ ),'..')))

from src .robot .sensor_wrapper import (
get_camera_image ,
get_imu_data ,
get_joint_states ,
get_lidar_data ,
)
from src .environment .world_builder import build_world 
from src .modules .perception import (
process_sensor_data ,
init_camera_transform ,
)
from src .modules .state_estimation import ParticleFilter 
from src .modules .motion_control import PIDController ,GraspPlanner 
from src .modules .action_planning import ActionSequencer 
from src .modules .knowledge_reasoning import KnowledgeBase 
from src .modules .learning import Learning 




logging .basicConfig (
level =logging .INFO ,
format ="%(asctime)s [%(name)s] %(message)s",
datefmt ="%H:%M:%S",
)
log =logging .getLogger ("CogArch")




WHEEL_JOINTS =[0 ,1 ,2 ,3 ]
ARM_JOINT_INDICES =list (range (5 ,12 ))
GRIPPER_JOINTS =[13 ,14 ]
END_EFFECTOR_LINK =12 
CAMERA_LINK_INDEX =None 




def get_joint_map (robot_id ):
    
    jmap ={}
    for i in range (p .getNumJoints (robot_id )):
        info =p .getJointInfo (robot_id ,i )
        jmap [info [1 ].decode ('utf-8')]=i 
    return jmap 


def apply_wheel_velocities (robot_id ,v_left ,v_right ):
    
    for idx in [0 ,2 ]:
        p .setJointMotorControl2 (
        robot_id ,idx ,p .VELOCITY_CONTROL ,
        targetVelocity =-v_left ,force =1500.0 ,
        )
    for idx in [1 ,3 ]:
        p .setJointMotorControl2 (
        robot_id ,idx ,p .VELOCITY_CONTROL ,
        targetVelocity =v_right ,force =1500.0 ,
        )


def stop_wheels (robot_id ):
    
    for idx in WHEEL_JOINTS :
        p .setJointMotorControl2 (
        robot_id ,idx ,p .VELOCITY_CONTROL ,
        targetVelocity =0 ,force =1000.0 ,
        )


def stow_arm (robot_id ):
    
    stow_positions =[0 ,0 ,0 ,0 ,1.57 ,0 ,0 ]
    for i ,joint_idx in enumerate (ARM_JOINT_INDICES ):
        p .setJointMotorControl2 (
        robot_id ,joint_idx ,p .POSITION_CONTROL ,
        targetPosition =stow_positions [i ],force =100 ,
        )


def open_gripper (robot_id ):
    
    for g in GRIPPER_JOINTS :
        p .setJointMotorControl2 (
        robot_id ,g ,p .POSITION_CONTROL ,
        targetPosition =0.04 ,force =20 ,
        )


def close_gripper (robot_id ):
    
    for g in GRIPPER_JOINTS :
        p .setJointMotorControl2 (
        robot_id ,g ,p .POSITION_CONTROL ,
        targetPosition =0.0 ,force =20 ,
        )


def compute_distance (a ,b ):
    
    return math .hypot (a [0 ]-b [0 ],a [1 ]-b [1 ])


def save_camera_data (rgb ,depth ,filename_prefix ="frame"):
    
    rgb_array =np .reshape (rgb ,(240 ,320 ,4 )).astype (np .uint8 )
    rgb_bgr =cv2 .cvtColor (rgb_array ,cv2 .COLOR_RGBA2BGR )
    depth_normalized =cv2 .normalize (np .array (depth ),None ,0 ,255 ,cv2 .NORM_MINMAX )
    depth_uint8 =depth_normalized .astype (np .uint8 )
    cv2 .imwrite (f"{filename_prefix }_rgb.png",rgb_bgr )
    cv2 .imwrite (f"{filename_prefix }_depth.png",depth_uint8 )




def setup_simulation ():
    
    scene_map =build_world ()

    robot_id =scene_map ['robot_id']
    table_id =scene_map ['table_id']
    room_id =scene_map ['room_id']
    target_id =scene_map ['target_id']
    table_pos =scene_map ['table_position']
    obstacle_ids =scene_map ['obstacle_ids']
    obstacle_pos =scene_map ['obstacle_positions']
    obstacle_col =scene_map ['obstacle_colors']
    landmark_map =scene_map ['landmark_map']
    robot_pos =scene_map ['robot_position']

    return (robot_id ,table_id ,room_id ,target_id ,
    table_pos ,obstacle_ids ,obstacle_pos ,obstacle_col ,
    landmark_map ,robot_pos )




def main ():

    (robot_id ,table_id ,room_id ,target_id ,
    table_pos ,obstacle_ids ,obstacle_pos ,obstacle_col ,
    landmark_map ,robot_start )=setup_simulation ()

    joint_map =get_joint_map (robot_id )
    global CAMERA_LINK_INDEX 
    CAMERA_LINK_INDEX =joint_map .get ("camera_joint",15 )
    log .info ("Joint map: %s",joint_map )
    log .info ("Robot at %s  |  Table at %s",robot_start ,table_pos )
    log .info ("Camera link index dynamically set to: %s",CAMERA_LINK_INDEX )




    init_camera_transform (robot_id ,camera_link_name ="camera_link")


    initial_pose =(robot_start [0 ],robot_start [1 ],0.0 )

    lm ={}
    for lid ,pos in landmark_map .items ():
        lm [lid ]=(pos [0 ],pos [1 ])
    pf =ParticleFilter (
    num_particles =200 ,
    initial_pose =initial_pose ,
    map_landmarks =lm ,
    motion_noise =(0.05 ,0.05 ,0.1 ),
    measurement_noise =0.3 ,
    )


    pid =PIDController (
    kp_lin =0.8 ,ki_lin =0.01 ,kd_lin =0.05 ,
    kp_ang =2.0 ,ki_ang =0.0 ,kd_ang =0.1 ,
    max_lin =2.0 ,max_ang =3.0 ,
    )
    grasp_planner =GraspPlanner (robot_id ,ARM_JOINT_INDICES ,END_EFFECTOR_LINK )


    sequencer =ActionSequencer ()


    kb =KnowledgeBase ()

    scene_config ={
    "table":{"pos":tuple (table_pos ),"mass":10.0 ,"color":"brown","shape":"box"},
    }
    for i ,(opos ,ocol )in enumerate (zip (obstacle_pos ,obstacle_col )):
        scene_config [f"obstacle_{i }"]={
        "pos":tuple (opos ),"mass":10.0 ,
        "color":ocol .lower (),"shape":"cube",
        }
    kb .load_initial_map (scene_config )
    log .info ("Knowledge base initialised with %d objects",len (scene_config ))


    learner =Learning (
    initial_params ={"Kp":pid .kp_lin ,"Ki":0.0 ,"Kd":0.0 },
    save_path ="q_learning_state.json",
    )


    step =0 
    dt =1.0 /240.0 
    nav_target =(table_pos [0 ],table_pos [1 ])
    target_world_pos =None 
    prev_v =0.0 
    prev_omega =0.0 
    grasp_in_progress =False 
    grasp_settled_steps =0 
    SEARCH_ANGULAR_VEL =1.5 
    last_target_found =False 
    last_target_local =None 


    open_gripper (robot_id )
    stow_arm (robot_id )

    log .info ("="*60 )
    log .info ("COGNITIVE ARCHITECTURE RUNNING — Sense-Think-Act Loop")
    log .info ("="*60 )


    while p .isConnected ():

        state =sequencer .get_current_action ()





        if step %10 ==0 :
            rgb ,depth ,mask =get_camera_image (robot_id ,CAMERA_LINK_INDEX )

            sensor_data ={'rgb':rgb ,'depth':depth ,'mask':mask }
            perception_output =process_sensor_data (sensor_data ,robot_id =robot_id )


            if perception_output ['target']is not None :
                last_target_found =True 
                last_target_local =perception_output ['target']['center']
                log .info ("Perception: TARGET DETECTED at local %s",last_target_local )
            else :
                last_target_found =False 
                last_target_local =None 
                if step %240 ==0 :
                    log .info ("Perception: no target in view")


            if step %2400 ==0 :
                save_camera_data (rgb ,depth ,filename_prefix =f"frame_{step }")


        imu =get_imu_data (robot_id )

        v_forward =imu ['accelerometer_data'][0 ]
        omega_z =imu ['gyroscope_data'][2 ]
        pf .predict (v_forward ,omega_z ,dt )


        if step %10 ==0 and perception_output is not None :

            if perception_output ['table']is not None :

                dx_local =perception_output ['table']['center'][0 ]
                dy_local =perception_output ['table']['center'][1 ]
                pf .update (0 ,(dx_local ,dy_local ))


            if perception_output ['obstacles']:
                for i ,obs in enumerate (perception_output ['obstacles']):
                    lm_id =i +1 
                    if lm_id in lm and obs ['center']is not None :
                        dx_local =obs ['center'][0 ]
                        dy_local =obs ['center'][1 ]
                        pf .update (lm_id ,(dx_local ,dy_local ))

            pf .resample ()


        x_est ,y_est ,theta_est =pf .get_estimate ()
        current_pose =(x_est ,y_est ,theta_est )


        target_world_pos =None 
        if last_target_found and last_target_local is not None :

            rel_x ,rel_y ,_ =last_target_local 
            cos_t =np .cos (current_pose [2 ])
            sin_t =np .sin (current_pose [2 ])
            target_world_pos =(
            current_pose [0 ]+rel_x *cos_t -rel_y *sin_t ,
            current_pose [1 ]+rel_x *sin_t +rel_y *cos_t ,
            table_pos [2 ]+0.05 
            )


        if last_target_local is not None :
            dist_to_nav =math .hypot (last_target_local [0 ],last_target_local [1 ])
        else :
            dist_to_nav =compute_distance (current_pose ,nav_target )






        perception_info ={
        'target_found':last_target_found ,
        'in_gripper':False ,
        'target_pos':target_world_pos ,
        'table_pos':table_pos ,
        }


        if target_world_pos is not None and step %50 ==0 :
            kb .perceive_target (
            float (target_world_pos [0 ]),
            float (target_world_pos [1 ]),
            float (target_world_pos [2 ]),
            )


        sequencer .update_state (perception_info ,current_pose ,dist_to_nav )
        state =sequencer .get_current_action ()





        if state =="SEARCH":


            apply_wheel_velocities (robot_id ,-SEARCH_ANGULAR_VEL ,-SEARCH_ANGULAR_VEL )
            stow_arm (robot_id )

        elif state =="NAVIGATE":

            if last_target_local is not None :
                dx_local =last_target_local [0 ]
                dy_local =last_target_local [1 ]
                angle_to_target =math .atan2 (dy_local ,dx_local )
                distance =math .hypot (dx_local ,dy_local )


                speed_cap =max (0.2 ,(1.0 -abs (angle_to_target )/(math .pi /2 )))
                pid_v =np .clip (pid .kp_lin *distance ,0 ,pid .max_lin *speed_cap )


                v =pid_v 
                omega =np .clip (pid .kp_ang *angle_to_target ,
                -pid .max_ang ,pid .max_ang )
            else :

                v =0.5 *pid .max_lin 
                omega =0.0 











            wheel_base =0.45 
            wheel_radius =0.1 


            w_left_nom =(v -omega *wheel_base /2.0 )/wheel_radius 
            w_right_nom =(v +omega *wheel_base /2.0 )/wheel_radius 



            apply_wheel_velocities (robot_id ,w_left_nom ,-w_right_nom )
            stow_arm (robot_id )

        elif state =="GRASP":

            stop_wheels (robot_id )

            if not grasp_in_progress :
                grasp_in_progress =True 
                grasp_settled_steps =0 
                log .info ("Starting grasp sequence...")


                feasible ,reason =kb .verify_grasp_conditions ()
                log .info ("Grasp feasibility: %s — %s",feasible ,reason )


                open_gripper (robot_id )

            grasp_settled_steps +=1 


            if grasp_settled_steps ==60 :
                if target_world_pos is not None :
                    log .info ("Executing IK grasp at %s",target_world_pos )
                    success =grasp_planner .execute_grasp (
                    list (target_world_pos ),GRIPPER_JOINTS 
                    )
                    if success :
                        perception_info ['in_gripper']=True 
                        log .info ("Grasp executed successfully!")
                    else :
                        log .warning ("Grasp IK failed — will retry")
                        sequencer .mark_grasp_attempt ()
                        grasp_in_progress =False 
                else :
                    log .warning ("No target position for grasp — returning to SEARCH")
                    sequencer .mark_grasp_attempt ()
                    grasp_in_progress =False 


                sequencer .update_state (perception_info ,current_pose ,dist_to_nav )

        elif state =="COMPLETED":
            stop_wheels (robot_id )
            if step %240 ==0 :
                log .info ("Mission COMPLETED — object grasped!")




        if step >0 and step %2400 ==0 :
            trial_data ={
            'success':state =="COMPLETED",
            'collided':False ,
            'performance_metrics':{
            'overshoot':0.0 ,
            'steady_state_error':dist_to_nav ,
            'settling_time':step *dt ,
            'torque_violation':0.0 ,
            'energy_consumption':abs (v_forward )*dt ,
            },
            }
            learner .update_from_trial (trial_data )
            learned_params =learner .get_optimized_parameters ()
            pid .kp_lin =learned_params .get ('Kp',pid .kp_lin )
            log .info ("Learning update — PID Kp=%.3f, epsilon=%.3f",
            pid .kp_lin ,learner .tuner .epsilon )




        if step %240 ==0 :
            log .info (
            "Step %d | State: %-10s | Pose: (%.2f, %.2f, %.1f°) | "
            "Dist: %.2f | Target: %s",
            step ,state ,x_est ,y_est ,math .degrees (theta_est ),
            dist_to_nav ,
            "DETECTED"if perception_info ['target_found']else "searching...",
            )

        prev_v =v_forward 
        prev_omega =omega_z 
        step +=1 

        p .stepSimulation ()
        time .sleep (1. /240. )



if __name__ =="__main__":
    main ()
