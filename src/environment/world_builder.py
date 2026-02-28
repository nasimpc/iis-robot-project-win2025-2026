import pybullet as p 
import pybullet_data 
import numpy as np 
import os 
import random 


ENVIRONMENT_DIR =os .path .dirname (os .path .abspath (__file__ ))
ROBOT_DIR =os .path .join (os .path .dirname (ENVIRONMENT_DIR ),'robot')


OBSTACLE_COLORS =[
[0.0 ,0.0 ,1.0 ,1.0 ],
[1.0 ,0.4 ,0.7 ,1.0 ],
[1.0 ,0.5 ,0.0 ,1.0 ],
[1.0 ,1.0 ,0.0 ,1.0 ],
[0.1 ,0.1 ,0.1 ,1.0 ],
]


def check_collision (new_pos ,existing_positions ,min_distance =1.0 ):
    for pos in existing_positions :
        dist =np .sqrt ((new_pos [0 ]-pos [0 ])**2 +(new_pos [1 ]-pos [1 ])**2 )
        if dist <min_distance :
            return False 
    return True 


def generate_random_position (x_range ,y_range ,existing_positions ,min_distance =1.0 ,max_attempts =100 ):
    for _ in range (max_attempts ):
        x =random .uniform (x_range [0 ],x_range [1 ])
        y =random .uniform (y_range [0 ],y_range [1 ])
        if check_collision ([x ,y ],existing_positions ,min_distance ):
            return [x ,y ]


    x =random .uniform (x_range [0 ],x_range [1 ])
    y =random .uniform (y_range [0 ],y_range [1 ])
    return [x ,y ]


def build_world (physics_client =None ):
    

    if physics_client is None :
        physics_client =p .connect (p .GUI )

    p .setAdditionalSearchPath (pybullet_data .getDataPath ())
    p .setGravity (0 ,0 ,-9.81 )


    placed_positions =[]


    room_path =os .path .join (ENVIRONMENT_DIR ,'room.urdf')
    room_id =p .loadURDF (room_path ,[0 ,0 ,0 ],useFixedBase =True )


    p .changeDynamics (room_id ,-1 ,lateralFriction =0.5 )


    robot_position =[-3.0 ,-3.0 ,0.2 ]
    robot_path =os .path .join (ROBOT_DIR ,'robot.urdf')
    robot_id =p .loadURDF (robot_path ,robot_position ,useFixedBase =False )
    placed_positions .append (robot_position [:2 ])



    table_pos_2d =generate_random_position (
    x_range =(0.0 ,3.0 ),
    y_range =(0.0 ,3.0 ),
    existing_positions =placed_positions ,
    min_distance =2.0 
    )
    table_position =[table_pos_2d [0 ],table_pos_2d [1 ],0.625 ]

    table_path =os .path .join (ENVIRONMENT_DIR ,'table.urdf')
    table_id =p .loadURDF (table_path ,table_position ,useFixedBase =True )
    placed_positions .append (table_pos_2d )


    obstacle_ids =[]
    obstacle_positions =[]
    obstacle_color_names =['Blue','Pink','Orange','Yellow','Black']

    obstacle_path =os .path .join (ENVIRONMENT_DIR ,'obstacle.urdf')

    for i in range (5 ):

        obs_pos_2d =generate_random_position (
        x_range =(-4.0 ,4.0 ),
        y_range =(-4.0 ,4.0 ),
        existing_positions =placed_positions ,
        min_distance =1.0 
        )
        obs_position =[obs_pos_2d [0 ],obs_pos_2d [1 ],0.0 ]

        obstacle_id =p .loadURDF (obstacle_path ,obs_position ,useFixedBase =True )


        p .changeVisualShape (obstacle_id ,-1 ,rgbaColor =OBSTACLE_COLORS [i ])

        obstacle_ids .append (obstacle_id )
        obstacle_positions .append (obs_position )
        placed_positions .append (obs_pos_2d )




    target_offset_x =random .uniform (-0.6 ,0.6 )
    target_offset_y =random .uniform (-0.3 ,0.3 )
    target_position =[
    table_position [0 ]+target_offset_x ,
    table_position [1 ]+target_offset_y ,
    table_position [2 ]+0.025 +0.06 
    ]

    target_path =os .path .join (ENVIRONMENT_DIR ,'target.urdf')
    target_id =p .loadURDF (target_path ,target_position ,useFixedBase =False )
    landmark_map ={
    0 :table_position ,
    }
    for i ,pos in enumerate (obstacle_positions ):
        landmark_map [i +1 ]=pos 



    scene_map ={
    'room_id':room_id ,
    'robot_id':robot_id ,
    'robot_position':robot_position ,
    'table_id':table_id ,
    'table_position':table_position ,
    'obstacle_ids':obstacle_ids ,
    'obstacle_positions':obstacle_positions ,
    'obstacle_colors':obstacle_color_names ,
    'target_id':target_id ,
    'target_position':target_position ,
    'landmark_map':landmark_map ,

    }


    print ("="*60 )
    print ("WORLD BUILDER: Scene Generated Successfully")
    print ("="*60 )
    print (f"Robot Position: {robot_position }")
    print (f"Table Position: {table_position }")
    print ("Obstacles:")
    for i ,(pos ,color )in enumerate (zip (obstacle_positions ,obstacle_color_names )):
        print (f"  {color }: {pos }")
    print (f"Target ID: {target_id } (position must be perceived by robot)")
    print ("="*60 )

    return scene_map 


def get_object_position (body_id ):
    pos ,_ =p .getBasePositionAndOrientation (body_id )
    return list (pos )


def get_robot_pose(robot_id):
    """Authorised wrapper for p.getBasePositionAndOrientation().

    All other modules MUST call this instead of using the PyBullet
    function directly (project constraint #2).

    Returns
    -------
    pos   : tuple(float, float, float)
    orn   : tuple(float, float, float, float)  – quaternion
    euler : tuple(float, float, float)          – (roll, pitch, yaw)
    """
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    euler = p.getEulerFromQuaternion(orn)
    return pos, orn, euler


if __name__ =="__main__":
    import time 

    print ("Starting World Builder Test...")


    scene_map =build_world ()

    print ("\nSimulation running. Press Ctrl+C to exit.")
    print ("Run this script multiple times to see different random configurations.\n")


    try :
        while p .isConnected ():
            p .stepSimulation ()
            time .sleep (1. /240. )
    except KeyboardInterrupt :
        print ("\nSimulation terminated by user.")
        p .disconnect ()
