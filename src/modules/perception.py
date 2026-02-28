import numpy as np 
import cv2 
from sklearn .decomposition import PCA 
from typing import Tuple ,List ,Dict ,Optional 
import pybullet as p 



_camera_to_base_rot =None 
_camera_to_base_trans =None 

def init_camera_transform (robot_id ,camera_link_name ="camera_link"):
    global _camera_to_base_rot ,_camera_to_base_trans 

    camera_idx =None 
    for i in range (p .getNumJoints (robot_id )):
        info =p .getJointInfo (robot_id ,i )
        if info [12 ].decode ('utf-8')==camera_link_name :
            camera_idx =i 
            break 
    if camera_idx is None :
        raise ValueError (f"Camera link '{camera_link_name }' not found")


    link_state =p .getLinkState (robot_id ,camera_idx ,computeForwardKinematics =True )
    cam_pos =link_state [0 ]
    cam_orn =link_state [1 ]
    _camera_to_base_trans =np .array (cam_pos )
    _camera_to_base_rot =np .array (p .getMatrixFromQuaternion (cam_orn )).reshape (3 ,3 )


def rgb_to_point_cloud (rgb ,depth ,mask ,robot_id =None ,width =320 ,height =240 ,fov =60 ):

    far =10.0 
    near =0.1 

    depth_2d =np .array (depth ).reshape (height ,width )
    depth_meters =far *near /(far -(far -near )*depth_2d )


    f =height /(2 *np .tan (np .deg2rad (fov )/2 ))
    cx =width /2.0 
    cy =height /2.0 


    u ,v =np .meshgrid (np .arange (width ),np .arange (height ))



    z_cam =depth_meters 
    x_cam =(u -cx )*z_cam /f 
    y_cam =-(v -cy )*z_cam /f 


    points_cam =np .stack ([x_cam ,y_cam ,z_cam ],axis =-1 ).reshape (-1 ,3 )




    cam_to_robot =np .array ([
    [0 ,0 ,1 ],
    [-1 ,0 ,0 ],
    [0 ,-1 ,0 ]
    ])

    points_robot =points_cam @cam_to_robot .T 
    points =points_robot 


    rgb_array =np .array (rgb ).reshape (height ,width ,4 )
    colors =rgb_array [:,:,:3 ].reshape (-1 ,3 )/255.0 


    mask_array =np .array (mask ).reshape (-1 )


    valid_mask =(depth_meters .reshape (-1 )>near )&(depth_meters .reshape (-1 )<far )

    return points [valid_mask ],colors [valid_mask ],mask_array [valid_mask ]


def detect_objects_by_color (points ,colors ,target_color ,color_tolerance =0.3 ):
    target_color =np .array (target_color )


    color_diff =np .linalg .norm (colors -target_color ,axis =1 )


    color_mask =color_diff <color_tolerance 
    detected_points =points [color_mask ]

    if len (detected_points )==0 :
        return None ,None 


    centroid =np .mean (detected_points ,axis =0 )

    return detected_points ,centroid 


def ransac_plane_detection (points ,distance_threshold =0.05 ,num_iterations =100 ,min_inliers =100 ):
    if len (points )<3 :
        return None ,None ,None 

    best_plane =None 
    best_inliers =None 
    max_inlier_count =0 

    for _ in range (num_iterations ):

        sample_indices =np .random .choice (len (points ),3 ,replace =False )
        sample_points =points [sample_indices ]


        p1 ,p2 ,p3 =sample_points 


        v1 =p2 -p1 
        v2 =p3 -p1 


        normal =np .cross (v1 ,v2 )


        norm =np .linalg .norm (normal )
        if norm <1e-6 :
            continue 

        normal =normal /norm 


        a ,b ,c =normal 
        d =-np .dot (normal ,p1 )


        distances =np .abs (a *points [:,0 ]+b *points [:,1 ]+c *points [:,2 ]+d )


        inliers =distances <distance_threshold 
        inlier_count =np .sum (inliers )


        if inlier_count >max_inlier_count :
            max_inlier_count =inlier_count 
            best_plane =[a ,b ,c ,d ]
            best_inliers =inliers 

    if best_plane is None or max_inlier_count <min_inliers :
        return None ,None ,None 

    inlier_points =points [best_inliers ]

    return best_plane ,best_inliers ,inlier_points 


def estimate_pose_pca (points ):
    if len (points )<3 :
        return None ,None ,None 


    centroid =np .mean (points ,axis =0 )


    centered_points =points -centroid 


    pca =PCA (n_components =3 )
    pca .fit (centered_points )


    principal_axes =pca .components_ 


    projected =centered_points @principal_axes .T 
    dimensions =np .max (projected ,axis =0 )-np .min (projected ,axis =0 )

    return centroid ,principal_axes ,dimensions 


def detect_table (points ,colors ,expected_height =0.625 ,height_tolerance =0.15 ,brown_color =[0.5 ,0.3 ,0.1 ]):
    

    height_mask =np .abs (points [:,2 ]-expected_height )<height_tolerance 
    candidate_points =points [height_mask ]
    candidate_colors =colors [height_mask ]

    if len (candidate_points )==0 :
        return None ,None ,None 


    table_plane ,inliers ,table_points =ransac_plane_detection (
    candidate_points ,
    distance_threshold =0.03 ,
    num_iterations =200 ,
    min_inliers =50 
    )

    if table_plane is None :
        return None ,None ,None 


    table_center =np .mean (table_points ,axis =0 )

    return table_plane ,table_center ,table_points 


def detect_target_object (points ,colors ,target_color =[1.0 ,0.0 ,0.0 ],color_tolerance =0.5 ):
    


    red_channel =colors [:,0 ]
    green_channel =colors [:,1 ]
    blue_channel =colors [:,2 ]


    red_mask =(red_channel >0.6 )&(green_channel <0.3 )&(blue_channel <0.3 )


    target_color_arr =np .array (target_color )
    color_diff =np .linalg .norm (colors -target_color_arr ,axis =1 )


    color_mask =color_diff <0.35 


    combined_mask =red_mask &color_mask 
    red_ish_points =np .sum (combined_mask )


    if np .random .rand ()<0.02 :
        print (f"[TargetDetect] Total points: {len (points )}, Red-ish points: {red_ish_points }, Tolerance: {color_tolerance }")


    target_points =points [combined_mask ]

    if len (target_points )<5 :
        return None ,None ,None 


    target_center =np .mean (target_points ,axis =0 )


    if len (target_points )>=3 :
        centroid ,principal_axes ,dimensions =estimate_pose_pca (target_points )
    else :
        centroid =target_center 
        principal_axes =np .eye (3 )
        dimensions =np .array ([0.08 ,0.08 ,0.12 ])

    target_pose ={
    'centroid':centroid ,
    'principal_axes':principal_axes ,
    'dimensions':dimensions 
    }

    return target_center ,target_points ,target_pose 


def detect_obstacles (points ,colors ,obstacle_colors =None ,color_tolerance =0.3 ):
    
    if obstacle_colors is None :
        obstacle_colors ={
        'Blue':[0.0 ,0.0 ,1.0 ],
        'Pink':[1.0 ,0.4 ,0.7 ],
        'Orange':[1.0 ,0.5 ,0.0 ],
        'Yellow':[1.0 ,1.0 ,0.0 ],
        'Black':[0.1 ,0.1 ,0.1 ]
        }

    obstacles =[]

    for color_name ,color_rgb in obstacle_colors .items ():
        obstacle_points ,obstacle_center =detect_objects_by_color (
        points ,colors ,color_rgb ,color_tolerance 
        )

        if obstacle_points is not None :

            centroid ,principal_axes ,dimensions =estimate_pose_pca (obstacle_points )

            obstacles .append ({
            'color':color_name ,
            'center':obstacle_center ,
            'points':obstacle_points ,
            'centroid':centroid ,
            'principal_axes':principal_axes ,
            'dimensions':dimensions 
            })

    return obstacles 


def process_sensor_data (sensor_data ,robot_id =None ):
    
    rgb =sensor_data ['rgb']
    depth =sensor_data ['depth']
    mask =sensor_data ['mask']


    points ,colors ,masks =rgb_to_point_cloud (rgb ,depth ,mask ,robot_id =robot_id )


    table_plane ,table_center ,table_points =detect_table (points ,colors )


    target_center ,target_points ,target_pose =detect_target_object (points ,colors )


    obstacles =detect_obstacles (points ,colors )

    perception_output ={
    'target':{
    'center':target_center ,
    'points':target_points ,
    'pose':target_pose 
    }if target_center is not None else None ,
    'table':{
    'plane':table_plane ,
    'center':table_center ,
    'points':table_points 
    }if table_center is not None else None ,
    'obstacles':obstacles ,
    'point_cloud':{
    'points':points ,
    'colors':colors ,
    'masks':masks 
    }
    }

    return perception_output 


def process_sensor_data_relative (sensor_data ):
    
    global _camera_to_base_rot ,_camera_to_base_trans 
    if _camera_to_base_rot is None or _camera_to_base_trans is None :
        raise RuntimeError ("Camera transform not initialized. Call init_camera_transform() first.")

    rgb =sensor_data ['rgb']
    depth =sensor_data ['depth']
    mask =sensor_data ['mask']


    points_cam ,colors ,_ =rgb_to_point_cloud (rgb ,depth ,mask ,robot_id =None )


    target_center_cam ,_ ,_ =detect_target_object (points_cam ,colors )
    if target_center_cam is not None :

        target_base =_camera_to_base_rot @target_center_cam +_camera_to_base_trans 
        target_rel =(target_base [0 ],target_base [1 ])
    else :
        target_rel =None 


    obstacles_cam =detect_obstacles (points_cam ,colors )
    obstacles_rel =[]


    for obs in obstacles_cam :
        if obs is not None :
            center_cam =obs ['center']
            center_base =_camera_to_base_rot @center_cam +_camera_to_base_trans 
            obstacles_rel .append ((center_base [0 ],center_base [1 ]))
        else :
            obstacles_rel .append (None )



    table_rel =None 

    return {
    'target':target_rel ,
    'obstacles':obstacles_rel ,
    'table':table_rel 
    }



if __name__ =="__main__":
    print ("Perception module loaded successfully.")
    print ("Functions available:")
    print ("  - rgb_to_point_cloud()")
    print ("  - detect_objects_by_color()")
    print ("  - ransac_plane_detection()")
    print ("  - estimate_pose_pca()")
    print ("  - detect_table()")
    print ("  - detect_target_object()")
    print ("  - detect_obstacles()")
    print ("  - process_sensor_data()")
    print ("  - process_sensor_data_relative() (new for state estimation)")