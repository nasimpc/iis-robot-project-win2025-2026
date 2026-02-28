import time 


class ActionSequencer :
    

    def __init__ (self ):
        self .states =["SEARCH","NAVIGATE","GRASP","COMPLETED"]
        self .current_state ="SEARCH"
        self .retry_count =0 
        self .max_retries =3 
        self .grasp_started =False 
        self .grasp_step =0 
        self .overscan_counter =0 


    def update_state (self ,perception_output ,robot_pose ,target_dist ):
        
        if self .current_state =="SEARCH":
            if perception_output .get ('target_found',False ):
                self .overscan_counter +=1 
                if self .overscan_counter >50 :
                    self .current_state ="NAVIGATE"
                    print ("[FSM] Target found and verified — switching to NAVIGATE")
            else :
                self .overscan_counter =0 

        elif self .current_state =="NAVIGATE":
            if not perception_output .get ('target_found',False ):

                self .lost_target_counter =getattr (self ,'lost_target_counter',0 )+1 
                if self .lost_target_counter >50 :
                    self .current_state ="SEARCH"
                    print ("[FSM] Target lost — returning to SEARCH")
            else :
                self .lost_target_counter =0 

            if self .current_state =="NAVIGATE"and target_dist <0.65 :
                self .current_state ="GRASP"
                self .grasp_started =False 
                self .grasp_step =0 
                print ("[FSM] Close enough — switching to GRASP")


        elif self .current_state =="GRASP":
            if perception_output .get ('in_gripper',False ):
                self .current_state ="COMPLETED"
                print ("[FSM] Object grasped — COMPLETED!")
            elif self .retry_count >=self .max_retries :
                self .retry_count =0 
                self .current_state ="SEARCH"
                print ("[FSM] Grasp failed too many times — returning to SEARCH")


    def mark_grasp_attempt (self ):
        
        self .retry_count +=1 

    def get_current_action (self ):
        return self .current_state 
