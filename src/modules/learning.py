import numpy as np 
import json 
import os 
import logging 
import itertools 
from typing import Dict ,Any ,Tuple 

class QLearningTuner :

    def __init__ (
    self ,
    initial_params :Dict [str ,float ],
    step_sizes :Dict [str ,float ],
    param_bounds :Dict [str ,Tuple [float ,float ]],
    state_thresholds :Dict [str ,float ],
    alpha :float =0.1 ,
    gamma :float =0.9 ,
    initial_epsilon :float =0.5 ,
    epsilon_decay :float =0.995 ,
    min_epsilon :float =0.01 
    ):
        self .logger =logging .getLogger ("QLearningTuner")

        self .alpha =alpha 
        self .gamma =gamma 


        self .epsilon =initial_epsilon 
        self .epsilon_decay =epsilon_decay 
        self .min_epsilon =min_epsilon 

        self .param_names =list (initial_params .keys ())
        self .current_params =initial_params .copy ()
        self .step_sizes =step_sizes 
        self .bounds =param_bounds 
        self .thresholds =state_thresholds 


        self .action_space =list (itertools .product ([-1 ,0 ,1 ],repeat =len (self .param_names )))
        self .num_actions =len (self .action_space )

        self .q_table :Dict [str ,np .ndarray ]={}
        self .last_state :str =None 
        self .last_action_idx :int =None 

    def _discretize_state (self ,metrics :Dict [str ,float ])->str :
        
        if not metrics :
            return "INIT"

        overshoot ="High"if metrics .get ("overshoot",0.0 )>self .thresholds ["overshoot"]else "Low"
        error ="High"if metrics .get ("steady_state_error",0.0 )>self .thresholds ["error"]else "Low"
        settling ="Slow"if metrics .get ("settling_time",0.0 )>self .thresholds ["settling"]else "Fast"
        torque ="High"if metrics .get ("torque_violation",0.0 )>0.0 else "Safe"

        return f"OS:{overshoot }_ERR:{error }_SET:{settling }_TRQ:{torque }"

    def _get_q_values (self ,state :str )->np .ndarray :
        if state not in self .q_table :
            self .q_table [state ]=np .zeros (self .num_actions )
        return self .q_table [state ]

    def choose_action (self ,current_state :str )->int :
        
        q_values =self ._get_q_values (current_state )

        if np .random .uniform (0 ,1 )<self .epsilon :
            return np .random .randint (0 ,self .num_actions )
        return int (np .argmax (q_values ))

    def apply_action_to_params (self ,action_idx :int ):
        
        adjustments =self .action_space [action_idx ]

        for i ,param in enumerate (self .param_names ):
            if adjustments [i ]==-1 :
                self .current_params [param ]-=self .step_sizes [param ]
            elif adjustments [i ]==1 :
                self .current_params [param ]+=self .step_sizes [param ]


            min_val ,max_val =self .bounds [param ]
            self .current_params [param ]=max (min_val ,min (max_val ,self .current_params [param ]))

    def decay_epsilon (self ):
        
        self .epsilon =max (self .min_epsilon ,self .epsilon *self .epsilon_decay )

    def update_q_table (self ,reward :float ,next_state :str ):
        
        if self .last_state is None or self .last_action_idx is None :
            return 

        current_q =self ._get_q_values (self .last_state )
        next_q =self ._get_q_values (next_state )

        old_value =current_q [self .last_action_idx ]
        next_max =np .max (next_q )

        new_value =old_value +self .alpha *(reward +self .gamma *next_max -old_value )
        self .q_table [self .last_state ][self .last_action_idx ]=new_value 

class Learning :
    

    def __init__ (
    self ,
    initial_params :Dict [str ,float ]=None ,
    step_sizes :Dict [str ,float ]=None ,
    param_bounds :Dict [str ,Tuple [float ,float ]]=None ,
    state_thresholds :Dict [str ,float ]=None ,
    rewards :Dict [str ,float ]=None ,
    save_path :str ="q_learning_state.json"
    ):
        self .save_path =save_path 
        self .logger =logging .getLogger ("LearningSystem")


        initial_params =initial_params or {"Kp":1.0 ,"Ki":0.0 ,"Kd":0.1 }
        step_sizes =step_sizes or {"Kp":0.1 ,"Ki":0.01 ,"Kd":0.05 }
        param_bounds =param_bounds or {"Kp":(0.0 ,10.0 ),"Ki":(0.0 ,5.0 ),"Kd":(0.0 ,5.0 )}
        state_thresholds =state_thresholds or {"overshoot":0.2 ,"error":0.1 ,"settling":5.0 }

        self .rewards =rewards or {
        "success":100.0 ,
        "collision":-100.0 ,
        "timeout":-5.0 ,
        "overshoot_penalty_weight":10.0 ,
        "error_penalty_weight":20.0 ,
        "torque_penalty_weight":50.0 ,
        "energy_penalty_weight":0.1 
        }

        self .tuner =QLearningTuner (initial_params ,step_sizes ,param_bounds ,state_thresholds )
        self .load_state ()

    def get_optimized_parameters (self )->Dict [str ,float ]:
        return self .tuner .current_params .copy ()

    def update_from_trial (self ,trial_data :Dict [str ,Any ]):
        success =trial_data .get ('success',False )
        collided =trial_data .get ('collided',False )


        if collided :
            reward =self .rewards ["collision"]
        elif success :
            reward =self .rewards ["success"]
        else :
            reward =self .rewards ["timeout"]


        perf =trial_data .get ('performance_metrics',{})
        reward -=perf .get ('overshoot',0.0 )*self .rewards ["overshoot_penalty_weight"]
        reward -=perf .get ('steady_state_error',0.0 )*self .rewards ["error_penalty_weight"]


        reward -=perf .get ('torque_violation',0.0 )*self .rewards ["torque_penalty_weight"]
        reward -=perf .get ('energy_consumption',0.0 )*self .rewards ["energy_penalty_weight"]


        next_state =self .tuner ._discretize_state (perf )
        self .tuner .update_q_table (reward ,next_state )


        next_action_idx =self .tuner .choose_action (next_state )
        self .tuner .apply_action_to_params (next_action_idx )
        self .tuner .decay_epsilon ()

        self .tuner .last_state =next_state 
        self .tuner .last_action_idx =next_action_idx 

        self .save_state ()

    def save_state (self ):
        q_table_serializable ={k :v .tolist ()for k ,v in self .tuner .q_table .items ()}
        state ={
        "q_table":q_table_serializable ,
        "params":self .tuner .current_params ,
        "epsilon":self .tuner .epsilon 
        }
        try :
            with open (self .save_path ,'w')as f :
                json .dump (state ,f ,indent =4 )
        except Exception as e :
            self .logger .error (f"Error saving: {e }")

    def load_state (self ):
        if os .path .exists (self .save_path ):
            try :
                with open (self .save_path ,'r')as f :
                    state =json .load (f )

                self .tuner .current_params =state .get ("params",self .tuner .current_params )
                self .tuner .epsilon =state .get ("epsilon",self .tuner .epsilon )

                saved_q =state .get ("q_table",{})
                self .tuner .q_table ={k :np .array (v )for k ,v in saved_q .items ()}
                self .logger .info (f"Loaded offline learning state. Current epsilon: {self .tuner .epsilon :.3f}")
            except Exception as e :
                self .logger .error (f"Error loading: {e }")