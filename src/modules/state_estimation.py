import numpy as np 
from typing import List ,Tuple ,Dict ,Optional 


class Particle :
    
    def __init__ (self ,x :float ,y :float ,theta :float ,weight :float =1.0 ):
        self .x =x 
        self .y =y 
        self .theta =theta 
        self .weight =weight 


class ParticleFilter :
    
    def __init__ (self ,
    num_particles :int ,
    initial_pose :Tuple [float ,float ,float ],
    map_landmarks :Dict [int ,Tuple [float ,float ]],
    motion_noise :Tuple [float ,float ,float ]=(0.02 ,0.02 ,0.05 ),
    measurement_noise :float =0.1 ):
        
        self .num_particles =num_particles 
        self .map_landmarks =map_landmarks 
        self .motion_noise =motion_noise 
        self .measurement_noise =measurement_noise 


        self .particles =[]
        for _ in range (num_particles ):
            x =initial_pose [0 ]+np .random .normal (0 ,0.05 )
            y =initial_pose [1 ]+np .random .normal (0 ,0.05 )
            theta =initial_pose [2 ]+np .random .normal (0 ,0.05 )
            self .particles .append (Particle (x ,y ,theta ,1.0 /num_particles ))

    def predict (self ,v :float ,omega :float ,dt :float ):
        
        for p in self .particles :

            v_noisy =v +np .random .normal (0 ,self .motion_noise [0 ])
            omega_noisy =omega +np .random .normal (0 ,self .motion_noise [2 ])


            p .x +=v_noisy *dt *np .cos (p .theta )
            p .y +=v_noisy *dt *np .sin (p .theta )
            p .theta +=omega_noisy *dt 


            p .theta =(p .theta +np .pi )%(2 *np .pi )-np .pi 


            p .x +=np .random .normal (0 ,self .motion_noise [0 ]*dt )
            p .y +=np .random .normal (0 ,self .motion_noise [1 ]*dt )
            p .theta +=np .random .normal (0 ,self .motion_noise [2 ]*dt )
            p .theta =(p .theta +np .pi )%(2 *np .pi )-np .pi 

    def update (self ,landmark_id :int ,observed_rel_pos :Tuple [float ,float ]):
        
        if landmark_id not in self .map_landmarks :
            return 

        landmark_true =self .map_landmarks [landmark_id ]
        dx_obs ,dy_obs =observed_rel_pos 

        for p in self .particles :

            dx_world =landmark_true [0 ]-p .x 
            dy_world =landmark_true [1 ]-p .y 


            cos_t =np .cos (p .theta )
            sin_t =np .sin (p .theta )
            dx_pred =dx_world *cos_t +dy_world *sin_t 
            dy_pred =-dx_world *sin_t +dy_world *cos_t 


            err =np .sqrt ((dx_pred -dx_obs )**2 +(dy_pred -dy_obs )**2 )


            likelihood =(1.0 /(np .sqrt (2 *np .pi )*self .measurement_noise ))*np .exp (-0.5 *(err /self .measurement_noise )**2 )

            p .weight *=likelihood 


        total_weight =sum (p .weight for p in self .particles )
        if total_weight >0 :
            for p in self .particles :
                p .weight /=total_weight 
        else :

            for p in self .particles :
                p .weight =1.0 /self .num_particles 

    def resample (self ):
        
        new_particles =[]
        N =self .num_particles 


        cum_weights =np .cumsum ([p .weight for p in self .particles ])


        step =1.0 /N 
        r =np .random .uniform (0 ,step )
        j =0 
        for i in range (N ):
            u =r +i *step 
            while u >cum_weights [j ]:
                j +=1 

            p =self .particles [j ]
            x =p .x +np .random .normal (0 ,0.01 )
            y =p .y +np .random .normal (0 ,0.01 )
            theta =p .theta +np .random .normal (0 ,0.01 )
            new_particles .append (Particle (x ,y ,theta ,1.0 /N ))

        self .particles =new_particles 

    def get_estimate (self )->Tuple [float ,float ,float ]:
        
        x_mean =np .average ([p .x for p in self .particles ],weights =[p .weight for p in self .particles ])
        y_mean =np .average ([p .y for p in self .particles ],weights =[p .weight for p in self .particles ])

        sin_sum =np .average ([np .sin (p .theta )for p in self .particles ],weights =[p .weight for p in self .particles ])
        cos_sum =np .average ([np .cos (p .theta )for p in self .particles ],weights =[p .weight for p in self .particles ])
        theta_mean =np .arctan2 (sin_sum ,cos_sum )
        return x_mean ,y_mean ,theta_mean 

    def get_particles (self )->List [Particle ]:
        
        return self .particles 