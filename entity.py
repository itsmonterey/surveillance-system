# -*- coding: utf-8 -*-
"""
Created on Sun May 27 13:33:20 2018

@author: stephen
"""

#0:car(bus)
#1:pedestrian(motorbike,bicycle)

from constants import labels

class Entity(object):
    
    def __init__(self,rect,type_='car'):
        x,y,w,h = rect
        self.index = -1
        self.pos = (x+w/2,y+h/2)
        self.size = (w,h)
        self.type =  0 if type_ in ['car','bus'] else 1
        self.poslist = []
        self.direction = 0
        self.velocity = 0
        self.accelerarity = 0
        self.estimated_nextpos = (0,0)
        
        self.spf_x = 0
        self.spf_y = 0
        
        self.life = 2#discounted by one for any frame without corresponding pair
        
        self.lp_str = 'unkown'
        self.colr = 'unknown'
        self.brand = 'unknown'
        self.name = 'unknown'
        self.gender = 'unknown'
        
        self.mismatching_cnts = 0

    def decrease_health(self):
        self.life -= 1

    def estimate_next_pos(self):
        self.estimated_nextpos = (self.pos[0]+self.spf_x,self.pos[1]+self.spf_y)

    def get_mismatching_cnts(self):
        return self.mismatching_cnts

    def increase_mismatching(self):
        self.mismatching_cnts += 1
 
    def increase_health(self):
        self.life += 1
        self.life = max(self.life,3)        
        
    def initialize_mismatching(self):
        self.mismatching_cnts = 0

        
    def isMergable(self,neighbor):
        x,y=self.pos
        x_,y_=neighbor.pos
        w,h=self.size
        w_,h_=neighbor.size
        ratio_diff = abs(w/h-w_/h_)
        size_diff = (w*h)/(w_*h_)
        distance = (x-x_)**2 + (y-y)**2
        if self.type != neighbor.type or\
           ratio_diff > 0.4 or\
           size_diff > 2 or\
           size_diff < 0.5 or\
           distance > min(w,w_,h_,h)/2:
            return False
        return True

    def update(self,rect):
        x_,y_=self.pos
        x,y,w,h = rect
        self.index = 0
        self.pos = (x+w/2,y+h/2)
        self.size = (w,h)
        self.spf_x = x - x_
        self.spf_y = y - y_
        self.increase_health()

    def setindex(self,index):
        self.index = index
        
    def setpos(self,pos):
        self.pos =  pos
    
    def setsize(self,size):
        self.size = size
          
class EntityManager(object):
    
    def __init__(self):
        self.entitylist = []
    
    def initialize(self):
        for entity in self.entitylist:
            entity.initialize_mismatching()
    
    def finialize(self,obj_cnts):
        for entity in self.entitylist:
            if entity.get_mismatching_cnts() == obj_cnts:
                entity.decrease_health()
    
    def refresh(self,candidates):
        self.initialize()
        cnts = 0
        for top,left,bottom,right,width,height,label in candidates:
            isNew = True
            incoming = Entity([left,top,width,height],label)
            for entity in self.entitylist:
                if entity.life < 0:
                    continue
                if entity.isMergable(incoming):
                    entity.update([left,top,width,height])
                    isNew = False
                else:
                    entity.increase_mismatching()
            if isNew:
                cnts += 1
                incoming.setindex(len(self.entitylist))
                self.entitylist(incoming)
        self.finialize(self,obj_cnts)
            
                
    
    