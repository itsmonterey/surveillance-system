# -*- coding: utf-8 -*-
"""
Created on Sun May 27 13:33:20 2018

@author: stephen
"""

#0:car(bus)
#1:pedestrian(motorbike,bicycle)

min_threshold = 5

import math

class Entity(object):
    
    def __init__(self,rect,label='car'):
        x,y,w,h = rect
        self.index = -1
        self.pos = (x+w/2,y+h/2)
        self.size = (w,h)
        self.poslist = []
        self.poslist.append(self.pos)
        
        self.spf_x = 0
        self.spf_y = 0
        self.nextpos = self.pos
                
        self.stopframes_cnts = 0
        
        self.life = 2#discounted by one for any frame without corresponding pair
        
        self.label =  label
        self.lp_str = 'unkown'
        self.colr = 'unknown'
        self.brand = 'unknown'
        self.name = 'unknown'
        self.gender = 'unknown'
        self.status = 'staying'#staying,moving
        
        self.mismatching_cnts = 0
        
        self.screen_size = (1200,680)

    def calc_speed(self):
        # speed
        if len(self.poslist) > 0:
            length = len(self.poslist)
            self.spf_x = self.pos[0] - self.poslist[length - 3][0] if length > 2 else\
                         self.pos[0] - self.poslist[length - 2][0]
            self.spf_y = self.pos[1] - self.poslist[length - 3][1] if length > 2 else\
                         self.pos[1] - self.poslist[length - 2][1]
        if abs(self.spf_x) < min_threshold and\
           abs(self.spf_y) < min_threshold:
            self.status = 'staying'
            self.stopframes_cnts += 1
        else:
            self.status = 'moving'
            self.stopframes_cnts = 0

    def calc_time(self,fps):
        return self.stopframes_cnts/fps

    def decrease_health(self,force_kill=0):
        if force_kill:
            self.life = 0
        self.life -= 1

    def estimate_next_pos(self):
        self.nextpos = (self.pos[0]+self.spf_x,self.pos[1]+self.spf_y)

    def get_mismatching_cnts(self):
        return self.mismatching_cnts

    def getIndex(self):
        return self.index

    def get_label(self):
        return self.label
        
    def get_rect(self):
        xc,yc = self.pos#self.nextpos
        w,h = self.size
        if xc>=self.screeen_size[0] or\
           yc>=self.screeen_size[1] or\
           xc+w < 0 or\
           yc+h < 0:
            self.decrease_health(force_kill=1)
        return [int(xc-w/2),int(yc-h/2),int(w),int(h)]

    def get_keys(self):
        w,h=self.size
        x,y=self.pos       
        return y-h/2,x-w/2,y+h/2,x+w/2,w,h,self.label
    
    def increase_mismatching(self):
        self.mismatching_cnts += 1
 
    def increase_health(self):
        self.life += 1
        self.life = max(self.life,3)        
        
    def initialize_mismatching(self):
        self.mismatching_cnts = 0

    def isAlive(self):
        if self.life > 0:
            return True
        else:
            return False

    def is_in_restricted(self,rect):
        x,y=self.pos
        w,h=self.size
        l,t,r,b=x,y,x+w,y+h
        left,top,width,height=rect
        right,bottom=left+width,top+height
        if r < left or\
           right < l or\
           b < top or\
           bottom < t:
            return False
        return True
    
    def is_in_wrong_direction(self,direction=(2,1)):
        delta_x,delta_y=direction
        if abs(delta_x) >= abs(delta_y):
            if self.spf_x*delta_x > 0:
                return False
            else:
                return True
        else:
            if self.spf_y*delta_y > 0:
                return False
            else:
                return True
        return False
 
    def is_over_red_line(self,pt1,pt2,direction=(2,0)):
        x,y=pt1
        x_,y_=pt2
        if abs(x-x_) > abs(y-y_):
            y_avg = (y+y_)/2
            if self.pos[1] > y_avg:
                return True
            else:
                return False
        else:
            x_avg = (x+x_)/2
            if self.pos[0] > x_avg:
                return True
            else:
                return False
        return False

    def isMergable(self,neighbor):
        xc,yc=self.pos
        xc_,yc_=neighbor.pos
        w,h=self.size
        w_,h_=neighbor.size
        ratio_diff = abs(w/h-w_/h_)
        size_diff = (w*h)/(w_*h_)
        distance = math.sqrt((xc-xc_)**2 + (yc-yc_)**2)
        
        labels = set([self.label,neighbor.label])
        
        if len(labels) == 1 or all(label in labels for label in ['pedestrian','motorbike']):
            if ratio_diff > 0.4 or\
               size_diff > 3 or\
               size_diff < 0.3:
                return False
            if any(case in labels for case in ['car','bus']):
                if distance > min(w,w_,h_,h)/2:
                    return False
            else:
                if distance > 3 * min(w,w_,h_,h):
                    return False
        else:
            return False
        return True

    def merge(self,neighbor):
        xc,yc=self.pos#self.nextpos#
        xc_,yc_=neighbor.pos
        w,h=self.size
        w_,h_=neighbor.size
        left,top,right,bottom=xc-w/2,yc-h/2,xc+w/2,yc+h/2
        left_,top_,right_,bottom_=xc_-w_/2,yc_-h_/2,xc_+w_/2,yc_+h_/2
        l,t,r,b=max(0,min(left,left_)),\
                max(0,min(top,top_)),\
                min(max(right,right_),self.screeen_size[0]),\
                min(max(bottom,bottom_),self.screeen_size[1])
        width,height = r-l,b-t
        self.pos = (l+width/2,t+height/2)
        self.size = (width,height)
        
        labels = set([self.label,neighbor.label])
        if all(label in labels for label in ['pedestrian','motorbike']):
            self.label = 'motorbike'

    def update(self,rect=None):
        if rect is None:
            self.pos = self.nextpos
            self.decrease_health()
        else:            
            x,y,w,h = rect
            self.pos = (x+w/2,y+h/2)
            self.size = (w,h)
            self.increase_health()
        self.poslist.append(self.pos)
        # calculate speed,next_pos
        self.calc_speed()
        self.estimate_next_pos()
        
    def setindex(self,index):
        self.index = index
        
    def setpos(self,pos):
        self.pos =  pos
    
    def setScreenSize(self,size):
        self.screeen_size=size
    
    def setsize(self,size):
        self.size = size
          
class EntityManager(object):
    
    def __init__(self):
        self.entitylist = []
        self.fps = 10
        self.screen_size = (1200,680)

    def add_entities(self,entities):
        for entity  in entities:
            self.entitylist.append(entity)

    def filtering(self,candidates):
        filtered = []
        for top,left,bottom,right,width,height,label in candidates:
            if abs(top) >2*10**4 or abs(left) > 2*10**4:
                continue
            e = Entity([left,top,width,height],label)
            e.setScreenSize(self.screen_size)
            isNew = True
            for i,[top_,left_,bottom_,right_,width_,height_,label_] in enumerate(filtered):
                e_ = Entity([left,top,width,height],label)
                e_.setScreenSize(self.screen_size)
                if e.isMergable(e_):
                    e.merge(e_)
                    filtered[i] = e.get_keys()
                    isNew = False
                    break
            filtered.append([top,left,bottom,right,width,height,label]) if isNew else None
        return filtered

    def finialize(self,obj_cnts):
        for entity in self.entitylist:
            if entity.get_mismatching_cnts() == obj_cnts:
                entity.update()
                
    def get_alives(self):
        return [entity for entity in self.entitylist if entity.isAlive()]
    
    def initialize(self):
        for entity in self.entitylist:
            entity.initialize_mismatching()
          
    def refresh(self,candidates):
        # 
        self.initialize()
        new_entities = []
        for top,left,bottom,right,width,height,label in self.filtering(candidates):
            isNew = True
            incoming = Entity([left,top,width,height],label)
            for entity in self.entitylist:
                if not entity.isAlive():
                    continue
                if entity.isMergable(incoming):
                    entity.update([left,top,width,height])
                    isNew = False
                    break
                else:
                    entity.increase_mismatching()
            if isNew:
                incoming.setindex(len(self.entitylist))
                incoming.setScreenSize(self.screen_size)
                new_entities.append(incoming)
        # finalize old entities
        self.finialize(len(candidates))
        # add new entties
        self.add_entities(new_entities)
    
    def setScreenSize(self,size):
        self.screen_size = size
                
    
    