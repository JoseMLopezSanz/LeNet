# Mute tensorflow debugging information on console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from keras import backend as K
import numpy as np
import cv2

class BlackBoard():
    def __init__(self, predictions):
        self.drawing = False
        self.background = True
        self.ix,self.iy = -1,-1
        self.size=400
        self.pad=60
        self.max_x, self.max_y, self.min_x, self.min_y = 0,0,np.inf,np.inf
        self.line=25
        self.img = np.zeros((self.size,self.size,1), np.uint8)+self.color()[1]
        cv2.namedWindow('image_black')
        cv2.setMouseCallback('image_black',self.draw)
        self.predictions=predictions
        print("[INFO] Press c to erase the board or esc to quit")


    def show(self):
        cv2.imshow('image_black',self.img)
    
    def erase(self):
        self.img = np.zeros((self.size,self.size,1), np.uint8)+self.color()[1]

    def blackboard_size(self,new_size):
        print("[INFO] Blackboard size set to {}".format(new_size))
        self.size=new_size
        self.img = np.zeros((self.size,self.size,1), np.uint8)+self.color()[1]
    
    def line_size(self, line_size):
        print("[INFO] Line thickness set to {}".format(line_size))
        self.line=line_size

    def color(self):
        if self.background==True:
            return 255,0
        elif self.background==False:
            return 0,255

    def change_color(self):
        self.background = not self.background
        if self.background:
            print("[INFO] Blackboard color changed to Black")
        else:
            print("[INFO] Blackboard color changed to White")
        self.img = np.zeros((size,size,1), np.uint8)+self.color()[1]

    def draw(self,event,x,y,flags,param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.line(self.img,(x,y),(x,y),self.color()[0],self.line)
            self.ix, self.iy=x, y
            self.max_x,self.max_y, self.min_x, self.min_y=max(self.ix, self.max_x),max(self.iy, self.max_y),min(self.ix, self.min_x),min(self.iy, self.min_y)
    
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.line(self.img,(self.ix,self.iy),(x,y),self.color()[0],self.line)
                self.ix, self.iy=x, y
                self.max_x,self.max_y, self.min_x, self.min_y=max(self.ix, self.max_x),max(self.iy, self.max_y),min(self.ix, self.min_x),min(self.iy, self.min_y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.line(self.img,(self.ix,self.iy),(x,y),self.color()[0],self.line)
            cv2.rectangle(self.img, (self.min_x-self.pad, self.max_y+self.pad),(self.max_x+self.pad, self.min_y-self.pad), self.color()[0], 1)

            #Obtaining prediction
            pred=self.predictions.predict_number(self.crop_image())
            cv2.putText(self.img, str(pred)[1], (10,40), cv2.FONT_HERSHEY_SIMPLEX,1,self.color()[0])
            #reset zone
            self.max_x, self.max_y, self.min_x, self.min_y = 0,0,np.inf,np.inf

    def crop_image(self):
        img_crop=self.img[max(0,self.min_y):min(self.size,self.max_y), max(0,self.min_x):min(self.size,self.max_x)]
        tot_size=max(self.max_x-self.min_x+2*self.pad,self.max_y-self.min_y+2*self.pad)
        img_crop_new=np.zeros((tot_size,tot_size,1), np.uint8)+self.color()[1]
        W=(tot_size-np.shape(img_crop)[0])/2
        H=(tot_size-np.shape(img_crop)[1])/2
        img_crop_new[W:W+np.shape(img_crop)[0],H:H+np.shape(img_crop)[1],:]=img_crop
        img_crop=cv2.resize(img_crop_new, (28, 28), interpolation=cv2.INTER_AREA)
        return img_crop


class Prediction():
    def __init__(self, model):
        print("[INFO] loading the pre-trained network '{}'...".format(model))
        self.model = load_model(model)
    
    def predict_number(self, img, verbose=-1, size=28):
        #sacar todo esto a una funcion
        img = img.astype("float")/255.0
        if K.image_data_format() == "channels_first":
            img = img.reshape(1, 1, size, size)
        #otherwise we are using channels last ordering so the desing matrix should be
        # num_samples x rows x columns x depth
        else:
            img = img.reshape(1, size,size,1)
        
        if verbose!=-1:
            cv2.imshow("number", img[0,:,:,:])
        pred = self.model.predict(img, batch_size=1).argmax(axis=1)
        return pred
