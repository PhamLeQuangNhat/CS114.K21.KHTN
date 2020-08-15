import os
import cv2
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from joblib import load
import numpy as np
import fruit_pred
root = Tk() 

# Set Title as Image Loader 
root.title("Image Loader") 
  
# Set the resolution of window 
root.geometry("300x400") 
root.configure(bg='#34495e')
# Allow Window to be resizable 
root.resizable(width = True, height = True) 
 
display1 = Label(root,text = "MACHINE LEARNING \n \n PHẠM LÊ QUANG NHẬT - 18520120", font = (20))
display1.grid(row = 0, column = 0,columnspan = 2, padx = 200, pady =25)

#display2 = Label(root,text = "PHẠM LÊ QUANG NHẬT - 18520120", font = (20))
#display2.grid(row = 1, column = 5,columnspan = 2, padx = 10, pady =25)

# Create a button and place it into the window using grid layout 
label = Label(root, text = "Upload the file") 
btn = Button(root, text ='BROWSE', font = (20), command = lambda: get_link()).grid(column = 0,
                                       row = 1, padx = 5, pady = 25) 
btn1 = Button(root, text ='PREDICT', font = (20), command = lambda: open_img()).grid( column =0,
                                       row = 2,  columnspan =2, padx = 5, pady = 25) 
display = Label(root,text = "", font = (10))
display.grid(row = 3,columnspan = 2, pady = 10)
label.grid(row = 1, column = 1, padx = 5, pady =5)
def get_link():

    
    global x 
    x = openfilename()
    label.config(text=x)

def open_img(): 
    # Select the Imagename  from a folder  
    #x = openfilename() 
    # opens the image 
    img = Image.open(x) 
      
    # resize the image and apply a high-quality down sampling filter 
    img = img.resize((250, 250), Image.ANTIALIAS) 
  
    # PhotoImage class is used to add image to widgets, icons etc 
    img = ImageTk.PhotoImage(img) 
   
    # create a label 
    panel = Label(root, image = img) 
      
    # set the image as img  
    panel.image = img 
    panel.grid( pady = 5, columnspan = 2,row =2)

    predict = fruit_pred.predict(x)
    print(predict)

    display.config(text=predict)

def openfilename(): 
  
    # open file dialog box to select image 
    # The dialogue box has a title "Open" 
    filename = filedialog.askopenfilename(title ='PREDICTING TABLE') 
    return filename 

root.mainloop() 
