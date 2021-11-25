import torch
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
# dir = ''https://github.com/ultralytics/yolov5/raw/master/data/;
dir = 'https://github.com/AlexUKMGo/yolov5/raw/main/'
imgs = [dir + f for f in ('0.jpg', '1.jpg','2.jpg','3.jpg','4.jpg')]  # batch of images

# Inference
results = model(imgs)
#results.print() # or .show(), .save()
results.save()
#print(len(results.pandas()))
#print("-----------------------------------------------------------------------")
for num in range(len(results.pandas())):
    list=results.pandas().xyxy[num].values.tolist()
    count=0
    for i in list:
        if(i[-1]=="person"):
             count=count+1;
    #print(count)
    # 编辑图片路径
    #bk_img = cv2.imread(str(num)+".jpg")
    bk_img = cv2.imread("C:\\Users\\Dylan\\Desktop\\runs\\detect\\exp\\"+str(num)+".jpg")
    #设置需要显示的字体
    fontpath = "font/simsun.ttc"# 14为字体大小
    font = ImageFont.truetype(fontpath, 14)
    img_pil = Image.fromarray(bk_img)
    draw = ImageDraw.Draw(img_pil)
    #绘制文字信息# (100,300/350)为字体的位置，(255,255,255)为白色，(0,0,0)为黑色
    draw.text((0, 0),  "the number of the person is:"+str(count), font = font, fill = (255,255,255))
    bk_img = np.array(img_pil)
    
    cv2.imshow("add_text",bk_img)
    cv2.waitKey()# 保存图片路径
    cv2.imwrite(str(num)+".jpg",bk_img)
    print("image"+str(num+1)+"totally have "+str(count)+" people.")
    
#print(results.pandas().xyxy[0].values.tolist())
#count=0
#for i in list:
#    if(i[-1]=="person"):
#        count=count+1;
