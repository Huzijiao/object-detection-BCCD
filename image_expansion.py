import pandas as pd
from PIL import Image
import os
import os.path

rootdir = "data/crop_img/Platelets/"  # 指明想要扩充的文件夹

firstName = "F"
for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        print('filename is :' + filename)
        currentPath = os.path.join(parent, filename)
        #name = filename.split(sep='-')
        temp = Image.open(currentPath).resize((256, 256))
        #newname = name[0] + "-0" + filename[1:]
        newname = rootdir +firstName+"-0"+filename;
        temp.save(newname)

        #newname = name[0] + "-1" + filename[1:]
        out= temp.transpose(Image.FLIP_LEFT_RIGHT) # 重设宽120，高120
        newname = rootdir +firstName+"-1"+filename;
        out.save(newname)

        #newname = name[0] + "-2" + filename[1:]
        out= temp.transpose(Image.ROTATE_90)
        newname = rootdir +firstName+"-2"+filename;
        out.save(newname)

        #newname = name[0] + "-3" + filename[1:]
        out= temp.transpose(Image.ROTATE_180)
        newname = rootdir +firstName+"-3"+filename;
        out.save(newname)

        #newname = name[0] + "-4" + filename[1:]
        out= temp.transpose(Image.ROTATE_270)
        newname = rootdir +firstName+"-4"+filename;
        out.save(newname)

        #newname = name[0] + "-5" + filename[1:]
        out= temp.transpose(Image.FLIP_TOP_BOTTOM)
        newname = rootdir +firstName+"-5"+filename;
        out.save(newname)




