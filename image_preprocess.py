from PIL import Image
import os.path
import glob
#提取目录下所有图片,更改尺寸后保存到另一目录
def convertjpg(jpgfile,outdir,width=224,height=224):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
# for jpgfile in glob.glob("data\\crop_img\\Platelets\\*.jpg"):
#     convertjpg(jpgfile,"data\\crop_img\\Platelets_resize")
for jpgfile in glob.glob("data\\crop_img\\RBC\\*.jpg"):
    convertjpg(jpgfile,"data\\crop_img\\RBC_resize")
for jpgfile in glob.glob("data\\crop_img\\WBC\\*.jpg"):
    convertjpg(jpgfile,"data\\crop_img\\WBC_resize")
