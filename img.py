import imageio
import os
path=r"C:\Users\lenovo\Desktop\img"

IMG=[]

for i in os.listdir(path):

    IMG.append(imageio.imread(path+'/'+i))
imageio.mimsave(path+"/"+"h1.gif",IMG,"GIF",duration=2)


