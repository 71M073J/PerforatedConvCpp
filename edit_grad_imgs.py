import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
prefix = "allTests_last2"
for x in os.listdir(f"./{prefix}/"):
    if os.path.isdir(f"./{prefix}/" + x) and len(x) > 10:
        cnt1 = 0
        fig = ax = None
        for imagename in os.listdir(f"./{prefix}/" + x):
            imagepath = f"./{prefix}/" + x + "/" + imagename
            if not imagename.startswith("_"): #TODO CHECK IF RUNNING ON NEW IMAGES IF THEY ALREADY HAVE UNDERSCORES ELSE COMMENT THIS IF OUT
                os.remove(imagepath)
                continue
            image = cv2.imread(imagepath)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[0] == 2000:
                starts = 255 - image[:, 56, 0]
                inds = np.nonzero(starts)[0]
                startinds = inds[1:][np.diff(inds) > 20] - 1
                ns = [image[x:x+15, 35:60, :] for x in startinds]
                starts = startinds //5
                title = image[:37, :, :]
                bottom = image[1942:, :, :]
                yaxis = image[866:1138,:32,:]
                content = image[37:1941:5, :, :]
                base = np.ones((500, 500, 3), dtype=np.uint8) * 255
                base[20:20+37, :, :] = title
                base[-58:, :, :] = bottom
                base[-58 - content.shape[0]:-58, :, :] = content
                base[:,35:63, :] = 255
                base[250 - (yaxis.shape[0]//2):250+yaxis.shape[0]//2, 5:5+32, :] = yaxis
                for i, s in enumerate(starts):
                    base[s:s+15, 35:60, :] = ns[i]
                cv2.imwrite(f"./{prefix}/" + x + "/_" + imagename, base)
            else:
                ...

