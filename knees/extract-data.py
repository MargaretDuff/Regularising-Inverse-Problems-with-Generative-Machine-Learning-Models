import numpy as np
import glob
import h5py

import PIL
import PIL.Image as Image
import matplotlib.pyplot as plt 
import numpy as np

list_files=glob.glob("./*train*/*.h5", recursive=True)
print(list_files)
count=0
hold=np.zeros([1000000, 128,128])
for i in range(len(list_files)):
    hf = h5py.File(list_files[i], 'r')
    if dict(hf.attrs)['acquisition']=='CORPD_FBK':
    	reconstruction_rss = hf['reconstruction_rss'][()]
    	for j in range(12,20):
        	image=Image.fromarray((reconstruction_rss[j]), 'F')
        	image=image.resize((128, 128), resample=Image.ANTIALIAS)
        	image=np.array(image.getdata()).reshape([128,128])
        	image=(image-np.min(image))/np.max(image)
        	hold[count]=image
        	count+=1
hold=hold[:count,:,:]
print(count)
np.save( 'knee_fastMRI_train_128_cleaned.npy', hold)


#images=np.load( 'knee_fastMRI_train_EXAMPLE.npy')
#images=(images-np.min(images))
#images*=255/np.max(images)
#images= np.round(images).astype(np.uint8)

#for i in range(30):
#    image=Image.fromarray(images[i,:,:], 'L')
    
#    image=image.convert("L")
 #   image.save('Example_patient_'+str(i)+'.png')
#
#    plt.figure()
#    plt.imshow(images[i,:,:])
#    plt.savefig('Example_'+str(i)+'.png')
    
list_files=glob.glob("./*val*/*.h5", recursive=True)
print(list_files)
count=0
hold=np.zeros([1000000, 128,128])
for i in range(len(list_files)):
    hf = h5py.File(list_files[i], 'r')
    if dict(hf.attrs)['acquisition']=='CORPD_FBK':
        reconstruction_rss = hf['reconstruction_rss'][()]
        for j in range(12,20):
                image=Image.fromarray((reconstruction_rss[j]), 'F')
                image=image.resize((128, 128), resample=Image.ANTIALIAS)
                image=np.array(image.getdata()).reshape([128,128])
                image=(image-np.min(image))/np.max(image)
                hold[count]=image
                count+=1
hold=hold[:count,:,:]
np.save( 'knee_fastMRI_test_128_cleaned.npy', hold)


#list_files=glob.glob("./*val*/*.h5", recursive=True)
#count=0
#hold=np.zeros([1000000, 128,128])
#for i in range(len(list_files)):
#    hf = h5py.File(list_files[i], 'r')
#    reconstruction_rss = hf['reconstruction_rss'][()]
#    for j in range(6,24):
#        image=Image.fromarray((reconstruction_rss[j]), 'F')
#        image=image.resize((128, 128), resample=Image.ANTIALIAS)
#        image=np.array(image.getdata()).reshape([128,128])
#        image=(image-np.min(image))/np.max(image)
#        hold[count]=image
#        count+=1
#hold=hold[:count,:,:]
#np.save( 'knee_fastMRI_test_128.npy', hold)


images=np.load( 'knee_fastMRI_train_128_cleaned.npy')
images=(images-np.min(images))*255/np.max(images)
images=(images-np.min(images))
images*=255/np.max(images)
images= np.round(images).astype(np.uint8)
for i in range(20):
    image=Image.fromarray(images[np.random.randint(0,400),:,:], 'L')
    
    image=image.convert("L")
    image.save('Example_cleaned_'+str(i)+'.png')



