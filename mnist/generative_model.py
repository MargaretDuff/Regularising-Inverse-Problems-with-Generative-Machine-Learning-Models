# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:47:34 2020

@author: magd21
"""

import numpy as np
#import matplotlib.pyplot as plt
#import cv2


class generative_model(object):
    def __init__(self, image_shape, latent_dim):
        self.image_shape=image_shape
        self.n_latent=latent_dim
        pass
    def generate(self, z):
        raise NotImplementedError
    def encode(self,x):
        raise NotImplementedError 

    def elephant(self):
        print('elephant')
        return(self.encode(np.random.normal(0,1,self.image_shape)))
    def observationLoss(self, z, y):  #calculate ||G(z)-y||^2
        raise NotImplementedError 
    def gradients(self, z,y):  # ||G(z)-y||^2 differentiated wrt to z
        raise NotImplementedError
    

    

    def gradientDescentBacktracking(self,initial_z,y, iteration_number, initial_L=20,upper=5, lower=0.6):
        loss= np.zeros(np.shape(initial_z)[0])
        z=np.zeros(np.shape(initial_z))
        for j in range(initial_z.shape[0]):

            z_0=initial_z[j]
            L=initial_L
            count=0
            for i in range(100000):
                z_1=np.zeros(np.shape(z_0))
                z_1=z_0-(1/L)*self.gradients(z_0,y)
                #            print('observation loss', self.observationLoss(z_0,y))
                #           print('L', L)
                #          print('Gradient', self.gradients(z_0,y))
                if np.all(self.observationLoss(z_1,y)<= \
                              self.observationLoss(z_0,y)\
                              -(1/(2*100*L))*np.linalg.norm(self.gradients(z_0,y), ord=2)**2): # debug +1 changed to -1 on 29-04-2021
                    z_0=z_1
                    count+=1
                    L=lower*L
                else:
                    L=upper*L
        #           print(z_1-z_0)

                if count==iteration_number:
                    break

            z[j]=z_0
            loss[j]=self.observationLoss(z_0,y)

        return z, loss


    def gradientDescent(self,initial_z,y, iteration_number, alpha=0.1):
        z_0=initial_z
        for i in range(iteration_number):
            z_0=z_0-alpha*self.gradients(z_0,y)
        loss= np.zeros(np.shape(initial_z)[0])
        for i in range(np.shape(initial_z)[0]):
            loss[i]=self.observationLoss(z_0[i,:],y[i,:,:])
        return z_0, loss
    
    
    def encodeGenerateLoss(self,x, saveName='test', restore=None, start=0):
        print(np.shape(x))
        if np.shape(np.shape(x))==(3,):
            if np.shape(x)[1:]==self.image_shape:
                number=np.shape(x)[0]
            else:
                print('Images are of incorrect size')
                raise
        elif np.shape(x)==self.image_shape:
            number=1
            x = np.expand_dims(x, axis=0)
        else:
            print('Images are of incorrect size')
            raise
        if restore==None:
            loss=np.zeros(number)
        else:
            loss=np.load(restore)
        for i in range(start,number):
            loss[i]=self.observationLoss(self.encode(x[i,:,:]), x[i,:,:])
            if i%50==0:
                np.save(saveName+'Checkpoint'+str(i)+'.npy', loss)
        np.save(saveName+'Final.npy', loss)
        return(loss)
    
    def latentSpaceValues(self,x, saveName='test', restore=True):
        print(np.shape(x))
        if np.shape(np.shape(x))==(3,):
            if np.shape(x)[1:]==self.image_shape:
                number=np.shape(x)[0]
            else:
                print('Images are of incorrect size')
                raise
        elif np.shape(x)==self.image_shape:
            number=1
            x = np.expand_dims(x, axis=0)
        else:
            print('Images are of incorrect size')
            raise
        if not restore:
            loss=np.zeros(number)
            encodings=np.zeros((number, self.n_latent))
            start=0
        else:
            import glob
            import os
            filesEncodings= glob.glob(saveName+'CheckpointEncodings*.npy')
            filesLoss=glob.glob(saveName+'CheckpointLoss*.npy')
            try:
                filesEncodings.sort(key=os.path.getmtime)
                filesLoss.sort(key=os.path.getmtime)
                encodings=np.load(filesEncodings[-1])
                loss=np.load(filesLoss[-1])
                start=int(filesEncodings[-1][len(saveName+'CheckpointEncodings'):-4])
                print('Restored encodings at number' +str(start))
            except:
                loss=np.zeros(number)
                encodings=np.zeros((number, self.n_latent))
                start=0

        for i in range(start,number):
            encodings[i,:]=self.encode(x[i,:,:])
            loss[i]=self.observationLoss(encodings[i,:], x[i,:,:])
            if i%50==0:
                try:
                    os.remove(saveName+'CheckpointEncodings'+str(i-50)+'.npy')
                except:	
                    pass
			
                np.save(saveName+'CheckpointLoss'+str(i)+'.npy', loss)
                try:
                    os.remove(saveName+'CheckpointLoss'+str(i-50)+'.npy')
                except:	
                    pass
        np.save(saveName+'Encodings.npy', encodings)
        np.save(saveName+'InOutLoss.npy',loss)
        return(encodings,loss)

    def projectManifold(self,x, saveName='test', noise_level=0.1, random_seed=99):
        print(np.shape(x))
        np.random.seed(random_seed)
        if np.shape(np.shape(x))==(3,):
            if np.shape(x)[1:]==self.image_shape:
                number=np.shape(x)[0]
            else:
                print('Images are of incorrect size')
                raise
        elif np.shape(x)==self.image_shape:
            number=1
            x = np.expand_dims(x, axis=0)
        else:
            print('Images are of incorrect size')
            raise
        if not restore:
            loss=np.zeros(number)
            encodings=np.zeros((number, self.n_latent))
            start=0
        else:
            import glob
            import os
            filesEncodings= glob.glob(saveName+'CheckpointNoisyEncodings*.npy')
            filesLoss=glob.glob(saveName+'CheckpointNoisyLoss*.npy')
            try:
                filesEncodings.sort(key=os.path.getmtime)
                filesLoss.sort(key=os.path.getmtime)
                encodings=np.load(filesEncodings[-1])
                loss=np.load(filesLoss[-1])
                start=int(filesEncodings[-1][len(saveName+'CheckpointEncodings'):-4])
                print('Restored encodings at number' +str(start))
            except:
                loss=np.zeros(number)
                encodings=np.zeros((number, self.n_latent))
                start=0



        x_noise=x+noise_level*np.random.normal(0,1, x.shape) #Add noise here!!
        for i in range(start,number):
            encodings[i,:]=self.encode(x_noise[i,:,:])
            loss[i]=self.observationLoss(encodings[i,:], x[i,:,:])
            if i%50==0:
                np.save(saveName+'CheckpointNoisyEncodings'+str(i)+'.npy', encodings)
                try:
                    os.remove(saveName+'CheckpointNoisyEncodings'+str(i-50)+'.npy')
                except:	
                    pass
			
                np.save(saveName+'CheckpointNoisyLoss'+str(i)+'.npy', loss)
                try:
                    os.remove(saveName+'CheckpointNoisyLoss'+str(i-50)+'.npy')
                except:	
                    pass
			
        np.save(saveName+'NoisyEncodings.npy', encodings)
        np.save(saveName+'NoisyInOutLoss.npy',loss)
        return(encodings,loss)
    
    def histInOutLoss(self, datasets, labels, bins=None, fileName='Test'):
        fig=plt.figure()
        if bins==None:
            bins=np.linspace(0,300,301)
        for i,dataset in enumerate(datasets): 
            data= self.encodeGenerateLoss(dataset, saveName=fileName+'_'+labels[i])
            plt.hist(data, bins,density=True, cumulative=False, alpha=0.6)
        plt.legend(labels)
        plt.savefig( 'Histogram encode generate loss.png'   )
        return(fig)
    
    def interpolation(self,z_0,z_1,z_2, intervals=100, save=False, save_name='test'):
        for i, lam in enumerate(np.linspace(0,1,intervals)):
            for j, lam2 in enumerate(np.linspace(0,1,intervals)): 
                z= z_0+ lam*(z_1-z_0)+lam2*(z_2-z_0)
                img=self.generate(z)[0,:,:]
                if save:
                    np.save(save_name+'_image_'+str(i)+'_'+str(j)+'.npy', img)
        
    def random_projection_image(self, image_set, save_name='Test'):
        images=np.zeros(image_set.shape)
        for i in range(images.shape[0]):
            images[i]=self.generate(np.random.normal(0,1, self.n_latent))[0,:,:]
        images=np.reshape(images, (-1,28*28))
        image_set=np.reshape(image_set, (-1,28*28))
        R=np.random.normal(0,1,(2,28*28))
        R_normed = R / np.linalg.norm(R,axis=0)
        generated_proj=np.matmul(R_normed, np.transpose(images)).transpose()
        test_proj=np.matmul(R_normed, np.transpose(image_set)).transpose()
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        fig.suptitle('Random projection into 2d of generated and test images - '+save_name)
        ax1.scatter(test_proj[:,0],test_proj[:,1], c='blue', marker='+', alpha=0.1)
        ax1.legend(['Test images'])
        ax2.scatter(generated_proj[:,0],generated_proj[:,1], c='red', marker='+', alpha=0.1)
        ax2.legend(['Generated images'])
        plt.savefig(save_name+'_image_projections.png')
        plt.show()   
         
    def emd_images(self, image_set, save_name='Test'):
        import ot 
        images=np.zeros(image_set.shape)
        for i in range(images.shape[0]):
            images[i]=self.generate(np.random.normal(0,1, self.n_latent))[0,:,:]
        images=np.reshape(images, (-1,28*28))
        image_set=np.reshape(image_set, (-1,28*28))
        M=ot.dist(images, image_set, metric='sqeuclidean')
        cost=ot.lp.emd2([], [], M, processes=36, numItermax=10000000, log=False, return_matrix=False)
        print(save_name+' = '+ str(cost))
        return(cost)

    def observationLossRandomPlots(self, image, dimensions=1, minimum=-3, maximum=3,save=False,  save_name='Test'):
        z1=np.random.uniform(0,1, size= self.n_latent)
        z1=z1 / np.linalg.norm(z1)
        z2=np.random.uniform(0,1, size= self.n_latent)
        z2=z2 / np.linalg.norm(z2)
        z=self.encode(image)[0]
        lam=np.linspace(minimum, maximum, 200)
        plot_hold=np.zeros((200,200))
        for i in range(200):
            for j in range(200):
                plot_hold[i,j]=self.observationLoss(z+lam[i]*z1+lam[j]*z2, image)
                
        plot_x, plot_y=np.meshgrid(lam, lam, indexing='ij')
        fig = plt.figure()
        plt.pcolormesh(plot_x, plot_y, plot_hold, shading='gouraud', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.contour(plot_x, plot_y, plot_hold, levels=10, cmap=plt.cm.Purples)
        
        if save:
            plt.savefig('Contour_plot_reconstruction_loss'+save_name+'.png')
        plt.show()
 
        
            
        
                

        
    # def randomZfixedY(self,number_z, y, method='gradientDescent'):
    #     fig=plt.figure()
    #     bins=np.linspace(0,300,301)
    #     if method=='gradientDescent':
    #         for i in range(np.shape(y)[0]):
    #             _,losses=self.gradientDescent(np.random.normal(0,1,(number_z, self.n_latent)),y[i,:,:],100)
    #             plt.hist(losses, bins, density=True, histtype='step', cumulative=True, label='Empirical')
       
    #     if method=='gradientDescentBacktracking':
    #         for i in range(np.shape(y)[0]):
    #             _,losses=self.gradientDescentBacktracking(np.random.normal(0,1,(number_z, self.n_latent)),y[i,:,:],100)
    #             plt.hist(losses, bins, density=True, histtype='step', cumulative=True, label='Empirical')
    #     plt.show()
    #     return(fig)
    
    

    

    # def basinOfAttraction(self, y):
    #     if np.shape(np.shape(y))==(3,):
    #         if np.shape(y)==(1,*self.image_shape):
    #             y=y[0,:,:]
    #     elif np.shape(y)==self.image_shape:
    #         pass
    #     else:
    #         print('Please give a sinlge 28x28 image')
    #         raise
    #     z_0,loss=self.gradientDescentBacktracking(self.encode(y),y, 20)
    #     if loss<15:
    #         pass
    #     elif loss>=15:
    #         plt.imshow(y)
    #         plt.show
    #         plt.imshow(self.generate(z_0)[0,:,:])
    #         plt.show()
    #         print('Have not found a good starting point to continue the test')
    #         raise
    #     successPlot=np.zeros(11)
    #     count=0
    #     successCount=0
    #     for item,distance in enumerate(np.linspace(0,1,11)):
    #         for repeat in range(100):
    #             z=z_0+np.random.uniform(distance, distance+0.1, (1,self.n_latent))
    #             _,loss=self.gradientDescentBacktracking(z,y, 20)
    #             count+=1
    #             if loss<15:
    #                 successCount+=1
    #         successPlot[item]=successCount/count
    #     plt.plot(np.linspace(0,1,11), successPlot)
            

        
    
    

