# # -*- coding: utf-8 -*-
# """
# Created on Mon Jun  1 12:30:12 2020





# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:30:12 2020

@author: marga
"""

import numpy as np
import odl

class invProb():
    def __init__(self,  generative_model, name='denoising',   kernel=(1/273)*np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]]),dim_comp=4800, angles=160, inpainting_frac=0.25):
        self.name=name
        
        self.generative_model=generative_model
        if self.name=='denoising': 
        ##For denosiing
            self.x_space=self.generative_model.x_space
            self.G=self.generative_model.generator
            self.y_space=self.x_space
            self.A=odl.operator.default_ops.IdentityOperator(self.y_space)
            self.A/=self.A.norm(estimate=True)
            

        elif self.name=='compressed_sensing':
            self.x_space=self.generative_model.x_space
            self.G=self.generative_model.generator
            self.dim_comp=dim_comp
            flatten=odl.operator.tensor_ops.FlatteningOperator(self.x_space)
            matrix=np.random.normal(0,1/self.dim_comp, (self.dim_comp, self.x_space.size) ).astype('float32')     
            cs=odl.operator.tensor_ops.MatrixOperator(matrix, domain=flatten.range)
            self.A=odl.operator.operator.OperatorComp(cs, flatten)
            self.A/=self.A.norm(estimate=True)
            self.y_space=self.A.range
            
        elif self.name=='convolution':
            ##For convolution 
            import convolution
            self.x_space=self.generative_model.x_space
            self.G=self.generative_model.generator
            self.kernel= kernel
            pad1=round((self.x_space.shape[0]-self.kernel.shape[0])/2)
            pad2=round((self.x_space.shape[1]-self.kernel.shape[1])/2)
            kernel_pad=np.pad(self.kernel, ((pad1,pad1+1), (pad2,pad2+1)),mode='constant')[:self.x_space.shape[0],:self.x_space.shape[1]]
            self.odl_kernel=self.x_space.element(kernel_pad)
            self.A = convolution.Convolution(self.x_space, self.odl_kernel)
            self.A/=self.A.norm(estimate=True)
            self.y_space=self.x_space
            
        elif self.name=='inpainting':  
            self.x_space=self.generative_model.x_space
            self.inpainting_frac=inpainting_frac
            self.G=self.generative_model.generator
            matrix=np.diag(np.random.binomial(1,self.inpainting_frac,self.x_space.size)).astype('float32')
            flatten=odl.operator.tensor_ops.FlatteningOperator(self.x_space)
            self.inv_flatten=flatten.inverse
            cs=odl.operator.tensor_ops.MatrixOperator(matrix, domain=flatten.range)
            self.A=odl.operator.operator.OperatorComp(cs, flatten)
            #self.A=odl.operator.operator.OperatorComp(flatten.inverse, A_part)
            self.A/=self.A.norm(estimate=True)
            self.y_space=self.A.range  
        
        elif self.name=='tomography':
            self.x_space=self.generative_model.tomo_space
            self.G=self.generative_model.generator_tomo
            geometry = odl.tomo.parallel_beam_geometry(self.x_space)
            self.A = odl.tomo.RayTransform(self.x_space, geometry)
            self.A/=self.A.norm(estimate=True)
            self.y_space=self.A.range
            
            
        elif self.name=='missing_angle_tomo':
            self.angles=angles
            self.x_space=self.generative_model.tomo_space
            self.G=self.generative_model.generator_tomo
            geometry = odl.tomo.parallel_beam_geometry(self.x_space)
            ray_transform = odl.tomo.RayTransform(self.x_space, geometry)
            matrix=np.diag(np.random.binomial(1,self.angles/ray_transform.range.shape[0],ray_transform.range.shape[0])).astype('float32')
            cs=odl.operator.tensor_ops.MatrixOperator(matrix, domain=ray_transform.range)
            self.A=odl.operator.operator.OperatorComp(cs, ray_transform) 
            
            self.A/=self.A.norm(estimate=True)
            self.y_space=self.A.range
            
        else:
            print('This is not currently supported. Please choose from: denosing, compressed_sensing, inpainting, tomography,   convolution, missing_angle_tomo')
            return(NameError)
    def set_data(self, data):
        self.data=self.y_space.element(data)
        f1= odl.solvers.L2NormSquared(self.y_space).translated(self.data)
        self.data_discrepancy=odl.solvers.FunctionalComp(f1,self.A)

    def observe_data(self, aim, noise_level=0.1, random_seed=99):
        np.random.seed(random_seed)
        self.aim=self.x_space.element(aim)
        self.noise_level=noise_level
        self.data=self.A(self.aim)+self.noise_level*np.random.normal(0,1, self.y_space.shape)
        f1= odl.solvers.L2NormSquared(self.y_space).translated(self.data)
        self.data_discrepancy=odl.solvers.FunctionalComp(f1,self.A)

        
    
