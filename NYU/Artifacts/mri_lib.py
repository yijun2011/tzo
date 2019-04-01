import numpy as np
import skimage
from keras.models import load_model
import random

laplacian = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]])

def hist_match(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, indices, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[indices].reshape(oldshape)

"""
Library to simulate repeatedly applying kernels to deblur an image

Constructor takes in:
@param image: The starting image, a numpy 2d array
"""
class MRILib:

    def __init__(self, image, path_to_model, dim = 16, cutoff = 0.9):
        self.filter = np.zeros((dim,dim))
        self.dim = dim
        self.image = image
        self.done = False
        self.cutoff = cutoff
        self.model = load_model(path_to_model)
        self.original_blur_score = self.model.predict(np.expand_dims(self.image, axis = 0))[0,0]

        self.starts = []
        for i in range(self.dim * self.dim):
            arr = np.zeros(self.dim * self.dim)
            arr[i] = 1.0
            self.starts.append(np.reshape(arr, (self.dim, self.dim)))


    
    """
    @param action1: The pixel number to activate
    @param action2: The pixel number to deactivate
    @return a tuple containing:
    filter: The current guessed filter
    done: Whether the current trajectory has finished
    blur_score: The reward of this guessed filter
    In that order
    """
    def step(self, action1):
        if (self.filter[int(action1/self.dim), action1 % self.dim] == 1):
            self.done = True
        self.filter[int(action1/self.dim), action1 % self.dim] = 1
        total = np.sum(self.filter, axis = None)
        normalized = self.filter / max(1, total)
        tmp = skimage.restoration.unsupervised_wiener(self.image, self.filter)[0]
        histmatched = hist_match(self.image, tmp)
        

        #REWARD FUNCTION GOES HERE:
        blur_score = self.model.predict(np.expand_dims(histmatched, axis = 0))[0,0] * 1000
        #blur_score = cv2.Laplacian(histmatched, cv2.CV_64F).var() * 10000
 
        return self.filter, self.done, blur_score 



    """
    Reset the guessed filter back to a random start state
    """
    def reset(self):
        self.done = False
        self.filter = np.copy(random.choice(self.starts))
        return self.filter
        
