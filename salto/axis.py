import numpy as np
import numpy.typing as npt


class axis:
    '''
    
    Create a new axis using a vector from point A ('negative pole')
    to point B ('positive pole') as a span vector.
    
    Attributes
    ----------
    negative_pole: numpy.array
        A vector (point), which indicates negative side of the axis
    positive_pole: numpy.array
        A vector (point), which indicates positive side of the axis
    
    Examples
    --------
        
    '''
    def __init__(self, negative_pole, positive_pole):
        
        self.dims = len(negative_pole)
        
        # Original values
        self.A = negative_pole 
        self.B = positive_pole
        self.M = _midpoint(negative_pole, positive_pole)  
        
        # Transformation
        self.transform = _transformation_matrix(self.dims, self.M)       
        self.shifted_A = self.shift(negative_pole)
        self.shifted_B = self.shift(positive_pole)
        
        self.direction = self.shifted_B / np.linalg.norm(self.shifted_B)
        # shifted_M is (0, 0, ...) 

    def __call__(self, V):
        proj_V = self._project(V)
        dist = np.linalg.norm(proj_V)
        dist_to_A = np.linalg.norm(proj_V - self.shifted_A)
        dist_to_B = np.linalg.norm(proj_V - self.shifted_B)
        sign = 1 if dist_to_B < dist_to_A else -1
        value = sign * dist
        return value
    
    def plot(*args):
        return        

    def _shift(self, V):
        extended_V = _extend_with_one(V)
        return (self.transform @ extended_V)[0:self.dims]
    
    def _project(self, V):
        shifted_V = self.shift(V)
        return (shifted_V.T @ self.direction) / (self.direction.T @ self.direction) * self.direction
    
   

    
def _extend_with_one(V: npt.NDArray) -> npt.NDArray:
    V = V[np.newaxis].T
    return np.vstack([V, [1]])      

def _transformation_matrix(dims: int, M: npt.NDArray) -> npt.NDArray:
    mat = np.eye(dims + 1)
    mat[0:dims, -1] = -M
    return mat   
    
def _midpoint(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    if (len(x) != len(y)):
        raise ValueError(
            f'Vectors come from different spaces! ' + 
            f'x: {len(x)} dimensions, y: {len(y)} dimensions')
    return (x + y) / 2