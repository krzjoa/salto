import numpy as np
import numpy.typing as npt


class axis:
    '''
    
    Create a new axis using the line passing through the point A (negative pole)
    and B (positive pole).
    
    Parameters
    ----------
    negative_pole: numpy.array
        A vector (point), which indicates negative side of the axis
    positive_pole: numpy.array
        A vector (point), which indicates positive side of the axis
     
    Attributes
    ----------
    dims: int
        The expected number of dimensions    
    M: numpy.array
        The midpoint between negative pole and positive pole points
    transform: numpy.array
        A matrix to shift the space to place the midpoint in the origin
     
     
    Examples
    --------
    import numpy as np
    import spacy
    import salto

    nlp = spacy.load('en_core_web_md')

    fire = nlp('fire')
    ice = nlp('ice')

    ice_fire_axis = salto.axis(ice.vector, fire.vector)

    cold = ['ice cream', 'polar', 'snow', 'winter', 'fridge', 'Antarctica']
    warm = ['boiling water', 'tropical', 'sun', 'summer', 'oven', 'Africa']

    cold_vecs = [nlp(w).vector for w in cold]
    warm_vecs = [nlp(w).vector for w in warm]

    cold_values = [ice_fire_axis(p) for p in cold_vecs]
    warm_values = [ice_fire_axis(p) for p in warm_vecs]

    axis.plot(
        {values: cold_values, labels: cold, color: 'blue'},
        {values: warm_values, labels: warm, color: 'red'},
        show_poles = True
    )
        
    '''
    def __init__(self, negative_pole: npt.NDArray, positive_pole: npt.NDArray):
        
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

    def __call__(self, V: npt.NDArray):
        
        if (len(V) != self.dims):
            raise ValueError(
                f'Vector's lenth is {len(V)}, but it should equal {self.dims}' 
            )
        
        proj_V = self._project(V)
        dist = np.linalg.norm(proj_V)
        dist_to_shifted_A = np.linalg.norm(proj_V - self.shifted_A)
        dist_to_shifted_B = np.linalg.norm(proj_V - self.shifted_B)
        sign = 1 if dist_to_shifted_A < dist_to_shifted_B else -1
        value = sign * dist
        return value
    
    def plot(*args):
        return        

    def _shift(self, V: npt.NDArray):
        extended_V = _extend_with_one(V)
        return (self.transform @ extended_V)[0:self.dims]
    
    def _project(self, V: npt.NDArray):
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
