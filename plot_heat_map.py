import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))



score = np.array([[0.7930014 , 0.79391466, 0.68049663, 0.68049663, 0.68049663,
    0.68049663],
   [0.79932488, 0.82785568, 0.83686115, 0.68049663, 0.68049663,
    0.68049663],
   [0.84791279, 0.86098748, 0.86343826, 0.85052542, 0.71332786,
    0.68375663],
   [0.86754217, 0.8713455 , 0.87109118, 0.86147301, 0.83515023,
    0.79361409],
   [0.87390033, 0.87277898, 0.86844387, 0.85182017, 0.82887299,
    0.79282799],
   [0.87464019, 0.87135706, 0.86527635, 0.84512676, 0.82629504,
    0.79393778]])

plt.figure(figsize=(8,6))
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(score, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(6), [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], rotation=45)
plt.yticks(np.arange(6), [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
plt.title('Validation accuracy')
plt.show()
