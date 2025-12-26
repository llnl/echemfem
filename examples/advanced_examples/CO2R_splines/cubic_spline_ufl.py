from firedrake import conditional, And, Constant, ge
import numpy as np

def cubic_spline_ufl(x, x_data, coeffs):
    # Create the piecewise cubic function
    spline = 0
    for i in range(len(x_data) - 1):
        x0, x1 = x_data[i], x_data[i + 1]
        # Cubic polynomial for the interval
        cubic_poly = (
            coeffs[3, i]
            + coeffs[2, i] * (x - x0)
            + coeffs[1, i] * (x - x0)**2
            + coeffs[0, i] * (x - x0)**3
        )
        
        # Add the piece to the spline with the appropriate condition
        spline += conditional(And(x >= x0, x < x1), cubic_poly, 0)
    
    # for last point, maintain constant value outside data range
    spline += conditional(ge(x, x_data[-1]), coeffs[3, -1], 0)

    # for last point, maintain linear value outside data range
    #slope = (coeffs[3, -1] - coeffs[3, -2]) / (x_data[-1] - x_data[-2])
    #spline += conditional(ge(x, x_data[-1]), slope * (x - x_data[-1]) + coeffs[3, -1], 0)

    return spline

class CubicSplineUFL(object):
    def __init__(self, x_data, coeffs):
        self.x_data = x_data
        self.coeffs = np.vectorize(Constant)(coeffs)
    
    def spline(self, x):
        return cubic_spline_ufl(x, self.x_data, self.coeffs)

    def update_coeffs(self, coeffs):
        for i in range(self.coeffs.shape[0]):
            for j in range(self.coeffs.shape[1]):
                self.coeffs[i, j].assign(coeffs[i, j])

