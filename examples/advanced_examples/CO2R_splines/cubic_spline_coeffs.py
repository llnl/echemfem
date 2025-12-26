from scipy.interpolate import CubicSpline, PchipInterpolator
import pandas as pd
import numpy as np

def cubic_spline(csv_file, U_value, flux_column, a='aCO', bounds=(0,1), method='auto', n_max=50):
    """
    Load CSV file and return spline (CubicSpline or PchipInterpolator) for given U and flux_column.

    method: 'cubic' (default), 'pchip', or 'auto' (tries cubic, switches to pchip if overshoot detected)
    """
    data = pd.read_csv(csv_file)
    lo_bnd, up_bnd = bounds
    data = data[(data[a] >= lo_bnd) & (data[a] <= up_bnd)]

    if U_value in data['U'].values:
        a_data, flux_data = get_data(data, U_value, flux_column, a)
    else:
        # get closest voltage values with existing data
        data['diff'] = np.abs(data['U'] - U_value)
        data_unique = data.drop_duplicates(subset=['U'])
        closest_values = data_unique.nsmallest(2, 'diff')['U'].values
        a_data, flux_data0 = get_data(data, closest_values[0], flux_column, a)
        a_data, flux_data1 = get_data(data, closest_values[1], flux_column, a)
        # Log-Linear interpolation between existing data
        diff0 = np.abs(closest_values[0] - U_value)
        diff1 = np.abs(closest_values[1] - U_value)
        weight1 = diff0 / (diff0 + diff1)
        weight0 = 1 - weight1
        #flux_data = weight0 * flux_data0 + weight1 * flux_data1 # linear interpo
        # Avoid log(0)
        epsilon = 1e-20

        # Take log of absolute value of fluxes
        log_flux0 = np.log(np.abs(flux_data0) + epsilon)
        log_flux1 = np.log(np.abs(flux_data1) + epsilon)

        # Interpolate in log space
        log_flux_interp = weight0 * log_flux0 + weight1 * log_flux1

        # Recover interpolated flux, preserving sign of flux_data0
        flux_data = np.sign(flux_data0) * np.exp(log_flux_interp)

    if False: 
    #if flux_column == "flux_CH4":
        # Smooth data
        epsilon = 1e-20
        log_flux_data = np.log(np.abs(flux_data) + epsilon)

        from scipy.signal import savgol_filter

        log_flux_data_smooth = savgol_filter(log_flux_data, window_length=11, polyorder=3)

        # Recover smoothed flux, preserving sign
        flux_data = np.sign(flux_data) * np.exp(log_flux_data_smooth)

    # Select spline method
    if method == 'cubic':
        cs = CubicSpline(a_data, flux_data, bc_type='natural')
    elif method == 'pchip':
        cs = PchipInterpolator(a_data, flux_data)
    elif method == 'auto':
        # Try CubicSpline first
        cs_try = CubicSpline(a_data, flux_data, bc_type='natural')
        aCO_fine = np.linspace(a_data[0], a_data[-1], 1000)
        flux_fine = cs_try(aCO_fine)
        # If overshoot detected, switch to Pchip
        if np.any(flux_fine > np.max(flux_data)) or np.any(flux_fine < np.min(flux_data)):
            print("WARNING: CubicSpline overshoot detected: switching to PchipInterpolator.")
            cs = PchipInterpolator(a_data, flux_data)
            method = "pchip"
        else:
            cs = cs_try
            method = "cubic"
    else:
        raise ValueError("Invalid method. Choose 'cubic', 'pchip', or 'auto'.")

    if len(a_data) > n_max:
        # coarsen interpolation uniformly
        a_coarse = np.linspace(a_data[0], a_data[-1], n_max)
        flux_coarse = cs(a_coarse)
        if method == 'cubic':
            cs = CubicSpline(a_coarse, flux_coarse, bc_type='natural')
        elif method == 'pchip':
            cs = PchipInterpolator(a_coarse, flux_coarse)
        a_data = a_coarse

    return cs, a_data


def get_data(data, U_value, flux_column, a):
    # Filter the DataFrame for the specific U value
    filtered_data = data[data['U'] == U_value]
    
    # Extract the activity values and the selected flux values
    a_data = filtered_data[a].to_numpy()
    flux_data = filtered_data[flux_column].to_numpy()

    # Sort a_data and rearrange flux_data accordingly
    sorted_indices = np.argsort(a_data)
    a_data = a_data[sorted_indices]
    flux_data = flux_data[sorted_indices]

    return a_data, flux_data

def cubic_spline_coeffs(csv_file, U_value, flux_column, a='aCO', bounds=(0,1)):
# from csv file, return cubic spline coefficients for a specific U value a flux
    cs, a_data = cubic_spline(csv_file, U_value, flux_column, a=a, bounds=bounds, method='pchip')
    if hasattr(cs, 'c'):
        coeffs = cs.c  # CubicSpline: aalready in correct format
    else:
        # Assume PchipInterpolator: use helper to extract coeffs
        coeffs, a_data = pchip_to_coeffs(cs)

    return coeffs, a_data

def pchip_to_coeffs(pchip):
    """
    Extract piecewise cubic coefficients from a PchipInterpolator object.

    Returns:
        coeffs: shape (4, n_intervals), ordered as [cubic, quadratic, linear, constant]
        x_data: breakpoints (x0, x1, ..., xn)
    """
    x_data = pchip.x
    y_data = pchip.y
    d = pchip.derivatives(x_data)

    coeffs = np.zeros((4, len(x_data) - 1))
    dx = np.diff(x_data)
    dy = np.diff(y_data)

    for i in range(len(dx)):
        h = dx[i]
        delta = dy[i] / h

        d0 = d[i]
        d1 = d[i + 1]

        coeffs[0, i] = (d0 + d1 - 2 * delta) / (h ** 2)    # cubic
        coeffs[1, i] = (3 * delta - 2 * d0 - d1) / h       # quadratic
        coeffs[2, i] = d0                                  # linear
        coeffs[3, i] = y_data[i]                           # constant

    return coeffs, x_data

