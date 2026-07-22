import numpy as np
from scipy.interpolate import *
from scipy.spatial import *


def LATs_acquisition(cartoPoints, vertices):

    aux_LATs_positions = cartoPoints.Point_LAT.copy()
    aux_voltages = cartoPoints.Point_Bipolar.copy()
    if aux_LATs_positions.size > 1:

        indices_LATs_buenos = np.where((aux_LATs_positions >= 0) & (aux_voltages >= 0.1))[0]

        LATs_values = aux_LATs_positions[indices_LATs_buenos]
        LATs_pos = np.column_stack((cartoPoints.Point_X[indices_LATs_buenos], cartoPoints.Point_Y[indices_LATs_buenos]))
        LATs_pos = np.column_stack((LATs_pos, cartoPoints.Point_Z[indices_LATs_buenos]))

        # Removing the offset of the original points so that they are aligned with the interpolated mesh
        offset = np.mean(vertices, axis=0)
        if LATs_pos.size > 0:
            LATs_pos = LATs_pos - offset

        #Add the LATs to the position matrix
        LATs = np.column_stack((LATs_pos, LATs_values))
    else:
        LATs = np.array(0)
        LATs_pos = np.array(0)
        LATs_values = np.array(0)

    return [LATs_pos, LATs, LATs_values]


def Voltages_acquisition(cartoPoints, vertices):

    aux_voltages = cartoPoints.Point_Bipolar.copy()
    if aux_voltages.size > 1:
        indices_voltages_buenos = np.where(aux_voltages > 0.01)[0]
        Voltages_values = aux_voltages[indices_voltages_buenos]

        Voltages_pos = np.column_stack((cartoPoints.Point_X[indices_voltages_buenos], cartoPoints.Point_Y[indices_voltages_buenos]))
        Voltages_pos = np.column_stack((Voltages_pos, cartoPoints.Point_Z[indices_voltages_buenos]))

        # Removing the offset of the original points so that they are aligned with the interpolated mesh
        offset = np.mean(vertices, axis=0)
        if Voltages_pos.size > 0:
            Voltages_pos = Voltages_pos - offset

        #Add the LATs to the position matrix
        Voltages = np.column_stack((Voltages_pos, Voltages_values))
    else:
        Voltages = np.array(0)
        Voltages_pos = np.array(0)
        Voltages_values = np.array(0)

    return [Voltages_pos, Voltages, Voltages_values]

def interpolateLinearBarycentric(data_points, data_values, vertices, search_distance):

    interpolation = LinearNDInterpolator(data_points, data_values)
    LATs_interpolated = interpolation(vertices)

    # Making 0 the LATs interpolated of the vertices that do not have a close enough point
    Mesh_to_points_distance = distance.cdist(vertices, data_points, 'euclidean')
    Num_vertices = vertices.shape[0]
    for v in range(Num_vertices):
        aux_distance = Mesh_to_points_distance[v, :]
        aux_search_indices = np.where(aux_distance < search_distance)[0]
        if len(aux_search_indices) == 0:
            LATs_interpolated[v] = 0

    return LATs_interpolated

# nan_helper and nan_solver do a linear interpolation with the nan values obtained with the previous linear interpolation


def nan_helper(y):

    return np.isnan(y), lambda z: z.nonzero()[0]


def nan_solver(interpolated_data):

    nan_pos, x = nan_helper(interpolated_data)
    interpolated_data[nan_pos] = np.interp(x(nan_pos), x(~nan_pos), interpolated_data[~nan_pos])
    return interpolated_data


def interpolateRBF(data_points, data_values, vertices, search_distance):

    interpolation = RBFInterpolator(data_points, data_values)
    LATs_interpolated = interpolation(vertices)

    Mesh_to_points_distance = distance.cdist(vertices, data_points, 'euclidean')
    Num_vertices = vertices.shape[0]
    for v in range(Num_vertices):
        aux_distance = Mesh_to_points_distance[v, :]
        aux_search_indices = np.where(aux_distance < search_distance)[0]
        if len(aux_search_indices) == 0:
            LATs_interpolated[v] = 0

    return LATs_interpolated


