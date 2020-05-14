from pdb_utils import *
from collections import defaultdict
from scipy.spatial import ConvexHull, cKDTree
import numpy as np
from tqdm import tqdm


def keep_heavy_bbo_only(pdb_lines_dictionary):
    data = defaultdict(list)
    for idx, atom_name in enumerate(pdb_lines_dictionary['atom_name']):
        if atom_name.strip() in ['CA', 'O', 'N', 'C']:
            for column in pdb_lines_dictionary:
                try:
                    data[column].append(pdb_lines_dictionary[column][idx])
                except IndexError:
                    print(column)
    print('From {0} atoms, kept {1} backbone atoms "CA, O, N"'.format(len(pdb_lines_dictionary['element']),
                                                                      len(data['element'])))

    return data


def create_kdtree(data):
    coord_array = get_coord_array(data)
    data_kd = cKDTree(coord_array)
    print('Created KDTree')

    return data_kd


def create_convex_hull(data):
    coord_array = get_coord_array(data)
    data_convexhull = ConvexHull(coord_array, qhull_options='Qt')
    print('Created convex hull')
    vertices = coord_array[data_convexhull.vertices, :]
    simplices = coord_array[data_convexhull.simplices, :]
    print('Found {0} vertices'.format(len(vertices)))
    print('Found {0} simplices'.format(len(simplices)))

    return vertices, simplices


def get_outermost_surface(data):
    vertices, simplices = create_convex_hull(data)
    coord_array = get_coord_array(data)
    outer_surface_indices = []
    print('Checking if any point lies on the planes defined by the Convex Hull Simplices')
    for idx, xyz_point in enumerate(tqdm(coord_array)):
        for simplex in simplices:
            simplex_array = np.array(simplex)
            in_plane = is_in_plane(xyz_point, simplex_array)
            if in_plane is True:
                outer_surface_indices.append(idx)  # outer_surface.append(xyz_point)
                break
    # outer_surface_array = np.array(outer_surface)

    return outer_surface_indices


def fit_check(shell, crystal, fit=True):  # Can modify it to exit if the fit is found to be false
    shell_kd = create_kdtree(shell)
    shell_coord_array = get_coord_array(shell)
    crysatm_coord_array = get_coord_array(crystal)
    spill_crystal_indices = []  # spill_crystal_coords = []
    clash_shell_indices = []  # clash_shell_coords = []
    nn_distances = []
    for index in tqdm(range(len(crysatm_coord_array))):
        crysatm_coords = crysatm_coord_array[index]
        nn_dist, nn_idx = shell_kd.query(crysatm_coords)
        nn_coords = shell_coord_array[nn_idx, :]
        len_nncoord = np.linalg.norm(nn_coords)
        len_crysatm = np.linalg.norm(crysatm_coords)
        nn_distances.append(nn_dist)
        if len_nncoord < len_crysatm:
            spill_crystal_indices.append(index)  # spill_crystal_coords.append([i for i in crysatm_coords])
            clash_shell_indices.append(nn_idx)  # clash_shell_coords.append([i for i in nn_coords])
            fit = False
        # else:
            # spill_crystal_coords.append([None, None, None])
            # clash_shell_coords.append([None, None, None])
    print('Crystal fitting in cage = '+str(fit))

    return fit, spill_crystal_indices, clash_shell_indices, nn_distances


def fine_fit(shell, crystal, neighbor_cutoff_distance):
    shell_kd = create_kdtree(shell)
    shell_coord_array = get_coord_array(shell)
    crysatm_coord_array = get_coord_array(crystal)
    nn_data_in = []
    nn_data_out = []
    for index in tqdm(range(len(crysatm_coord_array))):
        crysatm_coords = crysatm_coord_array[index]
        len_crysatm = np.linalg.norm(crysatm_coords)
        nn_indices = shell_kd.query_ball_point(crysatm_coords, neighbor_cutoff_distance)
        nn_in_temp = []
        nn_out_temp = []
        for idx_2 in nn_indices:
            len_nncoord = np.linalg.norm(shell_coord_array[idx_2])
            if len_nncoord < len_crysatm:
                nn_out_temp.append(idx_2)
            else:
                nn_in_temp.append(idx_2)
        nn_data_in.append(nn_in_temp)
        nn_data_out.append(nn_out_temp)

    return nn_data_in, nn_data_out


def visualize_fine_fit(nn_data, shell_data, crystal_data):
    crystal_indices = []
    shell_indices = []
    for idx, sub_list in enumerate(nn_data):
        if len(sub_list) >= 1:
            crystal_indices.append(idx)
            for idx_2 in sub_list:
                shell_indices.append(idx_2)
    set_shell_indices = set(shell_indices)
    shell_indices = list(set_shell_indices)
    pdb_crystal_data = indices_to_data_structure(crystal_indices, crystal_data)
    pdb_shell_data = indices_to_data_structure(shell_indices, shell_data)
    data_structure_to_pdb('finefit_crystal.pdb', pdb_crystal_data)
    data_structure_to_pdb('finefit_shell.pdb', pdb_shell_data)


if __name__ == '__main__':
    # Load pdb data
    shell_pdb_file = 'shell.pdb'
    crystal_pdb_file = 'crystal.pdb'   # This crystal fits inside the nanocage
    # crystal_pdb_file = 'crystal_2.pdb'  # This is a larger crystal that spills some atoms out of the nanocage pores
    shell_data = read(shell_pdb_file)
    bbo_shell_data = keep_heavy_bbo_only(shell_data)  # Get backbone atoms only
    crystal_data = read(crystal_pdb_file)

    # If you only want to consider the atoms on the outermost surface use this block
    surface_crystal_indices = get_outermost_surface(crystal_data)
    surface_crystal_data = indices_to_data_structure(surface_crystal_indices, crystal_data)
    # data_structure_to_pdb('surface_crystal_2_test.pdb', surface_crystal_data)  # Visualize results in Pymol

    # # Coarse fitting metrics
    # fit, spill_crystal_indices, clash_shell_indices, nn_distances = fit_check(bbo_shell_data, surface_crystal_data)
    # nn_dist_array = np.array(nn_distances)
    # min_nn_distance = np.amin(nn_dist_array)
    # mean_nn_distance = np.mean(nn_dist_array)
    # std_nn_distance = np.std(nn_dist_array)
    # max_nn_distance = np.amax(nn_dist_array)
    # print('Metrics of nearest neighbor distances from crystal to the closest protein backbone atom')
    # print('Minimum\t\t\t\tAverage\t\t\tStandard Deviation\tMaximum')
    # print(min_nn_distance, mean_nn_distance, std_nn_distance, max_nn_distance)
    # percentage_poking = len(spill_crystal_indices)*100/len(surface_crystal_indices)
    # print('Percentage of crystal surface atoms poking through the protein nanocage: '+str(percentage_poking))

    # # In this case, write pdbs to visualize if the atoms are actually clashing, or poking through the pores.
    # spill_crystal_data = indices_to_data_structure(spill_crystal_indices, surface_crystal_data)
    # data_structure_to_pdb('spilled_crystal.pdb', spill_crystal_data)
    # clash_shell_data = indices_to_data_structure(clash_shell_indices, bbo_shell_data)
    # data_structure_to_pdb('clashing_atoms_shell.pdb', clash_shell_data)

    # Fine fitting metrics
    neighbor_cutoff_distance = 10  # In Angstroms
    nn_data_in, nn_data_out = fine_fit(bbo_shell_data, surface_crystal_data, neighbor_cutoff_distance)
    # print(nn_data_in)
    # print(nn_data_out)
    nonzero_interactions_in = [len(x) for x in nn_data_in if len(x) > 0]
    nonzero_interactions_out = [len(x) for x in nn_data_out if len(x) > 0]
    # print(nonzero_interactions_in)
    total_interactions_in = len(nonzero_interactions_in)
    number_interactions_out = len(nonzero_interactions_out)
    print(total_interactions_in)
    print(number_interactions_out)
    # number_non_zero_neighbors_in = len([x for x in nn_data_in if len(x) > 0]) 
    # print(number_non_zero_neighbors_in)

    min_interactions = min(nonzero_interactions_in)
    average_interactions = sum(nonzero_interactions_in)/len(nonzero_interactions_in)
    max_interactions = max(nonzero_interactions_in)
    print(min_interactions)
    print(average_interactions)
    print(max_interactions)
    # visualize_fine_fit(nn_data, bbo_shell_data, surface_crystal_data)

