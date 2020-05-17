#! /usr/bin/env python3

from scipy.spatial import ConvexHull, cKDTree
# from tqdm import tqdm

from pdb_utils import *

import subprocess
import argparse
import timeit
import glob
import os

def keep_heavy_bbo_only(pdb_lines_dictionary):
    data = defaultdict(list)
    for idx, atom_name in enumerate(pdb_lines_dictionary['atom_name']):
        if atom_name.strip() in ['CA', 'O', 'N', 'C']:
            for column in pdb_lines_dictionary:
                try:
                    data[column].append(pdb_lines_dictionary[column][idx])
                except IndexError:
                    print(column)
    print('From {0} atoms, kept {1} backbone atoms "CA, O, N, C"'.format(len(pdb_lines_dictionary['element']),
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
    # for idx, xyz_point in enumerate(tqdm(coord_array)):
    for idx, xyz_point in enumerate(coord_array):
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
    # for index in tqdm(range(len(crysatm_coord_array))):
    for index in range(len(crysatm_coord_array)):
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
    # for index in tqdm(range(len(crysatm_coord_array))):
    for index in range(len(crysatm_coord_array)):
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
    data_structure_to_pdb('finefit_nanocrystal.pdb', pdb_crystal_data)  # @pylesh maybe customize this output?
    data_structure_to_pdb('finefit_nanocage.pdb', pdb_shell_data)  # @pylesh maybe customize this output?

# Three new functions:
def sort_fits(metric_list):
    'HOW THE SORTING WORKS: '
    'INDIVIDUAL FITS OF NANOPARTICLE IN CAGE SORTED AS FOLLOWS '
    '   FIRST:  index 4 = fits without a clash ( min_nn < args.clash ) ranked before fits with clash '
    '   SECOND: index 3 = fits with NO total_interactions_out ranked before fits with ANY total_interactions_out'
    '   THIRD:  index 2 = fits with more nanoparticle atoms neighboring protein atoms ranked before fits with fewer neighbors'
    return [metric_list[0]] + sorted(metric_list[1:], key = lambda x:(float(x[4]), is_anything(int(x[3])), -1*float(x[2])) )

def copy_n_first_pdb_combos(metric_list, target_directory, number):
    global HEADER_LIST, args # grap global HEADER_LIST and args variable (this method is convenient but not recommended)
    if metric_list[0] == HEADER_LIST:
        metric_list = metric_list[1:]

    for line in metric_list[:number]:
        protein_nanocage_pdb = line[0]
        inorganic_nanoparticle_pdb = line[1]

        protein_source = args.cage+'/'+protein_nanocage_pdb.replace('//','/')
        protein_target = target_directory+'/'+protein_nanocage_pdb
        if not os.path.isfile(protein_target):
            subprocess.check_output(['cp', protein_source, protein_target])

        inorganic_source = args.np+'/'+inorganic_nanoparticle_pdb.replace('//','/')
        inorganic_target = target_directory+'/'+inorganic_nanoparticle_pdb
        if not os.path.isfile(inorganic_target):
            subprocess.check_output(['cp', inorganic_source, inorganic_target])


def is_anything(number):
    assert type(number) == int, 'Error in function is_anything; "number" argument must be an integer'
    assert not number < 0, 'Error in function is_anything; "number" argument must not be negative'
    if number > 0: return 1
    else: return 0

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(prog='Evaluates fit of inorganic nanoparticles inside protein nanocages')
    parser.add_argument('-cage', '-protein_cage_directory', help='directory containing protein nanopcages', type=str, required=True )
    parser.add_argument('-np', '-nanoparticle_directory', help='directory containing inorganic nanoparticles', type=str, required=True )
    parser.add_argument('-neighbor', help='neighbor cutoff distance (default 10 Angstroms)', type=float, default=10 )
    parser.add_argument('-clash', help='cutoff distance for clash check (default is 2 Angstroms)', type=float, default=2.5 )
    parser.add_argument('-out', help='directory for output (will be made if it does not exist already)', type=str, default='crystal_fit_output' )
    parser.add_argument('-local', help='number of best matches for each individual cage to copy to subdirectories in -out directory ', type=int, default=4 )
    parser.add_argument('-all', help='number of globally best matches to copy to -out directory', type=int, default=8 )
    args = parser.parse_args()

    # Verbose input parsing statments
    print('Globbing protein input pdbs with: {0}/*.pdb'.format(args.cage))
    protein_nanocages = sorted( glob.glob( '{0}/*.pdb'.format(args.cage) ) )
    print('Globbing nanoparticle input pdbs with: {0}/*.pdb'.format(args.np))
    inorganic_nanoparticle = sorted( glob.glob( '{0}/*.pdb'.format(args.np) ))

    print('Found {0} protein cages and {1} inorganic nanoparticles'.format(len(protein_nanocages), len(inorganic_nanoparticle)) )
    combo_number = len(protein_nanocages)*len(inorganic_nanoparticle)
    print('Beginning to evaluate {0} combinations'.format( combo_number ))
    
    '====================================='
    ' THIS IS WHERE THE HEADER IS DEFINED '
    HEADER_LIST = 'protein\tcrystal\tin_total_interactions\tout_total_interactions\tclash_boolean\tmin_nn_distance\tmean_nn_distance\tstd_nn_distance\tmax_nn_distance\tmin_interactions_nonzero\taverage_interactions_nonzero\tmax_interactions_nonzero\tneighbor_cutoff_distance\tclash_distance'.split('\t')
    '====================================='

    global_score_list = [HEADER_LIST[:]] #LIST[:] is easy way to copy list

    if not os.path.isdir('crystal_fit_output/'):
        os.mkdir('crystal_fit_output/')

    counter = 0
    for shell_pdb_file in protein_nanocages:
        shell_data = read(shell_pdb_file) # Shell == protein nanocage
        bbo_shell_data = keep_heavy_bbo_only(shell_data) # Get backbone atoms only

        local_score_list = [HEADER_LIST[:]] #LIST[:] is easy way to copy list

        for crystal_pdb_file in inorganic_nanoparticle:
            counter += 1
            print('Evaluating fit of {0} inside {1}'.format(crystal_pdb_file.split('/')[-1], shell_pdb_file.split('/')[-1]))
            print('Combination {0} of {1}'.format(counter, combo_number) )
            clash_boolean = 0 # gets set to 1 if a clash is detected
            crystal_data = read(crystal_pdb_file)

            # If you only want to consider the atoms on the outermost surface use this block
            surface_crystal_indices = get_outermost_surface(crystal_data)
            print('got outermost surface')
            surface_crystal_data = indices_to_data_structure(surface_crystal_indices, crystal_data)
            # data_structure_to_pdb('surface_crystal_2_test.pdb', surface_crystal_data)  # Visualize results in Pymol

            # Coarse fitting metrics
            fit, spill_crystal_indices, clash_shell_indices, nn_distances = fit_check(bbo_shell_data, surface_crystal_data)
            nn_dist_array = np.array(nn_distances)
            min_nn_distance = np.amin(nn_dist_array)
            mean_nn_distance = np.mean(nn_dist_array)
            std_nn_distance = np.std(nn_dist_array)
            max_nn_distance = np.amax(nn_dist_array)
            print('METRICS OF NEAREST NEIGHBOR DISTANCES (A) OF CRYSTAL TO CLOSEST PROTEIN BACKBONE ATOM')
            print('\tMINIMUM            ', min_nn_distance)
            print('\tAVERAGE            ', mean_nn_distance)
            print('\tSTANDARD DEVIATION ', std_nn_distance)
            print('\tMAXIMUM            ', max_nn_distance)
            percentage_poking = len(spill_crystal_indices)*100/len(surface_crystal_indices)
            print('PERCENTAGE OF CRYSTAL SURFACE ATOMS POKING THROUGH THE PROTEIN NANOCAGE: '+str(percentage_poking))

            if min_nn_distance < args.clash:
                clash_boolean = 1

            # # In this case, write pdbs to visualize if the atoms are actually clashing, or poking through the pores.
            # spill_crystal_data = indices_to_data_structure(spill_crystal_indices, surface_crystal_data)
            # data_structure_to_pdb('spilled_crystal.pdb', spill_crystal_data)
            # clash_shell_data = indices_to_data_structure(clash_shell_indices, bbo_shell_data)
            # data_structure_to_pdb('clashing_atoms_shell.pdb', clash_shell_data)

            # Fine fitting metrics
            neighbor_cutoff_distance = args.neighbor  # In Angstroms
            nn_data_in, nn_data_out = fine_fit(bbo_shell_data, surface_crystal_data, neighbor_cutoff_distance)
            # print(nn_data_in)
            # print(nn_data_out)
            nonzero_interactions_in = [len(x) for x in nn_data_in if len(x) > 0]
            nonzero_interactions_out = [len(x) for x in nn_data_out if len(x) > 0]
            # print(nonzero_interactions_in)
            # print(nonzero_interactions_out)
            total_interactions_in = len(nonzero_interactions_in)
            total_interactions_out = len(nonzero_interactions_out)
            print('\nINTERACTIONS WITHIN {0}A OF THE SURFACE OF THE NANOCRYSTAL'.format(str(neighbor_cutoff_distance)))
            print('\tOUTSIDE THE PROTEIN NANOCAGE')
            print('\t\tTOTAL: ', total_interactions_out)
            print('\tNONZERO INSIDE THE PROTEIN NANOCAGE')
            print('\t\tTOTAL: ', total_interactions_in)
            if total_interactions_in:
                min_interactions_nonzero = min(nonzero_interactions_in)
                average_interactions_nonzero = sum(nonzero_interactions_in)/len(nonzero_interactions_in)
                max_interactions_nonzero = max(nonzero_interactions_in)
                print('\t\tMINIMUM: ', min_interactions_nonzero)  #Not sure if this is relevant since they are nonzero
                print('\t\tAVERAGE: ', average_interactions_nonzero)
                print('\t\tMAXIMUM: ', max_interactions_nonzero)
            else:
                min_interactions_nonzero = float('NaN')
                average_interactions_nonzero = float('NaN')
                max_interactions_nonzero = float('NaN')
            print()

            # Building output scorefile lines
            protein = shell_pdb_file.split('/')[-1]
            crystal = crystal_pdb_file.split('/')[-1]
            local_score_list.append([str(x) for x in [protein, crystal, total_interactions_in, total_interactions_out, clash_boolean, min_nn_distance, mean_nn_distance, std_nn_distance, max_nn_distance, min_interactions_nonzero, average_interactions_nonzero, max_interactions_nonzero, neighbor_cutoff_distance, args.clash]])
            # visualize_fine_fit(nn_data_in, bbo_shell_data, surface_crystal_data)

        protein_pdb_stem = shell_pdb_file.split('/')[-1].strip('.pdb')
        # crystal_pdb_stem = shell_pdb_file.split('/')[-1].strip('.pdb')
        # combo = '{0}_{1}'.format( shell_pdb_file.split('/')[-1].strip('.pdb'), crystal_pdb_file.split('/')[-1].strip('.pdb') )
        
        # Make folder for each protein cage's output
        local_out_dir = '{0}/{1}'.format(args.out, protein_pdb_stem)
        if not os.path.isdir(local_out_dir):
            os.mkdir(local_out_dir)
        
        # Write unsorted fit metrics to file  
        with open('{0}/{1}_fit_metrics.tsv'.format(local_out_dir, protein_pdb_stem), 'w') as scorefile:
            print('\n'.join(['\t'.join(line) for line in local_score_list]), file=scorefile)

        sorted_local_score_list = sort_fits(local_score_list)

        # Write SORTED fit metrics to file          
        with open('{0}/{1}_fit_metrics_SORTED.tsv'.format(local_out_dir, protein_pdb_stem), 'w') as scorefile:
            print('\n'.join(['\t'.join(line) for line in sorted_local_score_list]), file=scorefile)

        copy_n_first_pdb_combos(sorted_local_score_list, local_out_dir, args.local)

        # Add local scores to global list
        global_score_list.extend(local_score_list[1:])


    # Write global (all versus all) unsorted fit metrics to file  
    with open('{0}/{1}_fit_metrics.tsv'.format(args.out, 'all_vs_all'), 'w') as scorefile:
        print('\n'.join(['\t'.join(line) for line in global_score_list]), file=scorefile)

    sorted_global_score_list = sort_fits(global_score_list)

    # Write global (all versus all) SORTED fit metrics to file  
    with open('{0}/{1}_fit_metrics_SORTED.tsv'.format(args.out, 'all_vs_all'), 'w') as scorefile:
        print('\n'.join(['\t'.join(line) for line in sorted_global_score_list]), file=scorefile)

    copy_n_first_pdb_combos(sorted_global_score_list, args.out, args.all)
