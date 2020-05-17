from collections import defaultdict
import numpy as np
from math import fabs
import string

# Following the PDB guidelines from https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
# Record type, column start, column end, data type of the record, is the record required
PDB_STRUCTURE = [
    ('record_name', 1, 6, str, True),
    ('serial', 7, 11, int, True),
    ('atom_name', 13, 16, str, True),
    ('alt_loc', 17, 17, str, False),
    ('res_name', 18, 20, str, True),
    ('chain_id', 22, 22, str, True),
    ('res_seq', 23, 26, int, True),
    ('i_code', 27, 27, str, False),
    ('x_coord', 31, 38, float, True),
    ('y_coord', 39, 46, float, True),
    ('z_coord', 47, 54, float, True),
    ('occupancy', 55, 60, float, False),
    ('temp_factor', 61, 66, float, False),
    ('element', 77, 78, str, True),
    ('charge', 79, 80, str, False)
]

PDB_STRUCTURE_COLUMN_NAMES = ('record_name', 'serial', 'atom_name',
                              'alt_loc', 'res_name', 'chain_id',
                              'res_seq', 'i_code', 'x_coord',
                              'y_coord', 'z_coord', 'occupancy',
                              'temp_factor', 'element', 'charge')

ALPHABET = list(string.ascii_uppercase)


def parse_line(line):
    column = PDB_STRUCTURE[0]
    record_type = line[column[1] - 1:column[2]].strip()
    if record_type not in ['HETATM', 'ATOM']:
        return None
    line_data = {}
    for column in PDB_STRUCTURE:
        column_name = column[0]
        start = column[1] - 1
        end = column[2]
        data_type = column[3]
        required = column[4]
        token = line[start:end].strip()
        if not required and not token:
            continue
        line_data[column_name] = data_type(line[start:end].strip())

    return line_data


def read(pdb_file):
    """
    :param pdb_file:
    :return:
    * A dictionary where each key is the pdb column, e.g. 'record_name', 'serial', 'atom_name'...
    * The value for each key is the list of entries in that column, e.g. ['HETATM',...] or [1, 2, ...] or ['C', 'H',...]
    * Order is preserved for entries of each value, i.e. order in each list is the order of lines in the pdb_file
    """
    data = defaultdict(list)
    with open(pdb_file, 'r') as fin:
        for line in fin:
            try:
                line_data = parse_line(line)
                if line_data is None:
                    continue
                for column in PDB_STRUCTURE:
                    column_name = column[0]
                    if column_name in line_data:
                        data[column_name].append(line_data[column_name])
                    else:
                        data[column_name].append(None)
            except Exception:
                print(line)
                raise

    return data


def format_pdb_string(j):
    """
    Formats input into template string according to PDB standards
    Reads in a list containing variables to be formatted into PDB
    Outputs string formatted according to the standard
    """
    # Template string according to PDB standards
    template_string = '{0}{1} {2}{3}{4} {5}{6}{7}   {8}{9}{10}{11}{12}          {13}{14}'
    j[0] = j[0].ljust(6)
    j[1] = str(j[1]).rjust(5)  # atom_serial_no
    j[2] = j[2].center(4)  # atom_name
    j[3] = j[3]  # altloc
    j[4] = j[4].rjust(3)  # res_name
    j[5] = j[5]  # chain_id
    j[6] = str(j[6]).rjust(4)  # res_seq_no
    j[7] = j[7]  # icode
    j[8] = str('%8.3f' % (float(j[8]))).rjust(8)  # x_coord
    j[9] = str('%8.3f' % (float(j[9]))).rjust(8)  # y_coord
    j[10] = str('%8.3f' % (float(j[10]))).rjust(8)  # z_coord
    j[11] = str('%6.2f' % (float(j[11]))).rjust(6)  # occupancy
    j[12] = str('%6.2f' % (float(j[12]))).rjust(6)  # temp_factor
    j[13] = str(j[13]).rjust(2)  # element_name
    j[14] = str(j[14]).rjust(2)  # charge
    formatted_string = template_string.format(j[0], j[1], j[2], j[3], j[4],
                                              j[5], j[6], j[7], j[8], j[9],
                                              j[10], j[11], j[12], j[13], j[14])

    return formatted_string


def coord_array_to_pdb(out_name, coordinates):
    with open(out_name, 'w') as fout:
        counter = 0
        k = 0
        for i in range(coordinates.shape[0]):
            chain_id = ALPHABET[counter]
            name = 'X'
            not_required_str = ' '
            not_required_float = 0
            j = ['HETATM', i+1, name, not_required_str, name, chain_id, k+1, not_required_str,
                 coordinates[i, 0], coordinates[i, 1], coordinates[i, 2],
                 not_required_float, not_required_float, name, not_required_float]
            if k+1 == 9999:
                k = 0
                counter += 1
            else:
                k += 1
            line = format_pdb_string(j)
            fout.write(line+'\n')


def data_structure_to_pdb(out_name, data):
    with open(out_name, 'w') as fout:
        for idx in range(len(data['serial'])):
            j = []
            for idx_2, column_name in enumerate(PDB_STRUCTURE_COLUMN_NAMES):
                str_value = data[column_name][idx]
                if str_value is None and PDB_STRUCTURE[idx_2][4] is False:
                    if PDB_STRUCTURE[idx_2][3] is str:
                        str_value = ' '
                    elif PDB_STRUCTURE[idx_2][3] is float:
                        str_value = 0
                elif str_value is None and PDB_STRUCTURE[idx_2][4] is True:
                    print('ERROR: Missing required data in the data set.')
                    raise ValueError
                j.append(str_value)
            line = format_pdb_string(j)
            fout.write(line+'\n')


def normalize(vector):
    vector_length = np.linalg.norm(vector)
    normalized_vector = vector / vector_length

    return normalized_vector


def is_in_plane(point, coord_array, in_plane=False, plane_threshold=0.2):
    # Get two vectors from the coordinates defining a plane
    a_point = coord_array[0]
    ab_vector = a_point - coord_array[1]
    bc_vector = coord_array[1] - coord_array[2]
    # Get the unit normal from those two vectors
    unit_normal = normalize(np.cross(ab_vector, bc_vector))
    # Evaluate if a point is on the plane
    evaluation = np.dot((point - a_point), unit_normal)
    # The evaluation should be very close to zero
    # plane_threshold allows atoms slightly below or above the plane
    in_plane = fabs(evaluation) < plane_threshold

    return in_plane


def get_coord_array(data):
    # Create coordinate numpy array of size n,3. Where n = number of entries in data
    coordinate_list = []
    for index, x in enumerate(data['x_coord']):
        y = data['y_coord'][index]
        z = data['z_coord'][index]
        coordinates = [x, y, z]
        coordinate_list.append(coordinates)
    coord_array = np.array(coordinate_list)
    print('Coordinates stored in numpy array with shape '+str(coord_array.shape))

    return coord_array


def indices_to_data_structure_renumber(index_list, original_dataset):
    index_list.sort()  # Make sure that index list is sorted in ascending order
    data = defaultdict(list)
    chain_id_counter = 0
    res_seq_counter = 0
    for e, index in enumerate(index_list):
        # Modify original dataset to reflect index, chain and residue sequence changes
        original_dataset['serial'][index] = e + 1
        original_dataset['chain_id'][index] = ALPHABET[chain_id_counter]
        original_dataset['res_seq'][index] = res_seq_counter + 1
        for column_name in PDB_STRUCTURE_COLUMN_NAMES:
            data[column_name].append(original_dataset[column_name][index])
        if res_seq_counter + 1 == 9999:
            res_seq_counter = 0
            chain_id_counter += 1
        else:
            res_seq_counter += 1

    return data


def indices_to_data_structure(index_list, original_dataset):
    index_list.sort()  # Make sure that index list is sorted in ascending order
    data = defaultdict(list)
    for index in index_list:
        for column_name in PDB_STRUCTURE_COLUMN_NAMES:
            data[column_name].append(original_dataset[column_name][index])

    return data


if __name__ == '__main__':
    exit()
