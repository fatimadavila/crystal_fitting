def fine_fit(shell, crystal, neighbor_cutoff_distance):
    shell_kd = create_kdtree(shell)
    shell_coord_array = get_coord_array(shell)
    crysatm_coord_array = get_coord_array(crystal)
    nn_data = defaultdict(list)
    for index in tqdm(range(len(crysatm_coord_array))):
        crysatm_coords = crysatm_coord_array[index]
        nn_indices = shell_kd.query_ball_point(crysatm_coords, neighbor_cutoff_distance)
        nn_data[index] = nn_indices

    return nn_data


# def fine_fit(shell, crystal):
#     shell_kd, shell_coord_array = create_kdtree(shell)
#     crystal_convex_hull = create_convex_hull(crystal)
#     distances = []
#     for index in tqdm(range(crystal_convex_hull.shape[0])):
#         crysatm_coords = crystal_convex_hull[index, :]
#         nn_dist, nn_idx = shell_kd.query(crysatm_coords)
#         distances.append(nn_dist)
#     print(np.mean(np.array(distances)))


if __name__ == '__main__':
    exit()
