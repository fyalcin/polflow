import os


def write_occmat(structure, occmat_dict):
    # get the directory of this file

    # get indices from the True entries in the site properties "polaron"
    indices = [i for i, v in enumerate(structure.site_properties["polaron"]) if v]

    # then, check if all the indices correspond to Ti sites
    for index in indices:
        site = structure[index]
        polaron_type = site.properties["polaron_type"]
        # check if polaron_type is one of the keys in occmat_dict
        if polaron_type is None or polaron_type not in occmat_dict:
            raise ValueError(f"Site with index {index} can not host a polaron.")

    # get current directory

    cwd = os.getcwd()
    # get path to an OCCMATRIX file
    occmat_path = os.path.join(cwd, "OCCMATRIX")

    # open an OCCMAT file in write mode
    with open(occmat_path, "w") as f:
        # finally, write the occmat
        # write the number of sites
        f.write(f"{len(indices)}\n")
        for index_index, index in enumerate(indices):
            site = structure[index]
            polaron_type = site.properties["polaron_type"]
            # first line, write the site index followed by 2 2
            f.write(f"{index + 1} 2 2\n")
            # get the corresponding occmat
            occmat = occmat_dict[polaron_type][0]
            # write a comment saying this is spin component 1
            f.write("# spin component  1\n")
            # write the occmat with space as delimiter
            for row in occmat:
                for col in row:
                    f.write(f" {col}")

                f.write("\n")

            # get the corresponding occmat
            occmat = occmat_dict[polaron_type][1]
            # write a comment saying this is spin component 2
            f.write("# spin component  2\n")
            # write the occmat with space as delimiter
            for row in occmat:
                for val in row:
                    f.write(f" {val}")
                f.write("\n")

            # # write a newline at the end only if it is not the last index
            # if index_index != len(indices) - 1:
            f.write("\n")
