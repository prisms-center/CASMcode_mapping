import numpy as np

import libcasm.mapping.mapsearch as mapsearch
import libcasm.mapping.methods as mapmethods
import libcasm.xtal as xtal
import libcasm.xtal.prims as xtal_prims


def test_MappingSearch_1_old():
    # Construct the parent crystal structure
    disp_dof = xtal.DoFSetBasis("disp")
    Hstrain_dof = xtal.DoFSetBasis("Hstrain")
    parent_xtal_prim = xtal_prims.HCP(
        a=1.0,
        occ_dof=["A"],
        local_dof=[disp_dof],
        global_dof=[Hstrain_dof],
    )

    search_min_cost = 0.0
    search_max_cost = 1e20
    search_k_best = 10
    lattice_mapping_min_cost = 0.0
    lattice_mapping_k_best = 100
    lattice_mapping_reorientation_range = 3
    cost_tol = 1e-5

    # construct child as superstructure of parent
    transformation_matrix_to_super = np.array(
        [
            [1, 0, 0],
            [1, 2, 0],
            [0, 0, 1],
        ],
        dtype="int",
    )

    ### ~~~~ Internal ~~~~ ###

    parent_search_data = mapsearch.PrimSearchData(
        prim=parent_xtal_prim,
        enable_symmetry_breaking_atom_cost=False,
    )
    prim_structure_data = mapsearch.StructureSearchData(
        lattice=parent_xtal_prim.lattice(),
        atom_coordinate_cart=parent_xtal_prim.coordinate_cart(),
        atom_type=[occ[0] for occ in parent_xtal_prim.occ_dof()],
        override_structure_factor_group=None,
    )

    def custom_make_atom_to_site_cost(
        # lattice: xtal.Lattice,
        displacement: np.ndarray,
        atom_type: str,
        allowed_atom_types: list[str],
        infinity: float,
    ) -> float:
        if atom_type in ["Va", "VA", "va"]:
            return 0.0
        if atom_type not in allowed_atom_types:
            return infinity
        return displacement @ displacement

    # Create a MappingSearch object.
    # This will hold a queue of possible mappings,
    # sorted by cost, as we generate them.
    search = mapsearch.MappingSearch(
        min_cost=search_min_cost,
        max_cost=search_max_cost,
        k_best=search_k_best,
        atom_cost_f=mapsearch.IsotropicAtomCost(),
        total_cost_f=mapsearch.WeightedTotalCost(lattice_cost_weight=0.5),
        atom_to_site_cost_f=custom_make_atom_to_site_cost,
        enable_remove_mean_displacement=False,
        infinity=1e20,
        cost_tol=cost_tol,
    )

    # for each child, make lattice mapping solutions
    child_search_data = mapsearch.make_superstructure_data(
        prim_structure_data=prim_structure_data,
        transformation_matrix_to_super=transformation_matrix_to_super,
    )

    lattice_mappings = mapmethods.map_lattices(
        lattice1=parent_search_data.prim_lattice(),
        lattice2=child_search_data.lattice(),
        transformation_matrix_to_super=child_search_data.transformation_matrix_to_super(),
        lattice1_point_group=parent_search_data.prim_crystal_point_group(),
        lattice2_point_group=child_search_data.structure_crystal_point_group(),
        min_cost=lattice_mapping_min_cost,
        max_cost=1e20,
        cost_method="isotropic_strain_cost",
        k_best=lattice_mapping_k_best,
        reorientation_range=lattice_mapping_reorientation_range,
        cost_tol=cost_tol,
    )

    for scored_lattice_mapping in lattice_mappings:
        lattice_mapping_data = mapsearch.LatticeMappingSearchData(
            prim_data=parent_search_data,
            structure_data=child_search_data,
            lattice_mapping=scored_lattice_mapping,
        )

        # for each lattice mapping, generate possible translations
        trial_translations = mapsearch.make_trial_translations(
            lattice_mapping_data=lattice_mapping_data,
        )

        # for each combination of lattice mapping and translation,
        # make and insert a mapping solution (MappingNode)
        for trial_translation in trial_translations:
            search.make_and_insert_mapping_node(
                lattice_cost=scored_lattice_mapping.lattice_cost(),
                lattice_mapping_data=lattice_mapping_data,
                trial_translation_cart=trial_translation,
                forced_on={},
                forced_off=[],
            )

    assert search.size() == 27


def test_MappingSearch_1():
    # Construct the parent crystal structure
    disp_dof = xtal.DoFSetBasis("disp")
    Hstrain_dof = xtal.DoFSetBasis("Hstrain")
    parent_xtal_prim = xtal_prims.HCP(
        a=1.0,
        occ_dof=["A"],
        local_dof=[disp_dof],
        global_dof=[Hstrain_dof],
    )

    search_min_cost = 0.0
    search_max_cost = 1e20
    search_k_best = 10
    lattice_mapping_min_cost = 0.0
    lattice_mapping_k_best = 100
    lattice_mapping_reorientation_range = 3
    cost_tol = 1e-5

    # construct child as superstructure of parent
    transformation_matrix_to_super = np.array(
        [
            [1, 0, 0],
            [1, 2, 0],
            [0, 0, 1],
        ],
        dtype="int",
    )

    ### ~~~~ Internal ~~~~ ###

    parent_search_data = mapsearch.PrimSearchData(
        prim=parent_xtal_prim,
        enable_symmetry_breaking_atom_cost=False,
    )
    prim_structure_data = mapsearch.StructureSearchData(
        lattice=parent_xtal_prim.lattice(),
        atom_coordinate_cart=parent_xtal_prim.coordinate_cart(),
        atom_type=[occ[0] for occ in parent_xtal_prim.occ_dof()],
        override_structure_factor_group=None,
    )

    # Create a MappingSearch object.
    # This will hold a queue of possible mappings,
    # sorted by cost, as we generate them.
    search = mapsearch.MappingSearch(
        min_cost=search_min_cost,
        max_cost=search_max_cost,
        k_best=search_k_best,
        atom_cost_f=mapsearch.IsotropicAtomCost(),
        total_cost_f=mapsearch.WeightedTotalCost(lattice_cost_weight=0.5),
        atom_to_site_cost_future_f=mapsearch.make_atom_to_site_cost_future,
        enable_remove_mean_displacement=False,
        infinity=1e20,
        cost_tol=cost_tol,
    )

    # check child structure, using libcasm.configuration:
    # prim = casmconfig.Prim(parent_xtal_prim)
    # supercell = casmconfig.Supercell(
    #     prim=prim,
    #     transformation_matrix_to_super=transformation_matrix_to_super,
    # )
    # child_structure = casmconfig.Configuration(supercell=supercell).to_structure()
    #
    # check child structure, using libcasm.xtal:
    # parent_structure = xtal.Structure(
    #     lattice=parent_xtal_prim.lattice(),
    #     atom_coordinate_frac=parent_xtal_prim.coordinate_frac(),
    #     atom_type=[occ[0] for occ in parent_xtal_prim.occ_dof()],
    # )
    # print(xtal.pretty_json(parent_structure.to_dict(frac=False)))
    # print(xtal.pretty_json(parent_structure.to_dict(frac=True)))
    # child_structure = xtal.make_superstructure(
    #     transformation_matrix_to_super=transformation_matrix_to_super,
    #     structure=parent_structure,
    # )
    # print(xtal.pretty_json(child_structure.to_dict(frac=False)))
    # print(xtal.pretty_json(child_structure.to_dict(frac=True)))

    # for each child, make lattice mapping solutions
    child_search_data = mapsearch.make_superstructure_data(
        prim_structure_data=prim_structure_data,
        transformation_matrix_to_super=transformation_matrix_to_super,
    )

    lattice_mappings = mapmethods.map_lattices(
        lattice1=parent_search_data.prim_lattice(),
        lattice2=child_search_data.lattice(),
        transformation_matrix_to_super=child_search_data.transformation_matrix_to_super(),
        lattice1_point_group=parent_search_data.prim_crystal_point_group(),
        lattice2_point_group=child_search_data.structure_crystal_point_group(),
        min_cost=lattice_mapping_min_cost,
        max_cost=1e20,
        cost_method="isotropic_strain_cost",
        k_best=lattice_mapping_k_best,
        reorientation_range=lattice_mapping_reorientation_range,
        cost_tol=cost_tol,
    )

    for scored_lattice_mapping in lattice_mappings:
        lattice_mapping_data = mapsearch.LatticeMappingSearchData(
            prim_data=parent_search_data,
            structure_data=child_search_data,
            lattice_mapping=scored_lattice_mapping,
        )

        # for each lattice mapping, generate possible translations
        trial_translations = mapsearch.make_trial_translations(
            lattice_mapping_data=lattice_mapping_data,
        )

        # for each combination of lattice mapping and translation,
        # make and insert a mapping solution (MappingNode)
        for trial_translation in trial_translations:
            search.make_and_insert_mapping_node(
                lattice_cost=scored_lattice_mapping.lattice_cost(),
                lattice_mapping_data=lattice_mapping_data,
                trial_translation_cart=trial_translation,
                forced_on={},
                forced_off=[],
            )

    assert search.size() == 26

    while search.size():
        search.partition()

    results = search.results()
    # print("n_results:", len(results))
    # print(xtal.pretty_json(results.to_dict()))

    assert len(results) == 13

    # print([x.total_cost() for x in results])
    expected_total_cost = [
        4.52428370788109e-32,
        0.10015680293184212,
        0.12999352433204403,
        0.1299935243320441,
        0.13493070669609494,
        0.13493070669609505,
        0.16002940363760038,
        0.16775806353212028,
        0.1777205424127767,
        0.21003096336081667,
        0.2100309633608167,
        0.2100309633608167,
        0.21003096336081678,
    ]
    assert np.allclose(expected_total_cost, [x.total_cost() for x in results])

    # print([x.lattice_cost() for x in results])
    expected_lattice_cost = [
        1.0271626370065258e-32,
        0.06389995577654096,
        0.002316820721706522,
        0.002316820721706522,
        0.2698614133921899,
        0.2698614133921901,
        0.18364515718805766,
        0.19910247697709726,
        0.309969868129839,
        0.23817705993877572,
        0.2381770599387758,
        0.2381770599387758,
        0.23817705993877572,
    ]
    assert np.allclose(expected_lattice_cost, [x.lattice_cost() for x in results])

    # print([x.atom_cost() for x in results])
    expected_atom_cost = [
        8.021404778755655e-32,
        0.1364136500871433,
        0.25767022794238154,
        0.2576702279423817,
        2.2739749456927957e-31,
        3.495696146724001e-31,
        0.13641365008714312,
        0.1364136500871433,
        0.04547121669571443,
        0.1818848667828576,
        0.1818848667828576,
        0.1818848667828576,
        0.18188486678285787,
    ]
    assert np.allclose(expected_atom_cost, [x.atom_cost() for x in results])
