"""Tests for the pure-function OpenFOAM generators in openfoam_case.py.

These need no Qt/VTK — they exercise string generation only.
"""

from mesh_prep import openfoam_case as oc


def test_of_key_only_quotes_digit_leading():
    # Regression guard for the helper the mesh dict relies on.
    assert oc._of_key("inlet") == "inlet"
    assert oc._of_key("outlet_2") == "outlet_2"
    assert oc._of_key("wall") == "wall"
    assert oc._of_key("2_outlet") == '"2_outlet"'


def test_classify_patch_real_names_unchanged():
    # Regression guard: every name the GUI/CLI actually produces must classify
    # exactly as before.
    assert oc.classify_patch("inlet") == "inlet"
    assert oc.classify_patch("outlet") == "outlet"
    assert oc.classify_patch("outlet_1") == "outlet"
    assert oc.classify_patch("outlet_2") == "outlet"
    assert oc.classify_patch("2_outlet") == "outlet"
    assert oc.classify_patch("inlet2") == "inlet"
    assert oc.classify_patch("wall") == "wall"
    assert oc.classify_patch("aorta_wall") == "wall"


def test_classify_patch_embedded_substring_not_matched():
    # A keyword buried inside a larger word (no separator) must NOT match —
    # the old substring logic wrongly classified these as inlet/outlet.
    assert oc.classify_patch("outletvalve") == "wall"
    assert oc.classify_patch("myinletish") == "wall"


def test_classify_patch_precedence_inlet_over_outlet():
    # When both tokens appear, inlet wins (documented precedence).
    assert oc.classify_patch("inlet_outlet") == "inlet"


def test_mesh_dict_normal_names_stay_unquoted():
    # The common case must not change: ordinary names are never quoted.
    out = oc.generate_mesh_dict("aorta.stl", ["inlet", "outlet", "wall"])
    assert '"inlet"' not in out
    assert '"outlet"' not in out
    assert '"wall"' not in out
    # names still present as bare keys
    assert "inlet" in out and "outlet" in out and "wall" in out


def test_mesh_dict_quotes_digit_leading_patch_names_like_bc_files():
    # A digit-leading solid name must be quoted everywhere it becomes a
    # dict key / word token, consistent with the 0/ BC generators — otherwise
    # the OpenFOAM parser reads the leading digit as a number.
    names = ["2_outlet", "wall"]

    mesh = oc.generate_mesh_dict("aorta.stl", names)
    # BC generators are the reference: they already quote digit-leading names.
    p = oc.generate_p(names, geo_name="")

    assert '"2_outlet"' in p, "sanity: BC generator quotes digit-leading name"
    # patchBoundaryLayers key, renameBoundary key, newName value must all quote.
    assert '"2_outlet"' in mesh
    # the raw unquoted form must NOT appear as a standalone key line
    assert "\n        2_outlet\n" not in mesh
    assert "newName 2_outlet;" not in mesh
