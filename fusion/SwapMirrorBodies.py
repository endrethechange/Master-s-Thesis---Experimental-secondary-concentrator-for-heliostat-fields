"""
Autodesk Fusion script for swapping mirror body visibility.

This script was used to toggle between two body variants in selected mirror
components by swapping the visibility of bodies named Body1 and Body3.
It was developed for workflow support during the master's thesis project.
"""


import adsk.core
import adsk.fusion
import traceback
import re

# Names of the two bodies whose visibility should be swapped.
# In this case, Fusion bodies named "Body1" and "Body3" are treated as
# the two alternative mirror states.
BODY_A = "Body1"
BODY_B = "Body3"


def is_named(body_name, target):
    """
    Check whether a Fusion body name matches the requested target name.

    Fusion often renames duplicated bodies by appending a number in parentheses,
    for example:
        Body1
        Body1 (2)
        Body1 (3)

    This function therefore accepts both the exact name and numbered copies.
    """
    return re.match(rf"^{re.escape(target)}(\s*\(\d+\))?$", body_name.strip()) is not None


def swap_body_visibility_in_component(comp):
    """
    Swap the visibility of BODY_A and BODY_B inside one Fusion component.

    The script searches through all BRep bodies in the component and separates
    bodies matching BODY_A from bodies matching BODY_B. It then reads the current
    visibility state of the first matching body in each group and applies the
    opposite group's state to all bodies in the other group.

    Example:
        If Body1 is visible and Body3 is hidden, Body1 becomes hidden and
        Body3 becomes visible.

    Returns:
        The number of body visibility entries that were changed or considered.
    """
    bodies_a = []
    bodies_b = []

    # Go through all solid/surface bodies in the component and collect the ones
    # whose names match BODY_A or BODY_B.
    for body in comp.bRepBodies:
        if is_named(body.name, BODY_A):
            bodies_a.append(body)
        elif is_named(body.name, BODY_B):
            bodies_b.append(body)

    # If the component does not contain either of the target bodies, there is
    # nothing to swap.
    if not bodies_a and not bodies_b:
        return 0

    # Determine the current visibility state.
    #
    # If several bodies match the same name, the first one is used as the
    # representative visibility state for that group.
    #
    # If one group is missing, its visibility is treated as False.
    a_visible = bodies_a[0].isLightBulbOn if bodies_a else False
    b_visible = bodies_b[0].isLightBulbOn if bodies_b else False

    # Apply Body3's previous visibility state to all Body1 bodies.
    for body in bodies_a:
        body.isLightBulbOn = b_visible

    # Apply Body1's previous visibility state to all Body3 bodies.
    for body in bodies_b:
        body.isLightBulbOn = a_visible

    # Return how many target bodies were found in this component.
    return len(bodies_a) + len(bodies_b)


def collect_occurrences_from_selection(ui):
    """
    Collect selected Fusion occurrences from the current user selection.

    The script is intended to be run after selecting mirror component rows in
    the Fusion Browser. However, Fusion selections may contain either component
    occurrences or bodies inside occurrences. This function supports both cases.

    Returns:
        A list of unique selected occurrences.
    """
    occs = []

    for i in range(ui.activeSelections.count):
        ent = ui.activeSelections.item(i).entity

        # Case 1:
        # The selected entity is already a component occurrence.
        occ = adsk.fusion.Occurrence.cast(ent)
        if occ:
            occs.append(occ)
            continue

        # Case 2:
        # The selected entity is a body inside an occurrence.
        # In that case, use the body's assembly context to find the occurrence
        # it belongs to.
        body = adsk.fusion.BRepBody.cast(ent)
        if body and body.assemblyContext:
            occs.append(body.assemblyContext)
            continue

    # Remove duplicate occurrences.
    #
    # This avoids processing the same occurrence more than once if both the
    # occurrence and one of its bodies were selected, or if multiple bodies
    # from the same occurrence were selected.
    unique = []
    seen = set()

    for occ in occs:
        key = occ.fullPathName
        if key not in seen:
            seen.add(key)
            unique.append(occ)

    return unique


def run(context):
    """
    Main Fusion script entry point.

    The script:
        1. Gets the active Fusion design.
        2. Reads the user's current Browser/model selection.
        3. Finds selected component occurrences.
        4. For each selected occurrence, swaps visibility between Body1 and Body3.
        5. Reports how many occurrences and body entries were processed.
    """
    ui = None

    try:
        # Get the running Fusion application and user interface.
        app = adsk.core.Application.get()
        ui = app.userInterface

        # Get the active Fusion design.
        design = adsk.fusion.Design.cast(app.activeProduct)

        # Stop if the active product is not a Fusion design.
        if not design:
            ui.messageBox("No active Fusion design.")
            return

        # Get the selected component occurrences from the current selection.
        selected_occs = collect_occurrences_from_selection(ui)

        # The script depends on the user selecting one or more mirror component
        # rows before running it.
        if not selected_occs:
            ui.messageBox(
                "Select the mirror component rows in the Browser first, then run the script."
            )
            return

        changed = 0
        processed = 0

        # Process each selected occurrence separately.
        for occ in selected_occs:
            comp = occ.component

            # Swap visibility between Body1 and Body3 inside this component.
            changed += swap_body_visibility_in_component(comp)
            processed += 1

        # Show a short summary when the script is finished.
        ui.messageBox(
            f"Done.\n\n"
            f"Selected mirror occurrences processed: {processed}\n"
            f"Body visibility entries swapped: {changed}"
        )

    except:
        # If anything fails, show the full Python traceback in Fusion.
        # This makes debugging easier than silently failing.
        if ui:
            ui.messageBox("Script failed:\n\n{}".format(traceback.format_exc()))
