"""
Autodesk Fusion script for extending component bodies down to a shared base plane.

The script finds occurrences containing exactly one solid body, projects each
body onto a plane below the XY plane, and extrudes the largest projected profile
back up to the body. This creates a joined support/base volume using each body's
plan-view outline.
"""

import adsk.core
import adsk.fusion
import traceback


def _largest_profile(sk: adsk.fusion.Sketch):
    """
    Return the largest closed profile in a sketch.

    Projection can create multiple profiles, so the largest one is assumed to
    be the main footprint of the body.
    """
    best = None
    best_area = -1.0

    for p in sk.profiles:
        a = abs(p.areaProperties().area)

        if a > best_area:
            best_area = a
            best = p

    return best


def run(context):
    """
    Main Fusion script entry point.

    Finds valid bodies, creates a shared offset plane, projects each body onto
    it, and extrudes the resulting outline upward to join the original body.
    """
    app = adsk.core.Application.get()
    ui = app.userInterface

    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        root = design.rootComponent

        ext_feats = root.features.extrudeFeatures

        xy_plane = root.xYConstructionPlane
        planes = root.constructionPlanes

        # First pass: collect bodies from occurrences that contain exactly one
        # solid body. Surface bodies and multi-body components are skipped.
        targets = []
        global_min_z = None

        for occ in root.occurrences:
            comp = occ.component
            solid_bodies = []

            for i in range(comp.bRepBodies.count):
                b = comp.bRepBodies.item(i)

                if b.isSolid:
                    solid_bodies.append(b)

            if len(solid_bodies) == 1:
                target_body = solid_bodies[0]
            else:
                continue

            # Convert the component body into an assembly-context body so that
            # the occurrence position is respected.
            body = target_body.createForAssemblyContext(occ)
            targets.append(body)

            bbox = body.boundingBox
            min_z = bbox.minPoint.z

            if global_min_z is None or min_z < global_min_z:
                global_min_z = min_z

        if global_min_z is None:
            return

        # Create one shared construction plane 7 mm below the XY plane.
        offset_in = adsk.core.ValueInput.createByString("-7 mm")
        plane_in = planes.createInput()
        plane_in.setByOffset(xy_plane, offset_in)
        offset_plane = planes.add(plane_in)

        for body in targets:
            # Project the body onto the shared offset plane to get its footprint.
            sk = root.sketches.add(offset_plane)

            try:
                sk.project(body)
            except:
                continue

            # Use the largest projected profile as the extrusion outline.
            prof = _largest_profile(sk)

            if not prof:
                continue

            ext_in = ext_feats.createInput(
                prof,
                adsk.fusion.FeatureOperations.JoinFeatureOperation
            )

            # Extrude the profile upward until it reaches the original body.
            to_ent = adsk.fusion.ToEntityExtentDefinition.create(body, False)
            ext_in.setOneSideExtent(
                to_ent,
                adsk.fusion.ExtentDirections.PositiveExtentDirection
            )

            ext_feats.add(ext_in)

    except:
        if ui:
            ui.messageBox(traceback.format_exc())
