# This script is an example of how you can run blender from the command line
# (in background mode with no interface) to automate tasks, in this example it
# creates a text object, camera and light, then renders and/or saves it.
# This example also shows how you can parse command line options to scripts.
#
# Example usage for this test.
#  blender --background --factory-startup --python $HOME/background_job.py -- \
#          --text="Hello World" \
#          --render="/tmp/hello" \
#          --save="/tmp/hello.blend"
#
# Notice:
# '--factory-startup' is used to avoid the user default settings from
#                     interfering with automated scene generation.
#
# '--' causes blender to ignore all following arguments so python can use them.
#
# See blender --help for details.


import bpy

import os
from os.path import join


def get_meshes(dir, target_scan):
    stages = ["v1/scans"]
    meshes = []

    # we have train, val and test stage-subfolders
    for stage in stages:
        path = join(dir, stage)
        if os.path.exists(path):
            # for each scan in the stage-subfolder
            for scan in os.listdir(path):
                # only collect this scan
                if target_scan and scan != target_scan:
                    continue
                scan_path = join(path, scan)
                suffix = '_uvs_blender'

                region_segmentation_path = join(scan_path, 'region_segmentations', scan, 'region_segmentations')
                house_segmentation_path = join(scan_path, 'house_segmentations', scan, 'house_segmentations')

                region_mesh_names = [f for f in os.listdir(region_segmentation_path) if 'ply' in f and suffix not in f]
                region_mesh_paths = [join(region_segmentation_path, f) for f in region_mesh_names]

                house_mesh_names = [f for f in os.listdir(house_segmentation_path) if 'ply' in f and suffix not in f]
                house_mesh_paths = [join(house_segmentation_path, f) for f in house_mesh_names]

                meshes.extend(region_mesh_paths)
                #meshes.extend(house_mesh_paths)

    return meshes


def reset():
    # Clear existing objects.
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load(mesh):
    # load mesh
    bpy.ops.import_mesh.ply(filepath=mesh)
    # get obj from blender internal data
    obj = get_obj()
    # get face info for this obj
    faces = len(obj.data.polygons)
    # set obj as the active one (might be the case already after loading, but to be sure explicitly do this)
    bpy.context.view_layer.objects.active = obj
    return obj, faces


def get_obj():
    # get object (is the online objects existing right now, so the objectList only contains one object)
    objectList = bpy.data.objects
    obj = objectList[0]
    if obj.type != "MESH":
        raise ValueError("Object is not of type MESH")
    return obj


def decimate(obj, faces, max_faces):
    ratio = 1.0 * max_faces / faces

    # cleanAllDecimateModifiers
    for m in obj.modifiers:
        if m.type == "DECIMATE":
            obj.modifiers.remove(modifier=m)

    print(f"{obj.name} has {faces} faces before decimation")
    # decimate
    modifier = obj.modifiers.new("DecimateMod", 'DECIMATE')
    modifier.ratio = ratio
    modifier.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier="DecimateMod")
    print(f"{obj.name} has {len(obj.data.polygons)} faces after decimation")


def uv_smart_project(angle_limit):
    # entering edit mode
    bpy.ops.object.editmode_toggle()
    # select all objects elements
    bpy.ops.mesh.select_all(action='SELECT')
    # the actual unwrapping operation, 1.2217 are 70 degrees
    bpy.ops.uv.smart_project(correct_aspect=False, angle_limit=angle_limit)
    # exiting edit mode
    bpy.ops.object.editmode_toggle()
    print("created uv parameterization")


def save(path):
    bpy.ops.export_mesh.ply(filepath=path,
                            use_ascii=False,
                            use_selection=True,
                            use_mesh_modifiers=True,
                            use_normals=True,
                            use_uv_coords=True,
                            use_colors=True)


def create_uvs(opt):
    print("Collect all meshes...")
    meshes = get_meshes(opt.dir, opt.scan)

    print(f"Create uvs for {len(meshes)} meshes...")
    for i, mesh in enumerate(meshes):
        # load mesh into empty blender environment
        reset()
        obj, faces = load(mesh)

        # reduce face count
        if not opt.no_decimate and faces > opt.max_faces:
            # do decimate
            decimate(obj, faces, opt.max_faces)
            # save decimated mesh
            decimate_mesh = mesh.split(".")[0] + f"_decimate_{opt.max_faces}.ply"
            save(decimate_mesh)
            # reload mesh (directly doing uv_smart_project without reload results in seg fault...)
            mesh = decimate_mesh
            reset()
            obj, faces = load(mesh)

        # get uv parameterization
        uv_smart_project(opt.angle_limit)

        # save uv mesh
        uv_mesh = mesh.split(".")[0] + "_uvs_blender.ply"
        save(uv_mesh)

        # print progress
        print(f"\nCompleted [{i+1}/{len(meshes)}]\n")


def main():
    import sys       # to get command line args
    import argparse  # to parse options for us and print a nice help message

    # get the args passed to blender after "--", all of which are ignored by
    # blender so scripts may receive their own arguments
    argv = sys.argv

    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # When --help or no args are given, print this help
    usage_text = (
        "Run blender in background mode with this script:"
        "  blender --background --python " + __file__ + " -- [options]"
    )

    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("-d", "--dir", dest="dir", help="path/to/scannet")
    parser.add_argument("-s", "--scan", dest="scan", required=False, default=None, help="Only create uv for this scan")
    parser.add_argument("-mf", "--max_faces", dest="max_faces", required=False, default=500000, help="reduce mesh to only contain this many faces at maximum")
    parser.add_argument("-al", "--angle_limit", dest="angle_limit", required=False, default=1.2217, help="angle_limit argument for uv.smart_project() method")
    parser.add_argument("-nd", "--no_decimate", dest="no_decimate", action="store_true", required=False, default=False, help="create uvs out of the unreduced mesh")
    args = parser.parse_args(argv)  # In this example we won't use the args

    if not argv:
        parser.print_help()
        return

    create_uvs(args)


if __name__ == "__main__":
    main()