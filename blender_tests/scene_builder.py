import bpy
import math
from pathlib import Path
from dataclasses import dataclass

@dataclass
class RenderConfig:
    width: int = 320
    height: int = 160
    fps: int = 240
    frame_start: int = 1
    frame_end: int = 30
    engine: str = "BLENDER_EEVEE_NEXT"
    

@dataclass
class PlaneConfig:
    name: str = "Plane"
    size: float = 3.0
    location: tuple = (0.0, 0.0, -5.0)
    rotation_deg: tuple = (0.0, 0.0, 0.0)
    color: tuple = (0.8, 0.8, 0.8, 1.0)

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    

def set_scene_settings(config: RenderConfig):
    scene = bpy.context.scene
    scene.frame_start = config.frame_start
    scene.frame_end = config.frame_end
    scene.render.fps = config.fps
    scene.render.resolution_x = config.width
    scene.render.resolution_y = config.height
    scene.render.resolution_percentage = 100
    
    try:
        scene.render.engine = config.engine
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
        

def create_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def add_camera(location=(0, 0, 0), rotation=(0, 0, 0), lens=24):
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.object
    camera.name = "ToF Camera"
    camera.data.lens = lens
    
    bpy.context.scene.camera = camera
    return camera


def add_light(location=(0, 0, 1), energy=300, size=5.0):
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = "Area Light"
    light.data.energy = energy
    light.data.size = size
    return light


def add_plane(config: PlaneConfig):
    bpy.ops.mesh.primitive_plane_add(size=config.size, location=config.location)
    plane = bpy.context.object
    plane.name = config.name
    
    rx, ry, rz = [math.radians(v) for v in config.rotation_deg]
    plane.rotation_euler = (rx, ry, rz)
    
    mat = create_material(f"{config.name}_Material", config.color)
    plane.data.materials.append(mat)
    
    return plane


def get_object_fcurves(obj):
    """
    Return the object's animation fcurves in both older Blender versions
    and newer Blender 5.x-style action/channelbag versions.
    """
    if obj.animation_data is None:
        return []

    action = obj.animation_data.action
    if action is None:
        return []

    # Older Blender API
    if hasattr(action, "fcurves"):
        return list(action.fcurves)

    # Blender 5.x API: fcurves live inside channelbags
    fcurves = []

    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip, "channelbags"):
                    for channelbag in strip.channelbags:
                        if hasattr(channelbag, "fcurves"):
                            fcurves.extend(list(channelbag.fcurves))

    return fcurves


def set_keyframe_interpolation(obj, interpolation="LINEAR"):
    fcurves = get_object_fcurves(obj)

    if not fcurves:
        print(f"Warning: no fcurves found for {obj.name}; leaving default interpolation.")
        return

    for fcurve in fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = interpolation


def set_linear_keyframes(obj):
    set_keyframe_interpolation(obj, "LINEAR")


def set_constant_keyframes(obj):
    set_keyframe_interpolation(obj, "CONSTANT")


def animate_location(obj, frame_locations, interpolation="LINEAR"):
    scene = bpy.context.scene
    
    for frame, loc in frame_locations:
        scene.frame_set(frame)
        obj.location = loc
        obj.keyframe_insert(data_path="location", frame=frame)
        
    if interpolation.upper() == "CONSTANT":
        set_constant_keyframes(obj)
    else:
        set_linear_keyframes(obj)


def save_blend(output_path):
    output_path = Path(output_path).resolve()
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    print(f"Saved scene to: {output_path}")


def build_basic_scene(
    output_path,
    render_config: RenderConfig,
    foreground_plane: PlaneConfig,
    foreground_motion=None,
    background_plane: PlaneConfig | None = None,
):
    clear_scene()
    set_scene_settings(render_config)
    
    add_camera()
    add_light()
    
    fg = add_plane(foreground_plane)
    
    if background_plane is not None:
        add_plane(background_plane)
        
    if foreground_motion is not None:
        animate_location(fg, foreground_motion["keyframes"], foreground_motion.get("interpolation", "LINEAR"))
        
    save_blend(output_path)