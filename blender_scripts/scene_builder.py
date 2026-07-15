import bpy
import math
from pathlib import Path
from dataclasses import dataclass
from mathutils import Vector
import bmesh

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
    

@dataclass
class CubeConfig:
    name: str = "Cube"
    size: tuple = (1.0, 1.0, 1.0)
    location: tuple = (0.0, 0.0, -5.0)
    rotation_deg: tuple = (0.0, 0.0, 0.0)
    color: tuple = (0.8, 0.8, 0.8, 1.0)
    
    
@dataclass
class CylinderConfig:
    name: str = "Cylinder"
    vertices: int = 32
    radius: float = 0.05
    depth: float = 2.0
    location: tuple = (0.0, 0.0, -5.0)
    rotation_deg: tuple = (0.0, 0.0, 0.0)
    color: tuple = (0.8, 0.8, 0.8, 1.0)
    

@dataclass
class SphereConfig:
    name: str = "Sphere"
    radius: float = 0.25
    location: tuple = (0.0, 0.0, -5.0)
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


def apply_rotation_deg(obj, rotation_deg):
    rx, ry, rz = [math.radians(v) for v in rotation_deg]
    obj.rotation_euler = (rx, ry, rz)
    
    
def assign_material(obj, name, color):
    mat = create_material(name, color)
    obj.data.materials.append(mat)
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


def add_cube(config: CubeConfig):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=config.location)
    cube = bpy.context.object
    cube.name = config.name
    
    cube.dimensions = config.size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    apply_rotation_deg(cube, config.rotation_deg)
    assign_material(cube, f"{config.name}_Material", config.color)
    
    return cube


def add_cylinder(config: CylinderConfig):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=config.vertices,
        radius=config.radius,
        depth=config.depth,
        location=config.location,
    )
    cylinder = bpy.context.object
    cylinder.name = config.name
    
    apply_rotation_deg(cylinder, config.rotation_deg)
    assign_material(cylinder, f"{config.name}_Material", config.color)
    
    return cylinder


def add_sphere(config: SphereConfig):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=32,
        ring_count=16,
        radius=config.radius,
        location=config.location,
    )
    sphere = bpy.context.object
    sphere.name = config.name
    
    assign_material(sphere, f"{config.name}_Material", config.color)
    
    return sphere


def add_wavy_pad(
    name="Wavy Pad",
    size_x=2.0,
    size_z=1.4,
    location=(0.0, -1.15, -7.6),
    amplitude=0.08,
    freq_x=3.0,
    freq_z=4.0,
    subdivisions=40,
    color=(0.70, 0.70, 0.70, 1.0),
):
    """
    Create a wavy landing pad in the X-Z plane.

    X and Z define the pad surface.
    Y is the height/roughness direction.
    """

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    verts = []
    faces = []

    nx = subdivisions
    nz = subdivisions

    for iz in range(nz + 1):
        z_norm = iz / nz
        z = (z_norm - 0.5) * size_z

        for ix in range(nx + 1):
            x_norm = ix / nx
            x = (x_norm - 0.5) * size_x

            y = amplitude * math.sin(freq_x * x) * math.cos(freq_z * z)

            verts.append((x, y, z))

    for iz in range(nz):
        for ix in range(nx):
            v0 = iz * (nx + 1) + ix
            v1 = v0 + 1
            v2 = v0 + (nx + 1) + 1
            v3 = v0 + (nx + 1)

            faces.append((v0, v1, v2, v3))

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location

    assign_material(obj, f"{name}_Material", color)

    return obj


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
  
        
def animate_rotation(obj, frame_rotations_deg, interpolation="LINEAR"):
    """
    Animate object Euler rotation using degrees.

    frame_rotations_deg:
        [
            (frame, (rx_deg, ry_deg, rz_deg)),
            ...
        ]
    """
    scene = bpy.context.scene

    for frame, rot_deg in frame_rotations_deg:
        scene.frame_set(frame)
        rx, ry, rz = [math.radians(v) for v in rot_deg]
        obj.rotation_euler = (rx, ry, rz)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    if interpolation.upper() == "CONSTANT":
        set_constant_keyframes(obj)
    else:
        set_linear_keyframes(obj)
        
        
def animate_camera_path(camera, frame_locations, interpolation="LINEAR"):
    animate_location(camera, frame_locations, interpolation)
    
    
def animate_camera_rotation(camera, frame_rotations_deg, interpolation="LINEAR"):
    animate_rotation(camera, frame_rotations_deg, interpolation)
    

def look_at(obj, target):
    """
    Rotate an object so its local -Z axis points toward target.

    This is useful for cameras, because Blender cameras look along -Z.
    """
    obj_location = Vector(obj.location)
    target_location = Vector(target)

    direction = target_location - obj_location

    if direction.length == 0:
        raise ValueError("look_at target must be different from object location.")

    quat = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = quat.to_euler()
    
    
def set_world_background(color=(1.0, 1.0, 1.0), strength=0.8):
    """
    Set Blender world/background lighting.
    Useful for making simple geometry visible in rendered preview.
    """
    world = bpy.context.scene.world

    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.color = color


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
    
    
def save_drone_path_config(
    name: str,
    frame_locations: list,
    segment_names: list[str] | None = None,
    description: str = "",
    output_dir: str | Path = "configs/drone_paths",
):
    """
    Save camera translation keyframes as a drone path YAML config.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{name}.yaml"

    if len(frame_locations) < 2:
        raise ValueError(
            "Drone path must contain at least two camera keyframes."
        )

    num_segments = len(frame_locations) - 1

    if segment_names is None:
        segment_names = [
            f"Segment {index + 1}"
            for index in range(num_segments)
        ]

    if len(segment_names) != num_segments:
        raise ValueError(
            "segment_names must contain one name for each path segment."
        )

    lines = [
        f"name: {name}",
        "",
        "description: >",
        f"  {description}" if description else "  Camera translation path.",
        "",
        "camera_path:",
    ]

    for frame, position in frame_locations:
        x, y, z = position

        lines.extend([
            f"  - frame: {frame}",
            f"    position: [{x}, {y}, {z}]",
        ])

    lines.extend([
        "",
        "segment_names:",
    ])

    for segment_name in segment_names:
        lines.append(f'  - "{segment_name}"')

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Saved drone path config: {output_path.resolve()}")

    return output_path