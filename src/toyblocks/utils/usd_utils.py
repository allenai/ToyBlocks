import numpy as np
import trimesh
import trimesh.visual.material
from pxr import Usd, UsdGeom, UsdShade

def _to_rgba_uint8(value):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size >= 4:
        rgba = arr[:4]
    elif arr.size == 3:
        rgba = np.concatenate([arr[:3], [1.0]])
    else:
        return None
    return np.clip(rgba * 255.0, 0, 255).astype(np.uint8)


def _resolve_surface_shader(material):
    """Resolve the material surface shader, trying common Omniverse/USD contexts."""
    if not material:
        return None

    # Omniverse / Isaac usually uses mdl
    for ctx in ["mdl", "universal", "glslfx", "ri", ""]:
        try:
            shader = material.ComputeSurfaceSource(ctx)
            if shader:
                return UsdShade.Shader(shader.GetPrim() if hasattr(shader, "GetPrim") else shader)
        except Exception:
            pass

    # Fallback: inspect authored surface outputs directly
    try:
        for out in material.GetSurfaceOutputs():
            src = out.GetConnectedSource()
            if src and src[0]:
                return UsdShade.Shader(src[0])
    except Exception:
        pass

    return None


def _asset_path_to_string(asset):
    if asset is None:
        return None
    if isinstance(asset, str):
        return asset
    if hasattr(asset, "resolvedPath") and asset.resolvedPath:
        return str(asset.resolvedPath)
    if hasattr(asset, "path") and asset.path:
        return str(asset.path)
    return None


def _extract_texture_color(asset):
    texture_path = _asset_path_to_string(asset)
    if not texture_path:
        return None

    # If a texture is referenced but cannot be read locally, use white as a
    # safer fallback than treating the material as "no color".
    if texture_path.startswith(("http://", "https://")):
        return np.array([255, 255, 255, 255], dtype=np.uint8)

    try:
        from PIL import Image
        with Image.open(texture_path) as image:
            rgba_pixels = np.asarray(image.convert("RGBA"), dtype=np.float32)
    except Exception:
        return np.array([255, 255, 255, 255], dtype=np.uint8)

    if rgba_pixels.size == 0:
        return np.array([255, 255, 255, 255], dtype=np.uint8)

    mean_rgba = rgba_pixels.reshape(-1, 4).mean(axis=0)
    return np.clip(mean_rgba, 0, 255).astype(np.uint8)


def _get_shader_input(shader, names):
    for name in names:
        inp = shader.GetInput(name)
        if inp:
            return inp
    return None


def _get_input_colorspace(inp):
    if not inp:
        return None
    attr = inp.GetAttr()
    if not attr:
        return None
    try:
        color_space = attr.GetColorSpace()
        if color_space:
            return str(color_space)
    except Exception:
        pass
    try:
        color_space = attr.GetMetadata("colorSpace")
        if color_space:
            return str(color_space)
    except Exception:
        pass
    return None


def _extract_texture_input(shader, names):
    inp = _get_shader_input(shader, names)
    if not inp:
        return None
    asset = inp.Get()
    if not _asset_path_to_string(asset):
        return None
    return {
        "asset": asset,
        "color_space": _get_input_colorspace(inp),
    }


def _extract_material_textures(material):
    shader = _resolve_surface_shader(material)
    if not shader:
        return {}

    textures = {}
    base_color = _extract_texture_input(
        shader,
        [
            "diffuse_texture",
            "albedo_texture",
            "base_color_texture",
            "basecolor_texture",
            "baseColor_texture",
        ],
    )
    if base_color is not None:
        textures["base_color"] = base_color

    normal = _extract_texture_input(
        shader,
        [
            "normalmap_texture",
            "normal_texture",
            "normalTexture",
        ],
    )
    if normal is not None:
        textures["normal"] = normal

    orm = _extract_texture_input(
        shader,
        [
            "ORM_texture",
            "orm_texture",
            "metallic_roughness_texture",
            "metallicRoughness_texture",
        ],
    )
    if orm is not None:
        textures["orm"] = orm

    return textures


def _linear_to_srgb_rgba(image):
    arr = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = arr[..., :3]
    srgb = np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.4) - 0.055,
    )
    arr[..., :3] = np.clip(srgb, 0.0, 1.0)
    out = (arr * 255.0).round().astype(np.uint8)
    from PIL import Image

    return Image.fromarray(out, mode="RGBA")


def _load_texture_image(asset, color_space=None, role="generic"):
    texture_path = _asset_path_to_string(asset)
    if not texture_path:
        return None
    if texture_path.startswith(("http://", "https://")):
        return None
    try:
        from PIL import Image
        with Image.open(texture_path) as image:
            image = image.convert("RGBA").copy()
    except Exception:
        return None

    cs = str(color_space).lower() if color_space is not None else None
    if role == "base_color":
        # glTF baseColorTexture is interpreted as sRGB by renderers. If USD
        # explicitly marks the texture as raw/linear, gamma-encode it.
        if cs in ("raw", "linear"):
            image = _linear_to_srgb_rgba(image)
        return image
    return image


def _resolve_st_primvar(mesh, time):
    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    candidates = [
        "st",
        "st_0",
        "uv",
        "UVMap",
    ]
    for name in candidates:
        pv = primvars_api.GetPrimvar(name)
        if pv and pv.HasValue():
            return pv

    for pv in primvars_api.GetPrimvars():
        name = pv.GetBaseName()
        if name.startswith("st") or "uv" in name.lower():
            if pv.HasValue():
                return pv
    return None


def _get_face_corner_uv(face_id, local_corner, global_corner, poly, st_values, st_indices, st_interp):
    if st_interp in ("faceVarying", "varying"):
        uv_idx = global_corner
        if st_indices is not None and global_corner < len(st_indices):
            uv_idx = int(st_indices[global_corner])
    elif st_interp == "vertex":
        vertex_idx = int(poly[local_corner])
        uv_idx = vertex_idx
        if st_indices is not None and vertex_idx < len(st_indices):
            uv_idx = int(st_indices[vertex_idx])
    elif st_interp == "uniform":
        uv_idx = face_id
        if st_indices is not None and face_id < len(st_indices):
            uv_idx = int(st_indices[face_id])
    elif st_interp == "constant":
        uv_idx = 0
        if st_indices is not None and len(st_indices) > 0:
            uv_idx = int(st_indices[0])
    else:
        vertex_idx = int(poly[local_corner])
        uv_idx = vertex_idx
        if st_indices is not None and vertex_idx < len(st_indices):
            uv_idx = int(st_indices[vertex_idx])

    if uv_idx < 0 or uv_idx >= len(st_values):
        return None
    return st_values[uv_idx]


def _extract_material_color(material):
    shader = _resolve_surface_shader(material)
    if not shader:
        return None

    candidate_inputs = [
        "diffuseColor",
        "baseColor",
        "diffuse_color_constant",
        "albedo_constant",
        "albedo",
        "diffuse_color",
        "color",
        "emissive_color",
        "emissiveColor",
    ]

    for name in candidate_inputs:
        inp = shader.GetInput(name)
        if not inp:
            continue

        val = inp.Get()
        rgba = _to_rgba_uint8(val)
        if rgba is not None:
            return rgba

    texture_inputs = [
        "diffuse_texture",
        "albedo_texture",
        "base_color_texture",
        "basecolor_texture",
        "baseColor_texture",
    ]
    for name in texture_inputs:
        inp = shader.GetInput(name)
        if not inp:
            continue
        rgba = _extract_texture_color(inp.Get())
        if rgba is not None:
            return rgba

    return None

def get_scale_from_prim(prim):
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            v = op.Get()
            return np.array(v, dtype=np.float64)
    return None
def bake_relative_transform_into_mesh(tm: trimesh.Trimesh, rel_tf: np.ndarray) -> trimesh.Trimesh:
    verts_h = np.concatenate([tm.vertices, np.ones((len(tm.vertices), 1))], axis=1)
    verts = (verts_h @ rel_tf.T)[:, :3]
    out = trimesh.Trimesh(vertices=verts, faces=tm.faces.copy(), process=False)
    if hasattr(tm.visual, "uv") and tm.visual.uv is not None and hasattr(tm.visual, "material"):
        material = tm.visual.material.copy() if hasattr(tm.visual.material, "copy") else tm.visual.material
        out.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.asarray(tm.visual.uv, dtype=np.float32).copy(),
            material=material,
        )
        return out
    if hasattr(tm.visual, "face_colors") and tm.visual.face_colors is not None:
        out.visual = trimesh.visual.ColorVisuals(mesh=out, face_colors=tm.visual.face_colors.copy())
    return out

def _apply_texture_alpha_mode(mesh):
    """Declare how a textured mesh's base color alpha should be composited.

    USD albedo/transparency textures carry cutout alpha: the letter blocks' decal sheet is ~70%
    transparent and only the glyph itself is opaque. glTF defaults alphaMode to OPAQUE, which
    ignores that channel and renders the whole sheet solid over the block face. Isaac reads the USD
    material directly and is unaffected, which is why only the viser view looks wrong.

    MASK rather than BLEND: the decal is a hard-edged glyph sitting on the block face, and alpha
    testing needs no back-to-front sort to composite correctly.

    trimesh.util.concatenate packs per-mesh textures into an atlas and builds a *new* PBRMaterial
    that does not carry alphaMode across, so this runs on the concatenated result as well as on the
    single-mesh one.
    """
    material = getattr(getattr(mesh, "visual", None), "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if texture is None or texture.mode not in ("RGBA", "LA"):
        return mesh
    # getextrema() rather than np.asarray: these atlases reach 8192x8192 and only the alpha band's
    # minimum decides this.
    if texture.getextrema()[-1][0] == 255:
        return mesh  # alpha channel present but fully opaque; leave it OPAQUE so nothing is tested
    material.alphaMode = "MASK"
    material.alphaCutoff = 0.5
    return mesh


def usd_mesh_to_trimesh(
    mesh_prim,
    time=Usd.TimeCode.Default(),
    default_rgba=(180, 180, 180, 255),
):
    """
    Convert a UsdGeom.Mesh prim to trimesh.Trimesh and assign face colors.

    Color priority:
    1. GeomSubset bound materials
    2. Mesh-level bound material
    3. displayColor primvar
    4. default_rgba

    If every face has to be assigned a default rgba value, then mesh.visual is set to None.
    """
    mesh = UsdGeom.Mesh(mesh_prim)
    if not mesh:
        raise TypeError(f"Prim is not a UsdGeom.Mesh: {mesh_prim.GetPath()}")

    points = np.asarray(mesh.GetPointsAttr().Get(time), dtype=np.float64)
    face_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(time), dtype=np.int64)
    face_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(time), dtype=np.int64)
        
    # Default fallback color
    fallback_rgba = np.array(default_rgba, dtype=np.uint8)

    # Mesh-level material fallback
    mesh_material, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
    mesh_mat_rgba = _extract_material_color(mesh_material)
    if mesh_mat_rgba is not None:
        fallback_rgba = mesh_mat_rgba

    # Mesh-level texture
    mesh_texture_info = _extract_material_textures(mesh_material)
    base_color_info = mesh_texture_info.get("base_color")
    normal_info = mesh_texture_info.get("normal")
    orm_info = mesh_texture_info.get("orm")
    mesh_texture_image = _load_texture_image(
        base_color_info["asset"] if base_color_info else None,
        color_space=base_color_info.get("color_space") if base_color_info else None,
        role="base_color",
    )
    normal_texture_image = _load_texture_image(
        normal_info["asset"] if normal_info else None,
        color_space=normal_info.get("color_space") if normal_info else None,
        role="normal",
    )
    orm_texture_image = _load_texture_image(
        orm_info["asset"] if orm_info else None,
        color_space=orm_info.get("color_space") if orm_info else None,
        role="orm",
    )

    st_primvar = _resolve_st_primvar(mesh, time) if mesh_texture_image is not None else None
    st_values = None
    st_interp = None
    st_indices = None
    if st_primvar:
        st_values = np.asarray(st_primvar.Get(time), dtype=np.float32)
        st_interp = st_primvar.GetInterpolation() or "faceVarying"
        st_index_attr = st_primvar.GetIndicesAttr()
        if st_index_attr and st_index_attr.HasAuthoredValue():
            st_indices = np.asarray(st_index_attr.Get(time), dtype=np.int64)
        if st_values.ndim != 2 or st_values.shape[1] < 2:
            st_values = None
            st_interp = None
            st_indices = None

    # displayColor fallback
    color_pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("displayColor")
    display_color = None
    display_interp = None
    if color_pv and color_pv.HasValue():
        display_color = np.asarray(color_pv.Get(time))
        display_interp = color_pv.GetInterpolation()
        if display_interp in ("constant", "uniform") and len(display_color) > 0:
            rgba = _to_rgba_uint8(display_color[0])
            if rgba is not None:
                fallback_rgba = rgba

    # Per-original-face subset colors
    subset_face_color = {}
    subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(mesh_prim))
    for subset in subsets:
        subset_prim = subset.GetPrim()
        material, _ = UsdShade.MaterialBindingAPI(subset_prim).ComputeBoundMaterial()
        rgba = _extract_material_color(material)
        if rgba is None:
            continue

        face_ids = subset.GetIndicesAttr().Get(time)
        if face_ids is None:
            continue

        for face_id in face_ids:
            subset_face_color[int(face_id)] = rgba

    # Triangulate and assign per-triangle colors
    triangles = []
    tri_colors = []
    textured_vertices = []
    textured_uvs = []
    textured_faces = []

    cursor = 0
    global_corner = 0
    missing_rgba_counts = 0
    for face_id, n in enumerate(face_counts):
        poly = face_indices[cursor:cursor + n]
        cursor += n

        # Choose color for this original USD face
        rgba = subset_face_color.get(face_id, None)

        if rgba is None and display_color is not None:
            if display_interp == "uniform" and face_id < len(display_color):
                rgba = _to_rgba_uint8(display_color[face_id])
            elif display_interp == "vertex":
                # For vertex colors, fall back to average color for this polygon
                cols = display_color[np.asarray(poly, dtype=np.int64)]
                rgba = _to_rgba_uint8(np.mean(cols, axis=0))
            elif display_interp == "faceVarying":
                # More exact handling requires mapping face-vertex colors through triangulation.
                # Use average over the polygon's face-varying values for a simple fallback.
                pass
        if rgba is None:
            missing_rgba_counts += 1
            rgba = fallback_rgba
        if n == 3:
            triangles.append(poly)
            tri_colors.append(rgba)
            if st_values is not None:
                tri_uvs = []
                tri_ok = True
                for local_corner in range(3):
                    uv = _get_face_corner_uv(
                        face_id=face_id,
                        local_corner=local_corner,
                        global_corner=global_corner + local_corner,
                        poly=poly,
                        st_values=st_values,
                        st_indices=st_indices,
                        st_interp=st_interp,
                    )
                    if uv is None:
                        tri_ok = False
                        break
                    tri_uvs.append(uv[:2])
                if tri_ok:
                    base_idx = len(textured_vertices)
                    textured_vertices.extend(points[np.asarray(poly, dtype=np.int64)])
                    textured_uvs.extend(tri_uvs)
                    textured_faces.append([base_idx, base_idx + 1, base_idx + 2])
        elif n > 3:
            # for non triangle USD faces, we use fan triangulation
            for i in range(1, n - 1):
                triangles.append([poly[0], poly[i], poly[i + 1]])
                tri_colors.append(rgba)
                if st_values is not None:
                    corner_ids = [0, i, i + 1]
                    tri_uvs = []
                    tri_ok = True
                    for local_corner in corner_ids:
                        uv = _get_face_corner_uv(
                            face_id=face_id,
                            local_corner=local_corner,
                            global_corner=global_corner + local_corner,
                            poly=poly,
                            st_values=st_values,
                            st_indices=st_indices,
                            st_interp=st_interp,
                        )
                        if uv is None:
                            tri_ok = False
                            break
                        tri_uvs.append(uv[:2])
                    if tri_ok:
                        base_idx = len(textured_vertices)
                        textured_vertices.extend(points[np.asarray([poly[0], poly[i], poly[i + 1]], dtype=np.int64)])
                        textured_uvs.extend(tri_uvs)
                        textured_faces.append([base_idx, base_idx + 1, base_idx + 2])
        global_corner += n

    if mesh_texture_image is not None and len(textured_faces) > 0:
        texture_material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=mesh_texture_image,
            baseColorFactor=[255, 255, 255, 255],
            normalTexture=normal_texture_image.convert("RGB") if normal_texture_image is not None else None,
            metallicRoughnessTexture=orm_texture_image.convert("RGB") if orm_texture_image is not None else None,
            occlusionTexture=orm_texture_image.convert("RGB") if orm_texture_image is not None else None,
            metallicFactor=1.0,
            roughnessFactor=1.0,
        )
        tm = trimesh.Trimesh(
            vertices=np.asarray(textured_vertices, dtype=np.float64),
            faces=np.asarray(textured_faces, dtype=np.int64),
            process=False,
        )
        tm.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.asarray(textured_uvs, dtype=np.float32),
            material=texture_material,
        )
        return _apply_texture_alpha_mode(tm)

    faces = np.asarray(triangles, dtype=np.int64)
    tm = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    usd_likely_has_visuals = missing_rgba_counts < len(faces)
    if usd_likely_has_visuals:
        face_colors = np.asarray(tri_colors, dtype=np.uint8)
        tm.visual = trimesh.visual.ColorVisuals(mesh=tm, face_colors=face_colors)
    else:
        tm._visual = None
    return tm
    
def usd_matrix_to_numpy(mat4):
    return np.array(mat4, dtype=np.float64).reshape(4, 4)

def usd_geom_to_mesh(prim):
    """
    Converts a USD geometry prim to a trimesh.Trimesh object. Visuals included if found. Otherwise trimesh visuals is set to None.
    """
    if prim.IsA(UsdGeom.Mesh):
        return usd_mesh_to_trimesh(prim)
    elif prim.IsA(UsdGeom.Cylinder):
        cylinder = UsdGeom.Cylinder(prim)
        radius = cylinder.GetRadiusAttr().Get()
        height = cylinder.GetHeightAttr().Get()
        mesh = trimesh.creation.cylinder(radius=radius, height=height)
        return mesh
    elif prim.IsA(UsdGeom.Sphere):
        sphere = UsdGeom.Sphere(prim)
        radius = sphere.GetRadiusAttr().Get()
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        sphere_mat_rgba = _extract_material_color(material)
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        if sphere_mat_rgba is not None:
            mesh.visual.face_colors = np.tile(sphere_mat_rgba, (len(mesh.faces), 1))
        return mesh
    elif prim.IsA(UsdGeom.Cube):
        cube = UsdGeom.Cube(prim)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        cube_mat_rgba = _extract_material_color(material)
        size_attr = cube.GetSizeAttr()
        size = float(size_attr.Get() if size_attr and size_attr.HasValue() else 2.0)
        mesh = trimesh.creation.box(extents=(size, size, size))
        if cube_mat_rgba is not None:
            mesh.visual.face_colors = np.tile(cube_mat_rgba, (len(mesh.faces), 1))
        return mesh
    elif prim.IsA(UsdGeom.Capsule):
        capsule = UsdGeom.Capsule(prim)
        radius = capsule.GetRadiusAttr().Get()
        height = capsule.GetHeightAttr().Get()
        mesh = trimesh.creation.capsule(radius=radius, height=height)
        return mesh
    elif prim.IsA(UsdGeom.Cone):
        cone = UsdGeom.Cone(prim)
        radius = cone.GetRadiusAttr().Get()
        height = cone.GetHeightAttr().Get()
    else:
        return None

def prim_to_trimesh_parts(prim_path: str, stage: Usd.Stage, visual_only: bool = True):
    """Collect a prim's visual meshes, one per sub-prim, each keeping its own material.

    Concatenating these is lossy for anything textured: trimesh packs the sub-meshes' images into
    a single atlas and rewrites their UVs to match, and a mesh whose appearance lives entirely in
    its UVs -- the letter blocks' glyph is a window into a shared decal sheet -- loses it in that
    rewrite. Callers that render should keep the parts separate; `prim_to_trimesh` concatenates
    for callers that only want geometry.
    """
    root = stage.GetPrimAtPath(prim_path) # root should correspond to a particular link or rigid object
    meshes = []
    for prim in Usd.PrimRange(root):
        prim.GetAttribute("visibility").Get()
        if prim.GetAttribute("visibility").Get() == "invisible":
            continue
        if visual_only and prim.GetAttribute("purpose").Get() == "guide":
            # NOTE: 0 clue where the "guide" purpose differentiation of collision vs not collision meshes is documented
            # wp says its from some tutorial https://www.youtube.com/playlist?list=PLn0vmt2OawcHgWDFBOzCg9djAXplYx0EW
            continue
        mesh = usd_geom_to_mesh(prim)
        if mesh is not None:
            xform_cache = UsdGeom.XformCache()
            root_tf = usd_matrix_to_numpy(xform_cache.GetLocalToWorldTransform(root))
            scales = np.eye(4)
            scales[[0, 1, 2], [0, 1, 2]] = np.sqrt(np.sum(root_tf[:3, :3] ** 2, axis=0))
            rel_tf_usd, _ = xform_cache.ComputeRelativeTransform(prim, root) # this is probably wrong!
            rel_tf = usd_matrix_to_numpy(rel_tf_usd)
            rel_tf = rel_tf @ scales
            # hack rotation fix for now...
            if "Gripper/Robotiq_2F_85" in prim.GetPath().pathString:
                if "base_link" in prim.GetPath().pathString or "left" in prim.GetPath().pathString:
                    # Rotate the mesh by 180 degrees around the y-axis
                    rot180_z = np.array([
                        [-1, 0, 0, 0],
                        [ 0, -1, 0, 0],
                        [ 0,  0, 1, 0],
                        [ 0,  0, 0, 1]
                    ], dtype=np.float64)
                    rel_tf = rot180_z @ rel_tf
            mesh = bake_relative_transform_into_mesh(mesh, rel_tf)
            meshes.append(mesh)
    return meshes


def prim_to_trimesh(prim_path: str, stage: Usd.Stage, visual_only: bool = True):
    parts = prim_to_trimesh_parts(prim_path, stage, visual_only=visual_only)
    return _apply_texture_alpha_mode(trimesh.util.concatenate(parts))