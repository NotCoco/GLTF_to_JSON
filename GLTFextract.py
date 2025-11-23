#!/usr/bin/env python3
# GLB/GLTF → baked JSON (skinned, animated) — correct hierarchical skinning + UI
# - Keeps vertex order (no reindexing)
# - Builds animated world matrices for ALL nodes each frame
# - jointMatrix = worldAnimated(joint) * inverseBindMatrix[j]
# - Optionally apply the mesh node's animated/world transform to the skinned result

import sys, json, math, struct, traceback, base64
from pathlib import Path
from io import BytesIO
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    Image = None

BASEDIR = None
TEXTURE_CACHE = {}

# ------------------------------ IO helpers ----------------------------------

def _read_uri(uri: str, basedir: Path):
    if uri.startswith("data:"):
        _, b64 = uri.split(",", 1)
        return base64.b64decode(b64)
    return (basedir / uri).read_bytes()

def load_gltf_or_glb(path: Path):
    data = path.read_bytes()
    basedir = path.parent
    if data[:4] == b'glTF':  # GLB
        if len(data) < 12: raise RuntimeError("Bad GLB header")
        ver = struct.unpack_from("<I", data, 4)[0]
        if ver != 2: raise RuntimeError("Only GLB v2 supported")
        total = struct.unpack_from("<I", data, 8)[0]
        off = 12
        js = None; bin = None
        while off + 8 <= len(data):
            n, typ = struct.unpack_from("<II", data, off); off += 8
            chunk = data[off:off+n]; off += n
            if typ == 0x4E4F534A: js = chunk
            elif typ == 0x004E4942: bin = chunk
        if js is None: raise RuntimeError("GLB missing JSON chunk")
        g = json.loads(js.decode("utf-8"))
        buffers = []
        for buf in g.get("buffers", []) or []:
            uri = buf.get("uri")
            buffers.append(bin if uri is None else _read_uri(uri, basedir))
        return g, buffers, basedir
    else:
        g = json.loads(data.decode("utf-8"))
        buffers = []
        for buf in g.get("buffers", []) or []:
            uri = buf.get("uri")
            buffers.append(b"" if uri is None else _read_uri(uri, basedir))
        return g, buffers, basedir

COMPONENT_DTYPES = {
    5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4),
}
TYPE_NUMCOMP = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16,
}

def load_buffer_view(g, buffers, idx):
    bv = (g.get("bufferViews") or [])[idx]
    b = buffers[bv["buffer"]]
    off = int(bv.get("byteOffset", 0)); ln = int(bv["byteLength"])
    return b[off:off+ln], int(bv.get("byteStride", 0))

def read_accessor(g, buffers, idx):
    if idx is None: return None
    a = (g.get("accessors") or [])[idx]
    comp = a["componentType"]; tag, csz = COMPONENT_DTYPES[comp]
    typ  = a["type"];           ncomp   = TYPE_NUMCOMP[typ]
    cnt  = int(a["count"]);     norm    = bool(a.get("normalized", False))
    bv_idx = a.get("bufferView"); boff = int(a.get("byteOffset", 0))

    def norm_val(v):
        if not norm: return v
        if comp == 5120: return max(-1.0, min(v/127.0, 1.0))
        if comp == 5121: return v/255.0
        if comp == 5122: return max(-1.0, min(v/32767.0, 1.0))
        if comp == 5123: return v/65535.0
        return v

    if bv_idx is not None:
        buf, stride = load_buffer_view(g, buffers, bv_idx)
        if stride == 0: stride = csz*ncomp
        out = []
        for i in range(cnt):
            base = boff + i*stride
            vals = struct.unpack_from("<"+tag*ncomp, buf, base)
            if ncomp == 1: out.append(norm_val(vals[0]))
            else: out.append(tuple(norm_val(v) for v in vals))
        return out
    else:
        # Rare in glTF; keep for completeness.
        out = []
        for i in range(cnt):
            base = boff + i*csz*ncomp
            vals = struct.unpack_from("<"+tag*ncomp, buffers[0], base)
            if ncomp == 1: out.append(norm_val(vals[0]))
            else: out.append(tuple(norm_val(v) for v in vals))
        return out

# ------------------------------ Math ----------------------------------------

def _mat4_identity():
    return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

def _mat4_mul(a, b):
    r = [[0]*4 for _ in range(4)]
    for i in range(4):
        ai = a[i]
        r[i][0] = ai[0]*b[0][0] + ai[1]*b[1][0] + ai[2]*b[2][0] + ai[3]*b[3][0]
        r[i][1] = ai[0]*b[0][1] + ai[1]*b[1][1] + ai[2]*b[2][1] + ai[3]*b[3][1]
        r[i][2] = ai[0]*b[0][2] + ai[1]*b[1][2] + ai[2]*b[2][2] + ai[3]*b[3][2]
        r[i][3] = ai[0]*b[0][3] + ai[1]*b[1][3] + ai[2]*b[2][3] + ai[3]*b[3][3]
    return r

def _mat4_vec3(m, v3):
    x = m[0][0]*v3[0] + m[0][1]*v3[1] + m[0][2]*v3[2] + m[0][3]
    y = m[1][0]*v3[0] + m[1][1]*v3[1] + m[1][2]*v3[2] + m[1][3]
    z = m[2][0]*v3[0] + m[2][1]*v3[1] + m[2][2]*v3[2] + m[2][3]
    return (x, y, z)

def _mat4_to_list(m):
    return [m[r][c] for r in range(4) for c in range(4)]

def _quat_to_mat4(q):
    (w,x,y,z) = q
    xx=x*x; yy=y*y; zz=z*z; xy=x*y; xz=x*z; yz=y*z; wx=w*x; wy=w*y; wz=w*z
    return [
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy),   0],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx),   0],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy), 0],
        [0,0,0,1]
    ]

def _trs_to_mat4(T=None, R=None, S=None):
    Tm = _mat4_identity()
    if T is not None: Tm[0][3],Tm[1][3],Tm[2][3] = T
    Rm = _quat_to_mat4(R) if R is not None else _mat4_identity()
    Sm = _mat4_identity()
    if S is not None: Sm[0][0],Sm[1][1],Sm[2][2] = S[0],S[1],S[2]
    return _mat4_mul(_mat4_mul(Tm, Rm), Sm)

def _as_mat4_from_gltf_colmajor(m16):
    return [
        [m16[0],  m16[4],  m16[8],  m16[12]],
        [m16[1],  m16[5],  m16[9],  m16[13]],
        [m16[2],  m16[6],  m16[10], m16[14]],
        [m16[3],  m16[7],  m16[11], m16[15]],
    ]

def _clamp(v, lo, hi): 
    return lo if v < lo else (hi if v > hi else v)

# ------------------------------ Animation sampling ---------------------------

def build_parent_map(g):
    parents = [-1] * len(g.get("nodes", []) or [])
    for i, nd in enumerate(g.get("nodes", []) or []):
        for ch in nd.get("children", []) or []:
            parents[ch] = i
    return parents

def build_tracks_per_animation(g, buffers):
    clips = []
    animations = g.get("animations", []) or []
    for idx, anim in enumerate(animations):
        name = anim.get("name") or f"clip_{idx}"
        tracks = {}
        duration = 0.0
        for ch in anim.get("channels", []) or []:
            tgt = ch.get("target") or {}
            node = tgt.get("node")
            path = tgt.get("path")
            if node is None or path is None:
                continue
            samplers = anim.get("samplers", []) or []
            samp = samplers[ch.get("sampler", 0)] if samplers else {}
            tin = read_accessor(g, buffers, samp.get("input"))
            tout = read_accessor(g, buffers, samp.get("output"))
            if not tin or not tout:
                continue
            duration = max(duration, float(tin[-1]))
            if path == "rotation":
                tout = [(float(q[3]), float(q[0]), float(q[1]), float(q[2])) for q in tout]
            else:
                tout = [tuple(float(v) for v in vec) for vec in tout]
            target_tracks = tracks.setdefault(node, {})
            target_tracks[path] = {"t": [float(x) for x in tin], "v": tout}
        if tracks:
            clips.append({"name": name, "tracks": tracks, "duration": duration})
    return clips


def build_tracks(g, buffers, sample_fps):
    """Aggregate animation tracks per node and record clip metadata."""
    tracks = {}
    clip_meta = []
    animations = g.get("animations", []) or []
    time_offset = 0.0
    frame_cursor = 0

    for idx, anim in enumerate(animations):
        name = anim.get("name") or f"clip_{idx}"
        samplers = anim.get("samplers", []) or []
        channels = anim.get("channels", []) or []

        clip_duration = 0.0
        for ch in channels:
            target = ch.get("target") or {}
            node = target.get("node")
            path = target.get("path")
            if node is None or path is None:
                continue
            if path not in {"translation", "rotation", "scale"}:
                continue

            sampler_idx = ch.get("sampler", 0)
            sampler = samplers[sampler_idx] if sampler_idx < len(samplers) else {}
            tin = read_accessor(g, buffers, sampler.get("input"))
            tout = read_accessor(g, buffers, sampler.get("output"))
            if not tin or not tout:
                continue

            clip_duration = max(clip_duration, float(tin[-1]))

            if path == "rotation":
                values = [
                    (
                        float(q[3]),
                        float(q[0]),
                        float(q[1]),
                        float(q[2])
                    ) if isinstance(q, (list, tuple)) and len(q) >= 4 else (
                        float(q[-1]),
                        float(q[0]),
                        float(q[1]) if len(q) > 1 else 0.0,
                        float(q[2]) if len(q) > 2 else 0.0,
                    )
                    for q in tout
                ]
            else:
                values = [
                    tuple(float(v) for v in vec) if isinstance(vec, (list, tuple)) else (float(vec),)
                    for vec in tout
                ]

            times = [float(t) + time_offset for t in tin]

            node_tracks = tracks.setdefault(node, {})
            track = node_tracks.setdefault(path, {"t": [], "v": [], "interp": (sampler.get("interpolation") or "LINEAR").upper()})

            if track["t"] and times:
                # Avoid duplicating the boundary sample when clips abut
                if abs(times[0] - track["t"][-1]) < 1e-6:
                    times = times[1:]
                    values = values[1:]
            if not times:
                continue

            track["t"].extend(times)
            track["v"].extend(values)

        if clip_duration > 0.0:
            frame_start = frame_cursor
            frame_count = max(1, int(round(clip_duration * sample_fps)) + 1)
            frame_end = frame_start + frame_count - 1
            clip_meta.append({
                "name": name,
                "startTime": time_offset,
                "duration": clip_duration,
                "startFrame": frame_start,
                "endFrame": frame_end,
                "frameCount": frame_count
            })
            frame_cursor = frame_end + 1
            time_offset += clip_duration

    return tracks, clip_meta

def sample_track(track, tsec, default):
    if not track: return default
    T = track["t"]; V = track["v"]
    if tsec <= T[0]:  return V[0]
    if tsec >= T[-1]: return V[-1]
    # linear
    lo = 0; hi = len(T)-1
    # small array – linear scan is fine
    for i in range(len(T)-1):
        if T[i] <= tsec <= T[i+1]:
            a = _clamp((tsec - T[i])/(T[i+1]-T[i]), 0.0, 1.0)
            A, B = V[i], V[i+1]
            if len(A) == 4:  # quaternion (simple lerp; good enough here)
                return ( (1-a)*A[0]+a*B[0],
                         (1-a)*A[1]+a*B[1],
                         (1-a)*A[2]+a*B[2],
                         (1-a)*A[3]+a*B[3] )
            return tuple( (1-a)*A[k] + a*B[k] for k in range(len(A)) )
    return default

def animated_world_mats(g, tsec, tracks, base_local):
    """Return world matrix for every node at time tsec using animated local TRS."""
    nodes = g.get("nodes", []) or []
    parents = build_parent_map(g)

    # build animated local first
    local = [None]*len(nodes)
    for i, nd in enumerate(nodes):
        T0,R0,S0 = base_local[i]
        tr = sample_track((tracks.get(i) or {}).get("translation"), tsec, T0)
        rr = sample_track((tracks.get(i) or {}).get("rotation"),    tsec, R0)
        ss = sample_track((tracks.get(i) or {}).get("scale"),       tsec, S0)
        local[i] = _trs_to_mat4(tr, rr, ss)

    # compose to world
    world = [_mat4_identity() for _ in nodes]
    for i in range(len(nodes)):
        chain = []
        cur = i
        while cur != -1:
            chain.append(cur); cur = parents[cur]
        M = _mat4_identity()
        for idx in reversed(chain):
            M = _mat4_mul(M, local[idx])
        world[i] = M
    return world

def base_local_trs(g):
    nodes = g.get("nodes", []) or []
    out = []
    for nd in nodes:
        t = tuple(map(float, nd.get("translation", (0,0,0))))
        r = nd.get("rotation")
        if r is None: r = (1.0, 0.0, 0.0, 0.0)
        else:         r = (float(r[3]), float(r[0]), float(r[1]), float(r[2]))  # to w,x,y,z
        s = tuple(map(float, nd.get("scale", (1,1,1))))
        out.append((t,r,s))
    return out

# ------------------------------ Export core ----------------------------------

APPLY_MESH_NODE_WORLD = True  # apply world(mesh node) to baked positions (recommended)

def bake_frames(g, buffers, mesh_node_idx, skin_idx, positions, joints, weights, sample_fps, qscale):
    if skin_idx is None:
        # bind pose only
        pos = [(x*qscale, y*qscale, z*qscale) for (x,y,z) in positions]
        vx = [int(round(x)) for x,_,_ in pos]
        vy = [int(round(y)) for _,y,_ in pos]
        vz = [int(round(z)) for _,_,z in pos]
        return [{"vx": vx, "vy": vy, "vz": vz}], [], None

    skin = (g.get("skins", []) or [])[skin_idx]
    joints_nodes = skin.get("joints") or []
    inv_bind_idx = skin.get("inverseBindMatrices")
    inv_bind = []
    if inv_bind_idx is not None:
        mats = read_accessor(g, buffers, inv_bind_idx)
        if mats and isinstance(mats[0], (list, tuple)) and len(mats[0]) == 16:
            inv_bind = [_as_mat4_from_gltf_colmajor(m) for m in mats]
        elif mats and isinstance(mats, list) and len(mats) % 16 == 0:
            for i in range(0, len(mats), 16):
                inv_bind.append(_as_mat4_from_gltf_colmajor(mats[i:i+16]))
    if not inv_bind:
        inv_bind = [_mat4_identity() for _ in joints_nodes]

    baseLoc = base_local_trs(g)
    parents = build_parent_map(g)
    node_to_joint = {node: idx for idx, node in enumerate(joints_nodes)}
    nodes_list = (g.get("nodes", []) or [])

    def _bind_local_tuple(node_idx, comp_idx):
        if node_idx < len(baseLoc):
            comp = baseLoc[node_idx][comp_idx]
            return tuple(float(v) for v in comp)
        return (0.0, 0.0, 0.0) if comp_idx == 0 else ((1.0, 0.0, 0.0, 0.0) if comp_idx == 1 else (1.0, 1.0, 1.0))

    joint_entries = []
    for ji, jnode in enumerate(joints_nodes):
        node_info = nodes_list[jnode] if jnode is not None and jnode < len(nodes_list) else {}
        name = node_info.get("name") or f"joint_{ji}"
        parent_node = parents[jnode] if jnode < len(parents) else -1
        parent_joint = node_to_joint.get(parent_node, -1)
        ib_mat = inv_bind[ji if ji < len(inv_bind) else 0]
        t_loc = _bind_local_tuple(jnode, 0)
        r_loc = _bind_local_tuple(jnode, 1)
        s_loc = _bind_local_tuple(jnode, 2)
        joint_entries.append({
            "index": ji,
            "node": int(jnode),
            "name": name,
            "parent": parent_joint,
            "inverseBindMatrix": _mat4_to_list(ib_mat),
            "bindLocal": {
                "translation": list(t_loc),
                "rotation": list(r_loc),
                "scale": list(s_loc),
            }
        })
    if joint_entries:
        print(f"[skin] joints extracted: {len(joint_entries)} (skin index {skin_idx})")

    joint_frames = []
    tracks, clip_meta = build_tracks(g, buffers, sample_fps)

    max_time = 0.0
    for node_tracks in tracks.values():
        for track in node_tracks.values():
            if track["t"]:
                max_time = max(max_time, track["t"][-1])

    if max_time <= 0.0:
        # no animated samplers -> just bind pose skinned via the static nodes
        world0 = animated_world_mats(g, 0.0, {}, baseLoc)
        # build jointMatrix once
        J = []
        for ji, jnode in enumerate(joints_nodes):
            wm = world0[jnode]
            ib = inv_bind[ji if ji < len(inv_bind) else 0]
            J.append(_mat4_mul(wm, ib))
        joint_frames.append([_mat4_to_list(m) for m in J])
        out = []
        for i,(x,y,z) in enumerate(positions):
            p = (0.0,0.0,0.0)
            jset = joints[i]; wset = weights[i]
            px=py=pz=0.0
            for k in range(4):
                jj = int(jset[k]) if k < len(jset) else 0
                ww = float(wset[k]) if k < len(wset) else 0.0
                if ww <= 0.0: continue
                M = J[jj if jj < len(J) else 0]
                rx,ry,rz = _mat4_vec3(M, (x,y,z))
                px += ww*rx; py += ww*ry; pz += ww*rz
            out.append((px,py,pz))
        if APPLY_MESH_NODE_WORLD and mesh_node_idx is not None:
            worldMesh = world0[mesh_node_idx]
            out = [_mat4_vec3(worldMesh, p) for p in out]
        out = [(px*qscale, py*qscale, pz*qscale) for (px,py,pz) in out]
        return [{
            "vx": [int(round(px)) for (px,_,_) in out],
            "vy": [int(round(py)) for (_,py,_) in out],
            "vz": [int(round(pz)) for (_,_,pz) in out],
        }], clip_meta, {
            "joints": joint_entries,
            "jointMatrices": joint_frames,
            "jointNodes": [int(n) for n in joints_nodes],
        }

    frames = []
    t = 0.0; step = 1.0/float(sample_fps)
    while t <= max_time + 1e-6:
        world = animated_world_mats(g, t, tracks, baseLoc)

        # jointMatrices(t)
        J = []
        for ji, jnode in enumerate(joints_nodes):
            wm = world[jnode]
            ib = inv_bind[ji if ji < len(inv_bind) else 0]
            J.append(_mat4_mul(wm, ib))
        joint_frames.append([_mat4_to_list(m) for m in J])

        # skin
        out = []
        for i,(x,y,z) in enumerate(positions):   # POSITION index order preserved
            jset = joints[i]; wset = weights[i]
            px=py=pz=0.0
            for k in range(4):
                jj = int(jset[k]) if k < len(jset) else 0
                ww = float(wset[k]) if k < len(wset) else 0.0
                if ww <= 0.0: continue
                M = J[jj if jj < len(J) else 0]
                rx,ry,rz = _mat4_vec3(M, (x,y,z))
                px += ww*rx; py += ww*ry; pz += ww*rz
            out.append((px,py,pz))

        # apply mesh-node world (so clip moves the mesh if node is animated)
        if APPLY_MESH_NODE_WORLD and mesh_node_idx is not None:
            worldMesh = world[mesh_node_idx]
            out = [_mat4_vec3(worldMesh, p) for p in out]

        # quantize
        out = [(px*qscale, py*qscale, pz*qscale) for (px,py,pz) in out]
        frames.append({
            "vx": [int(round(px)) for (px,_,_) in out],
            "vy": [int(round(py)) for (_,py,_) in out],
            "vz": [int(round(pz)) for (_,_,pz) in out],
        })
        t += step
    return frames, clip_meta, {
        "joints": joint_entries,
        "jointMatrices": joint_frames,
        "jointNodes": [int(n) for n in joints_nodes],
    }

# ------------------------------ Colors / UVs --------------------------------

def _wrap_coord(val: float, mode: int) -> float:
    if mode == 10497 or mode is None:  # REPEAT (default)
        return val - math.floor(val)
    if mode == 33648:  # MIRRORED_REPEAT
        v = val % 2.0
        return v if v <= 1.0 else 2.0 - v
    # CLAMP_TO_EDGE or unknown
    return max(0.0, min(1.0, val))


def _to_byte(val: float) -> int:
    return max(0, min(255, int(round(val))))


def _load_texture_image(g, buffers, tex_index):
    if Image is None:
        return None
    if tex_index is None or tex_index < 0:
        return None
    key = tex_index
    if key in TEXTURE_CACHE:
        return TEXTURE_CACHE[key]
    textures = g.get("textures", []) or []
    if tex_index >= len(textures):
        return None
    tex = textures[tex_index]
    src_idx = tex.get("source")
    if src_idx is None:
        return None
    images = g.get("images", []) or []
    if src_idx >= len(images):
        return None
    img_info = images[src_idx]
    data = None
    if "uri" in img_info:
        if BASEDIR is None:
            return None
        data = _read_uri(img_info["uri"], BASEDIR)
    elif "bufferView" in img_info:
        buf_bytes, _ = load_buffer_view(g, buffers, img_info["bufferView"])
        data = bytes(buf_bytes)
    else:
        return None
    try:
        img = Image.open(BytesIO(data))
        img = img.convert("RGBA")
    except Exception:
        return None
    TEXTURE_CACHE[key] = img
    return img


def sample_vcol_for_prim(g, buffers, prim, count, uvs=None):
    rr = gg = bb = 255
    tex_img = None
    wrap_s = wrap_t = 10497  # default repeat
    tex_factor = (1.0, 1.0, 1.0)
    tex_info = None
    used_texture = False

    mi = prim.get("material")
    if mi is not None:
        materials = (g.get("materials", []) or [])
        if 0 <= mi < len(materials):
            m = materials[mi]
            pbr = (m.get("pbrMetallicRoughness") or {})
            factor = pbr.get("baseColorFactor") or [1, 1, 1, 1]
            tex_factor = (
                float(factor[0]) if len(factor) > 0 else 1.0,
                float(factor[1]) if len(factor) > 1 else 1.0,
                float(factor[2]) if len(factor) > 2 else 1.0,
            )
            rr = _to_byte(255 * tex_factor[0])
            gg = _to_byte(255 * tex_factor[1])
            bb = _to_byte(255 * tex_factor[2])

            tex_info = pbr.get("baseColorTexture")
            if tex_info is None:
                ext = (m.get("extensions") or {}).get("KHR_materials_pbrSpecularGlossiness")
                if ext is not None:
                    tex_info = ext.get("diffuseTexture")
                    diff_factor = ext.get("diffuseFactor")
                    if diff_factor:
                        tex_factor = (
                            float(diff_factor[0]) if len(diff_factor) > 0 else tex_factor[0],
                            float(diff_factor[1]) if len(diff_factor) > 1 else tex_factor[1],
                            float(diff_factor[2]) if len(diff_factor) > 2 else tex_factor[2],
                        )
                        rr = _to_byte(255 * tex_factor[0])
                        gg = _to_byte(255 * tex_factor[1])
                        bb = _to_byte(255 * tex_factor[2])

            if tex_info and uvs:
                tex_index = tex_info.get("index")
                tex_img = _load_texture_image(g, buffers, tex_index)
                if tex_img is not None:
                    textures = g.get("textures", []) or []
                    if tex_index is not None and 0 <= tex_index < len(textures):
                        sampler_idx = textures[tex_index].get("sampler")
                        samplers = g.get("samplers", []) or []
                        if sampler_idx is not None and 0 <= sampler_idx < len(samplers):
                            sampler = samplers[sampler_idx]
                            wrap_s = sampler.get("wrapS", wrap_s)
                            wrap_t = sampler.get("wrapT", wrap_t)

    if tex_img is None or not uvs or Image is None or len(uvs) != count:
        return [rr] * count, [gg] * count, [bb] * count, False

    w, h = tex_img.size
    pixels = tex_img.load()
    out_r = []
    out_g = []
    out_b = []
    for uv in uvs:
        if not uv:
            out_r.append(rr)
            out_g.append(gg)
            out_b.append(bb)
            continue
        u = _wrap_coord(float(uv[0]), wrap_s)
        v = _wrap_coord(float(uv[1]), wrap_t)
        x = max(0, min(w - 1, int(round(u * (w - 1)))))
        y = max(0, min(h - 1, int(round((1.0 - v) * (h - 1)))))
        pr, pg, pb, _ = pixels[x, y]
        out_r.append(_to_byte(pr * tex_factor[0]))
        out_g.append(_to_byte(pg * tex_factor[1]))
        out_b.append(_to_byte(pb * tex_factor[2]))
        used_texture = True
    return out_r, out_g, out_b, used_texture

# ------------------------------ Mesh/skin collection -------------------------

def select_mesh_and_collect_prims(g, buffers):
    meshes = g.get("meshes", []) or []
    best = (-1e9, None, None)
    for mi, m in enumerate(meshes):
        faces = 0; prims=[]
        for p in (m.get("primitives", []) or []):
            if "POSITION" not in (p.get("attributes") or {}): continue
            pos = read_accessor(g, buffers, p["attributes"]["POSITION"])
            pos = [(float(x), float(y), float(z)) for (x,y,z) in pos]
            idx = read_accessor(g, buffers, p["indices"]) if "indices" in p else None
            if idx is not None: faces += len(idx)//3
            prims.append({"prim": p, "pos": pos, "indices": idx})
        if faces and prims:  # pick the densest
            score = faces
            if score > best[0]: best = (score, mi, prims)
            
    if best[1] is None:
        # Graceful fallback: Return None to indicate no mesh found
        # This allows the extractor to proceed in "Animation-Only" mode
        print("[warn] No usable mesh/primitives found. Proceeding with animation-only export.")
        return None, []
        
    return best[1], best[2]

def find_node_with_mesh(g, mesh_index):
    for ni, nd in enumerate(g.get("nodes", []) or []):
        if nd.get("mesh") == mesh_index:
            return ni
    return None

def find_skin_for_node(g, node_idx):
    nd = (g.get("nodes", []) or [])[node_idx]
    return nd.get("skin")

# ------------------------------ Export wrapper + UI --------------------------

def export(src: Path, out_path: Path, sample_fps=20, target_int_h=2048, rotateX=0.0, pretty=False):
    global BASEDIR
    global TEXTURE_CACHE
    g, buffers, basedir = load_gltf_or_glb(src)
    BASEDIR = basedir
    TEXTURE_CACHE = {}

    mesh_idx, prims = select_mesh_and_collect_prims(g, buffers)
    
    # Handle case where no mesh exists
    if mesh_idx is not None:
        node_idx = find_node_with_mesh(g, mesh_idx)
        skin_idx = find_skin_for_node(g, node_idx) if node_idx is not None else None
        print(f"[pick] final mesh={mesh_idx} node={node_idx} skin={skin_idx}")
    else:
        node_idx = None
        # Fallback: try to find the first skin in the file
        skins = g.get("skins", [])
        if skins:
            skin_idx = 0
            print(f"[pick] No mesh found, defaulting to first skin found (index 0)")
        else:
            skin_idx = None
            print(f"[pick] No mesh and no skin found.")

    # gather
    pos_all=[]; vR=[]; vG=[]; vB=[]; faceA=[]; faceB=[]; faceC=[]
    J_all=[]; W_all=[]
    color_mode_counts = {"vertex": 0, "texture": 0, "texture-ref": 0, "fallback": 0}
    for prim_index, info in enumerate(prims):
        prim = info["prim"]; pos = info["pos"]; idx = info["indices"]
        off = len(pos_all)
        pos_all.extend(pos)
        if idx is None:
            raise RuntimeError("Primitive without indices not supported")
        for i in range(0, len(idx), 3):
            a,b,c = int(idx[i]), int(idx[i+1]), int(idx[i+2])
            faceA.append(off+a); faceB.append(off+b); faceC.append(off+c)

        attrs = prim.get("attributes") or {}

        colors = None
        tex_info_ref = None
        if attrs:
            for color_key in ("COLOR_0", "COLOR_1"):
                if color_key in attrs:
                    raw = read_accessor(g, buffers, attrs.get(color_key))
                    if raw and len(raw) == len(pos):
                        conv_r = []
                        conv_g = []
                        conv_b = []
                        for entry in raw:
                            if isinstance(entry, (list, tuple)):
                                r = float(entry[0]) if len(entry) > 0 else 0.0
                                g = float(entry[1]) if len(entry) > 1 else r
                                b = float(entry[2]) if len(entry) > 2 else r
                            else:
                                r = g = b = float(entry)
                            scale = 255.0 if max(abs(r), abs(g), abs(b)) <= 1.0 else 1.0
                            conv_r.append(_to_byte(r * scale))
                            conv_g.append(_to_byte(g * scale))
                            conv_b.append(_to_byte(b * scale))
                        colors = (conv_r, conv_g, conv_b)
                    break

        uvs = None
        if attrs and colors is None:
            # Respect baseColorTexture texCoord if present
            tex_coord_idx = 0
            mat_idx = prim.get("material")
            materials = (g.get("materials", []) or [])
            if mat_idx is not None and 0 <= mat_idx < len(materials):
                m = materials[mat_idx]
                tex = (m.get("pbrMetallicRoughness") or {}).get("baseColorTexture")
                if tex:
                    tex_coord_idx = int(tex.get("texCoord", 0))
                    tex_info_ref = tex
                else:
                    # Spec gloss extension
                    ext = (m.get("extensions") or {}).get("KHR_materials_pbrSpecularGlossiness")
                    if ext and "diffuseTexture" in ext:
                        tex = ext["diffuseTexture"]
                        tex_coord_idx = int(tex.get("texCoord", tex_coord_idx))
                        tex_info_ref = tex
            attr_key = f"TEXCOORD_{tex_coord_idx}"
            if attr_key not in attrs and "TEXCOORD_0" in attrs:
                attr_key = "TEXCOORD_0"
            if attr_key in attrs:
                uvs_raw = read_accessor(g, buffers, attrs.get(attr_key))
                if uvs_raw and len(uvs_raw) == len(pos):
                    uvs = [
                        (float(u[0]), float(u[1])) if isinstance(u, (list, tuple)) else (float(u), 0.0)
                        for u in uvs_raw
                    ]

        if colors is not None:
            vr, vg, vb = colors
            color_mode = "vertex"
        else:
            vr, vg, vb, used_texture = sample_vcol_for_prim(g, buffers, prim, len(pos), uvs)
            color_mode = "texture" if used_texture else ("texture-ref" if tex_info_ref else "fallback")
        color_mode_counts[color_mode] = color_mode_counts.get(color_mode, 0) + 1
        print(f"[color] primitive {prim_index}: mode={color_mode} verts={len(pos)}")
        vR.extend(vr); vG.extend(vg); vB.extend(vb)
        J = read_accessor(g, buffers, attrs.get("JOINTS_0"))
        W = read_accessor(g, buffers, attrs.get("WEIGHTS_0"))
        if J and W:
            Jc = [tuple(int(x) for x in j) if isinstance(j,(list,tuple)) else (int(j),0,0,0) for j in J]
            Wc = []
            for w in W:
                ww = tuple(float(x) for x in w) if isinstance(w,(list,tuple)) else (float(w),0.0,0.0,0.0)
                s = sum(ww); Wc.append(tuple(x/s for x in ww) if s>1e-8 else ww)
            J_all.extend(Jc); W_all.extend(Wc)
        else:
            J_all.extend([(0,0,0,0)]*len(pos)); W_all.extend([(1.0,0.0,0.0,0.0)]*len(pos))

    # quantization scale by height
    ys = [y for (_,y,_) in pos_all]
    if ys:
        ymin=min(ys); ymax=max(ys); h=max(1e-9, ymax-ymin)
        qscale = target_int_h/h
        print(f"[diag] pre-quant bbox: Y[{ymin:.6f},{ymax:.6f}]  qscale={qscale:.3f}")
    else:
        # Default scale if no mesh data present (1 unit = 128 units in Jagex coords typically)
        # But target_int_h is usually 2048 (character height).
        # If we assume 1 blender unit = 128 game units, then we multiply by 128?
        # Wait, qscale is used to scale blender units TO game units.
        # If target is 2048 (game units), and character is 2m (blender units), scale is ~1000.
        # Let's default to 128.0 as a safe guess if we have no mesh to measure.
        qscale = 128.0 
        print(f"[diag] No mesh vertices. Defaulting qscale={qscale}")

    frames, clip_meta, rig_meta = bake_frames(g, buffers, node_idx, skin_idx, pos_all, J_all, W_all, sample_fps, qscale)
    print(f"[diag:bake] baked frames: {len(frames)}; verts/frame={len(frames[0]['vx']) if frames else 0}")

    framesX=[f["vx"] for f in frames]; framesY=[f["vy"] for f in frames]; framesZ=[f["vz"] for f in frames]
    print("[color] summary:", ", ".join(f"{k}={v}" for k, v in color_mode_counts.items()))

    anim_clips = []
    total_frames = len(frames)
    if clip_meta and total_frames > 0:
        max_frame_index = total_frames - 1
        for info in clip_meta:
            start = max(0, min(max_frame_index, int(info.get("startFrame", 0))))
            end = max(start, min(max_frame_index, int(info.get("endFrame", start))))
            span = max(1, end - start + 1)
            anim_clips.append({
                "name": info.get("name", "clip"),
                "startFrame": start,
                "endFrame": end,
                "frameCount": span,
                "durationSeconds": float(info.get("duration", span / float(sample_fps))),
                "fps": int(sample_fps)
            })
        for clip in anim_clips:
            print(f"[diag:clip] {clip['name']} frames={clip['startFrame']}..{clip['endFrame']} ({clip['frameCount']} frames, {clip['durationSeconds']:.3f}s)")

    out = {
        "frames": frames,
        "faces": {"a": faceA, "b": faceB, "c": faceC},
        "vcol": {"r": vR, "g": vG, "b": vB},
        # legacy arrays used by your Java
        "baseX": framesX[0] if framesX else [], 
        "baseY": framesY[0] if framesY else [], 
        "baseZ": framesZ[0] if framesZ else [],
        "framesX": framesX, "framesY": framesY, "framesZ": framesZ,
        "faceA": faceA, "faceB": faceB, "faceC": faceC,
        "vR": vR, "vG": vG, "vB": vB,
        "fps": int(sample_fps),
        "animationClips": anim_clips,
        "qscale": qscale
    }
    if rig_meta:
        weight_data = {
            "joints": [list(map(int, j)) for j in J_all],
            "weights": [list(map(float, w)) for w in W_all],
        }
        out["skin"] = {
            "joints": rig_meta.get("joints", []),
            "jointNodes": rig_meta.get("jointNodes", []),
            "jointMatrices": rig_meta.get("jointMatrices", []),
            "weights": weight_data,
        }
    text = json.dumps(out, indent=2) if pretty else json.dumps(out)
    out_path.write_text(text)
    print(f"[ok] wrote {out_path} (frames={len(frames)}, verts={len(framesX[0]) if framesX else 0}, faces={len(faceA)})")
    # print(f"[peek] faceA[0:5]={faceA[:5]} baseX[0:5]={framesX[0][:5]}")
    return out

# ---------------------------------- UI --------------------------------------

class Tee:
    def __init__(self, widget=None): self.widget = widget
    def write(self, s):
        sys.__stdout__.write(s)
        if self.widget:
            self.widget.configure(state="normal")
            self.widget.insert("end", s)
            self.widget.configure(state="disabled")
            self.widget.see("end")
    def flush(self): pass

def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, scrolledtext, messagebox
    root = tk.Tk()
    root.title("GLTF/GLB → Baked JSON (correct skinning)")
    frm = ttk.Frame(root, padding=10); frm.grid(sticky="nsew")
    root.columnconfigure(0, weight=1); root.rowconfigure(0, weight=1)
    for c in range(3): frm.columnconfigure(c, weight=(1 if c==1 else 0))
    frm.rowconfigure(7, weight=1)

    in_v = tk.StringVar()
    out_v = tk.StringVar(value=str(Path.home()/"model_baked.json"))
    fps_v = tk.IntVar(value=20)
    tgt_v = tk.IntVar(value=2048)
    pretty_v = tk.BooleanVar(value=False)

    def choose_in():
        p = filedialog.askopenfilename(filetypes=[("glTF/GLB","*.gltf *.glb"),("All","*.*")])
        if p: in_v.set(p)
    def choose_out():
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if p: out_v.set(p)

    ttk.Label(frm, text="Input:").grid(row=0, column=0, sticky="w")
    ttk.Entry(frm, textvariable=in_v).grid(row=0, column=1, sticky="ew")
    ttk.Button(frm, text="Browse", command=choose_in).grid(row=0, column=2)

    ttk.Label(frm, text="Output:").grid(row=1, column=0, sticky="w")
    ttk.Entry(frm, textvariable=out_v).grid(row=1, column=1, sticky="ew")
    ttk.Button(frm, text="Browse", command=choose_out).grid(row=1, column=2)

    ttk.Label(frm, text="FPS:").grid(row=2, column=0, sticky="w")
    ttk.Entry(frm, textvariable=fps_v, width=8).grid(row=2, column=1, sticky="w")

    ttk.Label(frm, text="Target int height:").grid(row=3, column=0, sticky="w")
    ttk.Entry(frm, textvariable=tgt_v, width=10).grid(row=3, column=1, sticky="w")

    ttk.Checkbutton(frm, text="Pretty JSON (multi-line)", variable=pretty_v).grid(row=4, column=1, sticky="w")

    log = scrolledtext.ScrolledText(frm, width=100, height=26, state="disabled")
    log.grid(row=7, column=0, columnspan=3, sticky="nsew")
    sys.stdout = Tee(log)

    def go():
        try:
            src = Path(in_v.get()); out = Path(out_v.get())
            export(src, out, sample_fps=int(fps_v.get()), target_int_h=int(tgt_v.get()), pretty=bool(pretty_v.get()))
            messagebox.showinfo("Done", f"Wrote {out}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    ttk.Button(frm, text="Export", command=go).grid(row=6, column=2, sticky="e")
    root.mainloop()

def main(argv):
    if len(argv) == 1:
        try: run_gui(); return
        except Exception: traceback.print_exc()

    # CLI fallback
    src=None; out=Path.home()/"model_baked.json"; fps=20; tgt=2048; pretty=False
    i=1
    while i < len(argv):
        if argv[i] in ("-i","--in"): src=Path(argv[i+1]); i+=2
        elif argv[i] in ("-o","--out"): out=Path(argv[i+1]); i+=2
        elif argv[i]=="--fps": fps=int(argv[i+1]); i+=2
        elif argv[i]=="--target": tgt=int(argv[i+1]); i+=2
        elif argv[i]=="--pretty": pretty=True; i+=1
        else: print("unknown arg:", argv[i]); sys.exit(2)
    if src is None:
        print("Usage: exporter.py --in model.glb --out model_baked.json [--fps 20] [--target 2048] [--pretty]")
        sys.exit(2)
    export(src, out, sample_fps=fps, target_int_h=tgt, pretty=pretty)

if __name__ == "__main__":
    main(sys.argv)
