import os
import cv2
import numpy as np
import pickle
import argparse


SIZE = 1000
RED = [0, 0, 255]
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]
BLACK = [0, 0, 0]
GRAY = [100, 100, 100]
WHITE = [255, 255, 255]


# ====================================================================================================


def read_obj_file(filename):
    """
    OBJ file format:
    
    1. vertices
    v x/y/z
    
    2. texture coordinates (The origin is located at the lower left corner of the texture map.)
    vt u/v
    
    3. vertex normals
    vn nx/ny/nz

    4. faces (index from 1)
    f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
    """
    
    v, vt, vn = [], [], []
    fv, fvt, fvn = [], [], []
    
    with open(filename) as file:
        for line in file:
            items = line.split()
            if line.startswith("v "):
                v.append([float(item) for item in items[1:]])
            elif line.startswith("vt "):
                vt.append([float(item) for item in items[1:]])
            elif line.startswith("vn "):
                vn.append([float(item) for item in items[1:]])
            elif line.startswith("f "):
                fv.append([int(item.split("/")[0]) for item in items[1:]])
                fvt.append([int(item.split("/")[1]) for item in items[1:] if item.count("/") >= 1])
                fvn.append([int(item.split("/")[2]) for item in items[1:] if item.count("/") >= 2])

    v, vt, vn = np.array(v, dtype=float), np.array(vt, dtype=float), np.array(vn, dtype=float)
    fv, fvt, fvn = np.array(fv, dtype=int), np.array(fvt, dtype=int), np.array(fvn, dtype=int)

    return v, vt, vn, fv, fvt, fvn


def compute_vuv_vd(v, rmat, tvec, H, W, focal_length=5000):
    v = v.T                      # (N, 3) -> (3, N)
    rmat = rmat.reshape(3, 3)    # (1, 3, 3) -> (3, 3)
    tvec = tvec.reshape(3, 1)    # (1, 3) -> (3, 1)
    cmat = np.array([
        [focal_length,            0, W // 2],
        [           0, focal_length, H // 2],
        [           0,            0,      1]
    ], float)
    
    ms = v                       # model/world space (3, N)
    vs = rmat.dot(ms) + tvec     # view/projection space (3, N)
    ns = vs / vs[2, :]           # normalize space (3, N)
    ds = cmat.dot(ns)            # display space (3, N)
    
    v2p = ds[:2, :].astype(int)  # vertex to pixel (2, N)
    vd = vs[2, :]                # vertex depth (1, N)
    
    v2p = v2p.T                  # (2, N) -> (N, 2)
    vd = vd.T                    # (1, N) -> (N, 1)

    return v2p, vd


def compute_fn(v, fv):
    p1 = v[fv[:, 0]]
    p2 = v[fv[:, 1]]
    p3 = v[fv[:, 2]]
    
    p12 = p2 - p1
    p13 = p3 - p1
    
    fn_dir = np.cross(p12, p13, axis=1)
    fn_len = np.linalg.norm(fn_dir, axis=1)
    fn = fn_dir / fn_len[:, np.newaxis]
    
    return fn


# ====================================================================================================


def warping(
        image_path, mask_path, model_path, template_model_path, parameter_path, template_texture_path,
        all_faces_on_image_path, all_faces_on_texture_path, front_faces_on_image_path, front_faces_on_texture_path, image_texture_path, mask_texture_path, normal_texture_path
    ):

    """
    Read image and mask
    """
    
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    """
    Read obj file
    """

    v, __, _, __, ___, _ = read_obj_file(model_path)
    _, vt, _, fv, fvt, _ = read_obj_file(template_model_path)
    
    v[:, 1:] = -v[:, 1:]     # negate y and z
    vt[:, 1] = 1 - vt[:, 1]  # apply horizontal flip
    fv = fv - 1
    fvt = fvt - 1

    """
    Read pickle file
    """

    with open(parameter_path, "rb") as file:
        parameter = pickle.load(file)
    
    rmat = parameter["camera_rotation"]     # rotation matrix (1, 3, 3)
    tvec = parameter["camera_translation"]  # translation vector (1, 3)

    """
    Read or generate template texture
    """

    if os.path.exists(template_texture_path):

        template_texture = cv2.imread(template_texture_path)

    else:

        v2t = (vt * SIZE).astype(int)

        template_texture = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        for vtid in fvt:
            cv2.drawContours(image=template_texture, contours=[v2t[vtid]], contourIdx=0, color=GRAY, thickness=-1)
        
        cv2.imwrite(template_texture_path, template_texture)

    """
    Obtain dimensions
    """

    H, W, _ = image.shape
    V, _ = v.shape
    VT, _ = vt.shape
    F, _ = fv.shape

    """
    Step 1: Transformation
    """

    v2p, vd = compute_vuv_vd(v, rmat, tvec, H, W)  # vertex to pixel (V, 2) & vertex depth (V, )
    v2t = (vt * SIZE).astype(int)                  # vertex to texel (VT, 2)

    """
    Step 2: Rendering before z-test
    """

    all_faces_on_image = image.copy()               # all faces on image (H, W, 3)
    for fid in range(F):
        cv2.drawContours(image=all_faces_on_image, contours=[v2p[fv[fid]]], contourIdx=0, color=BLUE, thickness=1)
    
    cv2.imwrite(all_faces_on_image_path, all_faces_on_image)
    
    all_faces_on_texture = template_texture.copy()  # all faces on texture (SIZE, SIZE, 3)
    for fid in range(F):
        cv2.drawContours(image=all_faces_on_texture, contours=[v2t[fvt[fid]]], contourIdx=0, color=BLUE, thickness=1)
    
    cv2.imwrite(all_faces_on_texture_path, all_faces_on_texture)
    
    """
    Step 3: Z-test
    """

    fd = np.mean(vd[fv], axis=1)                 # face depth (F, )
    
    fids = np.argsort(-fd)                       # face id with fd in descending order
    
    p2f = np.ones((H, W), dtype=int) * -1        # pixel to face id (H, W)
    for fid in fids:
        cv2.drawContours(image=p2f, contours=[v2p[fv[fid]]], contourIdx=0, color=int(fid), thickness=-1)
    
    visible_fids = np.unique(p2f)[1:]            # visible face id
    
    t2f = np.ones((SIZE, SIZE), dtype=int) * -1  # texel to face id (SIZE, SIZE)
    for fid in visible_fids:
        cv2.drawContours(image=t2f, contours=[v2t[fvt[fid]]], contourIdx=0, color=int(fid), thickness=-1)

    """
    Step 4: Rendering after z-test
    """

    front_faces_on_image = image.copy()               # front faces on image (H, W, 3)
    for fid in visible_fids:
        cv2.drawContours(image=front_faces_on_image, contours=[v2p[fv[fid]]], contourIdx=0, color=BLUE, thickness=1)
    
    cv2.imwrite(front_faces_on_image_path, front_faces_on_image)
    
    front_faces_on_texture = template_texture.copy()  # front faces on texture (SIZE, SIZE, 3)
    for fid in visible_fids:
        cv2.drawContours(image=front_faces_on_texture, contours=[v2t[fvt[fid]]], contourIdx=0, color=BLUE, thickness=1)
    
    cv2.imwrite(front_faces_on_texture_path, front_faces_on_texture)

    """
    Step 5: Sample color from image to texture
    """

    image_texture = template_texture.copy()
    mask_texture = template_texture.copy()

    for i in range(SIZE):
        for j in range(SIZE):
            
            fid = t2f[j, i]
            if fid == -1:
                continue
                
            # solve t1tx = a * t1t2 + b * t1t3
            t1, t2, t3 = vt[fvt[fid]]
            t1t2 = t2 - t1
            t1t3 = t3 - t1
            tx = np.array([i / SIZE, j / SIZE], dtype=float)
            t1tx = tx - t1
            a, b = np.linalg.solve(np.array([t1t2, t1t3]).T, t1tx)
            
            # compute p1px = a * p1p2 + b * p1p3
            p1, p2, p3 = v2p[fv[fid]]
            p1p2 = p2 - p1
            p1p3 = p3 - p1
            p1px = a * p1p2 + b * p1p3
            px = p1 + p1px
            x, y = (px).astype(int)
            
            image_texture[j, i] = image[y, x]
            mask_texture[j, i] = mask[y, x]
    
    cv2.imwrite(image_texture_path, image_texture)
    cv2.imwrite(mask_texture_path, mask_texture)

    """
    Normal texture
    """

    fn = compute_fn(v, fv)       # face normal (F, 3)
    
    fa = np.dot(fn, [0, 0, -1])  # face angle (F, ) [-1, 1]
    
    fw = np.maximum(fa, 0)       # face weight (F, ) [0, 1]

    fp = fw * 255                # face priority (F, ) [0, 255]

    normal_texture = np.zeros((SIZE, SIZE), np.uint8)
    for fid in visible_fids:
        cv2.drawContours(image=normal_texture, contours=[v2t[fvt[fid]]], contourIdx=0, color=int(fp[fid]), thickness=-1)
    
    cv2.imwrite(normal_texture_path, normal_texture)


def extract_body(
        image_texture_path, mask_texture_path, normal_texture_path,
        texture_path, visible_path, weight_path
    ):
    
    image_texture = cv2.imread(image_texture_path)
    mask_texture = cv2.imread(mask_texture_path)
    normal_texture = cv2.imread(normal_texture_path, cv2.IMREAD_GRAYSCALE)

    red_mask = np.all(mask_texture == RED, axis=2)
    green_mask = np.all(mask_texture == GREEN, axis=2)
    blue_mask = np.all(mask_texture == BLUE, axis=2)
    
    texture = image_texture.copy()
    texture[(red_mask|blue_mask)] = GRAY
    cv2.imwrite(texture_path, texture)
    
    visible = np.zeros((SIZE, SIZE), dtype=np.uint8)
    visible[green_mask] = 255
    cv2.imwrite(visible_path, visible)

    weight = normal_texture.copy()
    weight[(red_mask|blue_mask)] = 0
    cv2.imwrite(weight_path, weight)


def blending(
        texture_paths, weight_paths, template_texture_path,
        blended_texture_path, blended_visible_path,
        blending_method
    ):

    textures = np.array([cv2.imread(texture_path) for texture_path in texture_paths])
    weights = np.array([cv2.imread(weight_path, cv2.IMREAD_GRAYSCALE) for weight_path in weight_paths])
    template_texture = cv2.imread(template_texture_path)

    """
    blended texture
    """

    blended_texture = template_texture.copy()

    if blending_method == "priority":

        for texture, weight in reversed(list(zip(textures, weights))):
            nonzero_mask = (weight > 0)
            blended_texture[nonzero_mask] = texture[nonzero_mask]
    
    elif blending_method == "average":

        weights_ = weights.copy()
        
        nonzero_mask = (weights > 0)
        weights[nonzero_mask] = 1

        sum_weights = np.sum(weights, axis=0)
        zero_mask = (sum_weights == 0)
        nonzero_mask = (sum_weights > 0)
        sum_weights[zero_mask] = 1
        normalized_weights = weights / sum_weights
        
        weighted_textures = textures * normalized_weights[:, :, :, np.newaxis]
        sum_weighted_texture = np.sum(weighted_textures, axis=0)
        blended_texture[nonzero_mask] = sum_weighted_texture[nonzero_mask]

        weights = weights_

    elif blending_method == "DINAR":

        sum_weights = np.sum(weights, axis=0)
        zero_mask = (sum_weights == 0)
        nonzero_mask = (sum_weights > 0)
        sum_weights[zero_mask] = 1
        normalized_weights = weights / sum_weights
        
        weighted_textures = textures * normalized_weights[:, :, :, np.newaxis]
        sum_weighted_texture = np.sum(weighted_textures, axis=0)
        blended_texture[nonzero_mask] = sum_weighted_texture[nonzero_mask]
        
    cv2.imwrite(blended_texture_path, blended_texture)

    """
    blended visible
    """

    blended_visible = np.zeros((SIZE, SIZE), dtype=np.uint8)
    
    sum_weights = np.sum(weights, axis=0)
    nonzero_mask = (sum_weights > 0)
    blended_visible[nonzero_mask] = 255
    
    cv2.imwrite(blended_visible_path, blended_visible)


def inpainting(
        blended_texture_path, blended_visible_path, template_texture_path,
        inpainted_texture_path
    ):

    blended_texture = cv2.imread(blended_texture_path)
    blended_visible = cv2.imread(blended_visible_path)
    template_texture = cv2.imread(template_texture_path)

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    inpainted_texture = blended_texture.copy()
    to_inpaint = np.all(blended_visible == BLACK, axis=2) & np.all(template_texture == GRAY, axis=2)
    while np.any(to_inpaint):
        new_inpainted_texture = inpainted_texture.copy()
        new_to_inpaint = to_inpaint.copy()
        for i in range(SIZE):
            for j in range(SIZE):
                if to_inpaint[i, j]:
                    neighbors = []
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < SIZE and 0 <= nj < SIZE and not to_inpaint[ni][nj] and np.any(template_texture[ni][nj] == GRAY):
                            neighbors.append((ni, nj))
                    neighbors = np.array(neighbors)
                    if neighbors.size > 0:
                        new_inpainted_texture[i, j] = np.mean(inpainted_texture[neighbors[:, 0], neighbors[:, 1]], axis=0)
                        new_to_inpaint[i, j] = False
        inpainted_texture = new_inpainted_texture
        to_inpaint = new_to_inpaint

    cv2.imwrite(inpainted_texture_path, inpainted_texture)


# ====================================================================================================


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_folder", type=str)
    parser.add_argument("--model_type", type=str)
    args = parser.parse_args()

    filenames = os.listdir(os.path.join(args.object_folder, "images"))                          # ["test1.jpg", "test10.jpg", "test2.jpg"]
    filenames.sort(key=lambda filename: int(filename.replace("test", "").replace(".jpg", "")))  # ["test1.jpg", "test2.jpg", "test10.jpg"]
    image_ids = [filename.split(".")[0] for filename in filenames]                              # ["test1", "test2", "test10"]

    save_dir = os.path.join(args.object_folder, f"texture_{args.model_type}")
    save_subdirs = []
    for image_id in image_ids:
        save_subdir = os.path.join(save_dir, image_id)
        save_subdirs.append(save_subdir)
        os.makedirs(save_subdir, exist_ok=True)
    
    for filename, image_id, save_subdir in zip(filenames, image_ids, save_subdirs):
        
        # input path
        image_path = os.path.join(args.object_folder, "images", filename)
        mask_path = os.path.join(args.object_folder, "PGN", f"{image_id}_PGN.png")
        model_path = os.path.join(args.object_folder, args.model_type, "meshes", image_id, "000.obj")
        template_model_path = os.path.join("template", f"template_model_{args.model_type}.obj")
        parameter_path = os.path.join(args.object_folder, args.model_type, "results", image_id, "000.pkl")
        template_texture_path = os.path.join("template", f"template_texture_{args.model_type}_{SIZE}.png")

        # output path
        all_faces_on_image_path = os.path.join(save_subdir, "all_faces_on_image.png")
        all_faces_on_texture_path = os.path.join(save_subdir, "all_faces_on_texture.png")
        front_faces_on_image_path = os.path.join(save_subdir, "front_faces_on_image.png")
        front_faces_on_texture_path = os.path.join(save_subdir, "front_faces_on_texture.png")
        image_texture_path = os.path.join(save_subdir, "image_texture.png")
        mask_texture_path = os.path.join(save_subdir, "mask_texture.png")
        normal_texture_path = os.path.join(save_subdir, "normal_texture.png")
        texture_path = os.path.join(save_subdir, "texture.png")
        visible_path = os.path.join(save_subdir, "visible.png")
        weight_path = os.path.join(save_subdir, "weight.png")

        # print information
        print(f"Start processing {filename}")

        # algorithm
        # warping(
        #     image_path, mask_path, model_path, template_model_path, parameter_path, template_texture_path,
        #     all_faces_on_image_path, all_faces_on_texture_path, front_faces_on_image_path, front_faces_on_texture_path, image_texture_path, mask_texture_path, normal_texture_path
        # )
        # extract_body(
        #     image_texture_path, mask_texture_path, normal_texture_path,
        #     texture_path, visible_path, weight_path
        # )
    
    for blending_method in ["priority", "average", "DINAR"]:

        if blending_method != "DINAR":
            continue
    
        # input path
        texture_paths = [os.path.join(save_subdir, "texture.png") for save_subdir in save_subdirs]
        weight_paths = [os.path.join(save_subdir, "weight.png") for save_subdir in save_subdirs]
        template_texture_path = os.path.join("template", f"template_texture_{args.model_type}_{SIZE}.png")

        # output path
        blended_texture_path = os.path.join(save_dir, f"blended_texture_({blending_method}_blending).png")
        blended_visible_path = os.path.join(save_dir, "blended_visible.png")
        inpainted_texture_path = os.path.join(save_dir, f"inpainted_texture_({blending_method}_blending).png")

        # select textures for blending
        selection = [1, 9, 2, 16, 6, 12]
        texture_paths = [texture_paths[i - 1] for i in selection]
        weight_paths = [weight_paths[i - 1] for i in selection]

        # print information
        print(f"Start {blending_method} blending")
        
        # algorithm
        blending(
            texture_paths, weight_paths, template_texture_path,
            blended_texture_path, blended_visible_path,
            blending_method
        )
        inpainting(
            blended_texture_path, blended_visible_path, template_texture_path,
            inpainted_texture_path
        )


if __name__ == "__main__":
    # python main.py --object_folder data/obj4 --model_type smpl
    # python main.py --object_folder data/obj4 --model_type smplx
    main()
