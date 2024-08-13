import os
import cv2
import numpy as np
import scipy
import pickle
import shutil
import scipy.special
import trimesh
import argparse
import collections


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


def compute_vuv_vd(v, rmat, tvec, H, W, degree_x, degree_y, degree_z, focal_length=5000):
    v = v.T                      # (N, 3) -> (3, N)
    rmat = rmat.reshape(3, 3)    # (1, 3, 3) -> (3, 3)
    tvec = tvec.reshape(3, 1)    # (1, 3) -> (3, 1)
    cmat = np.array([
        [focal_length,            0, W // 2],
        [           0, focal_length, H // 2],
        [           0,            0,      1]
    ], float)

    # delta_x, delta_y, delta_z = 0, 0, 0
    # degree_x, degree_y, degree_z = 0, 0, 0

    # delta_x, delta_y, delta_z = -0.05, 0, 0.5
    # degree_x, degree_y, degree_z = 0, 30, 10

    # delta_x, delta_y, delta_z = 0, 0, 0
    # degree_x, degree_y, degree_z = 0, 90, 0

    delta_x, delta_y, delta_z = 0.03, -0.47, 1
    degree_x, degree_y, degree_z = 20, 0, 0

    theta_x = np.pi * (degree_x / 180)
    rotate_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ], float)

    theta_y = np.pi * (degree_y / 180)
    rotate_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ], float)

    theta_z = np.pi * (degree_z / 180)
    rotate_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ], float)

    translate = np.array([
        [delta_x],
        [delta_y],
        [delta_z]
    ], float)
    
    ms = v         # model space (3, N)
    ws = rotate_x.dot(rotate_z.dot(rotate_y.dot(ms))) + translate         # world space (3, N)
    vs = rmat.dot(ws) + tvec     # view/projection space (3, N)
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
        degree_x, degree_y, degree_z
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

    v2p, vd = compute_vuv_vd(v, rmat, tvec, H, W, degree_x, degree_y, degree_z)  # vertex to pixel (V, 2) & vertex depth (V, )
    v2t = (vt * SIZE).astype(int)                  # vertex to texel (VT, 2)

    """
    Step 2: Rendering before z-test
    """

    all_faces_on_image = image.copy()               # all faces on image (H, W, 3)
    for fid in range(F):
        cv2.drawContours(image=all_faces_on_image, contours=[v2p[fv[fid]]], contourIdx=0, color=BLUE, thickness=1)
    
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
    
    # cv2.imwrite(all_faces_on_texture_path, all_faces_on_texture)

    """
    Step 4: Rendering after z-test
    """

    front_faces_on_image = image.copy()               # front faces on image (H, W, 3)
    for fid in visible_fids:
        cv2.drawContours(image=front_faces_on_image, contours=[v2p[fv[fid]]], contourIdx=0, color=BLUE, thickness=1)
    
    cv2.imwrite("test.jpg", front_faces_on_image)


# ====================================================================================================


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_folder", type=str)
    parser.add_argument("--model_type", type=str)
    args = parser.parse_args()

    filenames = os.listdir(os.path.join(args.object_folder, "images"))  # ["test1.jpg", "test2.jpg"]
    filenames.sort(key=lambda filename: int(filename.replace("test", "").replace(".jpg", "")))
    image_ids = [filename.split(".")[0] for filename in filenames]      # ["test1", "test2"]

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
        model_path = os.path.join(args.object_folder, args.model_type, "meshes", "test1", "000.obj")
        template_model_path = os.path.join("template", f"template_model_{args.model_type}.obj")
        parameter_path = os.path.join(args.object_folder, args.model_type, "results", image_id, "000.pkl")
        template_texture_path = os.path.join("template", f"template_texture_{args.model_type}_{SIZE}.png")

        # output path
        all_faces_on_image_path = os.path.join(save_subdir, "all_faces_on_image.png")

        # print information
        print(f"Start processing {filename}")

        if image_id != "test1":
            continue

        # algorithm
        degree_x = 0
        degree_y = 0
        degree_z = 0
        warping(
            image_path, mask_path, template_model_path, template_model_path, parameter_path, template_texture_path,
            degree_x, degree_y, degree_z
        )
        # for axis in ["Y", "Z", "X"]:
        #     cmd = input(f"Rotate in {axis}-axis\n")
        #     while cmd != "OK":
        #         if axis == "X":
        #             degree_x = int(cmd)
        #         elif axis == "Y":
        #             degree_y = int(cmd)
        #         elif axis == "Z":
        #             degree_z = int(cmd)
        #         warping(
        #             image_path, mask_path, template_model_path, template_model_path, parameter_path, template_texture_path,
        #             degree_x, degree_y, degree_z
        #         )
        #         cmd = input()


if __name__ == "__main__":
    # python test.py --object_folder data/obj4 --model_type smpl
    # python main.py --object_folder data/obj4 --model_type smplx
    main()
