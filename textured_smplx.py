# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:46:26 2020

@author: qzane
"""

import os
import pickle
import collections
import numpy as np
import scipy.ndimage
import cv2
import utils
from utils import obj_vv,obj_vt,obj_fv,obj_ft,fv2norm,cv2_triangle

      
def get_texture_SMPL(fname_img, fname_obj, fname_pkl, npath, obj_name, template_obj, is_mask):
    ''' 
        fname_img: the image file
        fname_obj: the .obj file that smplify-x produced
        fname_pkl: the .pkl file that smplify-x produced
        npath: the output path
        obj_name: name for the output files
        template_obj: the texture template for smpl or smplx model
        
        output: texture image and face normals
    '''
    if not os.path.isdir(npath):
        os.makedirs(npath)
        
    print('get_texture', fname_img, fname_obj, fname_pkl, npath, obj_name, template_obj)
    
    #obj_name = os.path.split(fname_img)[1][:-4] # image name without .png
    f_obj = fname_obj
    f_img = fname_img
    img = cv2.imread(f_img)
    
    height,width = img.shape[:2]
    
    vv = obj_vv(f_obj)
    vv[:, 1:] = -vv[:, 1:]
    
    vt = obj_vt(template_obj)
    vt[:, 1] = 1-vt[:, 1] # smplx texture is flipped
    fv = obj_fv(template_obj)
    ft = obj_ft(template_obj)
    vtuv = (vt*1000).astype(int)
    
    fnorm = fv2norm(fv, vv)
    
    
    with open(fname_pkl,'rb') as f:
        data = pickle.load(f)
    
    rmat = np.array(data['camera_rotation']).reshape(3,3)
    tvec = np.array(data['camera_translation']).reshape(3)
    cameraMatrix = np.array([[5000,0,width//2],[0,5000,height//2],[0,0,1]], float)
    
    vuv = utils.p3d_p2d(vv, rmat, tvec, cameraMatrix).reshape(-1,3)
    
    vuv, vd = vuv[:, :2], vuv[:, 2]
    # vuv: vertices in texture space (u,v)
    # vd: vertices depth
    
    vuv = vuv.astype(int) 
    
    timg = img.copy() # img with triangles
    for _fv in fv:
        cv2_triangle(timg, vuv[_fv])
        
    #cv2.imwrite(os.path.join(npath, 'T%05d_all_V.png'%frame), timg)
    
    
    texture = np.zeros((1000,1000, 3), np.uint8)
    
    for _ft in ft:
        cv2.drawContours(texture, [vtuv[_ft]], 0, (100,100,100), -1)
        
    #cv2.imwrite(os.path.join(npath, 'T%05d_all_texture.png'%frame), texture)        
    
            
    
    fd = np.zeros(fv.shape[0], dtype=float)
    for no, _fv in enumerate(fv): #todo: vectorization
        fd[no] = max(vd[_fv]) # most far away point
    f_order = np.argsort(-fd) # f index from far to near
    
    img_f = np.ones(img.shape[:2], dtype=int) * -1
    for f in f_order:
        cv2.drawContours(img_f, [vuv[fv[f]]], 0, int(f), -1)
    visible_f = np.unique(img_f.flatten())[1:] # remove -1
    
    
    
    timg = img.copy() # img with triangles
    for f in visible_f:
        cv2_triangle(timg, vuv[fv[f]])
        
    #cv2.imwrite(os.path.join(npath, 'T%05d_V.png'%frame), timg)  
    
    vis_texture = np.zeros((1000,1000), np.uint8)
    for f in visible_f:
        cv2_triangle(texture, vtuv[ft[f]])
        cv2.drawContours(vis_texture, [vtuv[ft[f]]], 0, 255, -1)
        
        
    #cv2.imwrite(os.path.join(npath, 'T%05d_v_texture.png'%frame), texture)  
    #cv2.imwrite(os.path.join(npath, '%s_vis_texture.png'%obj_name), vis_texture)  
    
    img_f_texture = np.ones((1000,1000), dtype=int) * -1
    for f in visible_f:
        cv2.drawContours(img_f_texture, [vtuv[ft[f]]], 0, int(f), -1)
        
    
    norm_texture = np.ones((1000,1000,3), np.uint8)# * 255
    depth_texture = np.zeros((1000,1000), float)# * 255
    norm_photo = np.ones(img.shape, np.uint8)# * 255
    
     
    for x in range(1000):
        for y in range(1000):
            f = img_f_texture[y,x]
            if f != -1:
                _vuv = vuv[fv[f]] #(3, 2)
                
                _vtuv = vt[ft[f]] # (3,2)
                px = np.array((x/1000.0, y/1000.0))
                
                # solve P1Px = u*P1P2 + v*P1P3                
                p1,p2,p3 = _vtuv
                p1px = px-p1
                p1p2 = p2-p1
                p1p3 = p3-p1
                _u,_v = np.linalg.solve(((p1p2[0],p1p3[0]),(p1p2[1],p1p3[1])),p1px)
                
                u,v = (_vuv[0] + _u*(_vuv[1]-_vuv[0]) + _v*(_vuv[2]-_vuv[0])).astype(int)
                
                _vv = vv[fv[f]]
                
                xyz = (_vv[0] + _u*(_vv[1]-_vv[0]) + _v*(_vv[2]-_vv[0]))
                
                vxyz = xyz
                vxyz = vxyz / np.sqrt((vxyz*vxyz).sum())    
                
                _fnorm = fnorm[f]
                
                cosA = (vxyz*_fnorm).sum()
                #abscosA = np.abs(cosA)
                                    
                A = np.arccos(cosA) #between 0 to pi/2
                
                if A < np.pi* 0.5:
                    colorA = (255, 255, int(A / np.pi * 2 * 255))
                else:
                    colorA = (int((np.pi-A) / np.pi * 2 * 255),255, 255)
                    
                #colorA = int(A / np.pi * 2 * 255)
                norm_texture[y,x]=colorA
                depth_texture[y,x]=(xyz)[2]
                try:
                    norm_photo[v, u] = colorA
                except:
                    pass
                
                try: # when uv is outside the image, just continue
                    texture[y,x]=img[v, u]
                except:
                    pass
    if not is_mask:
        cv2.imwrite(os.path.join(npath, '%s_raw_texture.png'%obj_name), texture)
    else:
        cv2.imwrite(os.path.join(npath, '%s_mask.png'%obj_name), texture)
        
    
def extract_body(npath, obj_name):
    raw_texture = cv2.imread(os.path.join(npath, '%s_raw_texture.png'%obj_name))
    mask = cv2.imread(os.path.join(npath, '%s_mask.png'%obj_name))
    # 
    red_mask = np.all(mask == [0, 0, 255], axis=2)
    green_mask = np.all(mask == [0, 255, 0], axis=2)
    blue_mask = np.all(mask == [255, 0, 0], axis=2)
    # 
    intm_texture = raw_texture.copy()
    intm_texture[red_mask] = [100, 100, 100]
    cv2.imwrite(os.path.join(npath, '%s_intm_texture.png'%obj_name), intm_texture)
    # 
    intm_visible = mask.copy()
    intm_visible[(green_mask|blue_mask)] = [255, 255, 255]
    intm_visible[~(green_mask|blue_mask)] = [0, 0, 0]
    cv2.imwrite(os.path.join(npath, '%s_intm_visible.png'%obj_name), intm_visible)
    # 
    texture = raw_texture.copy()
    texture[(red_mask|blue_mask)] = [100, 100, 100]
    cv2.imwrite(os.path.join(npath, '%s_texture.png'%obj_name), texture)
    # 
    visible = mask.copy()
    visible[green_mask] = [255, 255, 255]
    visible[~green_mask] = [0, 0, 0]
    cv2.imwrite(os.path.join(npath, '%s_visible.png'%obj_name), visible)
    
    
def combine_texture_SMPL(inPath, frames=None):
    front_texture = cv2.imread(os.path.join(inPath, 'front_texture.png'))
    front_visible = cv2.imread(os.path.join(inPath, 'front_visible.png'))
    back_texture = cv2.imread(os.path.join(inPath, 'back_texture.png'))
    back_visible = cv2.imread(os.path.join(inPath, 'back_visible.png'))
    combined_texture = front_texture
    combined_visible = front_visible
    selected = np.all(front_visible == [0, 0, 0], axis=2) & np.all(back_visible == [255, 255, 255], axis=2)
    combined_texture[selected] = back_texture[selected]
    combined_visible[selected] = [255, 255, 255]
    cv2.imwrite(os.path.join(inPath, 'combined_texture.png'), combined_texture)
    cv2.imwrite(os.path.join(inPath, 'combined_visible.png'), combined_visible)
        
        
def complete_texture(f_texture, f_vis, f_mask):
    ''' texture_acc and 0000_texture_acc_vis'''
    texture = cv2.imread(f_texture)
    vis = cv2.imread(f_vis)[:,:,0]
    mask = cv2.imread(f_mask)[:,:,0]
    labels, num_label = scipy.ndimage.measurements.label(mask)    
    texture_sum = np.zeros(texture.shape, float)
    texture_count = np.zeros((texture.shape[0], texture.shape[1], 1), int)
    h,w = vis.shape
    
    dxdy = ((1,0),(-1,0),(0,1),(0,-1))
    dxdy = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1))
    for label in range(1, num_label):
        ''' use bfs to complete the texture for each body part '''
        print('completing label:', label)
        queue = collections.deque() # (x,y) for pixel to visit
        to_visit = set()
        for x,y in zip(*np.where(labels==label)): # todo: use erosion on labels first
            if not vis[x][y]:
                continue
            border = 0
            for dx,dy in dxdy:
                nx, ny = x+dx, y+dy
                if nx>=0 and nx<h and ny>=0 and ny<w and vis[nx][ny]==0 and labels[nx][ny]==label:
                    border += 1
            if border:
                queue.append((x,y))
                texture_sum[x][y]=texture[x][y]
                texture_count[x][y][0]=1
        while(queue):
            x,y = queue.popleft()
            texture[x][y]=(texture_sum[x][y] / texture_count[x][y]).astype(np.uint8)
            vis[nx][ny]=255
            
            for dx,dy in dxdy:
                nx, ny = x+dx, y+dy
                if nx>=0 and nx<h and ny>=0 and ny<w and vis[nx][ny]==0 and labels[nx][ny]==label:
                    if (nx,ny) not in to_visit:
                        queue.append((nx,ny))
                        to_visit.add((nx,ny))
                    texture_sum[nx][ny]+=texture[x][y]
                    texture_count[nx][ny][0]+=1
    cv2.imwrite(f_texture[:-20]+'completed_texture.png', texture)
