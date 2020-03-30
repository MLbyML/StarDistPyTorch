import math
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import tifffile
import sys
from stardistPytorch.utils import polygons_to_label, _normalize_grid, random_label_cmap
    
device = torch.device("cuda")

def _plot_polygon(x,y,score,color):
    import matplotlib.pyplot as plt
    a,b = list(x),list(y)
    a += a[:1]
    b += b[:1]
    plt.plot(a,b,'--', alpha=1, linewidth=score, zorder=1, color=color)


def draw_polygons(coord, score, poly_idx, grid=(1,1), cmap=None, show_dist=False):
    """poly_idx is a N x 2 array with row-col coordinate indices"""
    return _draw_polygons(polygons=coord[poly_idx[:,0],poly_idx[:,1]],
                         points=poly_idx,
                         scores=score[poly_idx[:,0],poly_idx[:,1]],
                         grid=grid, cmap=cmap, show_dist=show_dist)


def _draw_polygons(polygons, points=None, scores=None, grid=(1,1), cmap=None, show_dist=False):
    """
        polygons is a list/array of x,y coordinate lists/arrays
        points is a list/array of x,y coordinates
        scores is a list/array of scalar values between 0 and 1
    """
    # TODO: better name for this function?
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    grid = _normalize_grid(grid,2)
    if points is None:
        points = [None]*len(polygons)
    if scores is None:
        scores = np.ones(len(polygons))
    if cmap is None:
        cmap = random_label_cmap(len(polygons)+1)
        #cmap='plasma'
    assert len(polygons) == len(scores)
    assert len(cmap.colors[1:]) >= len(polygons)
    assert not show_dist or all(p is not None for p in points)

    for point,poly,score,c in zip(points,polygons,scores,cmap.colors[1:]):
        if point is not None:
            plt.plot(point[1]*grid[1], point[0]*grid[0], '.', markersize=8*score, color=c)

        if show_dist:
            dist_lines = np.empty((poly.shape[-1],2,2))
            dist_lines[:,0,0] = poly[1]
            dist_lines[:,0,1] = poly[0]
            dist_lines[:,1,0] = point[1]*grid[1]
            dist_lines[:,1,1] = point[0]*grid[0]
            plt.gca().add_collection(LineCollection(dist_lines, colors=c, linewidths=0.4))

        _plot_polygon(poly[1], poly[0], 3*score, color=c)


def predict(img, net, nms_thresh=0.5, prob_thresh=0.4):
    imgTorch=torch.zeros(1, 1, img.shape[0], img.shape[1]).to(device)
    imgTorch[0, 0, ...]=torch.tensor(img).to(device)
    dist, prob=net(imgTorch)
    dist=dist.cpu().detach().numpy()
    dist=np.moveaxis(dist, 1, -1)
    prob=prob.cpu().detach().numpy()
    prob=prob[0, 0, ...]
    coord = dist_to_coord(dist, grid=(1,1))
    points = non_maximum_suppression(coord, prob, grid=(1,1), nms_thresh=nms_thresh, prob_thresh=prob_thresh)
    labels = polygons_to_label(coord, prob, points, shape=img.shape)
    return labels, dict(coord=coord[points[:,0],points[:,1]], points=points, prob=prob[points[:,0],points[:,1]])

def predict_tile(img, net, tileSize=192):
    pass
      # TODO
#     pad_width=256-tileSize//2
#     largeImg=np.pad(img, pad_width=pad_width, mode='reflect')
#     imgTorch=torch.zeros(1, 1, largeImg.shape[0], largeImg.shape[1]).to(device)
#     imgTorch[0, 0, ...]=torch.tensor(img).to(device)
    


def dist_to_coord(rhos, grid=(1,1)):
    """convert from polar to cartesian coordinates for a single image (3-D array) or multiple images (4-D array)"""

    is_single_image=1
    n_images, h, w, n_rays = rhos.shape
    coord = np.empty((n_images,h,w,2,n_rays),dtype=np.float32)

    start = np.indices((h,w))
    for i in range(2):
        coord[...,i,:] = grid[i] * np.broadcast_to(start[i].reshape(1,h,w,1), (n_images,h,w,n_rays))

    phis = ray_angles(n_rays).reshape(1,1,1,n_rays)
    coord[...,0,:] += rhos * np.sin(phis) # row coordinate
    coord[...,1,:] += rhos * np.cos(phis) # col coordinate

    return coord[0] if is_single_image else coord

def ray_angles(n_rays=32):
    return np.linspace(0,2*np.pi,n_rays,endpoint=False)


def non_maximum_suppression(coord, prob, grid=(1,1), b=2, nms_thresh=0.5, prob_thresh=0.5, verbose=False, max_bbox_search=True):
    """2D coordinates of the polys that survive from a given prediction (prob, coord)

    prob.shape = (Ny,Nx)
    coord.shape = (Ny,Nx,2,n_rays)

    b: don't use pixel closer than b pixels to the image boundary
    """
    from .lib.stardist2d import c_non_max_suppression_inds

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    #assert prob.ndim == 2
    #assert coord.ndim == 4
    grid = _normalize_grid(grid,2)

    mask = prob > prob_thresh
    if b is not None and b > 0:
        _mask = np.zeros_like(mask)
        _mask[b:-b,b:-b] = True
        mask &= _mask

    polygons = coord[mask]
    scores   = prob[mask]

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    survivors = np.zeros(len(ind), np.bool)
    polygons = polygons[ind]
    scores = scores[ind]

    if max_bbox_search:
        # map pixel indices to ids of sorted polygons (-1 => polygon at that pixel not a candidate)
        mapping = -np.ones(mask.shape,np.int32)
        mapping.flat[ np.flatnonzero(mask)[ind] ] = range(len(ind))
    else:
        mapping = np.empty((0,0),np.int32)

    if verbose:
        t = time()

    survivors[ind] = c_non_max_suppression_inds(polygons.astype(np.int32),
                    mapping, np.float32(nms_thresh), np.int32(max_bbox_search),
                    np.int32(grid[0]), np.int32(grid[1]),np.int32(verbose))

    if verbose:
        print("keeping %s/%s polygons" % (np.count_nonzero(survivors), len(polygons)))
        print("NMS took %.4f s" % (time() - t))

    points = np.stack([ii[survivors] for ii in np.nonzero(mask)],axis=-1)
    return points
