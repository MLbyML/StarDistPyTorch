from skimage.measure import regionprops
from skimage.draw import polygon
import numpy as np
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from tqdm import tqdm
from stardistPytorch.lib import stardist2d
from scipy.optimize import linear_sum_assignment
from numba import jit
from collections import namedtuple


matching_criteria = dict()

def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x

def _raise(e):
    raise e

def gputools_available():
    try:
        import gputools
    except:
        return False
    return True
    
def calculate_extents(lbl, func=np.median):
    """ Aggregate bounding box sizes of objects in label images. """
    if isinstance(lbl,(tuple,list)) or (isinstance(lbl,np.ndarray) and lbl.ndim==4):
        return func(np.stack([calculate_extents(_lbl,func) for _lbl in lbl], axis=0), axis=0)

    n = lbl.ndim
    n in (2,3) or _raise(ValueError("label image should be 2- or 3-dimensional (or pass a list of these)"))

    regs = regionprops(lbl)
    if len(regs) == 0:
        return np.zeros(n)
    else:
        extents = np.array([np.array(r.bbox[n:])-np.array(r.bbox[:n]) for r in regs])
        return func(extents, axis=0)

def precision(tp,fp,fn):
    return tp/(tp+fp) if tp > 0 else 0
def recall(tp,fp,fn):
    return tp/(tp+fn) if tp > 0 else 0
def accuracy(tp,fp,fn):
    return tp/(tp+fp+fn) if tp > 0 else 0
def f1(tp,fp,fn):
    return (2*tp)/(2*tp+fp+fn) if tp > 0 else 0
    
def intersection_over_union(overlap):
    _check_label_array(overlap,'overlap')
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return overlap / (n_pixels_pred + n_pixels_true - overlap)

matching_criteria['iou'] = intersection_over_union

def relabel_sequential(label_field, offset=1):
    offset = int(offset)
    if offset <= 0:
        raise ValueError("Offset must be strictly positive.")
    if np.min(label_field) < 0:
        raise ValueError("Cannot relabel array that contains negative values.")
    m = label_field.max()
    if not np.issubdtype(label_field.dtype, np.integer):
        new_type = np.min_scalar_type(int(m))
        label_field = label_field.astype(new_type)
        m = m.astype(new_type)  # Ensures m is an integer
    labels = np.unique(label_field)
    labels0 = labels[labels != 0]
    required_type = np.min_scalar_type(offset + len(labels0))
    if np.dtype(required_type).itemsize > np.dtype(label_field.dtype).itemsize:
        label_field = label_field.astype(required_type)
    new_labels0 = np.arange(offset, offset + len(labels0))
    if np.all(labels0 == new_labels0):
        return label_field, labels, labels
    forward_map = np.zeros(int(m + 1), dtype=label_field.dtype)
    forward_map[labels0] = new_labels0
    if not (labels == 0).any():
        labels = np.concatenate(([0], labels))
    inverse_map = np.zeros(offset - 1 + len(labels), dtype=label_field.dtype)
    inverse_map[(offset - 1):] = labels
    relabeled = forward_map[label_field]
    return relabeled, forward_map, inverse_map

def intersection_over_true(overlap):
    _check_label_array(overlap,'overlap')
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return overlap / n_pixels_true

matching_criteria['iot'] = intersection_over_true

def label_overlap(x, y, check=True):
    if check:
        _check_label_array(x,'x',True)
        _check_label_array(y,'y',True)
        x.shape == y.shape or _raise(ValueError("x and y must have the same shape"))
    return _label_overlap(x, y)

@jit(nopython=True)
def _label_overlap(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def intersection_over_pred(overlap):
    _check_label_array(overlap,'overlap')
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    return overlap / n_pixels_pred

matching_criteria['iop'] = intersection_over_pred

def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask_filled = binary_fill_holes(grown_mask,**kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled

def matching_dataset(y_true, y_pred, thresh=0.5, criterion='iou', by_image=False, show_progress=True, parallel=False):
    len(y_true) == len(y_pred) or _raise(ValueError("y_true and y_pred must have the same length."))
    return matching_dataset_lazy (
        tuple(zip(y_true,y_pred)), thresh=thresh, criterion=criterion, by_image=by_image, show_progress=show_progress, parallel=parallel,
    )

def _cpp_star_dist(a, n_rays=32):
    #TODO
    (np.isscalar(n_rays) and 0 < int(n_rays)) or _raise(ValueError())
    return stardist2d.c_star_dist(a.astype(np.uint16,copy=False), int(n_rays))

def star_dist(a, n_rays=32, mode='cpp'):
    """'a' assumbed to be a label image with integer values that encode object ids. id 0 denotes background."""
    #TODO
    n_rays >= 3 or _raise(ValueError("need 'n_rays' >= 3"))

    if mode == 'python':
        return _py_star_dist(a, n_rays)
    elif mode == 'cpp':
        return _cpp_star_dist(a, n_rays)
    elif mode == 'opencl':
        return _ocl_star_dist(a, n_rays)
    else:
        _raise(ValueError("Unknown mode %s" % mode))

def polygons_to_label(coord, prob, points, shape=None, thr=-np.inf):
    sh = coord.shape[:2] if shape is None else shape
    lbl = np.zeros(sh,np.int32)
    ind = np.argsort([ prob[p[0],p[1]] for p in points ])
    points = points[ind]

    i = 1
    for p in points:
        if prob[p[0],p[1]] < thr:
            continue
        rr,cc = polygon(coord[p[0],p[1],0], coord[p[0],p[1],1], sh)
        lbl[rr,cc] = i
        i += 1

    return lbl

def relabel_image_stardist(lbl, n_rays, **kwargs):
    """relabel each label region in `lbl` with its star representation"""
    _check_label_array(lbl, "lbl")
    dist = star_dist(lbl, n_rays, **kwargs)
    coord = dist_to_coord(dist)
    points = np.array(tuple(np.array(r.centroid).astype(int) for r in regionprops(lbl)))
    return polygons_to_label(coord, np.ones_like(lbl), points, shape=lbl.shape)

def _check_label_array(y, name=None, check_sequential=False):
    err = ValueError("{label} must be an array of {integers}.".format(
        label = 'labels' if name is None else name,
        integers = ('sequential ' if check_sequential else '') + 'non-negative integers',
    ))
    is_array_of_integers(y) or _raise(err)
    if check_sequential:
        label_are_sequential(y) or _raise(err)
    else:
        y.min() >= 0 or _raise(err)
    return True

def is_array_of_integers(y):
    return isinstance(y,np.ndarray) and np.issubdtype(y.dtype, np.integer)

def ray_angles(n_rays=32):
    return np.linspace(0,2*np.pi,n_rays,endpoint=False)


def random_label_cmap(n=2**16):
    import matplotlib
    import colorsys
    h,l,s = np.random.uniform(0,1,n), 0.4 + np.random.uniform(0,0.6,n), 0.2 + np.random.uniform(0,0.8,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

def _normalize_grid(grid,n):
    try:
        grid = tuple(grid)
        (len(grid) == n and
         all(map(np.isscalar,grid)) and
         all(map(_is_power_of_2,grid))) or _raise(TypeError())
        return tuple(int(g) for g in grid)
    except (TypeError, AssertionError):
        raise ValueError("grid = {grid} must be a list/tuple of length {n} with values that are power of 2".format(grid=grid, n=n))
        
def _is_power_of_2(i):    
    assert i > 0
    e = np.log2(i)
    return e == int(e)

def matching_dataset_lazy(y_gen, thresh=0.5, criterion='iou', by_image=False, show_progress=True, parallel=False):

    expected_keys = set(('fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion', 'thresh', 'n_true', 'n_pred', 'mean_true_score'))

    single_thresh = False
    if np.isscalar(thresh):
        single_thresh = True
        thresh = (thresh,)

    tqdm_kwargs = {}
    tqdm_kwargs['disable'] = not bool(show_progress)
    if int(show_progress) > 1:
        tqdm_kwargs['total'] = int(show_progress)

    # compute matching stats for every pair of label images
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        fn = lambda pair: matching(*pair, thresh=thresh, criterion=criterion, report_matches=False)
        with ThreadPoolExecutor() as pool:
            stats_all = tuple(pool.map(fn, tqdm(y_gen,**tqdm_kwargs)))
    else:
        stats_all = tuple (
            matching(y_t, y_p, thresh=thresh, criterion=criterion, report_matches=False)
            for y_t,y_p in tqdm(y_gen,**tqdm_kwargs)
        )

    # accumulate results over all images for each threshold separately
    n_images, n_threshs = len(stats_all), len(thresh)
    accumulate = [{} for _ in range(n_threshs)]
    for stats in stats_all:
        for i,s in enumerate(stats):
            acc = accumulate[i]
            for k,v in s._asdict().items():
                if k == 'mean_true_score' and not bool(by_image):
                    # convert mean_true_score to "sum_true_score"
                    acc[k] = acc.setdefault(k,0) + v * s.n_true
                else:
                    try:
                        acc[k] = acc.setdefault(k,0) + v
                    except TypeError:
                        pass

    # normalize/compute 'precision', 'recall', 'accuracy', 'f1'
    for thr,acc in zip(thresh,accumulate):
        set(acc.keys()) == expected_keys or _raise(ValueError("unexpected keys"))
        acc['criterion'] = criterion
        acc['thresh'] = thr
        acc['by_image'] = bool(by_image)
        if bool(by_image):
            for k in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score'):
                acc[k] /= n_images
        else:
            tp, fp, fn = acc['tp'], acc['fp'], acc['fn']
            acc.update(
                precision       = precision(tp,fp,fn),
                recall          = recall(tp,fp,fn),
                accuracy        = accuracy(tp,fp,fn),
                f1              = f1(tp,fp,fn),
                mean_true_score = acc['mean_true_score'] / acc['n_true'] if acc['n_true'] > 0 else 0.0,
            )

    accumulate = tuple(namedtuple('DatasetMatching',acc.keys())(*acc.values()) for acc in accumulate)
    return accumulate[0] if single_thresh else accumulate

def matching(y_true, y_pred, thresh=0.5, criterion='iou', report_matches=False):
    """
    if report_matches=True, return (matched_pairs,matched_scores) are independent of 'thresh'
    """
    _check_label_array(y_true,'y_true')
    _check_label_array(y_pred,'y_pred')
    y_true.shape == y_pred.shape or _raise(ValueError("y_true ({y_true.shape}) and y_pred ({y_pred.shape}) have different shapes".format(y_true=y_true, y_pred=y_pred)))
    criterion in matching_criteria or _raise(ValueError("Matching criterion '%s' not supported." % criterion))
    if thresh is None: thresh = 0
    thresh = float(thresh) if np.isscalar(thresh) else map(float,thresh)

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)

    overlap = label_overlap(y_true, y_pred, check=False)
    scores = matching_criteria[criterion](overlap)
    assert 0 <= np.min(scores) <= np.max(scores) <= 1

    # ignoring background
    scores = scores[1:,1:]
    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    def _single(thr):
        not_trivial = n_matched > 0 and np.any(scores >= thr)
        if not_trivial:
            # compute optimal matching with scores as tie-breaker
            costs = -(scores >= thr).astype(float) - scores / (2*n_matched)
            true_ind, pred_ind = linear_sum_assignment(costs)
            assert n_matched == len(true_ind) == len(pred_ind)
            match_ok = scores[true_ind,pred_ind] >= thr
            tp = np.count_nonzero(match_ok)
        else:
            tp = 0
        fp = n_pred - tp
        fn = n_true - tp
        stats_dict = dict (
            criterion       = criterion,
            thresh          = thr,
            fp              = fp,
            tp              = tp,
            fn              = fn,
            precision       = precision(tp,fp,fn),
            recall          = recall(tp,fp,fn),
            accuracy        = accuracy(tp,fp,fn),
            f1              = f1(tp,fp,fn),
            n_true          = n_true,
            n_pred          = n_pred,
            mean_true_score = np.sum(scores[true_ind,pred_ind][match_ok]) / n_true if not_trivial else 0.0,
        )
        if bool(report_matches):
            if not_trivial:
                stats_dict.update (
                    # int() to be json serializable
                    matched_pairs  = tuple((int(map_rev_true[i]),int(map_rev_pred[j])) for i,j in zip(1+true_ind,1+pred_ind)),
                    matched_scores = tuple(scores[true_ind,pred_ind]),
                    matched_tps    = tuple(map(int,np.flatnonzero(match_ok))),
                )
            else:
                stats_dict.update (
                    matched_pairs  = (),
                    matched_scores = (),
                    matched_tps    = (),
                )
        return namedtuple('Matching',stats_dict.keys())(*stats_dict.values())

    return _single(thresh) if np.isscalar(thresh) else tuple(map(_single,thresh))
