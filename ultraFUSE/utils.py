import torch.nn as nn
import torch
import numpy as np
import math
import cv2
from skimage import morphology
import tqdm

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
import scipy
import copy
import pandas as pd
import skimage
import torch.nn.functional as F
from scipy import signal
import shutil
import gc
from ultraFUSE.NSCT import NSCTdec, NSCTrec
import matplotlib.pyplot as plt
import ants
import tifffile


def fusion_perslice(topSlice, bottomSlice, GFr, Gaussianr, kernel, boundary, device):
    GFbase, GFdetail = GuidedFilter(r=GFr, eps=1), GuidedFilter(r=9, eps=1e-6)
    topSlice = torch.from_numpy(topSlice).to(device)
    bottomSlice = torch.from_numpy(bottomSlice).to(device)
    topBase = torch.conv2d(
        F.pad(
            topSlice,
            (Gaussianr // 2, Gaussianr // 2, Gaussianr // 2, Gaussianr // 2),
            "reflect",
        ),
        kernel,
    )
    bottomBase = torch.conv2d(
        F.pad(
            bottomSlice,
            (Gaussianr // 2, Gaussianr // 2, Gaussianr // 2, Gaussianr // 2),
            "reflect",
        ),
        kernel,
    )
    topDetail, bottomDetail = topSlice - topBase, bottomSlice - bottomBase
    mask = torch.arange(topSlice.shape[2], device=device)[None, None, :, None]
    mask0, mask1 = (mask > boundary).to(torch.float), (mask <= boundary).to(torch.float)
    result0base, result1base = GFbase(bottomBase, mask0), GFbase(topBase, mask1)
    result0detail, result1detail = GFdetail(bottomDetail, mask0), GFdetail(
        topDetail, mask1
    )
    t = result0base + result1base + 1e-3
    result0base, result1base = result0base / t, result1base / t
    t = result0detail + result1detail + 1e-3
    result0detail, result1detail = result0detail / t, result1detail / t
    minn, maxx = min(topSlice.min(), bottomSlice.min()), max(
        topSlice.max(), bottomSlice.max()
    )
    result = torch.clip(
        result0base * bottomBase
        + result1base * topBase
        + result0detail * bottomDetail
        + result1detail * topDetail,
        minn,
        maxx,
    )
    return result.squeeze().cpu().data.numpy().astype(np.uint16)


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def diff_x(self, input, r):
        left = input[:, :, r : 2 * r + 1]
        middle = input[:, :, 2 * r + 1 :] - input[:, :, : -2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1 : -r - 1]
        output = torch.cat([left, middle, right], dim=2)
        return output

    def diff_y(self, input, r):
        left = input[:, :, :, r : 2 * r + 1]
        middle = input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1 : -r - 1]
        output = torch.cat([left, middle, right], dim=3)
        return output

    def forward(self, x):
        return self.diff_y(self.diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()
        self.r, self.eps = r, eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        x, y = 0.001 * x, 0.001 * y
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()
        N = self.boxfilter(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0))
        mean_x = self.boxfilter(x) / N
        mean_y = self.boxfilter(y) / N
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N
        return (mean_A * x + mean_b) / 0.001


def extendBoundary(
    boundary,
    resampleRatio,
    window_size,
    poly_order,
    cSize,
    boundarySmoothed=None,
    _illu=True,
):
    for i in range(boundary.shape[0]):
        tmp = (
            copy.deepcopy(boundary[i, :])
            if _illu
            else copy.deepcopy(boundarySmoothed[i, :])
        )
        p = np.where(tmp != 0)[0]
        if len(p) != 0:
            p0, p1 = p[0], p[-1]
            if _illu:
                if (p1 - p0) > 3:
                    tmp[:p0], tmp[p1:] = np.nan, np.nan
                    h = (p1 - p0) // 2
                    h = window_size // 2 if (window_size // 2 < h) else h
                    m = list(set(range(boundary.shape[1])) - set(range(p0, p1)))
                    edge = np.zeros_like(tmp)
                    edge[: p0 + h] = tmp[
                        p0
                    ]  # pd.Series(tmp[:p0+h]).interpolate(method="spline", order=3, limit_direction="both").values#[m]
                    edge[p1 - h :] = tmp[
                        p1 - 1
                    ]  # pd.Series(tmp[p1-h:]).interpolate(method="spline", order=3, limit_direction="both").values#[m]
                    boundary[i, m] = edge[m]
                else:
                    m = list(set(range(boundary.shape[1])) - set(range(p0, p1)))
                    boundary[i, :] = tmp[p0]
            else:
                t = window_size // 2 if (p1 - p0) > window_size else (p1 - p0) // 2
                boundary[i, : p0 + t] = boundary[i, p0 + t]
                boundary[i, p1 - t :] = boundary[i, p1 - t]
        else:
            boundary[i, :] = boundary[i, :]
    if _illu:
        pass
    else:
        for i in range(boundary.shape[1]):
            tmp = (
                copy.deepcopy(boundary[:, i])
                if _illu
                else copy.deepcopy(boundarySmoothed[:, i])
            )
            p = np.where(boundary[:, i] != 0)[0]
            if len(p) != 0:
                p0, p1 = p[0], p[-1]
                if (p1 - p0) > window_size:
                    tmp[p0:p1] = signal.savgol_filter(
                        signal.savgol_filter(tmp[p0:p1], window_size, 1), window_size, 1
                    )
                t = window_size // 2 if (p1 - p0) > window_size else (p1 - p0) // 2
                boundary[: p0 + t, i] = boundary[p0 + t, i]
                boundary[p1 - t :, i] = boundary[p1 - t, i]
            else:
                boundary[:, i] = boundary[:, i]
    boundaryE = (
        F.interpolate(
            torch.from_numpy(boundary[None, None, :, :]),
            size=cSize,
            mode="bilinear",
            align_corners=True,
        )
        .squeeze()
        .data.numpy()
    )
    return boundaryE * resampleRatio if _illu else boundaryE


def EM2DPlus(
    segMask,
    segMask_more,
    empty_list,
    f0,
    f1,
    stripeMask,
    Lambda,
    window_size,
    poly_order,
    kernel2d,
    allow_break,
    maxEpoch,
    device,
    _xy,
    _fastMode,
):

    def preComputePrior(
        segMask, classMap, th0, th1, index, mask, tmpToTops, tmpToBottoms, zeroEpoch
    ):
        m, s, n = segMask.size()
        if zeroEpoch == 1:
            classMap[th0 > th1] = 2
        else:
            classMap[:] = 1
            classMap[mask > index] = 2
        classMap[:] = classMap * segMask
        classRatio = torch.stack(
            (torch.sum(classMap == 1, 0), torch.sum(classMap == 2, 0))
        )  # (2, S, N)
        mask12 = (classRatio[0] != 0) * (classRatio[1] != 0)  # (S, N)
        mask1 = (classRatio[0] != 0) * (classRatio[1] == 0)
        mask2 = (classRatio[0] == 0) * (classRatio[1] != 0)
        tmp = torch.zeros(mask12.sum(), m).to(device)
        tmp[segMask[:, mask12].T == 1] = torch.repeat_interleave(
            -0.5 / classRatio[:, mask12].T.reshape(-1),
            classRatio[:, mask12].T.reshape(-1),
        )
        tmpToTops[:, mask12] = tmp.T
        if mask1.sum() != 0:
            tmpToTops[:, mask1] = segMask[:, mask1] * (-1 / classRatio[0, mask1])
        if mask2.sum() != 0:
            tmpToTops[:, mask2] = segMask[:, mask2] * (-1 / classRatio[1, mask2])
        tmpToBottoms = copy.deepcopy(tmpToTops).view(m, -1)
        tmpToTops = tmpToTops.view(m, -1)
        tmp = torch.arange(s * n, device=device)
        tmpToBottoms[
            last_nonzero(
                tmpToBottoms.cpu().data.numpy(), None, axis=0, invalid_val=-1
            ).reshape(-1),
            tmp,
        ] = 1
        tmpToTops[
            first_nonzero(tmpToTops.cpu().data.numpy(), None, axis=0, invalid_val=-1),
            tmp,
        ] = 1
        tmpToTops, tmpToBottoms = tmpToTops.view(m, s, n), tmpToBottoms.view(m, s, n)
        tmpToTops[:] = torch.cumsum(tmpToTops, 0) * segMask
        tmpToBottoms[:] = (
            tmpToBottoms
            + torch.sum(tmpToBottoms, dim=0, keepdims=True)
            - torch.cumsum(tmpToBottoms, dim=0)
        ) * segMask
        del tmp, classRatio, mask12, mask1, mask2
        return tmpToTops, tmpToBottoms

    def maskfor2d(
        segMask, segMask_more, empty_list, stripeMask, window_size, m1, s1, n1
    ):
        validMask = (segMask.sum(0) != 0).astype(np.float32)
        tmp = np.repeat(np.arange(s1)[:, None], n1, 1)
        maskrow = (tmp >= first_nonzero(validMask, None, axis=0, invalid_val=-1)) * (
            tmp <= last_nonzero(validMask, None, axis=0, invalid_val=-1)
        )
        tmp = np.repeat(np.arange(n1)[None, :], s1, 0)
        maskcol = (
            tmp >= first_nonzero(validMask, None, axis=1, invalid_val=-1)[:, None]
        ) * (tmp <= last_nonzero(validMask, None, axis=1, invalid_val=-1)[:, None])
        validMask[:] = (maskrow * maskcol).astype(np.float32)

        validMask_more = copy.deepcopy(validMask)
        validMask_more[empty_list, :] = 1

        valid_edge1 = skimage.util.view_as_windows(
            np.pad(
                validMask_more,
                ((0, 0), (window_size[1] // 2, window_size[1] // 2)),
                "constant",
                constant_values=0,
            ),
            (1, window_size[1]),
        ).sum((-2, -1))
        valid_edge1 = (
            (valid_edge1 < window_size[1])
            * (valid_edge1 > 0)
            * validMask_more.astype(bool)
        )

        valid_edge2 = skimage.util.view_as_windows(
            np.pad(
                validMask,
                (
                    (window_size[0] // 2, window_size[0] // 2),
                    (window_size[1] // 2, window_size[1] // 2),
                ),
                "constant",
                constant_values=0,
            ),
            window_size,
        ).sum((-2, -1))
        valid_edge2 = (
            (valid_edge2 < window_size[0] * window_size[1])
            * (valid_edge2 > 0)
            * validMask.astype(bool)
        )

        validMask += scipy.ndimage.binary_dilation(
            stripeMask, structure=np.ones((1, 5)), iterations=5
        ).astype(np.float32)
        validMask_more += scipy.ndimage.binary_dilation(
            stripeMask, structure=np.ones((1, 5)), iterations=5
        ).astype(np.float32)
        validFor2D = skimage.util.view_as_windows(
            np.pad(
                validMask,
                (
                    (window_size[0] // 2, window_size[0] // 2),
                    (window_size[1] // 2, window_size[1] // 2),
                ),
                "constant",
                constant_values=0,
            ),
            window_size,
        ).sum((-2, -1))
        validFor2D = validFor2D == window_size[0] * window_size[1]
        validFor2D += valid_edge2

        validFor1D = skimage.util.view_as_windows(
            np.pad(
                validMask_more,
                ((0, 0), (window_size[1] // 2, window_size[1] // 2)),
                "constant",
                constant_values=0,
            ),
            (1, window_size[1]),
        ).sum((-2, -1))
        validFor1D = validFor1D == window_size[1]
        validFor1D += valid_edge1

        missingMask = (
            (segMask.sum(0) == 0)
            * (
                np.arange(n1)[None, :]
                >= first_nonzero(np.sum(segMask != 0, 0), None, axis=1, invalid_val=-1)[
                    :, None
                ]
            )
            * (
                np.arange(n1)[None, :]
                <= last_nonzero(np.sum(segMask != 0, 0), None, axis=1, invalid_val=-1)[
                    :, None
                ]
            )
        )
        return (
            validFor2D,
            validFor1D,
            missingMask,
            torch.arange(m1, device=device)[:, None, None],
        )

    def missingBoundary(boundaryTMP, missingMask, s1, n1):
        boundaryTMP[missingMask] = np.nan
        return (
            pd.Series(boundaryTMP.reshape(-1))
            .interpolate(method="polynomial", order=1)
            .values.reshape(s1, n1)
        )

    def sgolay2d(Z, validFor2D, window_size, kernel, device):
        w0, w1 = window_size
        if _xy:
            content = torch.conv2d(
                F.pad(Z[None, None], (w1 // 2, w1 // 2, w0 // 2, w0 // 2), "reflect"),
                kernel,
            )
            weight = torch.conv2d(
                F.pad(
                    torch.from_numpy(validFor2D.astype(np.float32)).to(device)[
                        None, None
                    ],
                    (w1 // 2, w1 // 2, w0 // 2, w0 // 2),
                    "reflect",
                ),
                kernel,
            )
            return (content / weight).squeeze().cpu().detach().numpy()
        else:
            return (
                torch.conv2d(
                    F.pad(Z, (w1 // 2, w1 // 2, w0 // 2, w0 // 2))[None, None, :, :],
                    kernel,
                )
                .squeeze()
                .cpu()
                .detach()
                .numpy()
            )

    def sgolay1d(boundaryTMP, validBoundMask, window_size, poly_order, m, s, n):
        for i in range(s):
            l = validBoundMask[i, :]
            df = pd.DataFrame({"A": l})
            df["block"] = (df["A"] == 0).astype(int).cumsum()
            df = df.reset_index()
            df = df[df.A != 0]
            ind = (
                df.groupby(["block"], group_keys=True)["index"].apply(np.array).tolist()
            )
            for c in ind:
                if len(c) <= poly_order:
                    boundaryTMP[i, c] = np.mean(boundaryTMP[i, c])
                else:
                    tmp = np.pad(boundaryTMP[i, c], window_size // 2, mode="reflect")
                    tmp = signal.savgol_filter(
                        tmp,
                        window_size,
                        poly_order,
                    )
                    boundaryTMP[i, c] = tmp[window_size // 2 : -(window_size // 2)]
        return boundaryTMP

    print("start optimizing...")
    m, s, n = segMask.shape
    tmpToTops, tmpToBottoms, classMap = (
        torch.zeros(m, s, n).to(device),
        torch.zeros(m, s, n).to(device),
        torch.ones(m, s, n).to(device),
    )
    boundary, boundaryLS, boundaryOld, boundaryTMP, cn = (
        torch.zeros((s, n)).to(device),
        torch.zeros((s, n)).to(device),
        torch.zeros((s, n)).to(device),
        np.zeros((s, n)),
        0,
    )
    validFor2D, validFor1D, missingMask, coorMask = maskfor2d(
        segMask, segMask_more, empty_list, stripeMask, window_size, m, s, n
    )
    for e in range(maxEpoch):
        tmpToTops[:], tmpToBottoms[:] = preComputePrior(
            segMask_more,
            classMap,
            f0,
            f1,
            boundary if e else None,
            coorMask if e else None,
            tmpToTops,
            tmpToBottoms,
            zeroEpoch=1 if e == 0 else 0,
        )
        tmpToBottoms[:], tmpToTops[:] = f0 * tmpToBottoms, f1 * tmpToTops
        if e == 0:
            boundary[:] = torch.argmax(
                tmpToBottoms
                + torch.sum(tmpToBottoms, dim=0, keepdims=True)
                - torch.cumsum(tmpToBottoms, dim=0)
                + torch.cumsum(tmpToTops, 0),
                0,
            )
        else:
            boundary[:] = torch.argmax(
                tmpToBottoms
                + torch.sum(tmpToBottoms, 0, True)
                - torch.cumsum(tmpToBottoms, 0)
                + torch.cumsum(tmpToTops, 0)
                - Lambda * (boundaryLS - coorMask) ** 2,
                0,
            )
        changes = 100 if e == 0 else torch.abs(boundaryOld - boundary).max().item()
        boundaryOld[:], boundaryTMP[:] = (
            copy.deepcopy(boundary),
            boundary.cpu().detach().numpy(),
        )
        boundaryTMP[:] = missingBoundary(boundaryTMP, missingMask, s, n)
        if _xy == False:
            boundaryRaw = copy.deepcopy(boundaryTMP)
        boundaryTMP[validFor2D] = sgolay2d(
            torch.from_numpy(boundaryTMP * validFor2D).to(torch.float).to(device),
            validFor2D,
            window_size,
            kernel2d,
            device,
        )[validFor2D]
        if _xy:
            boundaryTMP[:] = sgolay1d(
                boundaryTMP, validFor1D, window_size[1], poly_order[1], m, s, n
            )
        boundaryLS[:] = torch.from_numpy(boundaryTMP).to(device)
        cn = cn + 1 if changes <= 2 else 0
        if cn >= 10:
            break
        print(
            "\rNo.{:0>3d} iteration EM: maximum changes = {}".format(e, changes), end=""
        )
    del segMask, segMask_more, f0, f1, tmpToTops, tmpToBottoms, classMap, boundaryOld
    gc.collect()
    return boundaryTMP, boundaryRaw if _xy == False else None


def waterShed(segMask, vol0, th, maxv, minv, s, m, n, view, _log, _xy):
    thresh, x = np.zeros((m, n), dtype=np.uint8), np.zeros((m, n), dtype=np.float32)
    fg, bg = np.zeros((m, n), dtype=np.uint8), np.zeros((m, n), dtype=np.uint8)
    marker32, mm = np.zeros((m, n), dtype=np.int32), np.zeros((m, n), dtype=np.uint8)
    tmpMask = np.zeros((m, n), dtype=bool)
    if "top" in view:
        f_name = "top/left"
    elif "bottom" in view:
        f_name = "bottom/right"
    else:
        f_name = view
    for ind in tqdm.tqdm(range(s), desc="watershed {} view: ".format(f_name)):
        xo = (
            np.log(np.clip(vol0[ind], 1, None))
            if _log
            else vol0[ind].astype(np.float32)
        )
        thresh[:] = 255 * (xo > th).astype(np.uint8)
        if xo.max() > 0:
            x[:] = (
                255
                * (xo - math.log(np.clip(minv, 1, None)))
                / (math.log(maxv) - math.log(np.clip(minv, 1, None) + 1))
                if _log
                else 255 * (xo - minv) / (maxv - minv)
            )
        fg[:] = 255 * morphology.remove_small_objects(
            thresh.astype(bool), 25
        )  # cv2.erode(thresh, np.ones((2, 2), np.uint8), iterations = 1)
        _, bg[:] = cv2.threshold(
            cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=10), 1, 128, 1
        )
        marker32[:] = cv2.watershed(
            cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_GRAY2BGR),
            np.int32(cv2.add(fg, bg)),
        )
        mm[:] = cv2.convertScaleAbs(marker32)
        tmpMask[:] = (mm != 128) * (mm != 1)
        segMask[ind] = morphology.remove_small_objects(tmpMask, 9)


def refineShape(segMaskTop, segMaskBottom, s, m, n, _xy):

    def outlierFilling(points, outlier, n):
        points = points.astype(np.float32)
        index = np.where(points != outlier)[0]
        tmp = np.linspace(0, n - 1, n, dtype=np.int32)
        if len(index) == 0:
            pass
        else:
            points[(points == outlier) * (tmp >= index[0]) * (tmp <= index[-1])] = (
                np.nan
            )
        return points

    def fillHole(_maskl, _maskm):
        z, h, w = _maskl.shape
        result = np.zeros(_maskl.shape, dtype=bool)
        for i in tqdm.tqdm(range(z), desc="filling the hole: "):
            ts = 0.1 * sum(sum(_maskl[i]))
            aaaa, num = scipy.ndimage.label(
                morphology.remove_small_objects(
                    np.abs(_maskl[i].astype(np.uint8) - _maskm[i].astype(np.uint8)),
                    1000,
                )
            )
            _mask, _set = np.zeros(aaaa.shape), set(np.arange(1, num))
            for ii in range(1, num):
                if sum(sum(aaaa == ii)) > ts:
                    _mask[aaaa == ii], _set = 1, _set - set([ii])
            _mask = (_mask + _maskl[i].astype(np.uint8)).astype(np.uint8)
            _set_ = copy.deepcopy(_set)
            for s in _set:
                fnz, lnz = first_nonzero(None, _mask, 0, 0), last_nonzero(
                    None, _mask, 0, 0
                )
                tmp = _mask + (aaaa == s)
                fnzt, lnzt = first_nonzero(None, tmp, 0, 0), last_nonzero(
                    None, tmp, 0, 0
                )
                if (fnz == fnzt).all() and (lnz == lnzt).all():
                    _mask[:], _set_ = tmp, _set_ - set([s])
                    continue
                if ((np.abs(fnzt - fnz) / (fnz + 1)) > 0.2).any() or (
                    (np.abs(lnzt - lnz) / (lnz + 1)) > 0.2
                ).any():
                    _mask[:], _set_ = tmp, _set_ - set([s])
                    continue
            im_floodfill = 255 * (_mask.astype(np.uint8)).copy()
            mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(im_floodfill, mask, seedPoint=(0, 0), newVal=255)
            result[i, :, :] = (_mask + (~im_floodfill)).astype(bool)
        return result.transpose(2, 0, 1)

    tmp = np.linspace(0, m - 1, m, dtype=np.int32)[:, None]
    _mask, _maskl, _maskm = (
        np.zeros((s, m, n), dtype=np.uint8),
        np.zeros((s, m, n), dtype=bool),
        np.zeros((s, m, n), dtype=bool),
    )
    for i in tqdm.tqdm(range(s), desc="refine pair-wise segmentation result: "):
        boundaryCoordsBottom = last_nonzero(
            None, segMaskBottom[i, :, :], axis=0, invalid_val=0
        )
        boundaryCoordsTop = first_nonzero(
            None, segMaskTop[i, :, :], axis=0, invalid_val=m - 1
        )
        mask = boundaryCoordsBottom < boundaryCoordsTop
        boundaryCoordsBottom[mask] = 0
        boundaryCoordsTop[mask] = m - 1
        pointsoBottom = (
            pd.Series(outlierFilling(boundaryCoordsBottom, outlier=0, n=n))
            .interpolate(method="polynomial", order=1)
            .values
        )
        pointsoTop = (
            pd.Series(outlierFilling(boundaryCoordsTop, outlier=m - 1, n=n))
            .interpolate(method="polynomial", order=1)
            .values
        )
        segMask = (tmp >= pointsoTop) * ((tmp <= pointsoBottom))
        tmpMaskMore = segMask * (
            (
                (segMaskBottom[i, :, :] == 0).astype(np.float32)
                + (segMaskTop[i, :, :] == 0).astype(np.float32)
            )
            <= 1
        )
        tmpMaskLess = segMask * (
            (
                (segMaskBottom[i, :, :] == 0).astype(np.float32)
                + (segMaskTop[i, :, :] == 0).astype(np.float32)
            )
            < 1
        )
        if _xy:
            _mask[i, :, :] = segMask
        else:
            _maskl[i, :, :], _maskm[i, :, :] = tmpMaskLess, tmpMaskMore
        del mask, segMask, tmpMaskMore, tmpMaskLess
    if _xy == False:
        return fillHole(_maskl.transpose(1, 2, 0), _maskm.transpose(1, 2, 0))
    else:
        with tqdm.tqdm(
            total=_mask.shape[1] * _mask.shape[2], desc="refine along z: "
        ) as pbar:
            for i in range(_mask.shape[1]):
                for j in range(_mask.shape[2]):
                    _mask[:, i, j] = morphology.remove_small_holes(
                        _mask[:, i, j] > 0, 5
                    )
                    pbar.update(1)
        return _mask


def sgolay2dkernel(window_size, order):
    n_terms = (order + 1) * (order + 2) / 2.0
    half_size = window_size // 2
    exps = []
    for row in range(order[0] + 1):
        for column in range(order[1] + 1):
            if (row + column) > max(*order):
                continue
            exps.append((row, column))
    indx = np.arange(-half_size[0], half_size[0] + 1, dtype=np.float64)
    indy = np.arange(-half_size[1], half_size[1] + 1, dtype=np.float64)
    dx = np.repeat(indx, window_size[1])
    dy = np.tile(indy, [window_size[0], 1]).reshape(
        window_size[0] * window_size[1],
    )
    A = np.empty((window_size[0] * window_size[1], len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])
    return np.linalg.pinv(A)[0].reshape((window_size[0], -1))


def first_nonzero(arr, mask, axis, invalid_val=np.nan):
    if mask is None:
        mask = arr != 0
    if type(mask) is not np.ndarray:
        mask = mask.cpu().detach().numpy()
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, mask, axis, invalid_val=np.nan):
    if mask is None:
        mask = arr != 0
    if type(mask) is not np.ndarray:
        mask = mask.cpu().detach().numpy()
    val = mask.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)
