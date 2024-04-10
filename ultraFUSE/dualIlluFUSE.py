import pandas as pd

pd.set_option("display.width", 10000)
import dask.array as da
from typing import Union, Tuple, Optional, List, Dict
import numpy as np
import torch
import os
from aicsimageio import AICSImage

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
import matplotlib.patches as patches
import tqdm
import gc
import tifffile
from ultraFUSE.NSCT import NSCTdec, NSCTrec
from ultraFUSE.utils import (
    sgolay2dkernel,
    waterShed,
    refineShape,
    extendBoundary,
    EM2DPlus,
    fusion_perslice,
)


class dualIlluFUSE:
    def __init__(
        self,
        require_precropping: bool = True,
        precropping_params: list[int, int, int, int] = [],
        require_flipping: bool = False,
        resampleRatio: int = 2,
        Lambda: float = 0.1,
        window_size: list[int, int] = [5, 59],
        poly_order: list[int, int] = [3, 3],
        n_epochs: int = 150,
        Gaussian_kernel_size: int = 49,
        GF_kernel_size: int = 29,
        require_segmentation: bool = True,
        allow_break: bool = False,
        fast_mode: bool = False,
        require_log: bool = True,
        device: str = "cuda",
    ):
        self.train_params = {
            "require_precropping": require_precropping,
            "precropping_params": precropping_params,
            "require_flipping": require_flipping,
            "resampleRatio": resampleRatio,
            "Lambda": Lambda,
            "window_size": window_size,
            "poly_order": poly_order,
            "n_epochs": n_epochs,
            "Gaussian_kernel_size": Gaussian_kernel_size,
            "GF_kernel_size": GF_kernel_size,
            "require_segmentation": require_segmentation,
            "allow_break": allow_break,
            "fast_mode": fast_mode,
            "require_log": require_log,
            "device": device,
        }
        self.train_params["kernel2d"] = torch.from_numpy(
            sgolay2dkernel(
                np.array(self.train_params["window_size"]),
                np.array(self.train_params["poly_order"]),
            )
        ).to(self.train_params["device"])

    def train(
        self,
        data_path: str = "",
        sample_name: str = "",
        top_illu_data: Union[np.ndarray, da.core.Array, str] = None,
        bottom_illu_data: Union[np.ndarray, da.core.Array, str] = None,
        left_illu_data: Union[np.ndarray, da.core.Array, str] = None,
        right_illu_data: Union[np.ndarray, da.core.Array, str] = None,
        save_path: str = "",
        save_folder: str = "",
        camera_position: str = "",
    ):
        data_path = os.path.join(data_path, sample_name)
        if not os.path.exists(save_path):
            print("saving path does not exist.")
            return
        save_path = os.path.join(save_path, save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.sample_params = {}
        print("Read in...")
        if (left_illu_data is not None) and (right_illu_data is not None):
            T_flag = 1
            if isinstance(left_illu_data, str):
                left_illu_path = os.path.join(data_path, left_illu_data)
                left_illu_handle = AICSImage(left_illu_path)
                self.sample_params["topillu_saving_name"] = os.path.splitext(
                    left_illu_data
                )[0]
            else:
                left_illu_handle = AICSImage(left_illu_data)
                self.sample_params["topillu_saving_name"] = "left_illu{}".format(
                    "+" + camera_position if len(camera_position) != 0 else ""
                )
            rawPlanes_top = np.transpose(
                left_illu_handle.get_image_data("ZYX", T=0, C=0), (0, 2, 1)
            )
            if isinstance(right_illu_data, str):
                right_illu_path = os.path.join(data_path, right_illu_data)
                right_illu_handle = AICSImage(right_illu_path)
                self.sample_params["bottomillu_saving_name"] = os.path.splitext(
                    right_illu_data
                )[0]
            else:
                right_illu_handle = AICSImage(right_illu_data)
                self.sample_params["bottomillu_saving_name"] = "right_illu{}".format(
                    "+" + camera_position if len(camera_position) != 0 else ""
                )
            rawPlanes_bottom = np.transpose(
                right_illu_handle.get_image_data("ZYX", T=0, C=0), (0, 2, 1)
            )
        elif (top_illu_data is not None) and (bottom_illu_data is not None):
            T_flag = 0
            if isinstance(top_illu_data, str):
                top_illu_path = os.path.join(data_path, top_illu_data)
                top_illu_handle = AICSImage(top_illu_path)
                self.sample_params["topillu_saving_name"] = os.path.splitext(
                    top_illu_data
                )[0]
            else:
                top_illu_handle = AICSImage(top_illu_data)
                self.sample_params["topillu_saving_name"] = "top_illu{}".format(
                    "+" + camera_position if len(camera_position) != 0 else ""
                )
            rawPlanes_top = top_illu_handle.get_image_data("ZYX", T=0, C=0)
            if isinstance(bottom_illu_data, str):
                bottom_illu_path = os.path.join(data_path, bottom_illu_data)
                bottom_illu_handle = AICSImage(bottom_illu_path)
                self.sample_params["bottomillu_saving_name"] = os.path.splitext(
                    bottom_illu_data
                )[0]
            else:
                bottom_illu_handle = AICSImage(bottom_illu_data)
                self.sample_params["bottomillu_saving_name"] = "bottom_illu{}".format(
                    "+" + camera_position if len(camera_position) != 0 else ""
                )
            rawPlanes_bottom = bottom_illu_handle.get_image_data("ZYX", T=0, C=0)
        else:
            print("input(s) missing, please check.")
            return
        for k in self.sample_params.keys():
            if "saving_name" in k:
                sub_folder = os.path.join(save_path, self.sample_params[k])
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)
        if self.train_params["require_flipping"]:
            rawPlanes_top[:] = np.flip(rawPlanes_top, 1)
            rawPlanes_bottom[:] = np.flip(rawPlanes_bottom, 1)

        print("\nLocalize sample...")
        cropInfo, MIP_info = self.localizingSample(rawPlanes_top, rawPlanes_bottom)
        print(cropInfo)
        if self.train_params["require_precropping"]:
            if len(self.train_params["precropping_params"]) == 0:
                xs, xe, ys, ye = cropInfo.loc[
                    "in summary", ["startX", "endX", "startY", "endY"]
                ].astype(int)
            else:
                if T_flag:
                    ys, ye, xs, xe = self.train_params["precropping_params"]
                else:
                    xs, xe, ys, ye = self.train_params["precropping_params"]
        else:
            xs, xe, ys, ye = None, None, None, None
        s_o, m_o, n_o = rawPlanes_top.shape
        rawPlanes_top_crop = rawPlanes_top[:, xs:xe, ys:ye]
        rawPlanes_bottom_crop = rawPlanes_bottom[:, xs:xe, ys:ye]
        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=200)
        MIP_top = rawPlanes_top.max(0)
        if self.train_params["require_flipping"]:
            MIP_top = np.flip(MIP_top, 0)
        if T_flag:
            MIP_top = MIP_top.T
        MIP_bottom = rawPlanes_bottom.max(0)
        if self.train_params["require_precropping"]:
            top_left_point = [ys, xs]
            if self.train_params["require_flipping"]:
                top_left_point[1] = m_o - top_left_point[1] - (xe - xs)
            if T_flag:
                top_left_point = [top_left_point[1], top_left_point[0]]
        if self.train_params["require_flipping"]:
            MIP_bottom = np.flip(MIP_bottom, 0)
        if T_flag:
            MIP_bottom = MIP_bottom.T
        ax1.imshow(MIP_top)
        if self.train_params["require_precropping"]:
            rect = patches.Rectangle(
                tuple(top_left_point),
                (ye - ys) if (not T_flag) else (xe - xs),
                (xe - xs) if (not T_flag) else (ye - ys),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax1.add_patch(rect)
        ax1.set_title("{} illu".format("left" if T_flag else "top"), fontsize=8, pad=1)
        ax1.axis("off")
        ax2.imshow(MIP_bottom)
        if self.train_params["require_precropping"]:
            rect = patches.Rectangle(
                tuple(top_left_point),
                (ye - ys) if (not T_flag) else (xe - xs),
                (xe - xs) if (not T_flag) else (ye - ys),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax2.add_patch(rect)
        ax2.set_title(
            "{} illu".format("right" if T_flag else "bottom"), fontsize=8, pad=1
        )
        ax2.axis("off")
        plt.show()
        rawPlanesTop = da.from_array(rawPlanes_top)
        rawPlanesBottom = da.from_array(rawPlanes_bottom)
        del rawPlanes_top, rawPlanes_bottom

        print("\nCalculate volumetric measurements...")
        self.measureSample(
            rawPlanes_top_crop,
            "top",
            os.path.join(save_path, self.sample_params["topillu_saving_name"]),
            MIP_info,
        )
        self.measureSample(
            rawPlanes_bottom_crop,
            "bottom",
            os.path.join(save_path, self.sample_params["bottomillu_saving_name"]),
            MIP_info,
        )
        s, m_c, n_c = rawPlanes_top_crop.shape
        m = len(np.arange(m_c)[:: self.train_params["resampleRatio"]])
        n = len(np.arange(n_c)[:: self.train_params["resampleRatio"]])

        print("\nSegment sample...")
        if self.train_params["require_segmentation"]:
            segMask = self.segmentSample(
                topView=os.path.join(
                    save_path, self.sample_params["topillu_saving_name"]
                ),
                bottomView=os.path.join(
                    save_path, self.sample_params["bottomillu_saving_name"]
                ),
                topVol=rawPlanes_top_crop[
                    :,
                    :: self.train_params["resampleRatio"],
                    :: self.train_params["resampleRatio"],
                ],
                bottomVol=rawPlanes_bottom_crop[
                    :,
                    :: self.train_params["resampleRatio"],
                    :: self.train_params["resampleRatio"],
                ],
            )
        else:
            segMask = np.ones((s, m, n), dtype=np.uint8)

        print("\nExtract features...")
        topF, bottomF, stripeMask = self.extractNSCTF(
            s,
            m,
            n,
            topVol=rawPlanes_top_crop,
            bottomVol=rawPlanes_bottom_crop,
            segMask=segMask,
            Max=cropInfo.loc["in summary", "maxv"],
        )

        print("\nDual-illumination fusion...")
        boundary = self.dualViewFusion(topF, bottomF, segMask, stripeMask)
        boundary = extendBoundary(
            boundary,
            self.train_params["resampleRatio"],
            window_size=self.train_params["window_size"][1],
            poly_order=self.train_params["poly_order"][1],
            cSize=(s, n_c),
        )
        boundaryE = np.zeros((s, n_o))
        boundaryE[:, ys:ye] = boundary
        if ys is not None:
            boundaryE[:, :ys], boundaryE[:, ye:] = np.nan, np.nan
            for i in range(boundaryE.shape[0]):
                boundaryE[i, :] = (
                    pd.Series(boundaryE[i, :])
                    .interpolate(method="spline", order=1, limit_direction="both")
                    .values
                )
        if xs is not None:
            boundaryE += xs
        boundaryE = np.clip(boundaryE, 0, m_o).astype(np.uint16)
        if self.train_params["fast_mode"]:
            tifffile.imwrite(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_saving_name"],
                    "fusionBoundary_xy_simple_{}.tif".format(
                        "allowBreak" if self.train_params["allow_break"] else "noBreak"
                    ),
                ),
                boundaryE,
            )
        else:
            tifffile.imwrite(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_saving_name"],
                    "fusionBoundary_xy_full_{}.tif".format(
                        "allowBreak" if self.train_params["allow_break"] else "noBreak"
                    ),
                ),
                boundaryE,
            )
        del rawPlanes_top_crop, rawPlanes_bottom_crop

        print("\nStitching...")
        boundaryE = tifffile.imread(
            os.path.join(
                save_path,
                self.sample_params["topillu_saving_name"],
                "fusionBoundary_xy_full_{}.tif".format(
                    "allowBreak" if self.train_params["allow_break"] else "noBreak"
                ),
            )
        )
        reconVol = fusionResult(
            rawPlanesTop,
            rawPlanesBottom,
            boundaryE,
            self.train_params["device"],
            Gaussianr=self.train_params["Gaussian_kernel_size"],
            GFr=self.train_params["GF_kernel_size"],
        )
        if self.train_params["require_flipping"]:
            reconVol[:] = np.flip(reconVol, 1)
        if T_flag:
            result = reconVol.transpose(0, 2, 1)
        else:
            result = reconVol
        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=200)
        xyMIP = result.max(0)
        ax1.imshow(xyMIP)
        ax1.set_title("result", fontsize=8, pad=1)
        ax1.axis("off")
        ax2.imshow(np.zeros_like(xyMIP))
        ax2.axis("off")
        plt.show()

        print("Save...")
        tifffile.imwrite(
            os.path.join(save_path, self.sample_params["topillu_saving_name"])
            + "/illuFusionResult_{}_{}.tif".format(
                "simple" if self.train_params["fast_mode"] else "full",
                "allowBreak" if self.train_params["allow_break"] else "noBreak",
            ),
            result,
        )

        del reconVol, result

    def dualViewFusion(self, topF, bottomF, segMaskvUINT8, stripeMask):
        print("to GPU...")
        segMaskGPU = torch.from_numpy(segMaskvUINT8.transpose(1, 0, 2)).to(
            self.train_params["device"]
        )
        topFGPU, bottomFGPU = torch.from_numpy(topF**2).to(
            self.train_params["device"]
        ), torch.from_numpy(bottomF**2).to(self.train_params["device"])
        boundary, _ = EM2DPlus(
            segMaskGPU,
            bottomFGPU,
            topFGPU,
            stripeMask,
            self.train_params["Lambda"],
            self.train_params["window_size"],
            self.train_params["poly_order"],
            self.train_params["kernel2d"][None, None, :, :],
            self.train_params["allow_break"],
            (
                self.train_params["n_epochs"]
                if self.train_params["fast_mode"] == False
                else 1
            ),
            device=self.train_params["device"],
            _xy=True,
            _fastMode=self.train_params["fast_mode"],
        )
        del segMaskGPU, topFGPU, bottomFGPU
        return boundary

    def extractNSCTF(self, s, m, n, topVol, bottomVol, segMask, Max):
        r = self.train_params["resampleRatio"]
        featureExtrac = NSCTdec(levels=[3, 3, 3], device=self.train_params["device"])
        topF, bottomF = np.empty((s, m, n), dtype=np.float32), np.empty(
            (s, m, n), dtype=np.float32
        )
        topFBase, bottomFBase = np.empty((s, m, n), dtype=np.float32), np.empty(
            (s, m, n), dtype=np.float32
        )
        topSTD, bottomSTD = np.empty((m, s, n), dtype=np.float32), np.empty(
            (m, s, n), dtype=np.float32
        )
        tmp0, tmp1 = np.arange(0, s, 10), np.arange(10, s + 10, 10)
        for p, q in tqdm.tqdm(zip(tmp0, tmp1), desc="NSCT: ", total=len(tmp0)):
            topDataFloat, bottomDataFloat = topVol[p:q, :, :].astype(
                np.float32
            ), bottomVol[p:q, :, :].astype(np.float32)
            topDataGPU, bottomDataGPU = torch.from_numpy(
                topDataFloat[:, None, :, :]
            ).to(self.train_params["device"]), torch.from_numpy(
                bottomDataFloat[:, None, :, :]
            ).to(
                self.train_params["device"]
            )
            topDataGPU[:], bottomDataGPU[:] = topDataGPU / Max, bottomDataGPU / Max
            a, b, c = featureExtrac.nsctDec(topDataGPU, r, _forFeatures=True)
            topF[p:q], topFBase[p:q], topSTD[:, p:q, :] = (
                a.cpu().detach().numpy(),
                b.cpu().detach().numpy(),
                c.cpu().detach().numpy().transpose(1, 0, 2),
            )
            a[:], b[:], c[:] = featureExtrac.nsctDec(
                bottomDataGPU, r, _forFeatures=True
            )
            bottomF[p:q], bottomFBase[p:q], bottomSTD[:, p:q, :] = (
                a.cpu().detach().numpy(),
                b.cpu().detach().numpy(),
                c.cpu().detach().numpy().transpose(1, 0, 2),
            )
            del topDataFloat, bottomDataFloat, topDataGPU, bottomDataGPU, a, b, c
        if self.train_params["allow_break"]:
            print("detect stripes based on base signals from NSCT...")
            fBase = np.maximum(topFBase, bottomFBase)
            stripeMask = (np.abs(topFBase - bottomFBase) / (fBase + 1) * segMask).sum(
                1
            ) / (segMask.sum(1) + 1)
            stripeMask = morphology.remove_small_holes(
                morphology.remove_small_objects(
                    stripeMask > np.quantile(np.unique(stripeMask), 0.9), 200
                ),
                200,
            )
            del fBase
        del segMask
        del topF, bottomF, topFBase, bottomFBase
        gc.collect()
        return (
            topSTD,
            bottomSTD,
            (
                stripeMask
                if self.train_params["allow_break"]
                else np.zeros((s, n), dtype=bool)
            ),
        )

    def segmentSample(self, topView, bottomView, topVol, bottomVol):
        s, m, n = topVol.shape
        topf = np.load(topView + "/info.npy", allow_pickle=True).item()
        bottomf = np.load(bottomView + "/info.npy", allow_pickle=True).item()
        topSegMask, bottomSegMask = np.zeros((s, m, n), dtype=bool), np.zeros(
            (s, m, n), dtype=bool
        )
        waterShed(
            topSegMask,
            topVol,
            topf["thvol_log"] if self.train_params["require_log"] else topf["thvol"],
            topf["maxvol"],
            topf["minvol"],
            s,
            m,
            n,
            "top",
            _log=self.train_params["require_log"],
            _xy=True,
        )
        waterShed(
            bottomSegMask,
            bottomVol,
            (
                bottomf["thvol_log"]
                if self.train_params["require_log"]
                else bottomf["thvol"]
            ),
            bottomf["maxvol"],
            bottomf["minvol"],
            s,
            m,
            n,
            "bottom",
            _log=self.train_params["require_log"],
            _xy=True,
        )
        segMask = refineShape(topSegMask, bottomSegMask, s, m, n, _xy=True)
        """
        for i in range(0, topVol.shape[0], 3):
            plt.subplot(1, 3, 1)
            plt.imshow(topVol[i])
            plt.subplot(1, 3, 2)
            plt.imshow(bottomVol[i])
            plt.subplot(1, 3, 3)
            plt.imshow(segMask[i])
            plt.show()
        """
        del topSegMask, bottomSegMask, topVol, bottomVol
        return segMask

    def measureSample(self, rawPlanes, f, save_path, MIP_info):
        if "top" in f:
            f_name = "top/left"
        if "bottom" in f:
            f_name = "bottom/right"
        print("\r{} view: ".format(f_name), end="")
        maxvvol = rawPlanes.max()
        minvvol = rawPlanes.min()
        thvol = filters.threshold_otsu(rawPlanes[:, ::4, ::4].astype(np.float32))
        thvol_log = filters.threshold_otsu(
            np.log(np.clip(rawPlanes[:, ::4, ::4], 1, None).astype(np.float32))
        )
        print(
            "minimum intensity = {:.1f}, maximum intensity = {:.1f}, OTSU threshold = {:.1f}, OTSU threshold in log = {:.1f}".format(
                minvvol, maxvvol, thvol, thvol_log
            )
        )
        np.save(
            save_path + "/info.npy",
            {
                **{
                    "thvol": thvol,
                    "thvol_log": thvol_log,
                    "maxvol": maxvvol,
                    "minvol": minvvol,
                },
                **MIP_info,
            },
        )
        return rawPlanes

    def localizingSample(self, rawPlanes_top, rawPlanes_bottom):
        cropInfo = pd.DataFrame(
            columns=["startX", "endX", "startY", "endY", "maxv"],
            index=["top", "bottom"],
        )
        for f in ["top", "bottom"]:
            if "top" in f:
                f_name = "top/left"
            if "bottom" in f:
                f_name = "bottom/right"
            print("\rlocalize {} view: ".format(f_name), end="")
            maximumProjection = locals()["rawPlanes_" + f].max(0).astype(np.float32)
            maximumProjection = np.log(np.clip(maximumProjection, 1, None))
            m, n = maximumProjection.shape
            maxv, minv, th = (
                maximumProjection.max(),
                maximumProjection.min(),
                filters.threshold_otsu(maximumProjection),
            )
            print(
                "minimum intensity = {:.1f}, maximum intensity = {:.1f}, OTSU threshold = {:.1f}".format(
                    minv, maxv, th
                )
            )
            thresh = maximumProjection > th
            segMask = morphology.remove_small_objects(thresh, min_size=25)
            d1, d2 = (
                np.where(np.sum(segMask, axis=0) != 0)[0],
                np.where(np.sum(segMask, axis=1) != 0)[0],
            )
            a, b, c, d = (
                max(0, d1[0] - 100),
                min(n, d1[-1] + 100),
                max(0, d2[0] - 100),
                min(m, d2[-1] + 100),
            )
            cropInfo.loc[f, :] = [c, d, a, b, np.exp(maxv)]
        cropInfo.loc["in summary"] = (
            min(cropInfo["startX"]),
            max(cropInfo["endX"]),
            min(cropInfo["startY"]),
            max(cropInfo["endY"]),
            max(cropInfo["maxv"]),
        )
        return cropInfo, {"MIP_min": minv, "MIP_max": maxv, "MIP_th": th}


def fusionResult(topVol, bottomVol, boundary, device, Gaussianr=49, GFr=49):
    boundary = torch.from_numpy(boundary[None, None, :, :]).to(device)
    kernel = torch.ones(1, 1, Gaussianr, Gaussianr).to(device) / Gaussianr / Gaussianr
    reconVol = np.empty(topVol.shape, dtype=np.uint16)
    for s in tqdm.tqdm(range(topVol.shape[0]), desc="fusion: "):  # topVol.shape[0]
        reconVol[s, :, :] = fusion_perslice(
            topVol[s, :, :].compute().astype(np.float32)[None, None, :, :],
            bottomVol[s, :, :].compute().astype(np.float32)[None, None, :, :],
            GFr,
            Gaussianr,
            kernel,
            boundary[:, :, s : s + 1, :],
            device,
        )
    return reconVol
