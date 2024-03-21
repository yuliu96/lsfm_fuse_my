from ultraFUSE.dualIlluFUSE import dualIlluFUSE
from ultraFUSE.utils import (
    GuidedFilter,
    sgolay2dkernel,
    waterShed,
    refineShape,
    extendBoundary,
    EM2DPlus,
)
from ultraFUSE.NSCT import NSCTdec, NSCTrec

from typing import Union, Tuple, Optional, List, Dict
import dask
import numpy as np
import torch
import os
from aicsimageio import AICSImage
import scipy.ndimage as ndimage
import ants
import scipy.io as scipyio

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from skimage import morphology
import shutil
import io
import copy
import SimpleITK as sitk
import tqdm
import gc
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.width", 10000)
import matplotlib.patches as patches
import torch.nn as nn
import torch.nn.functional as F
import tifffile


class pseudoDualCameraFUSE:
    def __init__(
        self,
        require_precropping: bool = True,
        precropping_params: list[int, int, int, int] = [],
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
        skip_illuFusion: bool = True,
        destripe_preceded: bool = False,
        destripe_params: Dict = None,
        device: str = "cuda",
    ):
        self.train_params = {
            "require_precropping": require_precropping,
            "precropping_params": precropping_params,
            "require_flipping": False,
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
        self.modelFront = dualIlluFUSE(**self.train_params)
        self.modelBack = dualIlluFUSE(**self.train_params)
        self.train_params.update(
            {
                "skip_illuFusion": skip_illuFusion,
                "destripe_preceded": destripe_preceded,
                "destripe_params": destripe_params,
            }
        )
        self.train_params["kernel2d"] = torch.from_numpy(
            sgolay2dkernel(
                np.array([window_size[1], window_size[1]]),
                np.array([poly_order[1], poly_order[1]]),
            )
        ).to(device)

    def train(
        self,
        xy_spacing: float,
        z_spacing: float,
        data_path: str = "",
        sample_name: str = "",
        ventral_angle_in_degree: float = None,
        dorsal_angle_in_degree: float = None,
        top_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        bottom_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        top_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        bottom_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        left_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        right_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        left_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        right_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        save_path: str = "",
        save_folder: str = "",
    ):
        illu_name = "illuFusionResult_{}_{}".format(
            "simple" if self.train_params["fast_mode"] else "full",
            "allowBreak" if self.train_params["allow_break"] else "noBreak",
        )
        if not os.path.exists(save_path):
            print("saving path does not exist.")
            return
        save_path = os.path.join(save_path, save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.sample_params = {}
        if (
            (top_illu_ventral_det_data is not None)
            and (bottom_illu_ventral_det_data is not None)
            and (top_illu_dorsal_det_data is not None)
            and (bottom_illu_dorsal_det_data is not None)
        ):
            T_flag = 0
            if isinstance(top_illu_ventral_det_data, str):
                self.sample_params["topillu_ventraldet_data_saving_name"] = (
                    os.path.splitext(top_illu_ventral_det_data)[0]
                )
            else:
                self.sample_params["topillu_ventraldet_data_saving_name"] = (
                    "top_illu+ventral_det"
                )
            if isinstance(bottom_illu_ventral_det_data, str):
                self.sample_params["bottomillu_ventraldet_data_saving_name"] = (
                    os.path.splitext(bottom_illu_ventral_det_data)[0]
                )
            else:
                self.sample_params["bottomillu_ventraldet_data_saving_name"] = (
                    "bottom_illu+ventral_det"
                )
            if isinstance(top_illu_dorsal_det_data, str):
                self.sample_params["topillu_dorsaldet_data_saving_name"] = (
                    os.path.splitext(top_illu_dorsal_det_data)[0]
                )
            else:
                self.sample_params["topillu_dorsaldet_data_saving_name"] = (
                    "top_illu+dorsal_det"
                )
            if isinstance(bottom_illu_dorsal_det_data, str):
                self.sample_params["bottomillu_dorsaldet_data_saving_name"] = (
                    os.path.splitext(bottom_illu_dorsal_det_data)[0]
                )
            else:
                self.sample_params["bottomillu_dorsaldet_data_saving_name"] = (
                    "bottom_illu+dorsal_det"
                )
        elif (
            (left_illu_ventral_det_data is not None)
            and (right_illu_ventral_det_data is not None)
            and (left_illu_dorsal_det_data is not None)
            and (right_illu_dorsal_det_data is not None)
        ):
            T_flag = 1
            if isinstance(left_illu_ventral_det_data, str):
                self.sample_params["topillu_ventraldet_data_saving_name"] = (
                    os.path.splitext(left_illu_ventral_det_data)[0]
                )
            else:
                self.sample_params["topillu_ventraldet_data_saving_name"] = (
                    "left_illu+ventral_det"
                )
            if isinstance(right_illu_ventral_det_data, str):
                self.sample_params["bottomillu_ventraldet_data_saving_name"] = (
                    os.path.splitext(right_illu_ventral_det_data)[0]
                )
            else:
                self.sample_params["bottomillu_ventraldet_data_saving_name"] = (
                    "right_illu+ventral_det"
                )
            if isinstance(left_illu_dorsal_det_data, str):
                self.sample_params["topillu_dorsaldet_data_saving_name"] = (
                    os.path.splitext(left_illu_dorsal_det_data)[0]
                )
            else:
                self.sample_params["topillu_dorsaldet_data_saving_name"] = (
                    "left_illu+dorsal_det"
                )
            if isinstance(right_illu_dorsal_det_data, str):
                self.sample_params["bottomillu_dorsaldet_data_saving_name"] = (
                    os.path.splitext(right_illu_dorsal_det_data)[0]
                )
            else:
                self.sample_params["bottomillu_dorsaldet_data_saving_name"] = (
                    "right_illu+dorsal_det"
                )
        else:
            print("input(s) missing, please check.")
            return

        if (ventral_angle_in_degree is None) and (dorsal_angle_in_degree is None):
            ventral_info = os.path.join(
                data_path,
                sample_name,
                "infoTXT",
                self.sample_params["topillu_ventraldet_data_saving_name"]
                + "_Settings.txt",
            )
            dorsal_info = os.path.join(
                data_path,
                sample_name,
                "infoTXT",
                self.sample_params["topillu_dorsaldet_data_saving_name"]
                + "_Settings.txt",
            )
            if os.path.exists(ventral_info) and os.path.exists(dorsal_info):
                with io.open(ventral_info, "r", encoding="utf-8") as my_file:
                    txt_content = my_file.readlines()
                    for l in txt_content:
                        if "Angle (degrees) =" in l:
                            ventral_angle_in_degree = float(
                                l.split("Angle (degrees) = ")[1]
                                .strip()
                                .replace("\n", "")
                                .replace("\r", "")
                            )
                with io.open(dorsal_info, "r", encoding="utf-8") as my_file:
                    txt_content = my_file.readlines()
                    for l in txt_content:
                        if "Angle (degrees) =" in l:
                            dorsal_angle_in_degree = float(
                                l.split("Angle (degrees) = ")[1]
                                .strip()
                                .replace("\n", "")
                                .replace("\r", "")
                            )
            else:
                print("angle information is missing.")
                return

        for k in self.sample_params.keys():
            if "saving_name" in k:
                sub_folder = os.path.join(save_path, self.sample_params[k])
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)
        if self.train_params["skip_illuFusion"]:
            if os.path.exists(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_ventraldet_data_saving_name"],
                    illu_name + ".tif",
                )
            ):
                print("Skip dual-illu fusion for ventral det...")
                illu_flag_ventral = 0
            else:
                print("Cannot skip dual-illu fusion for ventral det...")
                illu_flag_ventral = 1
            if os.path.exists(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_dorsaldet_data_saving_name"],
                    illu_name + ".tif",
                )
            ):
                print("Skip dual-illu fusion for dorsal det...")
                illu_flag_dorsal = 0
            else:
                print("Cannot skip dual-illu fusion for dorsal det...")
                illu_flag_dorsal = 1
        else:
            illu_flag_dorsal = 1
            illu_flag_ventral = 1
        if illu_flag_ventral:
            print("\nFusion along illumination for ventral camera...")
            self.modelFront.train(
                data_path=data_path,
                sample_name=sample_name,
                top_illu_data=top_illu_ventral_det_data,
                bottom_illu_data=bottom_illu_ventral_det_data,
                left_illu_data=left_illu_ventral_det_data,
                right_illu_data=right_illu_ventral_det_data,
                save_path=save_path,
                save_folder="",
                camera_position="ventral_det",
            )
        if illu_flag_dorsal:
            print("\nFusion along illumination for dorsal camera...")
            self.modelBack.train(
                data_path=data_path,
                sample_name=sample_name,
                top_illu_data=top_illu_dorsal_det_data,
                bottom_illu_data=bottom_illu_dorsal_det_data,
                left_illu_data=left_illu_dorsal_det_data,
                right_illu_data=right_illu_dorsal_det_data,
                save_path=save_path,
                save_folder="",
                camera_position="dorsal_det",
            )

        data_path = os.path.join(data_path, sample_name)
        if not os.path.exists(
            os.path.join(
                save_path, self.sample_params["topillu_ventraldet_data_saving_name"]
            )
            + "/regInfo{}.npy".format(
                "" if (not self.train_params["destripe_preceded"]) else "_destripe"
            )
        ):
            print("\nRegister...")
            print("read in...")
            if self.train_params["destripe_preceded"]:
                illu_name_ = illu_name + "+RESULT/{}.tif".format(
                    self.train_params["destripe_params"]
                )
            else:
                illu_name_ = illu_name + ".tif"
            respective_view_uint16_handle = AICSImage(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_ventraldet_data_saving_name"],
                    illu_name_,
                )
            )
            moving_view_uint16_handle = AICSImage(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_dorsaldet_data_saving_name"],
                    illu_name_,
                )
            )
            respective_view_uint16 = respective_view_uint16_handle.get_image_data(
                "ZXY" if T_flag else "ZYX", T=0, C=0
            )
            moving_view_uint16 = moving_view_uint16_handle.get_image_data(
                "ZXY" if T_flag else "ZYX", T=0, C=0
            )
            print("reg in zx...")
            yMP_respective = respective_view_uint16.max(2)
            yMP_moving = moving_view_uint16.max(2)
            AffineMapZX = coarseRegistrationZX(
                yMP_respective,
                yMP_moving,
                ventral_angle_in_degree,
                dorsal_angle_in_degree,
                r=z_spacing / xy_spacing,
            )
            del yMP_respective, yMP_moving
            print("reg in y...")
            zMP_respective = respective_view_uint16.max(0)
            zMP_moving = moving_view_uint16.max(0)
            AffineMapZXY, frontMIP, backMIP = coarseRegistrationY(
                zMP_respective,
                zMP_moving,
                AffineMapZX,
            )
            del zMP_respective, zMP_moving
            print("rigid registration in 3D...")
            _, m, n = respective_view_uint16.shape
            a0, b0, c0, d0 = self.segMIP(frontMIP)
            a1, b1, c1, d1 = self.segMIP(backMIP)
            infomax = float(max(frontMIP.max(), backMIP.max()))
            xs, xe, ys, ye = min(c0, c1), max(d0, d1), min(a0, a1), max(b0, b1)
            x = min(
                len(np.arange(0, xs)),
                len(np.arange(xe, m)),
            )
            y = min(
                len(np.arange(0, ys)),
                len(np.arange(ye, n)),
            )
            xs, xe = x, -x if x != 0 else None
            ys, ye = y, -y if y != 0 else None
            fineReg(
                respective_view_uint16,
                moving_view_uint16,
                xs,
                xe,
                ys,
                ye,
                AffineMapZXY,
                infomax,
                os.path.join(
                    save_path, self.sample_params["topillu_ventraldet_data_saving_name"]
                ),
                self.train_params["destripe_preceded"],
            )
            del respective_view_uint16, moving_view_uint16
        else:
            print("\nSkip registration...")
        print("\nTranslate data...")
        regInfo = np.load(
            os.path.join(
                save_path,
                self.sample_params["topillu_ventraldet_data_saving_name"],
                "regInfo{}.npy".format(
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe"
                ),
            ),
            allow_pickle=True,
        ).item()
        AffineMapZXY = regInfo["AffineMapZXY"]
        zfront = regInfo["zfront"]
        zback = regInfo["zback"]
        z = regInfo["z"]
        m = regInfo["m"]
        n = regInfo["n"]
        xcs, xce, ycs, yce = regInfo["region_for_reg"]
        padding_z = boundaryInclude(
            regInfo,
            z + int(np.ceil(AffineMapZXY[0])) if AffineMapZXY[0] > 0 else z,
            m,
            n,
        )

        for f, f_name in zip(
            ["top", "bottom"] if (not T_flag) else ["left", "right"], ["top", "bottom"]
        ):
            if not self.train_params["destripe_preceded"]:
                if isinstance(locals()[f + "_illu_dorsal_det_data"], str):
                    f_handle = AICSImage(
                        os.path.join(data_path, locals()[f + "_illu_dorsal_det_data"])
                    )
                else:
                    f_handle = AICSImage(locals()[f + "_illu_dorsal_det_data"])
            else:
                f0 = self.sample_params[f_name + "illu_dorsaldet_data_saving_name"]
                f_handle = AICSImage(
                    os.path.join(
                        save_path,
                        f0,
                        f0 + "+RESULT",
                        self.train_params["destripe_params"] + ".tif",
                    )
                )
            inputs = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
            trans_path = os.path.join(
                save_path,
                self.sample_params[f_name + "illu_dorsaldet_data_saving_name"],
                self.sample_params[f_name + "illu_dorsaldet_data_saving_name"]
                + "{}_reg.tif".format(
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe"
                ),
            )
            volumeTranslate(
                inputs,
                regInfo,
                AffineMapZXY,
                zback,
                z,
                m,
                n,
                padding_z,
                trans_path,
                T_flag,
                device=self.train_params["device"],
            )
            del inputs
        if not self.train_params["destripe_preceded"]:
            fl = illu_name + ".tif"
        else:
            fl = illu_name + "+RESULT/{}.tif".format(
                self.train_params["destripe_params"]
            )
        f_handle = AICSImage(
            os.path.join(
                save_path, self.sample_params["topillu_dorsaldet_data_saving_name"], fl
            )
        )
        inputs = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        trans_path = os.path.join(
            save_path,
            self.sample_params["topillu_dorsaldet_data_saving_name"],
            illu_name
            + "{}_reg.tif".format(
                "" if (not self.train_params["destripe_preceded"]) else "_destripe"
            ),
        )
        volumeTranslate(
            inputs,
            regInfo,
            AffineMapZXY,
            zback,
            z,
            m,
            n,
            padding_z,
            trans_path,
            T_flag,
            device=self.train_params["device"],
        )
        del inputs
        fl = "fusionBoundary_xy_{}_{}".format(
            "simple" if self.train_params["fast_mode"] else "full",
            "allowBreak" if self.train_params["allow_break"] else "noBreak",
        )
        f0 = os.path.join(
            save_path,
            self.sample_params["topillu_dorsaldet_data_saving_name"],
            fl + ".npy",
        )
        boundary = np.load(f0)
        mask = np.arange(m)[None, :, None] > boundary[:, None, :]
        trans_path = os.path.join(
            save_path,
            self.sample_params["topillu_dorsaldet_data_saving_name"],
            fl
            + "{}_reg.npy".format(
                "" if (not self.train_params["destripe_preceded"]) else "_destripe"
            ),
        )
        volumeTranslate(
            mask,
            regInfo,
            AffineMapZXY,
            zback,
            z,
            m,
            n,
            padding_z,
            trans_path,
            T_flag,
            device=self.train_params["device"],
        )

        print("\nLocalize sample...")
        print("read in...")
        if not self.train_params["destripe_preceded"]:
            fl = illu_name + ".tif"
        else:
            fl = illu_name + "+RESULT/{}.tif".format(
                self.train_params["destripe_params"]
            )
        f_handle = AICSImage(
            os.path.join(
                save_path, self.sample_params["topillu_ventraldet_data_saving_name"], fl
            )
        )
        illu_front = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        f_handle = AICSImage(
            os.path.join(
                save_path,
                self.sample_params["topillu_dorsaldet_data_saving_name"],
                illu_name
                + "{}_reg.tif".format(
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe"
                ),
            )
        )
        illu_back = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        cropInfo = self.localizingSample(illu_front.max(0), illu_back.max(0), save_path)
        print(cropInfo)
        if self.train_params["require_precropping"]:
            xs, xe, ys, ye = cropInfo.loc[
                "in summary", ["startX", "endX", "startY", "endY"]
            ].astype(int)
        else:
            xs, xe, ys, ye = None, None, None, None
        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=200)
        ax1.imshow(illu_front.max(0).T if T_flag else illu_front.max(0))
        if self.train_params["require_precropping"]:
            rect = patches.Rectangle(
                (ys, xs) if (not T_flag) else (xs, ys),
                (ye - ys) if (not T_flag) else (xe - xs),
                (xe - xs) if (not T_flag) else (ye - ys),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax1.add_patch(rect)
        ax1.set_title("ventral det", fontsize=8, pad=1)
        ax1.axis("off")
        ax2.imshow(illu_back.max(0).T if T_flag else illu_back.max(0))
        if self.train_params["require_precropping"]:
            rect = patches.Rectangle(
                (ys, xs) if (not T_flag) else (xs, ys),
                (ye - ys) if (not T_flag) else (xe - xs),
                (xe - xs) if (not T_flag) else (ye - ys),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax2.add_patch(rect)
        ax2.set_title("dorsal det", fontsize=8, pad=1)
        ax2.axis("off")
        plt.show()
        segMask = self.segmentSample(
            illu_front[:, xs:xe, ys:ye][
                :,
                :: self.train_params["resampleRatio"],
                :: self.train_params["resampleRatio"],
            ],
            illu_back[:, xs:xe, ys:ye][
                :,
                :: self.train_params["resampleRatio"],
                :: self.train_params["resampleRatio"],
            ],
            save_path,
        )
        del illu_front, illu_back

        print("\nFor top/left Illu...")
        print("read in...")
        if not self.train_params["destripe_preceded"]:
            if isinstance(
                locals()[
                    "{}_illu_ventral_det_data".format("left" if T_flag else "top")
                ],
                str,
            ):
                top_handle = AICSImage(
                    os.path.join(
                        data_path,
                        locals()[
                            "{}_illu_ventral_det_data".format(
                                "left" if T_flag else "top"
                            )
                        ],
                    )
                )
            else:
                top_handle = AICSImage(
                    locals()[
                        "{}_illu_ventral_det_data".format("left" if T_flag else "top")
                    ]
                )
        else:
            f0 = self.sample_params["topillu_ventraldet_data_saving_name"]
            top_handle = AICSImage(
                os.path.join(
                    save_path,
                    f0,
                    f0 + "+RESULT",
                    self.train_params["destripe_params"] + ".tif",
                )
            )
        rawPlanesTopO = top_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        rawPlanesToptmp = rawPlanesTopO[:, xs:xe, ys:ye]
        _, m_c, n_c = rawPlanesToptmp.shape
        m = len(np.arange(m_c)[:: self.train_params["resampleRatio"]])
        n = len(np.arange(n_c)[:: self.train_params["resampleRatio"]])
        del rawPlanesTopO
        bottom_handle = AICSImage(
            os.path.join(
                save_path,
                self.sample_params["bottomillu_dorsaldet_data_saving_name"],
                self.sample_params["bottomillu_dorsaldet_data_saving_name"]
                + "{}_reg.tif".format(
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe"
                ),
            )
        )
        rawPlanesBottomO = bottom_handle.get_image_data(
            "ZXY" if T_flag else "ZYX", T=0, C=0
        )
        m0, n0 = rawPlanesBottomO.shape[-2:]
        rawPlanesBottom = rawPlanesBottomO[:, xs:xe, ys:ye]
        del rawPlanesBottomO
        s = rawPlanesBottom.shape[0]
        if rawPlanesToptmp.shape[0] < rawPlanesBottom.shape[0]:
            rawPlanesTop = np.concatenate(
                (
                    rawPlanesToptmp,
                    np.zeros(
                        (
                            rawPlanesBottom.shape[0] - rawPlanesToptmp.shape[0],
                            rawPlanesBottom.shape[1],
                            rawPlanesBottom.shape[2],
                        ),
                        dtype=np.uint16,
                    ),
                ),
                0,
            )
        else:
            rawPlanesTop = rawPlanesToptmp
        del rawPlanesToptmp
        topF, bottomF = self.extractNSCTF(
            s,
            m,
            n,
            topVol=rawPlanesTop,
            bottomVol=rawPlanesBottom,
            device=self.train_params["device"],
            Max=cropInfo.loc["in summary", "maxv"],
        )
        boundaryRaw, boundarySmooth = self.dualViewFusion(
            topF.transpose(0, 2, 1),
            bottomF.transpose(0, 2, 1),
            segMask.transpose(1, 0, 2),
        )
        boundaryE = np.zeros((n0, m0))
        boundaryE[ys:ye, xs:xe] = extendBoundary(
            boundaryRaw,
            self.train_params["resampleRatio"],
            window_size=self.train_params["window_size"][1],
            poly_order=self.train_params["poly_order"][1],
            cSize=(
                int(ye - ys) if ye is not None else n0,
                int(xe - xs) if xe is not None else m0,
            ),
            boundarySmoothed=boundarySmooth,
            _illu=False,
        )
        if xs is not None:
            boundaryE[:, :xs], boundaryE[:, xe:] = (
                boundaryE[:, xs][:, None],
                boundaryE[:, xe - 1][:, None],
            )
        if ys is not None:
            boundaryE[:ys, :], boundaryE[ye:, :] = (
                boundaryE[ys, :][None, :],
                boundaryE[ye - 1, :][None, :],
            )
        boundaryETop = np.clip(boundaryE.T, 0, s)
        np.save(
            os.path.join(
                save_path,
                self.sample_params["topillu_ventraldet_data_saving_name"],
                "fusionBoundary_z_{}.npy".format(
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe"
                ),
            ),
            boundaryETop.T if T_flag else boundaryETop,
        )
        del topF, bottomF, rawPlanesTop, rawPlanesBottom

        print("\n\nFor bottom/right Illu...")
        print("read in...")
        if not self.train_params["destripe_preceded"]:
            if isinstance(
                locals()[
                    "{}_illu_ventral_det_data".format("right" if T_flag else "bottom")
                ],
                str,
            ):
                top_handle = AICSImage(
                    os.path.join(
                        data_path,
                        locals()[
                            "{}_illu_ventral_det_data".format(
                                "right" if T_flag else "bottom"
                            )
                        ],
                    )
                )
            else:
                top_handle = AICSImage(
                    locals()[
                        "{}_illu_ventral_det_data".format(
                            "right" if T_flag else "bottom"
                        )
                    ]
                )
        else:
            f0 = self.sample_params["bottomillu_ventraldet_data_saving_name"]
            top_handle = AICSImage(
                os.path.join(
                    save_path,
                    f0,
                    f0 + "+RESULT",
                    self.train_params["destripe_params"] + ".tif",
                )
            )
        rawPlanesTopO = top_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        rawPlanesToptmp = rawPlanesTopO[:, xs:xe, ys:ye]
        _, m_c, n_c = rawPlanesToptmp.shape
        m = len(np.arange(m_c)[:: self.train_params["resampleRatio"]])
        n = len(np.arange(n_c)[:: self.train_params["resampleRatio"]])
        del rawPlanesTopO
        bottom_handle = AICSImage(
            os.path.join(
                save_path,
                self.sample_params["topillu_dorsaldet_data_saving_name"],
                self.sample_params["topillu_dorsaldet_data_saving_name"]
                + "{}_reg.tif".format(
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe"
                ),
            )
        )
        rawPlanesBottomO = bottom_handle.get_image_data(
            "ZXY" if T_flag else "ZYX", T=0, C=0
        )
        m0, n0 = rawPlanesBottomO.shape[-2:]
        rawPlanesBottom = rawPlanesBottomO[:, xs:xe, ys:ye]
        del rawPlanesBottomO
        s = rawPlanesBottom.shape[0]
        if rawPlanesToptmp.shape[0] < rawPlanesBottom.shape[0]:
            rawPlanesTop = np.concatenate(
                (
                    rawPlanesToptmp,
                    np.zeros(
                        (
                            rawPlanesBottom.shape[0] - rawPlanesToptmp.shape[0],
                            rawPlanesBottom.shape[1],
                            rawPlanesBottom.shape[2],
                        ),
                        dtype=np.uint16,
                    ),
                ),
                0,
            )
        else:
            rawPlanesTop = rawPlanesToptmp
        del rawPlanesToptmp
        topF, bottomF = self.extractNSCTF(
            s,
            m,
            n,
            topVol=rawPlanesTop,
            bottomVol=rawPlanesBottom,
            device=self.train_params["device"],
            Max=cropInfo.loc["in summary", "maxv"],
        )
        boundaryRaw, boundarySmooth = self.dualViewFusion(
            topF.transpose(0, 2, 1),
            bottomF.transpose(0, 2, 1),
            segMask.transpose(1, 0, 2),
        )
        boundaryE = np.zeros((n0, m0))
        boundaryE[ys:ye, xs:xe] = extendBoundary(
            boundaryRaw,
            self.train_params["resampleRatio"],
            window_size=self.train_params["window_size"][1],
            poly_order=self.train_params["poly_order"][1],
            cSize=(
                int(ye - ys) if ye is not None else n0,
                int(xe - xs) if xe is not None else m0,
            ),
            boundarySmoothed=boundarySmooth,
            _illu=False,
        )
        if xs is not None:
            boundaryE[:, :xs], boundaryE[:, xe:] = (
                boundaryE[:, xs][:, None],
                boundaryE[:, xe - 1][:, None],
            )
        if ys is not None:
            boundaryE[:ys, :], boundaryE[ye:, :] = (
                boundaryE[ys, :][None, :],
                boundaryE[ye - 1, :][None, :],
            )
        boundaryEBottom = np.clip(boundaryE.T, 0, s)
        np.save(
            os.path.join(
                save_path,
                self.sample_params["bottomillu_ventraldet_data_saving_name"],
                "fusionBoundary_z_{}.npy".format(
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe"
                ),
            ),
            boundaryEBottom.T if T_flag else boundaryEBottom,
        )
        del topF, bottomF, rawPlanesTop, rawPlanesBottom

        print("\n\nStitching...")
        print("read in...")
        # "ZXY" if T_flag else "ZYX"
        boundaryEFront = np.load(
            os.path.join(
                save_path,
                self.sample_params["topillu_ventraldet_data_saving_name"],
                "fusionBoundary_xy_{}_{}.npy".format(
                    "simple" if self.train_params["fast_mode"] else "full",
                    "allowBreak" if self.train_params["allow_break"] else "noBreak",
                ),
            )
        )
        fl = "fusionBoundary_xy_{}_{}".format(
            "simple" if self.train_params["fast_mode"] else "full",
            "allowBreak" if self.train_params["allow_break"] else "noBreak",
        )
        boundaryEBack = np.load(
            os.path.join(
                save_path,
                self.sample_params["topillu_dorsaldet_data_saving_name"],
                fl
                + "{}_reg.npy".format(
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe"
                ),
            )
        )
        boundaryTop = np.load(
            os.path.join(
                save_path,
                self.sample_params["topillu_ventraldet_data_saving_name"],
                "fusionBoundary_z_{}.npy".format(
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe"
                ),
            )
        )
        boundaryBottom = np.load(
            os.path.join(
                save_path,
                self.sample_params["bottomillu_ventraldet_data_saving_name"],
                "fusionBoundary_z_{}.npy".format(
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe"
                ),
            )
        )
        if T_flag:
            boundaryBottom = boundaryBottom.T
            boundaryTop = boundaryTop.T
            boundaryEBack = boundaryEBack.swapaxes(1, 2)

        if not self.train_params["destripe_preceded"]:
            fl = illu_name + ".tif"
        else:
            fl = illu_name + "+RESULT/{}.tif".format(
                self.train_params["destripe_params"]
            )
        f_handle = AICSImage(
            os.path.join(
                save_path, self.sample_params["topillu_ventraldet_data_saving_name"], fl
            )
        )
        illu_front = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        f_handle = AICSImage(
            os.path.join(
                save_path,
                self.sample_params["topillu_dorsaldet_data_saving_name"],
                illu_name
                + "{}_reg.tif".format(
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe"
                ),
            )
        )
        illu_back = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        s, m0, n0 = illu_back.shape
        reconVol = fusionResultFour(
            boundaryTop,
            boundaryBottom,
            boundaryEFront,
            boundaryEBack,
            illu_front,
            illu_back,
            s,
            m0,
            n0,
            save_path,
            self.train_params["device"],
            self.sample_params,
            Gaussianr=self.train_params["Gaussian_kernel_size"],
            GFr=self.train_params["GF_kernel_size"],
        )
        if T_flag:
            result = reconVol.swapaxes(1, 2)
        else:
            result = reconVol
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=200)
        ax1.imshow(result.max(0))
        ax1.set_title("result in xy", fontsize=8, pad=1)
        ax1.axis("off")
        ax2.imshow(result.max(1))
        ax2.set_title("result in zy", fontsize=8, pad=1)
        ax2.axis("off")
        ax3.imshow(result.max(2))
        ax3.set_title("result in zx", fontsize=8, pad=1)
        ax3.axis("off")
        plt.show()
        print("Save...")
        tifffile.imwrite(
            os.path.join(
                save_path,
                self.sample_params["topillu_ventraldet_data_saving_name"],
                "quadrupleFusionResult_{}_{}{}.tif".format(
                    "simple" if self.train_params["fast_mode"] else "full",
                    "allowBreak" if self.train_params["allow_break"] else "noBreak",
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe",
                ),
            ),
            result,
        )
        del reconVol, illu_front, illu_back, result
        gc.collect()

    def dualViewFusion(self, topF, bottomF, segMaskvUINT8):
        print("to GPU...")
        segMaskGPU = torch.from_numpy(segMaskvUINT8).to(self.train_params["device"])
        topFGPU, bottomFGPU = torch.from_numpy(topF).to(
            self.train_params["device"]
        ), torch.from_numpy(bottomF).to(self.train_params["device"])
        stripeMask = np.zeros((bottomF.shape[1], bottomF.shape[2]))
        boundaryRaw, boundarySmooth = EM2DPlus(
            segMaskGPU,
            bottomFGPU,
            topFGPU,
            stripeMask,
            self.train_params["Lambda"] * 10,
            [self.train_params["window_size"][1], self.train_params["window_size"][1]],
            [self.train_params["poly_order"][1], self.train_params["poly_order"][1]],
            self.train_params["kernel2d"][None, None, :, :],
            False,
            1,
            device=self.train_params["device"],
            _xy=False,
            _fastMode=False,
        )
        del topFGPU, bottomFGPU, segMaskGPU
        return boundaryRaw, boundarySmooth

    def extractNSCTF(self, s, m, n, topVol, bottomVol, device, Max):
        r = self.train_params["resampleRatio"]
        featureExtrac = NSCTdec(levels=[3, 3, 3], device=device)
        topF, bottomF = np.empty((s, m, n), dtype=np.float32), np.empty(
            (s, m, n), dtype=np.float32
        )
        topFBase, bottomFBase = np.empty((s, m, n), dtype=np.float32), np.empty(
            (s, m, n), dtype=np.float32
        )
        tmp0, tmp1 = np.arange(0, s, 10), np.arange(10, s + 10, 10)
        for p, q in tqdm.tqdm(zip(tmp0, tmp1), desc="NSCT: ", total=len(tmp0)):
            topDataFloat, bottomDataFloat = topVol[p:q, :, :].astype(
                np.float32
            ), bottomVol[p:q, :, :].astype(np.float32)
            topDataGPU, bottomDataGPU = torch.from_numpy(
                topDataFloat[:, None, :, :]
            ).to(device), torch.from_numpy(bottomDataFloat[:, None, :, :]).to(device)
            topDataGPU[:], bottomDataGPU[:] = topDataGPU / Max, bottomDataGPU / Max
            a, b, c = featureExtrac.nsctDec(topDataGPU, r, _forFeatures=True)
            topF[p:q], topFBase[p:q] = (
                a.cpu().detach().numpy(),
                b.cpu().detach().numpy(),
            )
            a[:], b[:], c[:] = featureExtrac.nsctDec(
                bottomDataGPU,
                r,
                _forFeatures=True,
            )
            bottomF[p:q], bottomFBase[p:q] = (
                a.cpu().detach().numpy(),
                b.cpu().detach().numpy(),
            )
            del topDataFloat, bottomDataFloat, topDataGPU, bottomDataGPU, a, b, c
        gc.collect()
        return topFBase * topF, bottomFBase * bottomF

    def segmentSample(self, topVoltmp, bottomVol, info_path):
        m, n = topVoltmp.shape[-2:]
        zfront, zback = topVoltmp.shape[0], bottomVol.shape[0]
        if zfront < zback:
            topVol = np.concatenate(
                (topVoltmp, np.zeros((zback - zfront, m, n), dtype=np.uint16)), 0
            )
        else:
            topVol = topVoltmp
        del topVoltmp
        Min, Max, th = 65535, 0, 0
        allList = [
            value for key, value in self.sample_params.items() if "saving_name" in key
        ]
        for f in allList:
            t = np.load(
                os.path.join(info_path, f, "info.npy"), allow_pickle=True
            ).item()
            Min = min(Min, t["minvol"])
            Max = max(Max, t["maxvol"])
            th += t["thvol_log"] if self.train_params["require_log"] else t["thvol"]
        th = th / len(allList)
        s = zback
        topSegMask, bottomSegMask = np.zeros((zback, m, n), dtype=bool), np.zeros(
            (zback, m, n), dtype=bool
        )
        waterShed(
            topSegMask,
            topVol,
            th,
            Max,
            Min,
            s,
            m,
            n,
            "front",
            _log=self.train_params["require_log"],
            _xy=False,
        )
        waterShed(
            bottomSegMask,
            bottomVol,
            th,
            Max,
            Min,
            s,
            m,
            n,
            "back",
            _log=self.train_params["require_log"],
            _xy=False,
        )
        segMask = refineShape(
            topSegMask.transpose(2, 0, 1),
            bottomSegMask.transpose(2, 0, 1),
            n,
            s,
            m,
            _xy=False,
        )
        """
        for i in range(0, segMask.shape[1], 3):
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(topVol[i])
            plt.subplot(1, 3, 2)
            plt.imshow(bottomVol[i])
            plt.subplot(1, 3, 3)
            plt.imshow(segMask[:, i, :].T)
            plt.show()
        """
        del topSegMask, bottomSegMask, topVol, bottomVol
        return segMask.astype(np.uint8)

    def localizingSample(self, rawPlanes_ventral, rawPlanes_dorsal, info_path):
        cropInfo = pd.DataFrame(
            columns=["startX", "endX", "startY", "endY", "maxv"],
            index=["ventral", "dorsal"],
        )
        for f in ["ventral", "dorsal"]:
            maximumProjection = locals()["rawPlanes_" + f].astype(np.float32)
            maximumProjection = np.log(np.clip(maximumProjection, 1, None))
            m, n = maximumProjection.shape
            allList = [
                value
                for key, value in self.sample_params.items()
                if ("saving_name" in key) and (f in key)
            ]
            th = 0
            maxv = 0
            for l in allList:
                t = np.load(
                    os.path.join(info_path, l, "info.npy"), allow_pickle=True
                ).item()
                th += t["MIP_th"]
                maxv = max(maxv, t["MIP_max"])
            th = th / len(allList)
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
        return cropInfo

    def segMIP(self, maximumProjection, maxv=None, minv=None, th=None):
        m, n = maximumProjection.shape
        if th == None:
            maxv, minv, th = (
                maximumProjection.max(),
                maximumProjection.min(),
                filters.threshold_otsu(maximumProjection),
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
        return a, b, c, d


def fusionResultFour(
    boundaryTop,
    boundaryBottom,
    boundaryFront,
    boundaryBack,
    illu_front,
    illu_back,
    s,
    m,
    n,
    info_path,
    device,
    sample_params,
    Gaussianr=49,
    GFr=49,
):
    zmax = boundaryBack.shape[0]
    decModel = NSCTdec(levels=[3, 3, 3], device=device)
    recModel = NSCTrec(levels=[3, 3, 3], device=device)
    max_filter = nn.MaxPool2d((9, 9), stride=(1, 1), padding=(4, 4))
    k = torch.ones(8, 1, 3, 3).to(device)
    mask = np.arange(m)[None, :, None]
    if boundaryFront.shape[0] < zmax:
        boundaryFront = np.concatenate(
            (
                boundaryFront,
                np.zeros((zmax - boundaryFront.shape[0], boundaryFront.shape[1])),
            ),
            0,
        )
    mask_front = mask > boundaryFront[:, None, :]  ###1是下面，0是上面
    mask_back = ~boundaryBack
    mask_ztop = (
        np.arange(s)[:, None, None] > boundaryTop[None, :, :]
    )  ###1是后面，0是前面
    mask_zbottom = (
        np.arange(s)[:, None, None] > boundaryBottom[None, :, :]
    )  ###1是后面，0是前面
    GFbase, GFdetail = GuidedFilter(r=GFr, eps=1), GuidedFilter(r=9, eps=1e-6)
    kernel = torch.ones(1, 1, Gaussianr, Gaussianr).to(device) / Gaussianr / Gaussianr
    listPair1 = {"1": "4", "2": "3", "4": "1", "3": "2"}
    reconVol = np.empty(illu_back.shape, dtype=np.uint16)
    allList = [
        value
        for key, value in sample_params.items()
        if ("saving_name" in key) and ("dorsal" in key)
    ]
    volmin = 65535
    for l in allList:
        volmin = min(
            np.load(os.path.join(info_path, l, "info.npy"), allow_pickle=True).item()[
                "minvol"
            ],
            volmin,
        )
    for ii in tqdm.tqdm(range(s), desc="fusion: "):
        if ii < illu_front.shape[0]:
            s1, s2, s3, s4 = (
                copy.deepcopy(illu_front[ii]),
                copy.deepcopy(illu_front[ii]),
                copy.deepcopy(illu_back[ii]),
                copy.deepcopy(illu_back[ii]),
            )
        else:
            s3, s4 = copy.deepcopy(illu_back[ii]), copy.deepcopy(illu_back[ii])
            s1, s2 = np.zeros(s3.shape), np.zeros(s3.shape)
        s3[s3 < volmin] = s1[s3 < volmin]
        s4[s4 < volmin] = s1[s4 < volmin]
        tmp1 = (mask_front[ii] == 0) * (mask_ztop[ii] == 0)  ###top+front
        tmp2 = (mask_front[ii] == 1) * (mask_zbottom[ii] == 0)  ###bottom+front
        tmp3 = (mask_back[ii] == 0) * (mask_ztop[ii] == 1)  ###top+back
        tmp4 = (mask_back[ii] == 1) * (mask_zbottom[ii] == 1)  ###bottom+back
        vnameList = ["1", "2", "3", "4"]
        sg, maskList = {}, {}
        sg["1"], sg["2"] = torch.from_numpy(s1.astype(np.float32)[None, None, :, :]).to(
            device
        ), torch.from_numpy(s2.astype(np.float32)[None, None, :, :]).to(device)
        sg["3"], sg["4"] = torch.from_numpy(s3.astype(np.float32)[None, None, :, :]).to(
            device
        ), torch.from_numpy(s4.astype(np.float32)[None, None, :, :]).to(device)
        mask1234 = np.zeros(tmp1.shape, dtype=bool)
        for vname in vnameList:
            maskList[vname] = np.zeros((m, n), dtype=bool)
        flag_nsct = 0
        for vname in vnameList:
            maskList[vname] += locals()["tmp" + vname] * (
                ~locals()["tmp" + listPair1[vname]]
            )
            if vnameList.index(vname) < vnameList.index(listPair1[vname]):
                v = locals()["tmp" + vname] * locals()["tmp" + listPair1[vname]]
                if sum(sum(v)):
                    if flag_nsct == 0:
                        F1, _, _ = decModel.nsctDec(sg[vname], 1, _forFeatures=True)
                        F2, _, _ = decModel.nsctDec(
                            sg[listPair1[vname]], 1, _forFeatures=True
                        )
                    if ((F1 - F2).cpu().data.numpy() * v).sum() >= 0:
                        maskList[vname] += v
                    else:
                        maskList[listPair1[vname]] += v
                    flag_nsct = 1
        for mm in maskList:
            mask1234 += maskList[mm]
        mask1234 = ~mask1234
        vnameList_short = ["1", "3"]
        maskList_short = {}
        maskList_short["1"] = maskList["1"] + maskList["2"]
        maskList_short["3"] = maskList["3"] + maskList["4"]
        if sum(sum(mask1234)) != s1.size:
            sg["1234"] = torch.maximum(
                torch.maximum(sg["1"], sg["2"]), torch.maximum(sg["3"], sg["4"])
            )
            vnameList_short.append("1234")
            maskList_short["1234"] = mask1234
        t0, t1 = torch.zeros(sg["1"].shape).to(device), torch.zeros(sg["1"].shape).to(
            device
        )
        minn, maxx = 65535, 0
        for vname in vnameList_short:
            data, mask = sg[vname], maskList_short[vname]
            locals()["base" + vname] = torch.conv2d(
                F.pad(data, (24, 24, 24, 24), "reflect"), kernel
            )
            locals()["detail" + vname] = data - locals()["base" + vname]
            maskGPU = torch.from_numpy(mask.astype(np.float32)[None, None, :, :]).to(
                device
            )
            locals()["resultbase" + vname] = GFbase(data, maskGPU)
            locals()["resultdetail" + vname] = GFdetail(data, maskGPU)
            if data.min() > 0:
                minn = min(data.min(), minn)
            maxx = max(data.max(), maxx)
            tdata = data[data != 0]
            if tdata.numel():
                minn = min(tdata.min(), minn)
            else:
                minn = min(0, minn)
            t0 += locals()["resultbase" + vname]
            t1 += locals()["resultdetail" + vname]
        result = torch.zeros(sg["1"].shape).to(device)
        for vname in vnameList_short:
            locals()["resultbase" + vname] = locals()["resultbase" + vname] / t0
            locals()["resultdetail" + vname] = locals()["resultdetail" + vname] / t1
            result += (
                locals()["resultbase" + vname] * locals()["base" + vname]
                + locals()["detail" + vname] * locals()["resultdetail" + vname]
            )
        result = (
            torch.clip(result, minn, maxx)
            .squeeze()
            .cpu()
            .data.numpy()
            .astype(np.uint16)
        )
        reconVol[ii, :, :] = result
        del s1, s2, s3, s4, tmp1, tmp2, tmp3, tmp4, sg, maskList
    del mask_front, mask_ztop, mask_back, mask_zbottom
    del illu_front, illu_back
    return reconVol


def volumeTranslate(
    inputs,
    regInfo,
    AffineMapZXY,
    s,
    zmax,
    m,
    n,
    padding_z,
    save_path,
    T_flag,
    device,
):
    inputs_dtype = inputs.dtype
    if zmax > s:
        rawData = np.concatenate(
            (inputs, np.zeros((zmax - s, m, n), dtype=inputs.dtype)), 0
        )
    else:
        rawData = inputs
    del inputs
    rawData[:] = np.flip(rawData, axis=(0, 1))
    zs, ze, zss, zee = translatingParams(-int(round(AffineMapZXY[0])))
    xs, xe, xss, xee = translatingParams(-int(round(AffineMapZXY[1])))
    ys, ye, yss, yee = translatingParams(int(round(AffineMapZXY[2])))
    if AffineMapZXY[0] > 0:
        rawData2 = np.concatenate(
            (
                rawData,
                np.zeros((int(np.ceil(AffineMapZXY[0])), m, n), dtype=rawData.dtype),
            ),
            0,
        )
    else:
        rawData2 = rawData
    translatedData = np.zeros(rawData2.shape, dtype=np.float32)
    translatedData[zss:zee, xss:xee, yss:yee] = rawData2[zs:ze, xs:xe, ys:ye]
    del rawData, rawData2
    AffineTransform = regInfo["AffineTransform_float_3_3_inverse"][:, 0]
    afixed = regInfo["fixed_inverse"][:, 0]
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(AffineTransform[:9].astype(np.float64))
    affine.SetTranslation(AffineTransform[9:].astype(np.float64))
    affine.SetCenter(afixed.astype(np.float64))
    A = np.array(affine.GetMatrix()).reshape(3, 3)
    c = np.array(affine.GetCenter())
    t = np.array(affine.GetTranslation())
    T = np.eye(4, dtype=np.float32)
    T[0:3, 0:3] = A
    T[0:3, 3] = -np.dot(A, c) + t + c
    commonData = np.zeros((max(padding_z, zmax), m, n), dtype=inputs_dtype)
    xx, yy = np.meshgrid(
        np.arange(commonData.shape[1], dtype=np.float32),
        np.arange(commonData.shape[2], dtype=np.float32),
    )
    xx, yy = xx.T[None], yy.T[None]
    ss = torch.split(torch.arange(commonData.shape[0]), 10)
    for s in tqdm.tqdm(ss, desc="projecting: "):
        start, end = s[0], s[-1] + 1
        Z = (
            np.ones(xx.shape, dtype=np.float32)
            * np.arange(start, end, dtype=np.float32)[:, None, None]
        )
        X = xx.repeat(end - start, axis=0)
        Y = yy.repeat(end - start, axis=0)
        offset = np.ones(Z.shape, dtype=np.float32)
        coor = np.stack((Z, X, Y, offset))
        del Z, X, Y, offset
        coor_translated = np.dot(T, coor.reshape(4, -1))[:-1].reshape(
            3, coor.shape[1], coor.shape[2], coor.shape[3]
        )
        del coor
        if coor_translated[0, ...].max() < 0:
            continue
        if coor_translated[0, ...].min() >= translatedData.shape[0]:
            continue
        minn = int(np.clip(np.floor(coor_translated[0, ...].min()), 0, None))
        maxx = int(
            np.clip(
                np.ceil(coor_translated[0, ...].max()), None, translatedData.shape[0]
            )
        )
        smallData = translatedData[minn : maxx + 1, ...]
        coor_translated[0, ...] = coor_translated[0, ...] - minn
        coor_translated = (
            coor_translated
            / np.asarray(smallData.shape, dtype=np.float32)[:, None, None, None]
            - 0.5
        ) * 2
        translatedDatasmall = coordinate_mapping(
            smallData, coor_translated[[2, 1, 0], ...], device=device
        )
        commonData[start:end, ...] = translatedDatasmall
        del smallData, coor_translated, translatedDatasmall
        gc.collect()
    print("save...")
    if T_flag:
        result = commonData.swapaxes(1, 2)
    else:
        result = commonData
    if inputs_dtype == np.uint16:
        tifffile.imwrite(save_path, result)
    else:
        np.save(save_path, result)
    del translatedData, commonData, result


def coordinate_mapping(smallData, coor_translated, device):
    coor_translatedCupy = (
        torch.from_numpy(coor_translated).permute(1, 2, 3, 0)[None].to(device)
    )
    smallDataCupy = torch.from_numpy(smallData)[None, None, :, :].to(device)
    translatedDataCupy = F.grid_sample(
        smallDataCupy,
        coor_translatedCupy,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    translatedDatasmall = translatedDataCupy.squeeze().cpu().data.numpy()
    del translatedDataCupy, smallDataCupy, coor_translatedCupy
    return translatedDatasmall


def fineReg(
    respective_view_uint16,
    moving_view_uint16,
    xcs,
    xce,
    ycs,
    yce,
    AffineMapZXY,
    InfoMax,
    save_path,
    destripe_preceded,
):
    _, m, n = respective_view_uint16.shape
    zfront = respective_view_uint16.shape[0]
    zback = moving_view_uint16.shape[0]
    if zfront == zback:
        moving_view_uint16_padded = moving_view_uint16
        respective_view_uint16_padded = respective_view_uint16
    elif zfront > zback:
        moving_view_uint16_padded = np.concatenate(
            (moving_view_uint16, np.zeros((zfront - zback, m, n), dtype=np.uint16)),
            0,
        )
        respective_view_uint16_padded = respective_view_uint16
    else:
        respective_view_uint16_padded = np.concatenate(
            (
                respective_view_uint16,
                np.zeros((zback - zfront, m, n), dtype=np.uint16),
            ),
            0,
        )
        moving_view_uint16_padded = moving_view_uint16
    del respective_view_uint16, moving_view_uint16
    moving_view_uint16_padded[:] = np.flip(moving_view_uint16_padded, axis=(0, 1))
    zs, ze, zss, zee = translatingParams(-int(round(AffineMapZXY[0])))
    xs, xe, xss, xee = translatingParams(-int(round(AffineMapZXY[1])))
    ys, ye, yss, yee = translatingParams(int(round(AffineMapZXY[2])))
    moving_view_uint16_translated = np.empty(
        moving_view_uint16_padded.shape, dtype=np.uint16
    )
    moving_view_uint16_translated[zss:zee, xss:xee, yss:yee] = (
        moving_view_uint16_padded[zs:ze, xs:xe, ys:ye]
    )
    del moving_view_uint16_padded
    respective_view_cropped = respective_view_uint16_padded[:, xcs:xce, ycs:yce]
    moving_view_cropped = moving_view_uint16_translated[:, xcs:xce, ycs:yce]
    del respective_view_uint16_padded, moving_view_uint16_translated
    respective_view_uint8, moving_view_uint8 = np.empty(
        respective_view_cropped.shape, dtype=np.uint8
    ), np.empty(moving_view_cropped.shape, dtype=np.uint8)
    respective_view_uint8[:] = respective_view_cropped / InfoMax * 255
    moving_view_uint8[:] = moving_view_cropped / InfoMax * 255
    del moving_view_cropped, respective_view_cropped
    print("to ANTS...")
    A = np.clip(respective_view_uint8.shape[0] - 200, 0, None) // 2
    staticANTS = ants.from_numpy(respective_view_uint8[A : -A if A > 0 else None, :, :])
    movingANTS = ants.from_numpy(moving_view_uint8[A : -A if A > 0 else None, :, :])
    del moving_view_uint8, respective_view_uint8
    print("registration...")
    regModel = ants.registration(
        staticANTS,
        movingANTS,
        mask=None,
        moving_mask=None,
        type_of_transform="Rigid",
        mask_all_stages=True,
        random_seed=2022,
    )
    shutil.copyfile(regModel["fwdtransforms"][0], os.path.join(save_path, "reg.mat"))
    rfile = scipyio.loadmat(regModel["fwdtransforms"][0])
    rfile_inverse = scipyio.loadmat(regModel["invtransforms"][0])
    np.save(
        os.path.join(
            save_path,
            "regInfo{}.npy".format("" if (not destripe_preceded) else "_destripe"),
        ),
        {
            "AffineMapZXY": AffineMapZXY,
            "AffineTransform_float_3_3": np.squeeze(rfile["AffineTransform_float_3_3"]),
            "fixed": np.squeeze(rfile["fixed"]),
            "AffineTransform_float_3_3_inverse": rfile_inverse[
                "AffineTransform_float_3_3"
            ],
            "fixed_inverse": rfile_inverse["fixed"],
            "region_for_reg": np.array([xcs, xce, ycs, yce]),
            "zfront": zfront,
            "zback": zback,
            "m": m,
            "n": n,
            "z": max(zfront, zback),
        },
    )
    del regModel, movingANTS, staticANTS


def coarseRegistrationY(front, back, AffineMapZX):
    AffineMapZXY = np.zeros(3)
    AffineMapZXY[:2] = AffineMapZX
    front = front.astype(np.float32)
    back = back.astype(np.float32)
    xs, xe, xss, xee = translatingParams(int(round(AffineMapZX[1])))
    translatedBack = np.zeros(back.shape)
    translatedBack[xss:xee, :] = back[xs:xe, :]
    translatedBack[:] = np.flip(translatedBack, 0)
    regModel = ants.registration(
        ants.from_numpy(front),
        ants.from_numpy(translatedBack),
        type_of_transform="Translation",
        random_seed=2022,
    )
    AffineMapZXY[2] += scipyio.loadmat(regModel["fwdtransforms"][0])[
        "AffineTransform_float_2_2"
    ][-1, 0]
    AffineMapZXY[1] -= scipyio.loadmat(regModel["fwdtransforms"][0])[
        "AffineTransform_float_2_2"
    ][-2, 0]
    return AffineMapZXY, front, regModel["warpedmovout"].numpy()


def coarseRegistrationZX(yMPfrontO, yMPbackO, frontAngle, backAngle, r):
    angle = backAngle - frontAngle
    yMPfrontO = yMPfrontO.astype(np.float32)
    yMPbackO = yMPbackO.astype(np.float32)
    zfront, zback = yMPfrontO.shape[0], yMPbackO.shape[0]
    if zfront == zback:
        pass
    elif zfront > zback:
        yMPbackO = np.concatenate(
            (yMPbackO, np.zeros((zfront - zback, yMPbackO.shape[1]))), 0
        )
    else:
        yMPfrontO = np.concatenate(
            (yMPfrontO, np.zeros((zback - zfront, yMPfrontO.shape[1]))), 0
        )
    yMPfront = ndimage.zoom(yMPfrontO, [r, 1], order=1)
    yMPback = ndimage.rotate(
        ndimage.zoom(yMPbackO, [r, 1], order=1), order=1, angle=angle
    )
    _size = np.maximum(yMPfront.shape, yMPback.shape)
    for index in ["front", "back"]:
        locals()["yMPL" + index] = np.zeros(_size)
        a, b = locals()["yMP" + index].shape
        locals()["yMPL" + index][
            (_size[0] - a) // 2 : (_size[0] - a) // 2 + a,
            (_size[1] - b) // 2 : (_size[1] - b) // 2 + b,
        ] = locals()["yMP" + index]
    regModel = ants.registration(
        ants.from_numpy(locals()["yMPLfront"]),
        ants.from_numpy(locals()["yMPLback"]),
        type_of_transform="Translation",
        random_seed=2022,
    )
    transformedback = regModel["warpedmovout"].numpy()
    tmp = ndimage.rotate(transformedback, angle=-angle, order=1)
    a, b = tmp.shape
    s, m = yMPbackO.shape
    sp = round(s * r)
    transformedbackO = ndimage.zoom(
        tmp[(a - sp) // 2 : (a - sp) // 2 + sp, (b - m) // 2 : (b - m) // 2 + m],
        [1 / r, 1],
        order=1,
    )
    regModel = ants.registration(
        ants.from_numpy(transformedbackO),
        ants.from_numpy(yMPbackO),
        type_of_transform="Translation",
        random_seed=2022,
    )
    return scipyio.loadmat(regModel["fwdtransforms"][0])["AffineTransform_float_2_2"][
        -2:
    ][:, 0]


def translatingParams(x):
    if x == 0:
        xs, xe, xss, xee = None, None, None, None
    elif x > 0:
        xs, xe, xss, xee = x, None, None, -x
    else:
        xs, xe, xss, xee = None, x, -x, None
    return xs, xe, xss, xee


def boundaryInclude(ft, t, m, n):
    AffineTransform, afixed = (
        ft["AffineTransform_float_3_3_inverse"][:, 0],
        ft["fixed_inverse"][:, 0],
    )
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(AffineTransform[:9].astype(np.float64))
    affine.SetTranslation(AffineTransform[9:].astype(np.float64))
    affine.SetCenter(afixed.astype(np.float64))
    z = copy.deepcopy(t)
    while 1:
        transformed_point = [
            affine.TransformPoint([float(z), float(j), float(k)])[0]
            for j in [0, m]
            for k in [0, n]
        ]
        zz = min(transformed_point)
        if zz > t + 1:
            break
        z += 1
    return z
