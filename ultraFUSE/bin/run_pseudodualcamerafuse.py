#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This sample script will get deployed in the bin directory of the
users' virtualenv when the parent module is installed using pip.
"""

import argparse
import logging
import sys
import traceback

from pathlib import Path
from ultraFUSE import pseudoDualCameraFUSE, get_module_version

###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"
)

###############################################################################


class Args(argparse.Namespace):

    DEFAULT_FIRST = 10
    DEFAULT_SECOND = 20

    def __init__(self):
        # Arguments that could be passed in through the command line
        self.first = self.DEFAULT_FIRST
        self.second = self.DEFAULT_SECOND
        self.debug = False
        #
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(
            prog="run_pseudodualcamerafuse",
            description="run pseudodualcamerafuse for LSFM images",
        )

        p.add_argument(
            "-v",
            "--version",
            action="version",
            version="%(prog)s " + get_module_version(),
        )

        p.add_argument(
            "--require_precropping",
            action="store",
            dest="require_precropping",
            default=True,
            type=bool,
        )

        p.add_argument(
            "--precropping_params",
            action="store",
            dest="precropping_params",
            default=[],
            type=list,
        )

        p.add_argument(
            "--resampleRatio",
            action="store",
            dest="resampleRatio",
            default=2,
            type=int,
        )

        p.add_argument(
            "--Lambda",
            action="store",
            dest="Lambda",
            default=0.1,
            type=float,
        )

        p.add_argument(
            "--window_size",
            action="store",
            dest="window_size",
            default=[5, 59],
            type=list,
        )

        p.add_argument(
            "--poly_order",
            action="store",
            dest="poly_order",
            default=[3, 3],
            type=list,
        )

        p.add_argument(
            "--n_epochs",
            action="store",
            dest="n_epochs",
            default=150,
            type=int,
        )

        p.add_argument(
            "--Gaussian_kernel_size",
            action="store",
            dest="Gaussian_kernel_size",
            default=49,
            type=int,
        )

        p.add_argument(
            "--GF_kernel_size",
            action="store",
            dest="GF_kernel_size",
            default=29,
            type=int,
        )

        p.add_argument(
            "--require_segmentation",
            action="store",
            dest="require_segmentation",
            default=True,
            type=bool,
        )

        p.add_argument(
            "--allow_break",
            action="store",
            dest="allow_break",
            default=False,
            type=bool,
        )

        p.add_argument(
            "--fast_mode",
            action="store",
            dest="fast_mode",
            default=False,
            type=bool,
        )

        p.add_argument(
            "--require_log",
            action="store",
            dest="require_log",
            default=True,
            type=bool,
        )

        p.add_argument(
            "--skip_illuFusion",
            action="store",
            dest="skip_illuFusion",
            default=True,
            type=bool,
        )

        p.add_argument(
            "--destripe_preceded",
            action="store",
            dest="destripe_preceded",
            default=False,
            type=bool,
        )

        p.add_argument(
            "--destripe_params",
            action="store",
            dest="destripe_params",
            default=None,
            type=dict,
        )

        p.add_argument(
            "--device",
            action="store",
            dest="device",
            default="cuda",
            type=str,
        )

        p.add_argument(
            "--xy_spacing",
            action="store",
            dest="xy_spacing",
            type=float,
        )

        p.add_argument(
            "--z_spacing",
            action="store",
            dest="z_spacing",
            type=float,
        )

        p.add_argument(
            "--data_path",
            action="store",
            dest="data_path",
            type=str,
        )

        p.add_argument(
            "--sample_name",
            action="store",
            dest="sample_name",
            type=str,
        )

        p.add_argument(
            "--ventral_angle_in_degree",
            action="store",
            dest="ventral_angle_in_degree",
            default=None,
            type=float,
        )

        p.add_argument(
            "--dorsal_angle_in_degree",
            action="store",
            dest="dorsal_angle_in_degree",
            default=None,
            type=float,
        )

        p.add_argument(
            "--top_illu_ventral_det_data",
            action="store",
            dest="top_illu_ventral_det_data",
            default=None,
            type=str,
        )

        p.add_argument(
            "--bottom_illu_ventral_det_data",
            action="store",
            dest="bottom_illu_ventral_det_data",
            default=None,
            type=str,
        )

        p.add_argument(
            "--top_illu_dorsal_det_data",
            action="store",
            dest="top_illu_dorsal_det_data",
            default=None,
            type=str,
        )
        p.add_argument(
            "--bottom_illu_dorsal_det_data",
            action="store",
            dest="bottom_illu_dorsal_det_data",
            default=None,
            type=str,
        )

        p.add_argument(
            "--left_illu_ventral_det_data",
            action="store",
            dest="left_illu_ventral_det_data",
            default=None,
            type=str,
        )

        p.add_argument(
            "--right_illu_ventral_det_data",
            action="store",
            dest="right_illu_ventral_det_data",
            default=None,
            type=str,
        )

        p.add_argument(
            "--left_illu_dorsal_det_data",
            action="store",
            dest="left_illu_dorsal_det_data",
            default=None,
            type=str,
        )
        p.add_argument(
            "--right_illu_dorsal_det_data",
            action="store",
            dest="right_illu_dorsal_det_data",
            default=None,
            type=str,
        )

        p.add_argument(
            "--save_path",
            action="store",
            dest="save_path",
            type=str,
        )
        p.add_argument(
            "--save_folder",
            action="store",
            dest="save_folder",
            type=str,
        )

        p.add_argument(
            "--debug",
            action="store_true",
            dest="debug",
            help=argparse.SUPPRESS,
        )
        p.parse_args(namespace=self)


###############################################################################


def main():
    try:
        args = Args()
        dbg = args.debug

        exe = pseudoDualCameraFUSE(
            args.require_precropping,
            args.precropping_params,
            args.resampleRatio,
            args.Lambda,
            args.window_size,
            args.poly_order,
            args.n_epochs,
            args.Gaussian_kernel_size,
            args.GF_kernel_size,
            args.require_segmentation,
            args.allow_break,
            args.fast_mode,
            args.require_log,
            args.skip_illuFusion,
            args.destripe_preceded,
            args.destripe_params,
            args.device,
        )
        out = exe.train(
            args.xy_spacing,
            args.z_spacing,
            args.data_path,
            args.sample_name,
            args.ventral_angle_in_degree,
            args.dorsal_angle_in_degree,
            args.top_illu_ventral_det_data,
            args.bottom_illu_ventral_det_data,
            args.top_illu_dorsal_det_data,
            args.bottom_illu_dorsal_det_data,
            args.left_illu_ventral_det_data,
            args.right_illu_ventral_det_data,
            args.left_illu_dorsal_det_data,
            args.right_illu_dorsal_det_data,
            args.save_path,
            args.save_folder,
        )

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
