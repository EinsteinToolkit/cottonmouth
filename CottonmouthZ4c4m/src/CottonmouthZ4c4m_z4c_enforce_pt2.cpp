// Copyright (C) 2026 Lucas Timotheo Sanches, Max Morris, and Steven R. Brandt.
// This file is part of Cottonmouth, a suite of astrophysics codes for the Einstein Toolkit.
// Cottonmouth was created with the Einstein Engine.
// 
// Cottonmouth is hosted at: https://github.com/EinsteinToolkit/cottonmouth
// The Einstein Engine is hosted at: https://github.com/max-morris/EinsteinEngine
// 
// Cottonmouth is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// Cottonmouth is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.


#define CARPETX_GF3D5
#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>
#include <simd.hxx>
#include <defs.hxx>
#include <vect.hxx>
#include <cmath>
#include <tuple>
#include "../../../CarpetX/CarpetX/src/timer.hxx"
#ifdef __CUDACC__
#include <nvtx3/nvToolsExt.h>
#endif
#define access(GF, IDX) (GF(IDX))
#define store(GF, IDX, VAL) (GF.store(IDX, VAL))
#define stencil(GF, IDX) (GF(IDX))
#define CCTK_ASSERT(X) if(!(X)) { CCTK_Error(__LINE__, __FILE__, CCTK_THORNSTRING, "Assertion Failure: " #X); }
using namespace Arith;
using namespace Loop;
using std::cbrt,std::max,std::min,std::sqrt;
void z4c_enforce_pt2(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_z4c_enforce_pt2;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("z4c_enforce_pt2");
    #endif
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define AtDD00_layout VVV_layout
    #define AtDD01_layout VVV_layout
    #define AtDD02_layout VVV_layout
    #define AtDD11_layout VVV_layout
    #define AtDD12_layout VVV_layout
    #define AtDD22_layout VVV_layout
    #define gtDD00_layout VVV_layout
    #define gtDD01_layout VVV_layout
    #define gtDD02_layout VVV_layout
    #define gtDD11_layout VVV_layout
    #define gtDD12_layout VVV_layout
    #define gtDD22_layout VVV_layout
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // z4c_enforce_pt2 loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x38 = access(gtDD00, stencil_idx_0_0_0_VVV); // x38: Dependency! Liveness = 6; [gtDD00, x38, x41, x42, x43, x58]
        vreal x55 = access(gtDD22, stencil_idx_0_0_0_VVV); // x55: Dependency! Liveness = 9; [gtDD22, x38, x39, x41, x42, x43, x46, x55, x58]
        vreal x118 = (x38 * x55); // x118: Dependency! Liveness = 6; [x38, x39, x41, x42, x46, x55]
        vreal x42 = access(gtDD02, stencil_idx_0_0_0_VVV); // x42: Dependency! Liveness = 7; [gtDD02, x38, x39, x41, x42, x43, x58]
        vreal x119 = pow2(x42); // x119: Dependency! Liveness = 3; [x39, x41, x42]
        vreal x195 = (x118 + (-(x119))); // x195: Dependency! Liveness = 9; [x118, x119, x195, x830, x831, x832, x840, x841, x845]
        x118 = access(AtDD11, stencil_idx_0_0_0_VVV); // x832: Dependency! Liveness = 9; [AtDD11, x38, x39, x41, x42, x43, x46, x55, x58]
        x119 = (x118 * x195); // x836: Dependency! Liveness = 11; [x195, x209, x48, x58, x69, x830, x831, x832, x840, x841, x845]
        x195 = access(gtDD11, stencil_idx_0_0_0_VVV); // x46: Dependency! Liveness = 8; [gtDD11, x38, x39, x41, x42, x43, x46, x58]
        vreal x47 = (x195 * x42); // x47: Dependency! Liveness = 6; [x38, x39, x41, x42, x46, x55]
        vreal x39 = access(gtDD12, stencil_idx_0_0_0_VVV); // x39: Dependency! Liveness = 7; [gtDD12, x38, x39, x41, x42, x43, x58]
        vreal x41 = access(gtDD01, stencil_idx_0_0_0_VVV); // x41: Dependency! Liveness = 7; [gtDD01, x38, x39, x41, x42, x43, x58]
        vreal x45 = (x39 * x41); // x45: Dependency! Liveness = 6; [x38, x39, x41, x42, x46, x55]
        vreal x48 = (x45 + (-(x47))); // x48: Dependency! Liveness = 11; [x195, x209, x45, x47, x48, x830, x831, x832, x840, x841, x845]
        x45 = access(AtDD02, stencil_idx_0_0_0_VVV); // x841: Dependency! Liveness = 9; [AtDD02, x38, x39, x41, x42, x43, x46, x55, x58]
        x47 = (x45 * x48); // x855: Dependency! Liveness = 9; [x209, x48, x58, x69, x830, x831, x840, x841, x845]
        x48 = (x195 * x55); // x67: Dependency! Liveness = 6; [x38, x39, x41, x42, x46, x55]
        vreal x68 = pow2(x39); // x68: Dependency! Liveness = 3; [x39, x41, x42]
        vreal x69 = (x48 + (-(x68))); // x69: Dependency! Liveness = 13; [x195, x209, x48, x58, x67, x68, x69, x830, x831, x832, x840, x841, x845]
        x68 = access(AtDD00, stencil_idx_0_0_0_VVV); // x845: Dependency! Liveness = 9; [AtDD00, x38, x39, x41, x42, x43, x46, x55, x58]
        vreal x856 = (x68 * x69); // x856: Dependency! Liveness = 7; [x209, x58, x69, x830, x831, x840, x845]
        x69 = (x195 * x38); // x144: Dependency! Liveness = 6; [x38, x39, x41, x42, x46, x55]
        vreal x145 = pow2(x41); // x145: Dependency! Liveness = 3; [x39, x41, x42]
        vreal x209 = (x69 + (-(x145))); // x209: Dependency! Liveness = 10; [x144, x145, x195, x209, x830, x831, x832, x840, x841, x845]
        x145 = access(AtDD22, stencil_idx_0_0_0_VVV); // x840: Dependency! Liveness = 9; [AtDD22, x38, x39, x41, x42, x43, x46, x55, x58]
        vreal x875 = (x145 * x209); // x875: Dependency! Liveness = 5; [x209, x58, x830, x831, x840]
        x209 = access(AtDD12, stencil_idx_0_0_0_VVV); // x830: Dependency! Liveness = 9; [AtDD12, x38, x39, x41, x42, x43, x46, x55, x58]
        vreal x40 = (x38 * x39); // x40: Dependency! Liveness = 6; [x38, x39, x41, x42, x46, x55]
        vreal x43 = (x40 + (-1 * x41 * x42)); // x43: Dependency! Liveness = 5; [x40, x41, x42, x43, x58]
        x40 = (-(x43)); // x44: Dependency! Liveness = 8; [x38, x39, x41, x42, x43, x46, x55, x58]
        x43 = (x209 * x40); // x991: Dependency! Liveness = 4; [x44, x58, x830, x831]
        vreal x831 = access(AtDD01, stencil_idx_0_0_0_VVV); // x831: Dependency! Liveness = 9; [AtDD01, x38, x39, x41, x42, x43, x46, x55, x58]
        vreal x56 = (x41 * x55); // x56: Dependency! Liveness = 6; [x38, x39, x41, x42, x46, x55]
        vreal x57 = (x39 * x42); // x57: Dependency! Liveness = 6; [x38, x39, x41, x42, x46, x55]
        vreal x58 = (x56 + (-(x57))); // x58: Dependency! Liveness = 12; [x195, x209, x48, x56, x57, x58, x830, x831, x832, x840, x841, x845]
        x56 = (-(x58)); // x59: Dependency! Liveness = 7; [x38, x39, x41, x42, x46, x55, x58]
        x58 = (x56 * x831); // x992: Dependency! Liveness = 3; [x58, x59, x831]
        x57 = ((1.0 / 3.0) * x38); // x989: Dependency! Liveness = 5; [x38, x39, x41, x42, x46]
        vreal x990 = ((2.0 / 3.0) * x38); // x990: Dependency! Liveness = 5; [x38, x39, x41, x42, x46]
        store(AtDD00, stencil_idx_0_0_0_VVV, (x68 + (-1 * x119 * x57) + (-1 * x43 * x990) + (-1 * x47 * x990) + (-1 * x57 * x856) + (-1 * x57 * x875) + (-1 * x58 * x990)));
        x990 = ((1.0 / 3.0) * x41); // x993: Dependency! Liveness = 3; [x39, x41, x42]
        x38 = ((2.0 / 3.0) * x41); // x994: Dependency! Liveness = 3; [x39, x41, x42]
        store(AtDD01, stencil_idx_0_0_0_VVV, (x831 + (-1 * x119 * x990) + (-1 * x38 * x43) + (-1 * x38 * x47) + (-1 * x38 * x58) + (-1 * x856 * x990) + (-1 * x875 * x990)));
        x831 = ((1.0 / 3.0) * x42); // x995: Dependency! Liveness = 2; [x39, x42]
        x41 = ((2.0 / 3.0) * x42); // x996: Dependency! Liveness = 2; [x39, x42]
        store(AtDD02, stencil_idx_0_0_0_VVV, (x45 + (-1 * x119 * x831) + (-1 * x41 * x43) + (-1 * x41 * x47) + (-1 * x41 * x58) + (-1 * x831 * x856) + (-1 * x831 * x875)));
        x42 = ((1.0 / 3.0) * x195); // x997: Dependency! Liveness = 4; [x39, x41, x42, x46]
        vreal x998 = ((2.0 / 3.0) * x195); // x998: Dependency! Liveness = 4; [x39, x41, x42, x46]
        store(AtDD11, stencil_idx_0_0_0_VVV, (x118 + (-1 * x119 * x42) + (-1 * x42 * x856) + (-1 * x42 * x875) + (-1 * x43 * x998) + (-1 * x47 * x998) + (-1 * x58 * x998)));
        x998 = ((2.0 / 3.0) * x39); // x1000: Dependency! Liveness = 3; [x39, x41, x42]
        vreal x999 = ((1.0 / 3.0) * x39); // x999: Dependency! Liveness = 1; [x39]
        store(AtDD12, stencil_idx_0_0_0_VVV, (x209 + (-1 * x119 * x999) + (-1 * x43 * x998) + (-1 * x47 * x998) + (-1 * x58 * x998) + (-1 * x856 * x999) + (-1 * x875 * x999)));
        x999 = ((1.0 / 3.0) * x55); // x1001: Dependency! Liveness = 6; [x38, x39, x41, x42, x46, x55]
        x39 = ((2.0 / 3.0) * x55); // x1002: Dependency! Liveness = 6; [x38, x39, x41, x42, x46, x55]
        store(AtDD22, stencil_idx_0_0_0_VVV, (x145 + (-1 * x119 * x999) + (-1 * x39 * x43) + (-1 * x39 * x47) + (-1 * x39 * x58) + (-1 * x856 * x999) + (-1 * x875 * x999)));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}