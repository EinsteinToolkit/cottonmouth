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
void enforce_pt2(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_enforce_pt2;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("enforce_pt2");
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
    const auto DX = CCTK_DELTA_SPACE(0);
    const auto DY = CCTK_DELTA_SPACE(1);
    const auto DZ = CCTK_DELTA_SPACE(2);
    const auto DT = cctk_delta_time;
    const auto DXI = (1.0 / DX);
    const auto DYI = (1.0 / DY);
    const auto DZI = (1.0 / DZ);
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // enforce_pt2 loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x115 = access(gtDD11, stencil_idx_0_0_0_VVV); // x115: Dependency! Liveness = 7; [gtDD11, x115, x48, x49, x50, x51, x61]
        vreal x48 = access(gtDD00, stencil_idx_0_0_0_VVV); // x48: Dependency! Liveness = 7; [gtDD00, x115, x48, x49, x50, x51, x61]
        vreal x50 = access(gtDD01, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 7; [gtDD01, x115, x48, x49, x50, x51, x61]
        vreal x137 = pow2(x50); // x137: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x138 = ((-(x137)) + (x115 * x48)); // x138: Dependency! Liveness = 7; [x115, x137, x48, x49, x50, x51, x61]
        vreal x731 = access(AtDD22, stencil_idx_0_0_0_VVV); // x731: Dependency! Liveness = 7; [AtDD22, x115, x48, x49, x50, x51, x61]
        vreal x915 = (x138 * x731); // x915: Dependency! Liveness = 11; [x115, x138, x48, x49, x50, x51, x61, x724, x728, x729, x731]
        vreal x728 = access(AtDD02, stencil_idx_0_0_0_VVV); // x728: Dependency! Liveness = 7; [AtDD02, x115, x48, x49, x50, x51, x61]
        vreal x49 = access(gtDD12, stencil_idx_0_0_0_VVV); // x49: Dependency! Liveness = 7; [gtDD12, x115, x48, x49, x50, x51, x61]
        vreal x51 = access(gtDD02, stencil_idx_0_0_0_VVV); // x51: Dependency! Liveness = 7; [gtDD02, x115, x48, x49, x50, x51, x61]
        vreal x116 = ((x49 * x50) + (-1 * x115 * x51)); // x116: Dependency! Liveness = 6; [x115, x48, x49, x50, x51, x61]
        vreal x916 = (x116 * x728); // x916: Dependency! Liveness = 10; [x115, x116, x48, x49, x50, x51, x61, x724, x728, x729]
        vreal x61 = access(gtDD22, stencil_idx_0_0_0_VVV); // x61: Dependency! Liveness = 7; [gtDD22, x115, x48, x49, x50, x51, x61]
        vreal x62 = pow2(x51); // x62: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x63 = ((-(x62)) + (x48 * x61)); // x63: Dependency! Liveness = 7; [x115, x48, x49, x50, x51, x61, x62]
        vreal x729 = access(AtDD11, stencil_idx_0_0_0_VVV); // x729: Dependency! Liveness = 7; [AtDD11, x115, x48, x49, x50, x51, x61]
        vreal x921 = (x63 * x729); // x921: Dependency! Liveness = 9; [x115, x48, x49, x50, x51, x61, x63, x724, x729]
        vreal x194 = pow2(x49); // x194: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x195 = ((-(x194)) + (x115 * x61)); // x195: Dependency! Liveness = 7; [x115, x194, x48, x49, x50, x51, x61]
        vreal x724 = access(AtDD00, stencil_idx_0_0_0_VVV); // x724: Dependency! Liveness = 7; [AtDD00, x115, x48, x49, x50, x51, x61]
        vreal x924 = (x195 * x724); // x924: Dependency! Liveness = 8; [x115, x195, x48, x49, x50, x51, x61, x724]
        vreal x727 = access(AtDD01, stencil_idx_0_0_0_VVV); // x727: Dependency! Liveness = 7; [AtDD01, x115, x48, x49, x50, x51, x61]
        vreal x730 = access(AtDD12, stencil_idx_0_0_0_VVV); // x730: Dependency! Liveness = 7; [AtDD12, x115, x48, x49, x50, x51, x61]
        vreal x52 = ((x48 * x49) + (-1 * x50 * x51)); // x52: Dependency! Liveness = 6; [x115, x48, x49, x50, x51, x61]
        vreal x72 = ((x50 * x61) + (-1 * x49 * x51)); // x72: Dependency! Liveness = 6; [x115, x48, x49, x50, x51, x61]
        vreal x1045 = ((1.0 / 3.0) * x48); // x1045: Dependency! Liveness = 6; [x115, x48, x49, x50, x51, x61]
        vreal x1046 = ((2.0 / 3.0) * x48); // x1046: Dependency! Liveness = 6; [x115, x48, x49, x50, x51, x61]
        store(AtDD00, stencil_idx_0_0_0_VVV, (x724 + (-1 * x1045 * (x915 + x921 + x924)) + (-1 * x1046 * x916) + ((2.0 / 3.0) * x48 * ((x52 * x730) + (x72 * x727)))));
        vreal x1051 = ((1.0 / 3.0) * x115); // x1051: Dependency! Liveness = 5; [x115, x49, x50, x51, x61]
        vreal x1052 = ((2.0 / 3.0) * x115); // x1052: Dependency! Liveness = 5; [x115, x49, x50, x51, x61]
        store(AtDD11, stencil_idx_0_0_0_VVV, (x729 + (-1 * x1051 * (x915 + x921 + x924)) + (-1 * x1052 * x916) + ((2.0 / 3.0) * x115 * ((x52 * x730) + (x72 * x727)))));
        vreal x1055 = ((1.0 / 3.0) * x61); // x1055: Dependency! Liveness = 4; [x49, x50, x51, x61]
        vreal x1056 = ((2.0 / 3.0) * x61); // x1056: Dependency! Liveness = 4; [x49, x50, x51, x61]
        store(AtDD22, stencil_idx_0_0_0_VVV, (x731 + (-1 * x1055 * (x915 + x921 + x924)) + (-1 * x1056 * x916) + ((2.0 / 3.0) * x61 * ((x52 * x730) + (x72 * x727)))));
        vreal x1049 = ((1.0 / 3.0) * x51); // x1049: Dependency! Liveness = 2; [x49, x51]
        vreal x1050 = ((2.0 / 3.0) * x51); // x1050: Dependency! Liveness = 2; [x49, x51]
        store(AtDD02, stencil_idx_0_0_0_VVV, (x728 + (-1 * x1049 * (x915 + x921 + x924)) + (-1 * x1050 * x916) + ((2.0 / 3.0) * x51 * ((x52 * x730) + (x72 * x727)))));
        vreal x1047 = ((1.0 / 3.0) * x50); // x1047: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x1048 = ((2.0 / 3.0) * x50); // x1048: Dependency! Liveness = 3; [x49, x50, x51]
        store(AtDD01, stencil_idx_0_0_0_VVV, (x727 + (-1 * x1047 * (x915 + x921 + x924)) + (-1 * x1048 * x916) + ((2.0 / 3.0) * x50 * ((x52 * x730) + (x72 * x727)))));
        vreal x1053 = ((1.0 / 3.0) * x49); // x1053: Dependency! Liveness = 1; [x49]
        vreal x1054 = ((2.0 / 3.0) * x49); // x1054: Dependency! Liveness = 1; [x49]
        store(AtDD12, stencil_idx_0_0_0_VVV, (x730 + (-1 * x1053 * (x915 + x921 + x924)) + (-1 * x1054 * x916) + ((2.0 / 3.0) * x49 * ((x52 * x730) + (x72 * x727)))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}