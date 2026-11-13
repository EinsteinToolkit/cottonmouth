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
        vreal x129 = access(gtDD11, stencil_idx_0_0_0_VVV); // x129: Dependency! Liveness = 7; [gtDD11, x129, x48, x49, x50, x51, x63]
        vreal x48 = access(gtDD00, stencil_idx_0_0_0_VVV); // x48: Dependency! Liveness = 7; [gtDD00, x129, x48, x49, x50, x51, x63]
        vreal x50 = access(gtDD01, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 7; [gtDD01, x129, x48, x49, x50, x51, x63]
        vreal x155 = pow2(x50); // x155: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x156 = ((-(x155)) + (x129 * x48)); // x156: Dependency! Liveness = 7; [x129, x155, x48, x49, x50, x51, x63]
        vreal x908 = access(AtDD22, stencil_idx_0_0_0_VVV); // x908: Dependency! Liveness = 7; [AtDD22, x129, x48, x49, x50, x51, x63]
        vreal x1092 = (x156 * x908); // x1092: Dependency! Liveness = 11; [x129, x156, x48, x49, x50, x51, x63, x901, x905, x906, x908]
        vreal x905 = access(AtDD02, stencil_idx_0_0_0_VVV); // x905: Dependency! Liveness = 7; [AtDD02, x129, x48, x49, x50, x51, x63]
        vreal x49 = access(gtDD12, stencil_idx_0_0_0_VVV); // x49: Dependency! Liveness = 7; [gtDD12, x129, x48, x49, x50, x51, x63]
        vreal x51 = access(gtDD02, stencil_idx_0_0_0_VVV); // x51: Dependency! Liveness = 7; [gtDD02, x129, x48, x49, x50, x51, x63]
        vreal x130 = ((x49 * x50) + (-1 * x129 * x51)); // x130: Dependency! Liveness = 6; [x129, x48, x49, x50, x51, x63]
        vreal x1093 = (x130 * x905); // x1093: Dependency! Liveness = 10; [x129, x130, x48, x49, x50, x51, x63, x901, x905, x906]
        vreal x63 = access(gtDD22, stencil_idx_0_0_0_VVV); // x63: Dependency! Liveness = 7; [gtDD22, x129, x48, x49, x50, x51, x63]
        vreal x64 = pow2(x51); // x64: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x65 = ((-(x64)) + (x48 * x63)); // x65: Dependency! Liveness = 7; [x129, x48, x49, x50, x51, x63, x64]
        vreal x906 = access(AtDD11, stencil_idx_0_0_0_VVV); // x906: Dependency! Liveness = 7; [AtDD11, x129, x48, x49, x50, x51, x63]
        vreal x1098 = (x65 * x906); // x1098: Dependency! Liveness = 9; [x129, x48, x49, x50, x51, x63, x65, x901, x906]
        vreal x224 = pow2(x49); // x224: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x225 = ((-(x224)) + (x129 * x63)); // x225: Dependency! Liveness = 7; [x129, x224, x48, x49, x50, x51, x63]
        vreal x901 = access(AtDD00, stencil_idx_0_0_0_VVV); // x901: Dependency! Liveness = 7; [AtDD00, x129, x48, x49, x50, x51, x63]
        vreal x1101 = (x225 * x901); // x1101: Dependency! Liveness = 8; [x129, x225, x48, x49, x50, x51, x63, x901]
        vreal x904 = access(AtDD01, stencil_idx_0_0_0_VVV); // x904: Dependency! Liveness = 7; [AtDD01, x129, x48, x49, x50, x51, x63]
        vreal x907 = access(AtDD12, stencil_idx_0_0_0_VVV); // x907: Dependency! Liveness = 7; [AtDD12, x129, x48, x49, x50, x51, x63]
        vreal x52 = ((x48 * x49) + (-1 * x50 * x51)); // x52: Dependency! Liveness = 6; [x129, x48, x49, x50, x51, x63]
        vreal x76 = ((x50 * x63) + (-1 * x49 * x51)); // x76: Dependency! Liveness = 6; [x129, x48, x49, x50, x51, x63]
        vreal x1199 = ((1.0 / 3.0) * x48); // x1199: Dependency! Liveness = 6; [x129, x48, x49, x50, x51, x63]
        vreal x1200 = ((2.0 / 3.0) * x48); // x1200: Dependency! Liveness = 6; [x129, x48, x49, x50, x51, x63]
        store(AtDD00, stencil_idx_0_0_0_VVV, (x901 + (-1 * x1093 * x1200) + (-1 * x1199 * (x1092 + x1098 + x1101)) + ((2.0 / 3.0) * x48 * ((x52 * x907) + (x76 * x904)))));
        vreal x1205 = ((1.0 / 3.0) * x129); // x1205: Dependency! Liveness = 5; [x129, x49, x50, x51, x63]
        vreal x1206 = ((2.0 / 3.0) * x129); // x1206: Dependency! Liveness = 5; [x129, x49, x50, x51, x63]
        store(AtDD11, stencil_idx_0_0_0_VVV, (x906 + (-1 * x1093 * x1206) + (-1 * x1205 * (x1092 + x1098 + x1101)) + ((2.0 / 3.0) * x129 * ((x52 * x907) + (x76 * x904)))));
        vreal x1209 = ((1.0 / 3.0) * x63); // x1209: Dependency! Liveness = 4; [x49, x50, x51, x63]
        vreal x1210 = ((2.0 / 3.0) * x63); // x1210: Dependency! Liveness = 4; [x49, x50, x51, x63]
        store(AtDD22, stencil_idx_0_0_0_VVV, (x908 + (-1 * x1093 * x1210) + (-1 * x1209 * (x1092 + x1098 + x1101)) + ((2.0 / 3.0) * x63 * ((x52 * x907) + (x76 * x904)))));
        vreal x1203 = ((1.0 / 3.0) * x51); // x1203: Dependency! Liveness = 2; [x49, x51]
        vreal x1204 = ((2.0 / 3.0) * x51); // x1204: Dependency! Liveness = 2; [x49, x51]
        store(AtDD02, stencil_idx_0_0_0_VVV, (x905 + (-1 * x1093 * x1204) + (-1 * x1203 * (x1092 + x1098 + x1101)) + ((2.0 / 3.0) * x51 * ((x52 * x907) + (x76 * x904)))));
        vreal x1201 = ((1.0 / 3.0) * x50); // x1201: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x1202 = ((2.0 / 3.0) * x50); // x1202: Dependency! Liveness = 3; [x49, x50, x51]
        store(AtDD01, stencil_idx_0_0_0_VVV, (x904 + (-1 * x1093 * x1202) + (-1 * x1201 * (x1092 + x1098 + x1101)) + ((2.0 / 3.0) * x50 * ((x52 * x907) + (x76 * x904)))));
        vreal x1207 = ((1.0 / 3.0) * x49); // x1207: Dependency! Liveness = 1; [x49]
        vreal x1208 = ((2.0 / 3.0) * x49); // x1208: Dependency! Liveness = 1; [x49]
        store(AtDD12, stencil_idx_0_0_0_VVV, (x907 + (-1 * x1093 * x1208) + (-1 * x1207 * (x1092 + x1098 + x1101)) + ((2.0 / 3.0) * x49 * ((x52 * x907) + (x76 * x904)))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}