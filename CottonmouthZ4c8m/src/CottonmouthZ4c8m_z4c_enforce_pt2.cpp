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
    const auto DX = CCTK_DELTA_SPACE(0);
    const auto DY = CCTK_DELTA_SPACE(1);
    const auto DZ = CCTK_DELTA_SPACE(2);
    const auto DT = cctk_delta_time;
    const auto DXI = (1.0 / DX);
    const auto DYI = (1.0 / DY);
    const auto DZI = (1.0 / DZ);
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // z4c_enforce_pt2 loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x39 = access(gtDD11, stencil_idx_0_0_0_VVV); // x39: Dependency! Liveness = 7; [gtDD11, x36, x37, x38, x39, x52, x99]
        vreal x99 = access(gtDD00, stencil_idx_0_0_0_VVV); // x99: Dependency! Liveness = 7; [gtDD00, x36, x37, x38, x39, x52, x99]
        vreal x36 = access(gtDD01, stencil_idx_0_0_0_VVV); // x36: Dependency! Liveness = 7; [gtDD01, x36, x37, x38, x39, x52, x99]
        vreal x157 = pow2(x36); // x157: Dependency! Liveness = 3; [x36, x37, x38]
        vreal x158 = ((-(x157)) + (x39 * x99)); // x158: Dependency! Liveness = 7; [x157, x36, x37, x38, x39, x52, x99]
        vreal x1164 = access(AtDD22, stencil_idx_0_0_0_VVV); // x1164: Dependency! Liveness = 7; [AtDD22, x36, x37, x38, x39, x52, x99]
        vreal x1165 = (x1164 * x158); // x1165: Dependency! Liveness = 11; [x1164, x1166, x1171, x1174, x158, x36, x37, x38, x39, x52, x99]
        vreal x1166 = access(AtDD02, stencil_idx_0_0_0_VVV); // x1166: Dependency! Liveness = 7; [AtDD02, x36, x37, x38, x39, x52, x99]
        vreal x37 = access(gtDD12, stencil_idx_0_0_0_VVV); // x37: Dependency! Liveness = 7; [gtDD12, x36, x37, x38, x39, x52, x99]
        vreal x38 = access(gtDD02, stencil_idx_0_0_0_VVV); // x38: Dependency! Liveness = 7; [gtDD02, x36, x37, x38, x39, x52, x99]
        vreal x40 = ((x36 * x37) + (-1 * x38 * x39)); // x40: Dependency! Liveness = 6; [x36, x37, x38, x39, x52, x99]
        vreal x1167 = (x1166 * x40); // x1167: Dependency! Liveness = 10; [x1166, x1171, x1174, x36, x37, x38, x39, x40, x52, x99]
        vreal x52 = access(gtDD22, stencil_idx_0_0_0_VVV); // x52: Dependency! Liveness = 7; [gtDD22, x36, x37, x38, x39, x52, x99]
        vreal x193 = pow2(x38); // x193: Dependency! Liveness = 3; [x36, x37, x38]
        vreal x194 = ((-(x193)) + (x52 * x99)); // x194: Dependency! Liveness = 7; [x193, x36, x37, x38, x39, x52, x99]
        vreal x1174 = access(AtDD11, stencil_idx_0_0_0_VVV); // x1174: Dependency! Liveness = 7; [AtDD11, x36, x37, x38, x39, x52, x99]
        vreal x1181 = (x1174 * x194); // x1181: Dependency! Liveness = 9; [x1171, x1174, x194, x36, x37, x38, x39, x52, x99]
        vreal x65 = pow2(x37); // x65: Dependency! Liveness = 3; [x36, x37, x38]
        vreal x66 = ((-(x65)) + (x39 * x52)); // x66: Dependency! Liveness = 7; [x36, x37, x38, x39, x52, x65, x99]
        vreal x1171 = access(AtDD00, stencil_idx_0_0_0_VVV); // x1171: Dependency! Liveness = 7; [AtDD00, x36, x37, x38, x39, x52, x99]
        vreal x1186 = (x1171 * x66); // x1186: Dependency! Liveness = 8; [x1171, x36, x37, x38, x39, x52, x66, x99]
        vreal x1168 = access(AtDD12, stencil_idx_0_0_0_VVV); // x1168: Dependency! Liveness = 7; [AtDD12, x36, x37, x38, x39, x52, x99]
        vreal x1172 = access(AtDD01, stencil_idx_0_0_0_VVV); // x1172: Dependency! Liveness = 7; [AtDD01, x36, x37, x38, x39, x52, x99]
        vreal x100 = ((x37 * x99) + (-1 * x36 * x38)); // x100: Dependency! Liveness = 6; [x36, x37, x38, x39, x52, x99]
        vreal x53 = ((x36 * x52) + (-1 * x37 * x38)); // x53: Dependency! Liveness = 6; [x36, x37, x38, x39, x52, x99]
        vreal x1293 = ((1.0 / 3.0) * x99); // x1293: Dependency! Liveness = 6; [x36, x37, x38, x39, x52, x99]
        store(AtDD00, stencil_idx_0_0_0_VVV, (x1171 + (-1 * x1293 * (x1165 + x1181 + x1186)) + ((2.0 / 3.0) * x99 * ((-(x1167)) + (x100 * x1168) + (x1172 * x53)))));
        vreal x1294 = ((1.0 / 3.0) * x36); // x1294: Dependency! Liveness = 3; [x36, x37, x38]
        store(AtDD01, stencil_idx_0_0_0_VVV, (x1172 + (-1 * x1294 * (x1165 + x1181 + x1186)) + ((2.0 / 3.0) * x36 * ((-(x1167)) + (x100 * x1168) + (x1172 * x53)))));
        vreal x1295 = ((1.0 / 3.0) * x39); // x1295: Dependency! Liveness = 5; [x36, x37, x38, x39, x52]
        store(AtDD11, stencil_idx_0_0_0_VVV, (x1174 + (-1 * x1295 * (x1165 + x1181 + x1186)) + ((2.0 / 3.0) * x39 * ((-(x1167)) + (x100 * x1168) + (x1172 * x53)))));
        vreal x1296 = ((1.0 / 3.0) * x38); // x1296: Dependency! Liveness = 2; [x37, x38]
        store(AtDD02, stencil_idx_0_0_0_VVV, (x1166 + (-1 * x1296 * (x1165 + x1181 + x1186)) + ((2.0 / 3.0) * x38 * ((-(x1167)) + (x100 * x1168) + (x1172 * x53)))));
        vreal x1297 = ((1.0 / 3.0) * x37); // x1297: Dependency! Liveness = 1; [x37]
        store(AtDD12, stencil_idx_0_0_0_VVV, (x1168 + (-1 * x1297 * (x1165 + x1181 + x1186)) + ((2.0 / 3.0) * x37 * ((-(x1167)) + (x100 * x1168) + (x1172 * x53)))));
        vreal x1298 = ((1.0 / 3.0) * x52); // x1298: Dependency! Liveness = 4; [x36, x37, x38, x52]
        store(AtDD22, stencil_idx_0_0_0_VVV, (x1164 + (-1 * x1298 * (x1165 + x1181 + x1186)) + ((2.0 / 3.0) * x52 * ((-(x1167)) + (x100 * x1168) + (x1172 * x53)))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}