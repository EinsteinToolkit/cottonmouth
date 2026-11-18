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
        vreal x50 = access(gtDD22, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 7; [gtDD22, x36, x37, x38, x39, x50, x89]
        vreal x89 = access(gtDD00, stencil_idx_0_0_0_VVV); // x89: Dependency! Liveness = 7; [gtDD00, x36, x37, x38, x39, x50, x89]
        vreal x38 = access(gtDD02, stencil_idx_0_0_0_VVV); // x38: Dependency! Liveness = 7; [gtDD02, x36, x37, x38, x39, x50, x89]
        vreal x169 = pow2(x38); // x169: Dependency! Liveness = 3; [x36, x37, x38]
        vreal x170 = ((-(x169)) + (x50 * x89)); // x170: Dependency! Liveness = 7; [x169, x36, x37, x38, x39, x50, x89]
        vreal x1000 = access(AtDD11, stencil_idx_0_0_0_VVV); // x1000: Dependency! Liveness = 7; [AtDD11, x36, x37, x38, x39, x50, x89]
        vreal x1007 = (x1000 * x170); // x1007: Dependency! Liveness = 11; [x1000, x170, x36, x37, x38, x39, x50, x89, x990, x992, x997]
        vreal x39 = access(gtDD11, stencil_idx_0_0_0_VVV); // x39: Dependency! Liveness = 7; [gtDD11, x36, x37, x38, x39, x50, x89]
        vreal x37 = access(gtDD12, stencil_idx_0_0_0_VVV); // x37: Dependency! Liveness = 7; [gtDD12, x36, x37, x38, x39, x50, x89]
        vreal x61 = pow2(x37); // x61: Dependency! Liveness = 3; [x36, x37, x38]
        vreal x62 = ((-(x61)) + (x39 * x50)); // x62: Dependency! Liveness = 7; [x36, x37, x38, x39, x50, x61, x89]
        vreal x997 = access(AtDD00, stencil_idx_0_0_0_VVV); // x997: Dependency! Liveness = 7; [AtDD00, x36, x37, x38, x39, x50, x89]
        vreal x1012 = (x62 * x997); // x1012: Dependency! Liveness = 10; [x36, x37, x38, x39, x50, x62, x89, x990, x992, x997]
        vreal x36 = access(gtDD01, stencil_idx_0_0_0_VVV); // x36: Dependency! Liveness = 7; [gtDD01, x36, x37, x38, x39, x50, x89]
        vreal x139 = pow2(x36); // x139: Dependency! Liveness = 3; [x36, x37, x38]
        vreal x140 = ((-(x139)) + (x39 * x89)); // x140: Dependency! Liveness = 7; [x139, x36, x37, x38, x39, x50, x89]
        vreal x990 = access(AtDD22, stencil_idx_0_0_0_VVV); // x990: Dependency! Liveness = 7; [AtDD22, x36, x37, x38, x39, x50, x89]
        vreal x991 = (x140 * x990); // x991: Dependency! Liveness = 9; [x140, x36, x37, x38, x39, x50, x89, x990, x992]
        vreal x992 = access(AtDD02, stencil_idx_0_0_0_VVV); // x992: Dependency! Liveness = 7; [AtDD02, x36, x37, x38, x39, x50, x89]
        vreal x40 = ((x36 * x37) + (-1 * x38 * x39)); // x40: Dependency! Liveness = 6; [x36, x37, x38, x39, x50, x89]
        vreal x993 = (x40 * x992); // x993: Dependency! Liveness = 8; [x36, x37, x38, x39, x40, x50, x89, x992]
        vreal x994 = access(AtDD12, stencil_idx_0_0_0_VVV); // x994: Dependency! Liveness = 7; [AtDD12, x36, x37, x38, x39, x50, x89]
        vreal x998 = access(AtDD01, stencil_idx_0_0_0_VVV); // x998: Dependency! Liveness = 7; [AtDD01, x36, x37, x38, x39, x50, x89]
        vreal x51 = ((x36 * x50) + (-1 * x37 * x38)); // x51: Dependency! Liveness = 6; [x36, x37, x38, x39, x50, x89]
        vreal x90 = ((x37 * x89) + (-1 * x36 * x38)); // x90: Dependency! Liveness = 6; [x36, x37, x38, x39, x50, x89]
        vreal x1110 = ((1.0 / 3.0) * x89); // x1110: Dependency! Liveness = 6; [x36, x37, x38, x39, x50, x89]
        store(AtDD00, stencil_idx_0_0_0_VVV, (x997 + (-1 * x1110 * (x1007 + x1012 + x991)) + ((2.0 / 3.0) * x89 * ((-(x993)) + (x51 * x998) + (x90 * x994)))));
        vreal x1111 = ((1.0 / 3.0) * x36); // x1111: Dependency! Liveness = 3; [x36, x37, x38]
        store(AtDD01, stencil_idx_0_0_0_VVV, (x998 + (-1 * x1111 * (x1007 + x1012 + x991)) + ((2.0 / 3.0) * x36 * ((-(x993)) + (x51 * x998) + (x90 * x994)))));
        vreal x1112 = ((1.0 / 3.0) * x39); // x1112: Dependency! Liveness = 5; [x36, x37, x38, x39, x50]
        store(AtDD11, stencil_idx_0_0_0_VVV, (x1000 + (-1 * x1112 * (x1007 + x1012 + x991)) + ((2.0 / 3.0) * x39 * ((-(x993)) + (x51 * x998) + (x90 * x994)))));
        vreal x1113 = ((1.0 / 3.0) * x38); // x1113: Dependency! Liveness = 2; [x37, x38]
        store(AtDD02, stencil_idx_0_0_0_VVV, (x992 + (-1 * x1113 * (x1007 + x1012 + x991)) + ((2.0 / 3.0) * x38 * ((-(x993)) + (x51 * x998) + (x90 * x994)))));
        vreal x1114 = ((1.0 / 3.0) * x37); // x1114: Dependency! Liveness = 1; [x37]
        store(AtDD12, stencil_idx_0_0_0_VVV, (x994 + (-1 * x1114 * (x1007 + x1012 + x991)) + ((2.0 / 3.0) * x37 * ((-(x993)) + (x51 * x998) + (x90 * x994)))));
        vreal x1115 = ((1.0 / 3.0) * x50); // x1115: Dependency! Liveness = 4; [x36, x37, x38, x50]
        store(AtDD22, stencil_idx_0_0_0_VVV, (x990 + (-1 * x1115 * (x1007 + x1012 + x991)) + ((2.0 / 3.0) * x50 * ((-(x993)) + (x51 * x998) + (x90 * x994)))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}