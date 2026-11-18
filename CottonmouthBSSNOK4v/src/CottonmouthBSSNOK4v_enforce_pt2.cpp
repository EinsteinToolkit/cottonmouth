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
        vreal x101 = access(gtDD11, stencil_idx_0_0_0_VVV); // x101: Dependency! Liveness = 7; [gtDD11, x101, x48, x49, x50, x51, x59]
        vreal x48 = access(gtDD00, stencil_idx_0_0_0_VVV); // x48: Dependency! Liveness = 7; [gtDD00, x101, x48, x49, x50, x51, x59]
        vreal x50 = access(gtDD01, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 7; [gtDD01, x101, x48, x49, x50, x51, x59]
        vreal x119 = pow2(x50); // x119: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x120 = ((-(x119)) + (x101 * x48)); // x120: Dependency! Liveness = 7; [x101, x119, x48, x49, x50, x51, x59]
        vreal x554 = access(AtDD22, stencil_idx_0_0_0_VVV); // x554: Dependency! Liveness = 7; [AtDD22, x101, x48, x49, x50, x51, x59]
        vreal x738 = (x120 * x554); // x738: Dependency! Liveness = 11; [x101, x120, x48, x49, x50, x51, x547, x551, x552, x554, x59]
        vreal x551 = access(AtDD02, stencil_idx_0_0_0_VVV); // x551: Dependency! Liveness = 7; [AtDD02, x101, x48, x49, x50, x51, x59]
        vreal x49 = access(gtDD12, stencil_idx_0_0_0_VVV); // x49: Dependency! Liveness = 7; [gtDD12, x101, x48, x49, x50, x51, x59]
        vreal x51 = access(gtDD02, stencil_idx_0_0_0_VVV); // x51: Dependency! Liveness = 7; [gtDD02, x101, x48, x49, x50, x51, x59]
        vreal x102 = ((x49 * x50) + (-1 * x101 * x51)); // x102: Dependency! Liveness = 6; [x101, x48, x49, x50, x51, x59]
        vreal x739 = (x102 * x551); // x739: Dependency! Liveness = 10; [x101, x102, x48, x49, x50, x51, x547, x551, x552, x59]
        vreal x59 = access(gtDD22, stencil_idx_0_0_0_VVV); // x59: Dependency! Liveness = 7; [gtDD22, x101, x48, x49, x50, x51, x59]
        vreal x60 = pow2(x51); // x60: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x61 = ((-(x60)) + (x48 * x59)); // x61: Dependency! Liveness = 7; [x101, x48, x49, x50, x51, x59, x60]
        vreal x552 = access(AtDD11, stencil_idx_0_0_0_VVV); // x552: Dependency! Liveness = 7; [AtDD11, x101, x48, x49, x50, x51, x59]
        vreal x744 = (x552 * x61); // x744: Dependency! Liveness = 9; [x101, x48, x49, x50, x51, x547, x552, x59, x61]
        vreal x164 = pow2(x49); // x164: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x165 = ((-(x164)) + (x101 * x59)); // x165: Dependency! Liveness = 7; [x101, x164, x48, x49, x50, x51, x59]
        vreal x547 = access(AtDD00, stencil_idx_0_0_0_VVV); // x547: Dependency! Liveness = 7; [AtDD00, x101, x48, x49, x50, x51, x59]
        vreal x747 = (x165 * x547); // x747: Dependency! Liveness = 8; [x101, x165, x48, x49, x50, x51, x547, x59]
        vreal x550 = access(AtDD01, stencil_idx_0_0_0_VVV); // x550: Dependency! Liveness = 7; [AtDD01, x101, x48, x49, x50, x51, x59]
        vreal x553 = access(AtDD12, stencil_idx_0_0_0_VVV); // x553: Dependency! Liveness = 7; [AtDD12, x101, x48, x49, x50, x51, x59]
        vreal x52 = ((x48 * x49) + (-1 * x50 * x51)); // x52: Dependency! Liveness = 6; [x101, x48, x49, x50, x51, x59]
        vreal x68 = ((x50 * x59) + (-1 * x49 * x51)); // x68: Dependency! Liveness = 6; [x101, x48, x49, x50, x51, x59]
        vreal x845 = ((1.0 / 3.0) * x48); // x845: Dependency! Liveness = 6; [x101, x48, x49, x50, x51, x59]
        vreal x846 = ((2.0 / 3.0) * x48); // x846: Dependency! Liveness = 6; [x101, x48, x49, x50, x51, x59]
        store(AtDD00, stencil_idx_0_0_0_VVV, (x547 + (-1 * x739 * x846) + (-1 * x845 * (x738 + x744 + x747)) + ((2.0 / 3.0) * x48 * ((x52 * x553) + (x550 * x68)))));
        vreal x851 = ((1.0 / 3.0) * x101); // x851: Dependency! Liveness = 5; [x101, x49, x50, x51, x59]
        vreal x852 = ((2.0 / 3.0) * x101); // x852: Dependency! Liveness = 5; [x101, x49, x50, x51, x59]
        store(AtDD11, stencil_idx_0_0_0_VVV, (x552 + (-1 * x739 * x852) + (-1 * x851 * (x738 + x744 + x747)) + ((2.0 / 3.0) * x101 * ((x52 * x553) + (x550 * x68)))));
        vreal x855 = ((1.0 / 3.0) * x59); // x855: Dependency! Liveness = 4; [x49, x50, x51, x59]
        vreal x856 = ((2.0 / 3.0) * x59); // x856: Dependency! Liveness = 4; [x49, x50, x51, x59]
        store(AtDD22, stencil_idx_0_0_0_VVV, (x554 + (-1 * x739 * x856) + (-1 * x855 * (x738 + x744 + x747)) + ((2.0 / 3.0) * x59 * ((x52 * x553) + (x550 * x68)))));
        vreal x849 = ((1.0 / 3.0) * x51); // x849: Dependency! Liveness = 2; [x49, x51]
        vreal x850 = ((2.0 / 3.0) * x51); // x850: Dependency! Liveness = 2; [x49, x51]
        store(AtDD02, stencil_idx_0_0_0_VVV, (x551 + (-1 * x739 * x850) + (-1 * x849 * (x738 + x744 + x747)) + ((2.0 / 3.0) * x51 * ((x52 * x553) + (x550 * x68)))));
        vreal x847 = ((1.0 / 3.0) * x50); // x847: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x848 = ((2.0 / 3.0) * x50); // x848: Dependency! Liveness = 3; [x49, x50, x51]
        store(AtDD01, stencil_idx_0_0_0_VVV, (x550 + (-1 * x739 * x848) + (-1 * x847 * (x738 + x744 + x747)) + ((2.0 / 3.0) * x50 * ((x52 * x553) + (x550 * x68)))));
        vreal x853 = ((1.0 / 3.0) * x49); // x853: Dependency! Liveness = 1; [x49]
        vreal x854 = ((2.0 / 3.0) * x49); // x854: Dependency! Liveness = 1; [x49]
        store(AtDD12, stencil_idx_0_0_0_VVV, (x553 + (-1 * x739 * x854) + (-1 * x853 * (x738 + x744 + x747)) + ((2.0 / 3.0) * x49 * ((x52 * x553) + (x550 * x68)))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}