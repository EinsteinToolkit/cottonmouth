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
using std::cbrt,std::fmax,std::fmin,std::sqrt;
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
        vreal x100 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV)); // x100: Dependency! Liveness = 4; [gtDD00, gtDD01, gtDD02, gtDD22]
        vreal x101 = pow2(access(gtDD02, stencil_idx_0_0_0_VVV)); // x101: Dependency! Liveness = 3; [gtDD01, gtDD02, gtDD12]
        vreal x177 = (x100 + (-(x101))); // x177: Dependency! Liveness = 9; [AtDD00, AtDD01, AtDD02, AtDD11, AtDD12, AtDD22, x100, x101, x177]
        x100 = (access(AtDD11, stencil_idx_0_0_0_VVV) * x177); // x780: Dependency! Liveness = 11; [AtDD00, AtDD01, AtDD02, AtDD11, AtDD12, AtDD22, x177, x192, x31, x40, x51]
        x177 = (access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD11, stencil_idx_0_0_0_VVV)); // x30: Dependency! Liveness = 6; [gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22]
        x101 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV)); // x29: Dependency! Liveness = 6; [gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22]
        vreal x31 = (x101 + (-(x177))); // x31: Dependency! Liveness = 11; [AtDD00, AtDD01, AtDD02, AtDD11, AtDD12, AtDD22, x177, x192, x29, x30, x31]
        vreal x796 = (access(AtDD02, stencil_idx_0_0_0_VVV) * x31); // x796: Dependency! Liveness = 9; [AtDD00, AtDD01, AtDD02, AtDD12, AtDD22, x192, x31, x40, x51]
        x31 = (access(gtDD11, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV)); // x49: Dependency! Liveness = 5; [gtDD00, gtDD01, gtDD02, gtDD11, gtDD22]
        vreal x50 = pow2(access(gtDD12, stencil_idx_0_0_0_VVV)); // x50: Dependency! Liveness = 3; [gtDD01, gtDD02, gtDD12]
        vreal x51 = (x31 + (-(x50))); // x51: Dependency! Liveness = 13; [AtDD00, AtDD01, AtDD02, AtDD11, AtDD12, AtDD22, x177, x192, x31, x40, x49, x50, x51]
        x50 = (access(AtDD00, stencil_idx_0_0_0_VVV) * x51); // x797: Dependency! Liveness = 7; [AtDD00, AtDD01, AtDD12, AtDD22, x192, x40, x51]
        x51 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD11, stencil_idx_0_0_0_VVV)); // x126: Dependency! Liveness = 5; [gtDD00, gtDD01, gtDD02, gtDD11, gtDD22]
        vreal x127 = pow2(access(gtDD01, stencil_idx_0_0_0_VVV)); // x127: Dependency! Liveness = 3; [gtDD01, gtDD02, gtDD12]
        vreal x192 = (x51 + (-(x127))); // x192: Dependency! Liveness = 10; [AtDD00, AtDD01, AtDD02, AtDD11, AtDD12, AtDD22, x126, x127, x177, x192]
        x127 = (access(AtDD22, stencil_idx_0_0_0_VVV) * x192); // x822: Dependency! Liveness = 5; [AtDD01, AtDD12, AtDD22, x192, x40]
        x192 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV)); // x38: Dependency! Liveness = 6; [gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22]
        vreal x39 = (access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV)); // x39: Dependency! Liveness = 6; [gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22]
        vreal x40 = (x192 + (-(x39))); // x40: Dependency! Liveness = 12; [AtDD00, AtDD01, AtDD02, AtDD11, AtDD12, AtDD22, x177, x192, x31, x38, x39, x40]
        x39 = (-(x40)); // x41: Dependency! Liveness = 3; [gtDD01, gtDD02, x40]
        x40 = (access(AtDD01, stencil_idx_0_0_0_VVV) * x39); // x911: Dependency! Liveness = 4; [AtDD01, AtDD12, x40, x41]
        vreal x26 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV)); // x26: Dependency! Liveness = 6; [gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22]
        vreal x27 = (x26 + (-1 * access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD02, stencil_idx_0_0_0_VVV))); // x27: Dependency! Liveness = 5; [gtDD01, gtDD02, x26, x27, x40]
        x26 = (-(x27)); // x28: Dependency! Liveness = 4; [gtDD01, gtDD02, x27, x40]
        x27 = (access(AtDD12, stencil_idx_0_0_0_VVV) * x26); // x912: Dependency! Liveness = 3; [AtDD12, x28, x40]
        vreal x910 = ((2.0 / 3.0) * access(gtDD00, stencil_idx_0_0_0_VVV)); // x910: Dependency! Liveness = 6; [gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22]
        vreal x913 = ((1.0 / 3.0) * access(gtDD00, stencil_idx_0_0_0_VVV)); // x913: Dependency! Liveness = 6; [gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22]
        store(AtDD00, stencil_idx_0_0_0_VVV, (access(AtDD00, stencil_idx_0_0_0_VVV) + (-1 * x100 * x913) + (-1 * x127 * x913) + (-1 * x27 * x910) + (-1 * x40 * x910) + (-1 * x50 * x913) + (-1 * x796 * x910)));
        x910 = ((2.0 / 3.0) * access(gtDD01, stencil_idx_0_0_0_VVV)); // x914: Dependency! Liveness = 3; [gtDD01, gtDD02, gtDD12]
        x913 = ((1.0 / 3.0) * access(gtDD01, stencil_idx_0_0_0_VVV)); // x915: Dependency! Liveness = 3; [gtDD01, gtDD02, gtDD12]
        store(AtDD01, stencil_idx_0_0_0_VVV, (access(AtDD01, stencil_idx_0_0_0_VVV) + (-1 * x100 * x913) + (-1 * x127 * x913) + (-1 * x27 * x910) + (-1 * x40 * x910) + (-1 * x50 * x913) + (-1 * x796 * x910)));
        vreal x916 = ((2.0 / 3.0) * access(gtDD02, stencil_idx_0_0_0_VVV)); // x916: Dependency! Liveness = 2; [gtDD02, gtDD12]
        vreal x917 = ((1.0 / 3.0) * access(gtDD02, stencil_idx_0_0_0_VVV)); // x917: Dependency! Liveness = 2; [gtDD02, gtDD12]
        store(AtDD02, stencil_idx_0_0_0_VVV, (access(AtDD02, stencil_idx_0_0_0_VVV) + (-1 * x100 * x917) + (-1 * x127 * x917) + (-1 * x27 * x916) + (-1 * x40 * x916) + (-1 * x50 * x917) + (-1 * x796 * x916)));
        x916 = ((2.0 / 3.0) * access(gtDD11, stencil_idx_0_0_0_VVV)); // x918: Dependency! Liveness = 5; [gtDD01, gtDD02, gtDD11, gtDD12, gtDD22]
        x917 = ((1.0 / 3.0) * access(gtDD11, stencil_idx_0_0_0_VVV)); // x919: Dependency! Liveness = 5; [gtDD01, gtDD02, gtDD11, gtDD12, gtDD22]
        store(AtDD11, stencil_idx_0_0_0_VVV, (access(AtDD11, stencil_idx_0_0_0_VVV) + (-1 * x100 * x917) + (-1 * x127 * x917) + (-1 * x27 * x916) + (-1 * x40 * x916) + (-1 * x50 * x917) + (-1 * x796 * x916)));
        vreal x920 = ((2.0 / 3.0) * access(gtDD12, stencil_idx_0_0_0_VVV)); // x920: Dependency! Liveness = 1; [gtDD12]
        vreal x921 = ((1.0 / 3.0) * access(gtDD12, stencil_idx_0_0_0_VVV)); // x921: Dependency! Liveness = 1; [gtDD12]
        store(AtDD12, stencil_idx_0_0_0_VVV, (access(AtDD12, stencil_idx_0_0_0_VVV) + (-1 * x100 * x921) + (-1 * x127 * x921) + (-1 * x27 * x920) + (-1 * x40 * x920) + (-1 * x50 * x921) + (-1 * x796 * x920)));
        x920 = ((2.0 / 3.0) * access(gtDD22, stencil_idx_0_0_0_VVV)); // x922: Dependency! Liveness = 4; [gtDD01, gtDD02, gtDD12, gtDD22]
        x921 = ((1.0 / 3.0) * access(gtDD22, stencil_idx_0_0_0_VVV)); // x923: Dependency! Liveness = 4; [gtDD01, gtDD02, gtDD12, gtDD22]
        store(AtDD22, stencil_idx_0_0_0_VVV, (access(AtDD22, stencil_idx_0_0_0_VVV) + (-1 * x100 * x921) + (-1 * x127 * x921) + (-1 * x27 * x920) + (-1 * x40 * x920) + (-1 * x50 * x921) + (-1 * x796 * x920)));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}