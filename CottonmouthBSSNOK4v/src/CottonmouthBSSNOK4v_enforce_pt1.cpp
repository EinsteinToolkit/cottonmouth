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
void enforce_pt1(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_enforce_pt1;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("enforce_pt1");
    #endif
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define evo_lapse_layout VVV_layout
    #define gtDD00_layout VVV_layout
    #define gtDD01_layout VVV_layout
    #define gtDD02_layout VVV_layout
    #define gtDD11_layout VVV_layout
    #define gtDD12_layout VVV_layout
    #define gtDD22_layout VVV_layout
    #define w_layout VVV_layout
    const auto DX = CCTK_DELTA_SPACE(0);
    const auto DY = CCTK_DELTA_SPACE(1);
    const auto DZ = CCTK_DELTA_SPACE(2);
    const auto DT = cctk_delta_time;
    const auto DXI = (1.0 / DX);
    const auto DYI = (1.0 / DY);
    const auto DZI = (1.0 / DZ);
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // enforce_pt1 loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x101 = access(gtDD11, stencil_idx_0_0_0_VVV); // x101: Dependency! Liveness = 8; [gtDD11, x101, x48, x49, x50, x51, x59, x844]
        vreal x48 = access(gtDD00, stencil_idx_0_0_0_VVV); // x48: Dependency! Liveness = 8; [gtDD00, x101, x48, x49, x50, x51, x59, x844]
        vreal x49 = access(gtDD12, stencil_idx_0_0_0_VVV); // x49: Dependency! Liveness = 8; [gtDD12, x101, x48, x49, x50, x51, x59, x844]
        vreal x50 = access(gtDD01, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 8; [gtDD01, x101, x48, x49, x50, x51, x59, x844]
        vreal x51 = access(gtDD02, stencil_idx_0_0_0_VVV); // x51: Dependency! Liveness = 8; [gtDD02, x101, x48, x49, x50, x51, x59, x844]
        vreal x59 = access(gtDD22, stencil_idx_0_0_0_VVV); // x59: Dependency! Liveness = 8; [gtDD22, x101, x48, x49, x50, x51, x59, x844]
        vreal x119 = pow2(x50); // x119: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x164 = pow2(x49); // x164: Dependency! Liveness = 2; [x49, x51]
        vreal x60 = pow2(x51); // x60: Dependency! Liveness = 1; [x51]
        vreal x844 = pow(static_cast<vreal>(((x101 * ((-(x60)) + (x48 * x59))) + (-1 * x119 * x59) + (-1 * x164 * x48) + (2 * x49 * x50 * x51))), (-1.0 / 3.0)); // x844: Dependency! Liveness = 10; [x101, x119, x164, x48, x49, x50, x51, x59, x60, x844]
        vreal x555 = access(evo_lapse, stencil_idx_0_0_0_VVV); // x555: Dependency! Liveness = 8; [evo_lapse, x101, x48, x49, x50, x51, x59, x844]
        store(evo_lapse, stencil_idx_0_0_0_VVV, max(x555, evolved_lapse_floor));
        vreal x544 = access(w, stencil_idx_0_0_0_VVV); // x544: Dependency! Liveness = 8; [w, x101, x48, x49, x50, x51, x59, x844]
        store(w, stencil_idx_0_0_0_VVV, max(x544, conformal_factor_floor));
        store(gtDD00, stencil_idx_0_0_0_VVV, (x48 * x844));
        store(gtDD11, stencil_idx_0_0_0_VVV, (x101 * x844));
        store(gtDD22, stencil_idx_0_0_0_VVV, (x59 * x844));
        store(gtDD01, stencil_idx_0_0_0_VVV, (x50 * x844));
        store(gtDD02, stencil_idx_0_0_0_VVV, (x51 * x844));
        store(gtDD12, stencil_idx_0_0_0_VVV, (x49 * x844));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}