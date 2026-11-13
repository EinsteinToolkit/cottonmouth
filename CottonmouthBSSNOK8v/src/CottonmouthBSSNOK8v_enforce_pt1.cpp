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
        vreal x129 = access(gtDD11, stencil_idx_0_0_0_VVV); // x129: Dependency! Liveness = 8; [gtDD11, x1198, x129, x48, x49, x50, x51, x63]
        vreal x48 = access(gtDD00, stencil_idx_0_0_0_VVV); // x48: Dependency! Liveness = 8; [gtDD00, x1198, x129, x48, x49, x50, x51, x63]
        vreal x49 = access(gtDD12, stencil_idx_0_0_0_VVV); // x49: Dependency! Liveness = 8; [gtDD12, x1198, x129, x48, x49, x50, x51, x63]
        vreal x50 = access(gtDD01, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 8; [gtDD01, x1198, x129, x48, x49, x50, x51, x63]
        vreal x51 = access(gtDD02, stencil_idx_0_0_0_VVV); // x51: Dependency! Liveness = 8; [gtDD02, x1198, x129, x48, x49, x50, x51, x63]
        vreal x63 = access(gtDD22, stencil_idx_0_0_0_VVV); // x63: Dependency! Liveness = 8; [gtDD22, x1198, x129, x48, x49, x50, x51, x63]
        vreal x155 = pow2(x50); // x155: Dependency! Liveness = 3; [x49, x50, x51]
        vreal x224 = pow2(x49); // x224: Dependency! Liveness = 2; [x49, x51]
        vreal x64 = pow2(x51); // x64: Dependency! Liveness = 1; [x51]
        vreal x1198 = pow(static_cast<vreal>(((x129 * ((-(x64)) + (x48 * x63))) + (-1 * x155 * x63) + (-1 * x224 * x48) + (2 * x49 * x50 * x51))), (-1.0 / 3.0)); // x1198: Dependency! Liveness = 10; [x1198, x129, x155, x224, x48, x49, x50, x51, x63, x64]
        vreal x909 = access(evo_lapse, stencil_idx_0_0_0_VVV); // x909: Dependency! Liveness = 8; [evo_lapse, x1198, x129, x48, x49, x50, x51, x63]
        store(evo_lapse, stencil_idx_0_0_0_VVV, max(x909, evolved_lapse_floor));
        vreal x898 = access(w, stencil_idx_0_0_0_VVV); // x898: Dependency! Liveness = 8; [w, x1198, x129, x48, x49, x50, x51, x63]
        store(w, stencil_idx_0_0_0_VVV, max(x898, conformal_factor_floor));
        store(gtDD00, stencil_idx_0_0_0_VVV, (x1198 * x48));
        store(gtDD11, stencil_idx_0_0_0_VVV, (x1198 * x129));
        store(gtDD22, stencil_idx_0_0_0_VVV, (x1198 * x63));
        store(gtDD01, stencil_idx_0_0_0_VVV, (x1198 * x50));
        store(gtDD02, stencil_idx_0_0_0_VVV, (x1198 * x51));
        store(gtDD12, stencil_idx_0_0_0_VVV, (x1198 * x49));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}