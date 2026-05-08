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
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // enforce_pt1 loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x47 = access(gtDD00, stencil_idx_0_0_0_VVV); // x47: Dependency! Liveness = 8; [gtDD00, x47, x48, x50, x51, x62, x95, x998]
        vreal x48 = access(gtDD12, stencil_idx_0_0_0_VVV); // x48: Dependency! Liveness = 8; [gtDD12, x47, x48, x50, x51, x62, x95, x998]
        vreal x50 = access(gtDD01, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 8; [gtDD01, x47, x48, x50, x51, x62, x95, x998]
        vreal x51 = access(gtDD02, stencil_idx_0_0_0_VVV); // x51: Dependency! Liveness = 8; [gtDD02, x47, x48, x50, x51, x62, x95, x998]
        vreal x62 = access(gtDD22, stencil_idx_0_0_0_VVV); // x62: Dependency! Liveness = 8; [gtDD22, x47, x48, x50, x51, x62, x95, x998]
        vreal x95 = access(gtDD11, stencil_idx_0_0_0_VVV); // x95: Dependency! Liveness = 8; [gtDD11, x47, x48, x50, x51, x62, x95, x998]
        vreal x160 = pow2(x48); // x160: Dependency! Liveness = 3; [x48, x50, x51]
        vreal x185 = pow2(x50); // x185: Dependency! Liveness = 2; [x50, x51]
        vreal x64 = pow2(x51); // x64: Dependency! Liveness = 1; [x51]
        vreal x998 = pow(static_cast<vreal>(((-1 * x160 * x47) + (-1 * x185 * x62) + (-1 * x64 * x95) + (x47 * x62 * x95) + (2 * x48 * x50 * x51))), (-1.0 / 3.0)); // x998: Dependency! Liveness = 10; [x160, x185, x47, x48, x50, x51, x62, x64, x95, x998]
        vreal x581 = access(evo_lapse, stencil_idx_0_0_0_VVV); // x581: Dependency! Liveness = 8; [evo_lapse, x47, x48, x50, x51, x62, x95, x998]
        store(evo_lapse, stencil_idx_0_0_0_VVV, max(x581, evolved_lapse_floor));
        vreal x570 = access(w, stencil_idx_0_0_0_VVV); // x570: Dependency! Liveness = 8; [w, x47, x48, x50, x51, x62, x95, x998]
        store(w, stencil_idx_0_0_0_VVV, max(x570, conformal_factor_floor));
        store(gtDD00, stencil_idx_0_0_0_VVV, (x47 * x998));
        store(gtDD11, stencil_idx_0_0_0_VVV, (x95 * x998));
        store(gtDD22, stencil_idx_0_0_0_VVV, (x62 * x998));
        store(gtDD01, stencil_idx_0_0_0_VVV, (x50 * x998));
        store(gtDD02, stencil_idx_0_0_0_VVV, (x51 * x998));
        store(gtDD12, stencil_idx_0_0_0_VVV, (x48 * x998));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}