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
void z4c_enforce_pt1(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_z4c_enforce_pt1;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("z4c_enforce_pt1");
    #endif
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define chi_layout VVV_layout
    #define evo_lapse_layout VVV_layout
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
    // z4c_enforce_pt1 loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x101 = pow2(access(gtDD02, stencil_idx_0_0_0_VVV)); // x101: Dependency! Liveness = 3; [gtDD01, gtDD02, gtDD12]
        vreal x127 = pow2(access(gtDD01, stencil_idx_0_0_0_VVV)); // x127: Dependency! Liveness = 2; [gtDD01, gtDD12]
        vreal x50 = pow2(access(gtDD12, stencil_idx_0_0_0_VVV)); // x50: Dependency! Liveness = 1; [gtDD12]
        vreal x909 = pow(static_cast<vreal>(((-1 * access(gtDD00, stencil_idx_0_0_0_VVV) * x50) + (-1 * access(gtDD11, stencil_idx_0_0_0_VVV) * x101) + (-1 * access(gtDD22, stencil_idx_0_0_0_VVV) * x127) + (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD11, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV)) + (2 * access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV)))), (-1.0 / 3.0)); // x909: Dependency! Liveness = 10; [gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22, x101, x127, x50, x909]
        store(chi, stencil_idx_0_0_0_VVV, max(access(chi, stencil_idx_0_0_0_VVV), chi_floor));
        store(evo_lapse, stencil_idx_0_0_0_VVV, max(access(evo_lapse, stencil_idx_0_0_0_VVV), evolved_lapse_floor));
        store(gtDD00, stencil_idx_0_0_0_VVV, (access(gtDD00, stencil_idx_0_0_0_VVV) * x909));
        store(gtDD11, stencil_idx_0_0_0_VVV, (access(gtDD11, stencil_idx_0_0_0_VVV) * x909));
        store(gtDD22, stencil_idx_0_0_0_VVV, (access(gtDD22, stencil_idx_0_0_0_VVV) * x909));
        store(gtDD01, stencil_idx_0_0_0_VVV, (access(gtDD01, stencil_idx_0_0_0_VVV) * x909));
        store(gtDD02, stencil_idx_0_0_0_VVV, (access(gtDD02, stencil_idx_0_0_0_VVV) * x909));
        store(gtDD12, stencil_idx_0_0_0_VVV, (access(gtDD12, stencil_idx_0_0_0_VVV) * x909));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}