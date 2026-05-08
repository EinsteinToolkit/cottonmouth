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
void z4c_to_adm(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_z4c_to_adm;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("z4c_to_adm");
    #endif
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define AtDD00_layout VVV_layout
    #define AtDD01_layout VVV_layout
    #define AtDD02_layout VVV_layout
    #define AtDD11_layout VVV_layout
    #define AtDD12_layout VVV_layout
    #define AtDD22_layout VVV_layout
    #define alp_layout VVV_layout
    #define betax_layout VVV_layout
    #define betay_layout VVV_layout
    #define betaz_layout VVV_layout
    #define chi_layout VVV_layout
    #define evo_lapse_layout VVV_layout
    #define evo_shiftU0_layout VVV_layout
    #define evo_shiftU1_layout VVV_layout
    #define evo_shiftU2_layout VVV_layout
    #define gtDD00_layout VVV_layout
    #define gtDD01_layout VVV_layout
    #define gtDD02_layout VVV_layout
    #define gtDD11_layout VVV_layout
    #define gtDD12_layout VVV_layout
    #define gtDD22_layout VVV_layout
    #define gxx_layout VVV_layout
    #define gxy_layout VVV_layout
    #define gxz_layout VVV_layout
    #define gyy_layout VVV_layout
    #define gyz_layout VVV_layout
    #define gzz_layout VVV_layout
    #define kxx_layout VVV_layout
    #define kxy_layout VVV_layout
    #define kxz_layout VVV_layout
    #define kyy_layout VVV_layout
    #define kyz_layout VVV_layout
    #define kzz_layout VVV_layout
    #define trK_layout VVV_layout
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // z4c_to_adm loop 0
    grid.loop_all_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x520 = pown<vreal>(access(chi, stencil_idx_0_0_0_VVV), -1); // x520: Dependency! Liveness = 8; [chi, gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22, x520]
        vreal x1051 = ((1.0 / 3.0) * access(trK, stencil_idx_0_0_0_VVV)); // x1051: Dependency! Liveness = 8; [gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22, trK, x520]
        store(kxx, stencil_idx_0_0_0_VVV, (x520 * (access(AtDD00, stencil_idx_0_0_0_VVV) + (access(gtDD00, stencil_idx_0_0_0_VVV) * x1051)))); // kxx: Liveness = 5; [AtDD00, gtDD00, kxx, x1051, x520]
        store(kxy, stencil_idx_0_0_0_VVV, (x520 * (access(AtDD01, stencil_idx_0_0_0_VVV) + (access(gtDD01, stencil_idx_0_0_0_VVV) * x1051)))); // kxy: Liveness = 6; [AtDD01, gtDD00, gtDD01, kxy, x1051, x520]
        store(kxz, stencil_idx_0_0_0_VVV, (x520 * (access(AtDD02, stencil_idx_0_0_0_VVV) + (access(gtDD02, stencil_idx_0_0_0_VVV) * x1051)))); // kxz: Liveness = 7; [AtDD02, gtDD00, gtDD01, gtDD02, kxz, x1051, x520]
        store(kyy, stencil_idx_0_0_0_VVV, (x520 * (access(AtDD11, stencil_idx_0_0_0_VVV) + (access(gtDD11, stencil_idx_0_0_0_VVV) * x1051)))); // kyy: Liveness = 8; [AtDD11, gtDD00, gtDD01, gtDD02, gtDD11, kyy, x1051, x520]
        store(kyz, stencil_idx_0_0_0_VVV, (x520 * (access(AtDD12, stencil_idx_0_0_0_VVV) + (access(gtDD12, stencil_idx_0_0_0_VVV) * x1051)))); // kyz: Liveness = 9; [AtDD12, gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, kyz, x1051, x520]
        store(kzz, stencil_idx_0_0_0_VVV, (x520 * (access(AtDD22, stencil_idx_0_0_0_VVV) + (access(gtDD22, stencil_idx_0_0_0_VVV) * x1051)))); // kzz: Liveness = 10; [AtDD22, gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22, kzz, x1051, x520]
        store(alp, stencil_idx_0_0_0_VVV, access(evo_lapse, stencil_idx_0_0_0_VVV)); // alp: Liveness = 9; [alp, evo_lapse, gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22, x520]
        store(betax, stencil_idx_0_0_0_VVV, access(evo_shiftU0, stencil_idx_0_0_0_VVV)); // betax: Liveness = 9; [betax, evo_shiftU0, gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22, x520]
        store(betay, stencil_idx_0_0_0_VVV, access(evo_shiftU1, stencil_idx_0_0_0_VVV)); // betay: Liveness = 9; [betay, evo_shiftU1, gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22, x520]
        store(betaz, stencil_idx_0_0_0_VVV, access(evo_shiftU2, stencil_idx_0_0_0_VVV)); // betaz: Liveness = 9; [betaz, evo_shiftU2, gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22, x520]
        store(gxx, stencil_idx_0_0_0_VVV, (access(gtDD00, stencil_idx_0_0_0_VVV) * x520)); // gxx: Liveness = 8; [gtDD00, gtDD01, gtDD02, gtDD11, gtDD12, gtDD22, gxx, x520]
        store(gxy, stencil_idx_0_0_0_VVV, (access(gtDD01, stencil_idx_0_0_0_VVV) * x520)); // gxy: Liveness = 7; [gtDD01, gtDD02, gtDD11, gtDD12, gtDD22, gxy, x520]
        store(gxz, stencil_idx_0_0_0_VVV, (access(gtDD02, stencil_idx_0_0_0_VVV) * x520)); // gxz: Liveness = 6; [gtDD02, gtDD11, gtDD12, gtDD22, gxz, x520]
        store(gyy, stencil_idx_0_0_0_VVV, (access(gtDD11, stencil_idx_0_0_0_VVV) * x520)); // gyy: Liveness = 5; [gtDD11, gtDD12, gtDD22, gyy, x520]
        store(gyz, stencil_idx_0_0_0_VVV, (access(gtDD12, stencil_idx_0_0_0_VVV) * x520)); // gyz: Liveness = 4; [gtDD12, gtDD22, gyz, x520]
        store(gzz, stencil_idx_0_0_0_VVV, (access(gtDD22, stencil_idx_0_0_0_VVV) * x520)); // gzz: Liveness = 3; [gtDD22, gzz, x520]    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}