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
void bssn2adm(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_bssn2adm;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("bssn2adm");
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
    #define w_layout VVV_layout
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // bssn2adm loop 0
    grid.loop_all_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x47 = access(gtDD00, stencil_idx_0_0_0_VVV); // x47: Dependency! Liveness = 8; [gtDD00, x47, x48, x50, x51, x572, x62, x95]
        vreal x573 = access(AtDD00, stencil_idx_0_0_0_VVV); // x573: Dependency! Liveness = 9; [AtDD00, x47, x48, x50, x51, x570, x572, x62, x95]
        vreal x570 = access(w, stencil_idx_0_0_0_VVV); // x570: Dependency! Liveness = 9; [w, x47, x48, x50, x51, x570, x572, x62, x95]
        vreal x571 = pow2(x570); // x571: Dependency! Liveness = 14; [x47, x48, x50, x51, x570, x571, x572, x574, x581, x582, x583, x584, x62, x95]
        vreal x572 = pown<vreal>(x571, -1); // x572: Dependency! Liveness = 13; [x47, x48, x50, x51, x571, x572, x574, x581, x582, x583, x584, x62, x95]
        vreal x574 = access(trK, stencil_idx_0_0_0_VVV); // x574: Dependency! Liveness = 10; [trK, x47, x48, x50, x51, x570, x572, x574, x62, x95]
        vreal x575 = ((1.0 / 3.0) * x574); // x575: Dependency! Liveness = 12; [x47, x48, x50, x51, x572, x574, x581, x582, x583, x584, x62, x95]
        store(kxx, stencil_idx_0_0_0_VVV, (x572 * (x573 + (x47 * x575)))); // kxx: Liveness = 5; [kxx, x47, x572, x573, x575]
        vreal x50 = access(gtDD01, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 8; [gtDD01, x47, x48, x50, x51, x572, x62, x95]
        vreal x576 = access(AtDD01, stencil_idx_0_0_0_VVV); // x576: Dependency! Liveness = 10; [AtDD01, x47, x48, x50, x51, x570, x572, x574, x62, x95]
        store(kxy, stencil_idx_0_0_0_VVV, (x572 * (x576 + (x50 * x575)))); // kxy: Liveness = 6; [kxy, x47, x50, x572, x575, x576]
        vreal x51 = access(gtDD02, stencil_idx_0_0_0_VVV); // x51: Dependency! Liveness = 8; [gtDD02, x47, x48, x50, x51, x572, x62, x95]
        vreal x577 = access(AtDD02, stencil_idx_0_0_0_VVV); // x577: Dependency! Liveness = 10; [AtDD02, x47, x48, x50, x51, x570, x572, x574, x62, x95]
        store(kxz, stencil_idx_0_0_0_VVV, (x572 * (x577 + (x51 * x575)))); // kxz: Liveness = 7; [kxz, x47, x50, x51, x572, x575, x577]
        vreal x578 = access(AtDD11, stencil_idx_0_0_0_VVV); // x578: Dependency! Liveness = 10; [AtDD11, x47, x48, x50, x51, x570, x572, x574, x62, x95]
        vreal x95 = access(gtDD11, stencil_idx_0_0_0_VVV); // x95: Dependency! Liveness = 14; [gtDD11, x47, x48, x50, x51, x570, x572, x574, x581, x582, x583, x584, x62, x95]
        store(kyy, stencil_idx_0_0_0_VVV, (x572 * (x578 + (x575 * x95)))); // kyy: Liveness = 8; [kyy, x47, x50, x51, x572, x575, x578, x95]
        vreal x48 = access(gtDD12, stencil_idx_0_0_0_VVV); // x48: Dependency! Liveness = 8; [gtDD12, x47, x48, x50, x51, x572, x62, x95]
        vreal x579 = access(AtDD12, stencil_idx_0_0_0_VVV); // x579: Dependency! Liveness = 10; [AtDD12, x47, x48, x50, x51, x570, x572, x574, x62, x95]
        store(kyz, stencil_idx_0_0_0_VVV, (x572 * (x579 + (x48 * x575)))); // kyz: Liveness = 9; [kyz, x47, x48, x50, x51, x572, x575, x579, x95]
        vreal x580 = access(AtDD22, stencil_idx_0_0_0_VVV); // x580: Dependency! Liveness = 10; [AtDD22, x47, x48, x50, x51, x570, x572, x574, x62, x95]
        vreal x62 = access(gtDD22, stencil_idx_0_0_0_VVV); // x62: Dependency! Liveness = 14; [gtDD22, x47, x48, x50, x51, x570, x572, x574, x581, x582, x583, x584, x62, x95]
        store(kzz, stencil_idx_0_0_0_VVV, (x572 * (x580 + (x575 * x62)))); // kzz: Liveness = 10; [kzz, x47, x48, x50, x51, x572, x575, x580, x62, x95]
        vreal x581 = access(evo_lapse, stencil_idx_0_0_0_VVV); // x581: Dependency! Liveness = 11; [evo_lapse, x47, x48, x50, x51, x570, x572, x574, x581, x62, x95]
        vreal x582 = access(evo_shiftU0, stencil_idx_0_0_0_VVV); // x582: Dependency! Liveness = 12; [evo_shiftU0, x47, x48, x50, x51, x570, x572, x574, x581, x582, x62, x95]
        vreal x583 = access(evo_shiftU1, stencil_idx_0_0_0_VVV); // x583: Dependency! Liveness = 13; [evo_shiftU1, x47, x48, x50, x51, x570, x572, x574, x581, x582, x583, x62, x95]
        vreal x584 = access(evo_shiftU2, stencil_idx_0_0_0_VVV); // x584: Dependency! Liveness = 14; [evo_shiftU2, x47, x48, x50, x51, x570, x572, x574, x581, x582, x583, x584, x62, x95]
        store(alp, stencil_idx_0_0_0_VVV, x581); // alp: Liveness = 12; [alp, x47, x48, x50, x51, x572, x581, x582, x583, x584, x62, x95]
        store(betax, stencil_idx_0_0_0_VVV, x582); // betax: Liveness = 11; [betax, x47, x48, x50, x51, x572, x582, x583, x584, x62, x95]
        store(betay, stencil_idx_0_0_0_VVV, x583); // betay: Liveness = 10; [betay, x47, x48, x50, x51, x572, x583, x584, x62, x95]
        store(betaz, stencil_idx_0_0_0_VVV, x584); // betaz: Liveness = 9; [betaz, x47, x48, x50, x51, x572, x584, x62, x95]
        vreal x585 = (x47 * x572); // x585: Dependency! Liveness = 7; [x47, x48, x50, x51, x572, x62, x95]
        store(gxx, stencil_idx_0_0_0_VVV, x585); // gxx: Liveness = 9; [gxx, x47, x48, x50, x51, x572, x585, x62, x95]
        vreal x586 = (x50 * x572); // x586: Dependency! Liveness = 6; [x48, x50, x51, x572, x62, x95]
        store(gxy, stencil_idx_0_0_0_VVV, x586); // gxy: Liveness = 9; [gxy, x47, x48, x50, x51, x572, x586, x62, x95]
        vreal x587 = (x51 * x572); // x587: Dependency! Liveness = 5; [x48, x51, x572, x62, x95]
        store(gxz, stencil_idx_0_0_0_VVV, x587); // gxz: Liveness = 9; [gxz, x47, x48, x50, x51, x572, x587, x62, x95]
        vreal x588 = (x572 * x95); // x588: Dependency! Liveness = 4; [x48, x572, x62, x95]
        store(gyy, stencil_idx_0_0_0_VVV, x588); // gyy: Liveness = 9; [gyy, x47, x48, x50, x51, x572, x588, x62, x95]
        vreal x589 = (x48 * x572); // x589: Dependency! Liveness = 3; [x48, x572, x62]
        store(gyz, stencil_idx_0_0_0_VVV, x589); // gyz: Liveness = 9; [gyz, x47, x48, x50, x51, x572, x589, x62, x95]
        vreal x590 = (x572 * x62); // x590: Dependency! Liveness = 2; [x572, x62]
        store(gzz, stencil_idx_0_0_0_VVV, x590); // gzz: Liveness = 9; [gzz, x47, x48, x50, x51, x572, x590, x62, x95]    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}