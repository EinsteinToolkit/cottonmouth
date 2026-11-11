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
    const auto DX = CCTK_DELTA_SPACE(0);
    const auto DY = CCTK_DELTA_SPACE(1);
    const auto DZ = CCTK_DELTA_SPACE(2);
    const auto DT = cctk_delta_time;
    const auto DXI = (1.0 / DX);
    const auto DYI = (1.0 / DY);
    const auto DZI = (1.0 / DZ);
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // bssn2adm loop 0
    grid.loop_all_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x48 = access(gtDD00, stencil_idx_0_0_0_VVV); // x48: Dependency! Liveness = 8; [gtDD00, x101, x48, x49, x50, x51, x546, x59]
        vreal x547 = access(AtDD00, stencil_idx_0_0_0_VVV); // x547: Dependency! Liveness = 9; [AtDD00, x101, x48, x49, x50, x51, x544, x546, x59]
        vreal x544 = access(w, stencil_idx_0_0_0_VVV); // x544: Dependency! Liveness = 9; [w, x101, x48, x49, x50, x51, x544, x546, x59]
        vreal x545 = pow2(x544); // x545: Dependency! Liveness = 14; [x101, x48, x49, x50, x51, x544, x545, x546, x548, x555, x556, x557, x558, x59]
        vreal x546 = pown<vreal>(x545, -1); // x546: Dependency! Liveness = 13; [x101, x48, x49, x50, x51, x545, x546, x548, x555, x556, x557, x558, x59]
        vreal x548 = access(trK, stencil_idx_0_0_0_VVV); // x548: Dependency! Liveness = 10; [trK, x101, x48, x49, x50, x51, x544, x546, x548, x59]
        vreal x549 = ((1.0 / 3.0) * x548); // x549: Dependency! Liveness = 12; [x101, x48, x49, x50, x51, x546, x548, x555, x556, x557, x558, x59]
        store(kxx, stencil_idx_0_0_0_VVV, (x546 * (x547 + (x48 * x549)))); // kxx: Liveness = 5; [kxx, x48, x546, x547, x549]
        vreal x50 = access(gtDD01, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 8; [gtDD01, x101, x48, x49, x50, x51, x546, x59]
        vreal x550 = access(AtDD01, stencil_idx_0_0_0_VVV); // x550: Dependency! Liveness = 10; [AtDD01, x101, x48, x49, x50, x51, x544, x546, x548, x59]
        store(kxy, stencil_idx_0_0_0_VVV, (x546 * (x550 + (x50 * x549)))); // kxy: Liveness = 6; [kxy, x48, x50, x546, x549, x550]
        vreal x51 = access(gtDD02, stencil_idx_0_0_0_VVV); // x51: Dependency! Liveness = 8; [gtDD02, x101, x48, x49, x50, x51, x546, x59]
        vreal x551 = access(AtDD02, stencil_idx_0_0_0_VVV); // x551: Dependency! Liveness = 10; [AtDD02, x101, x48, x49, x50, x51, x544, x546, x548, x59]
        store(kxz, stencil_idx_0_0_0_VVV, (x546 * (x551 + (x51 * x549)))); // kxz: Liveness = 7; [kxz, x48, x50, x51, x546, x549, x551]
        vreal x101 = access(gtDD11, stencil_idx_0_0_0_VVV); // x101: Dependency! Liveness = 8; [gtDD11, x101, x48, x49, x50, x51, x546, x59]
        vreal x552 = access(AtDD11, stencil_idx_0_0_0_VVV); // x552: Dependency! Liveness = 10; [AtDD11, x101, x48, x49, x50, x51, x544, x546, x548, x59]
        store(kyy, stencil_idx_0_0_0_VVV, (x546 * (x552 + (x101 * x549)))); // kyy: Liveness = 8; [kyy, x101, x48, x50, x51, x546, x549, x552]
        vreal x49 = access(gtDD12, stencil_idx_0_0_0_VVV); // x49: Dependency! Liveness = 8; [gtDD12, x101, x48, x49, x50, x51, x546, x59]
        vreal x553 = access(AtDD12, stencil_idx_0_0_0_VVV); // x553: Dependency! Liveness = 10; [AtDD12, x101, x48, x49, x50, x51, x544, x546, x548, x59]
        store(kyz, stencil_idx_0_0_0_VVV, (x546 * (x553 + (x49 * x549)))); // kyz: Liveness = 9; [kyz, x101, x48, x49, x50, x51, x546, x549, x553]
        vreal x554 = access(AtDD22, stencil_idx_0_0_0_VVV); // x554: Dependency! Liveness = 10; [AtDD22, x101, x48, x49, x50, x51, x544, x546, x548, x59]
        vreal x59 = access(gtDD22, stencil_idx_0_0_0_VVV); // x59: Dependency! Liveness = 14; [gtDD22, x101, x48, x49, x50, x51, x544, x546, x548, x555, x556, x557, x558, x59]
        store(kzz, stencil_idx_0_0_0_VVV, (x546 * (x554 + (x549 * x59)))); // kzz: Liveness = 10; [kzz, x101, x48, x49, x50, x51, x546, x549, x554, x59]
        vreal x555 = access(evo_lapse, stencil_idx_0_0_0_VVV); // x555: Dependency! Liveness = 11; [evo_lapse, x101, x48, x49, x50, x51, x544, x546, x548, x555, x59]
        vreal x556 = access(evo_shiftU0, stencil_idx_0_0_0_VVV); // x556: Dependency! Liveness = 12; [evo_shiftU0, x101, x48, x49, x50, x51, x544, x546, x548, x555, x556, x59]
        vreal x557 = access(evo_shiftU1, stencil_idx_0_0_0_VVV); // x557: Dependency! Liveness = 13; [evo_shiftU1, x101, x48, x49, x50, x51, x544, x546, x548, x555, x556, x557, x59]
        vreal x558 = access(evo_shiftU2, stencil_idx_0_0_0_VVV); // x558: Dependency! Liveness = 14; [evo_shiftU2, x101, x48, x49, x50, x51, x544, x546, x548, x555, x556, x557, x558, x59]
        store(alp, stencil_idx_0_0_0_VVV, x555); // alp: Liveness = 12; [alp, x101, x48, x49, x50, x51, x546, x555, x556, x557, x558, x59]
        store(betax, stencil_idx_0_0_0_VVV, x556); // betax: Liveness = 11; [betax, x101, x48, x49, x50, x51, x546, x556, x557, x558, x59]
        store(betay, stencil_idx_0_0_0_VVV, x557); // betay: Liveness = 10; [betay, x101, x48, x49, x50, x51, x546, x557, x558, x59]
        store(betaz, stencil_idx_0_0_0_VVV, x558); // betaz: Liveness = 9; [betaz, x101, x48, x49, x50, x51, x546, x558, x59]
        vreal x559 = (x48 * x546); // x559: Dependency! Liveness = 7; [x101, x48, x49, x50, x51, x546, x59]
        store(gxx, stencil_idx_0_0_0_VVV, x559); // gxx: Liveness = 9; [gxx, x101, x48, x49, x50, x51, x546, x559, x59]
        vreal x560 = (x50 * x546); // x560: Dependency! Liveness = 6; [x101, x49, x50, x51, x546, x59]
        store(gxy, stencil_idx_0_0_0_VVV, x560); // gxy: Liveness = 9; [gxy, x101, x48, x49, x50, x51, x546, x560, x59]
        vreal x561 = (x51 * x546); // x561: Dependency! Liveness = 5; [x101, x49, x51, x546, x59]
        store(gxz, stencil_idx_0_0_0_VVV, x561); // gxz: Liveness = 9; [gxz, x101, x48, x49, x50, x51, x546, x561, x59]
        vreal x562 = (x101 * x546); // x562: Dependency! Liveness = 4; [x101, x49, x546, x59]
        store(gyy, stencil_idx_0_0_0_VVV, x562); // gyy: Liveness = 9; [gyy, x101, x48, x49, x50, x51, x546, x562, x59]
        vreal x563 = (x49 * x546); // x563: Dependency! Liveness = 3; [x49, x546, x59]
        store(gyz, stencil_idx_0_0_0_VVV, x563); // gyz: Liveness = 9; [gyz, x101, x48, x49, x50, x51, x546, x563, x59]
        vreal x564 = (x546 * x59); // x564: Dependency! Liveness = 2; [x546, x59]
        store(gzz, stencil_idx_0_0_0_VVV, x564); // gzz: Liveness = 9; [gzz, x101, x48, x49, x50, x51, x546, x564, x59]    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}