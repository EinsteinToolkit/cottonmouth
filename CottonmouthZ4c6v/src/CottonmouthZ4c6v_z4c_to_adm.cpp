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
    const auto DX = CCTK_DELTA_SPACE(0);
    const auto DY = CCTK_DELTA_SPACE(1);
    const auto DZ = CCTK_DELTA_SPACE(2);
    const auto DT = cctk_delta_time;
    const auto DXI = (1.0 / DX);
    const auto DYI = (1.0 / DY);
    const auto DZI = (1.0 / DZ);
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // z4c_to_adm loop 0
    grid.loop_all_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x89 = access(gtDD00, stencil_idx_0_0_0_VVV); // x89: Dependency! Liveness = 13; [gtDD00, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x687, x712, x89]
        vreal x687 = access(chi, stencil_idx_0_0_0_VVV); // x687: Dependency! Liveness = 13; [chi, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x687, x712, x89]
        vreal x712 = pown<vreal>(x687, -1); // x712: Dependency! Liveness = 13; [x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x687, x712, x89, x978]
        vreal x729 = (x712 * x89); // x729: Dependency! Liveness = 7; [x36, x37, x38, x39, x50, x712, x89]
        store(gxx, stencil_idx_0_0_0_VVV, x729); // gxx: 
        vreal x36 = access(gtDD01, stencil_idx_0_0_0_VVV); // x36: Dependency! Liveness = 12; [gtDD01, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x712, x89]
        vreal x746 = (x36 * x712); // x746: Dependency! Liveness = 6; [x36, x37, x38, x39, x50, x712]
        store(gxy, stencil_idx_0_0_0_VVV, x746); // gxy: 
        vreal x39 = access(gtDD11, stencil_idx_0_0_0_VVV); // x39: Dependency! Liveness = 12; [gtDD11, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x712, x89]
        vreal x750 = (x39 * x712); // x750: Dependency! Liveness = 5; [x37, x38, x39, x50, x712]
        store(gyy, stencil_idx_0_0_0_VVV, x750); // gyy: 
        vreal x38 = access(gtDD02, stencil_idx_0_0_0_VVV); // x38: Dependency! Liveness = 12; [gtDD02, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x712, x89]
        vreal x754 = (x38 * x712); // x754: Dependency! Liveness = 4; [x37, x38, x50, x712]
        store(gxz, stencil_idx_0_0_0_VVV, x754); // gxz: 
        vreal x37 = access(gtDD12, stencil_idx_0_0_0_VVV); // x37: Dependency! Liveness = 12; [gtDD12, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x712, x89]
        vreal x759 = (x37 * x712); // x759: Dependency! Liveness = 3; [x37, x50, x712]
        store(gyz, stencil_idx_0_0_0_VVV, x759); // gyz: 
        vreal x50 = access(gtDD22, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 12; [gtDD22, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x712, x89]
        vreal x763 = (x50 * x712); // x763: Dependency! Liveness = 2; [x50, x712]
        store(gzz, stencil_idx_0_0_0_VVV, x763); // gzz: 
        vreal x997 = access(AtDD00, stencil_idx_0_0_0_VVV); // x997: Dependency! Liveness = 14; [AtDD00, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x687, x712, x89, x978]
        vreal x978 = access(trK, stencil_idx_0_0_0_VVV); // x978: Dependency! Liveness = 14; [trK, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x687, x712, x89, x978]
        vreal x1266 = ((1.0 / 3.0) * x978); // x1266: Dependency! Liveness = 12; [x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x712, x89, x978]
        store(kxx, stencil_idx_0_0_0_VVV, (x712 * (x997 + (x1266 * x89)))); // kxx: 
        vreal x998 = access(AtDD01, stencil_idx_0_0_0_VVV); // x998: Dependency! Liveness = 14; [AtDD01, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x687, x712, x89, x978]
        store(kxy, stencil_idx_0_0_0_VVV, (x712 * (x998 + (x1266 * x36)))); // kxy: 
        vreal x1000 = access(AtDD11, stencil_idx_0_0_0_VVV); // x1000: Dependency! Liveness = 8; [AtDD11, x36, x37, x38, x39, x50, x712, x89]
        store(kyy, stencil_idx_0_0_0_VVV, (x712 * (x1000 + (x1266 * x39)))); // kyy: 
        vreal x992 = access(AtDD02, stencil_idx_0_0_0_VVV); // x992: Dependency! Liveness = 14; [AtDD02, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x687, x712, x89, x978]
        store(kxz, stencil_idx_0_0_0_VVV, (x712 * (x992 + (x1266 * x38)))); // kxz: 
        vreal x994 = access(AtDD12, stencil_idx_0_0_0_VVV); // x994: Dependency! Liveness = 14; [AtDD12, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x687, x712, x89, x978]
        store(kyz, stencil_idx_0_0_0_VVV, (x712 * (x994 + (x1266 * x37)))); // kyz: 
        vreal x990 = access(AtDD22, stencil_idx_0_0_0_VVV); // x990: Dependency! Liveness = 14; [AtDD22, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x687, x712, x89, x978]
        store(kzz, stencil_idx_0_0_0_VVV, (x712 * (x990 + (x1266 * x50)))); // kzz: 
        vreal x1108 = access(evo_lapse, stencil_idx_0_0_0_VVV); // x1108: Dependency! Liveness = 9; [evo_lapse, x1108, x36, x37, x38, x39, x50, x712, x89]
        store(alp, stencil_idx_0_0_0_VVV, x1108); // alp: 
        vreal x1124 = access(evo_shiftU0, stencil_idx_0_0_0_VVV); // x1124: Dependency! Liveness = 10; [evo_shiftU0, x1108, x1124, x36, x37, x38, x39, x50, x712, x89]
        store(betax, stencil_idx_0_0_0_VVV, x1124); // betax: 
        vreal x1127 = access(evo_shiftU1, stencil_idx_0_0_0_VVV); // x1127: Dependency! Liveness = 11; [evo_shiftU1, x1108, x1124, x1127, x36, x37, x38, x39, x50, x712, x89]
        store(betay, stencil_idx_0_0_0_VVV, x1127); // betay: 
        vreal x1130 = access(evo_shiftU2, stencil_idx_0_0_0_VVV); // x1130: Dependency! Liveness = 12; [evo_shiftU2, x1108, x1124, x1127, x1130, x36, x37, x38, x39, x50, x712, x89]
        store(betaz, stencil_idx_0_0_0_VVV, x1130); // betaz:     
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}