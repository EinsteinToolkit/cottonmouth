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
        vreal x48 = access(gtDD00, stencil_idx_0_0_0_VVV); // x48: Dependency! Liveness = 8; [gtDD00, x115, x48, x49, x50, x51, x61, x723]
        vreal x724 = access(AtDD00, stencil_idx_0_0_0_VVV); // x724: Dependency! Liveness = 9; [AtDD00, x115, x48, x49, x50, x51, x61, x721, x723]
        vreal x721 = access(w, stencil_idx_0_0_0_VVV); // x721: Dependency! Liveness = 9; [w, x115, x48, x49, x50, x51, x61, x721, x723]
        vreal x722 = pow2(x721); // x722: Dependency! Liveness = 14; [x115, x48, x49, x50, x51, x61, x721, x722, x723, x725, x732, x733, x734, x735]
        vreal x723 = pown<vreal>(x722, -1); // x723: Dependency! Liveness = 13; [x115, x48, x49, x50, x51, x61, x722, x723, x725, x732, x733, x734, x735]
        vreal x725 = access(trK, stencil_idx_0_0_0_VVV); // x725: Dependency! Liveness = 10; [trK, x115, x48, x49, x50, x51, x61, x721, x723, x725]
        vreal x726 = ((1.0 / 3.0) * x725); // x726: Dependency! Liveness = 12; [x115, x48, x49, x50, x51, x61, x723, x725, x732, x733, x734, x735]
        store(kxx, stencil_idx_0_0_0_VVV, (x723 * (x724 + (x48 * x726)))); // kxx: Liveness = 5; [kxx, x48, x723, x724, x726]
        vreal x50 = access(gtDD01, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 8; [gtDD01, x115, x48, x49, x50, x51, x61, x723]
        vreal x727 = access(AtDD01, stencil_idx_0_0_0_VVV); // x727: Dependency! Liveness = 10; [AtDD01, x115, x48, x49, x50, x51, x61, x721, x723, x725]
        store(kxy, stencil_idx_0_0_0_VVV, (x723 * (x727 + (x50 * x726)))); // kxy: Liveness = 6; [kxy, x48, x50, x723, x726, x727]
        vreal x51 = access(gtDD02, stencil_idx_0_0_0_VVV); // x51: Dependency! Liveness = 8; [gtDD02, x115, x48, x49, x50, x51, x61, x723]
        vreal x728 = access(AtDD02, stencil_idx_0_0_0_VVV); // x728: Dependency! Liveness = 10; [AtDD02, x115, x48, x49, x50, x51, x61, x721, x723, x725]
        store(kxz, stencil_idx_0_0_0_VVV, (x723 * (x728 + (x51 * x726)))); // kxz: Liveness = 7; [kxz, x48, x50, x51, x723, x726, x728]
        vreal x115 = access(gtDD11, stencil_idx_0_0_0_VVV); // x115: Dependency! Liveness = 8; [gtDD11, x115, x48, x49, x50, x51, x61, x723]
        vreal x729 = access(AtDD11, stencil_idx_0_0_0_VVV); // x729: Dependency! Liveness = 10; [AtDD11, x115, x48, x49, x50, x51, x61, x721, x723, x725]
        store(kyy, stencil_idx_0_0_0_VVV, (x723 * (x729 + (x115 * x726)))); // kyy: Liveness = 8; [kyy, x115, x48, x50, x51, x723, x726, x729]
        vreal x49 = access(gtDD12, stencil_idx_0_0_0_VVV); // x49: Dependency! Liveness = 8; [gtDD12, x115, x48, x49, x50, x51, x61, x723]
        vreal x730 = access(AtDD12, stencil_idx_0_0_0_VVV); // x730: Dependency! Liveness = 10; [AtDD12, x115, x48, x49, x50, x51, x61, x721, x723, x725]
        store(kyz, stencil_idx_0_0_0_VVV, (x723 * (x730 + (x49 * x726)))); // kyz: Liveness = 9; [kyz, x115, x48, x49, x50, x51, x723, x726, x730]
        vreal x61 = access(gtDD22, stencil_idx_0_0_0_VVV); // x61: Dependency! Liveness = 8; [gtDD22, x115, x48, x49, x50, x51, x61, x723]
        vreal x731 = access(AtDD22, stencil_idx_0_0_0_VVV); // x731: Dependency! Liveness = 10; [AtDD22, x115, x48, x49, x50, x51, x61, x721, x723, x725]
        store(kzz, stencil_idx_0_0_0_VVV, (x723 * (x731 + (x61 * x726)))); // kzz: Liveness = 10; [kzz, x115, x48, x49, x50, x51, x61, x723, x726, x731]
        vreal x732 = access(evo_lapse, stencil_idx_0_0_0_VVV); // x732: Dependency! Liveness = 11; [evo_lapse, x115, x48, x49, x50, x51, x61, x721, x723, x725, x732]
        vreal x733 = access(evo_shiftU0, stencil_idx_0_0_0_VVV); // x733: Dependency! Liveness = 12; [evo_shiftU0, x115, x48, x49, x50, x51, x61, x721, x723, x725, x732, x733]
        vreal x734 = access(evo_shiftU1, stencil_idx_0_0_0_VVV); // x734: Dependency! Liveness = 13; [evo_shiftU1, x115, x48, x49, x50, x51, x61, x721, x723, x725, x732, x733, x734]
        vreal x735 = access(evo_shiftU2, stencil_idx_0_0_0_VVV); // x735: Dependency! Liveness = 14; [evo_shiftU2, x115, x48, x49, x50, x51, x61, x721, x723, x725, x732, x733, x734, x735]
        store(alp, stencil_idx_0_0_0_VVV, x732); // alp: Liveness = 12; [alp, x115, x48, x49, x50, x51, x61, x723, x732, x733, x734, x735]
        store(betax, stencil_idx_0_0_0_VVV, x733); // betax: Liveness = 11; [betax, x115, x48, x49, x50, x51, x61, x723, x733, x734, x735]
        store(betay, stencil_idx_0_0_0_VVV, x734); // betay: Liveness = 10; [betay, x115, x48, x49, x50, x51, x61, x723, x734, x735]
        store(betaz, stencil_idx_0_0_0_VVV, x735); // betaz: Liveness = 9; [betaz, x115, x48, x49, x50, x51, x61, x723, x735]
        vreal x736 = (x48 * x723); // x736: Dependency! Liveness = 7; [x115, x48, x49, x50, x51, x61, x723]
        store(gxx, stencil_idx_0_0_0_VVV, x736); // gxx: Liveness = 9; [gxx, x115, x48, x49, x50, x51, x61, x723, x736]
        vreal x737 = (x50 * x723); // x737: Dependency! Liveness = 6; [x115, x49, x50, x51, x61, x723]
        store(gxy, stencil_idx_0_0_0_VVV, x737); // gxy: Liveness = 9; [gxy, x115, x48, x49, x50, x51, x61, x723, x737]
        vreal x738 = (x51 * x723); // x738: Dependency! Liveness = 5; [x115, x49, x51, x61, x723]
        store(gxz, stencil_idx_0_0_0_VVV, x738); // gxz: Liveness = 9; [gxz, x115, x48, x49, x50, x51, x61, x723, x738]
        vreal x739 = (x115 * x723); // x739: Dependency! Liveness = 4; [x115, x49, x61, x723]
        store(gyy, stencil_idx_0_0_0_VVV, x739); // gyy: Liveness = 9; [gyy, x115, x48, x49, x50, x51, x61, x723, x739]
        vreal x740 = (x49 * x723); // x740: Dependency! Liveness = 3; [x49, x61, x723]
        store(gyz, stencil_idx_0_0_0_VVV, x740); // gyz: Liveness = 9; [gyz, x115, x48, x49, x50, x51, x61, x723, x740]
        vreal x741 = (x61 * x723); // x741: Dependency! Liveness = 2; [x61, x723]
        store(gzz, stencil_idx_0_0_0_VVV, x741); // gzz: Liveness = 9; [gzz, x115, x48, x49, x50, x51, x61, x723, x741]    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}