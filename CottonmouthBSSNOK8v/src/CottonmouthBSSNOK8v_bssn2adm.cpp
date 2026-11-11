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
        vreal x48 = access(gtDD00, stencil_idx_0_0_0_VVV); // x48: Dependency! Liveness = 8; [gtDD00, x129, x48, x49, x50, x51, x63, x900]
        vreal x901 = access(AtDD00, stencil_idx_0_0_0_VVV); // x901: Dependency! Liveness = 9; [AtDD00, x129, x48, x49, x50, x51, x63, x898, x900]
        vreal x898 = access(w, stencil_idx_0_0_0_VVV); // x898: Dependency! Liveness = 9; [w, x129, x48, x49, x50, x51, x63, x898, x900]
        vreal x899 = pow2(x898); // x899: Dependency! Liveness = 14; [x129, x48, x49, x50, x51, x63, x898, x899, x900, x902, x909, x910, x911, x912]
        vreal x900 = pown<vreal>(x899, -1); // x900: Dependency! Liveness = 13; [x129, x48, x49, x50, x51, x63, x899, x900, x902, x909, x910, x911, x912]
        vreal x902 = access(trK, stencil_idx_0_0_0_VVV); // x902: Dependency! Liveness = 10; [trK, x129, x48, x49, x50, x51, x63, x898, x900, x902]
        vreal x903 = ((1.0 / 3.0) * x902); // x903: Dependency! Liveness = 12; [x129, x48, x49, x50, x51, x63, x900, x902, x909, x910, x911, x912]
        store(kxx, stencil_idx_0_0_0_VVV, (x900 * (x901 + (x48 * x903)))); // kxx: Liveness = 5; [kxx, x48, x900, x901, x903]
        vreal x50 = access(gtDD01, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 8; [gtDD01, x129, x48, x49, x50, x51, x63, x900]
        vreal x904 = access(AtDD01, stencil_idx_0_0_0_VVV); // x904: Dependency! Liveness = 10; [AtDD01, x129, x48, x49, x50, x51, x63, x898, x900, x902]
        store(kxy, stencil_idx_0_0_0_VVV, (x900 * (x904 + (x50 * x903)))); // kxy: Liveness = 6; [kxy, x48, x50, x900, x903, x904]
        vreal x51 = access(gtDD02, stencil_idx_0_0_0_VVV); // x51: Dependency! Liveness = 8; [gtDD02, x129, x48, x49, x50, x51, x63, x900]
        vreal x905 = access(AtDD02, stencil_idx_0_0_0_VVV); // x905: Dependency! Liveness = 10; [AtDD02, x129, x48, x49, x50, x51, x63, x898, x900, x902]
        store(kxz, stencil_idx_0_0_0_VVV, (x900 * (x905 + (x51 * x903)))); // kxz: Liveness = 7; [kxz, x48, x50, x51, x900, x903, x905]
        vreal x129 = access(gtDD11, stencil_idx_0_0_0_VVV); // x129: Dependency! Liveness = 8; [gtDD11, x129, x48, x49, x50, x51, x63, x900]
        vreal x906 = access(AtDD11, stencil_idx_0_0_0_VVV); // x906: Dependency! Liveness = 10; [AtDD11, x129, x48, x49, x50, x51, x63, x898, x900, x902]
        store(kyy, stencil_idx_0_0_0_VVV, (x900 * (x906 + (x129 * x903)))); // kyy: Liveness = 8; [kyy, x129, x48, x50, x51, x900, x903, x906]
        vreal x49 = access(gtDD12, stencil_idx_0_0_0_VVV); // x49: Dependency! Liveness = 8; [gtDD12, x129, x48, x49, x50, x51, x63, x900]
        vreal x907 = access(AtDD12, stencil_idx_0_0_0_VVV); // x907: Dependency! Liveness = 10; [AtDD12, x129, x48, x49, x50, x51, x63, x898, x900, x902]
        store(kyz, stencil_idx_0_0_0_VVV, (x900 * (x907 + (x49 * x903)))); // kyz: Liveness = 9; [kyz, x129, x48, x49, x50, x51, x900, x903, x907]
        vreal x63 = access(gtDD22, stencil_idx_0_0_0_VVV); // x63: Dependency! Liveness = 8; [gtDD22, x129, x48, x49, x50, x51, x63, x900]
        vreal x908 = access(AtDD22, stencil_idx_0_0_0_VVV); // x908: Dependency! Liveness = 10; [AtDD22, x129, x48, x49, x50, x51, x63, x898, x900, x902]
        store(kzz, stencil_idx_0_0_0_VVV, (x900 * (x908 + (x63 * x903)))); // kzz: Liveness = 10; [kzz, x129, x48, x49, x50, x51, x63, x900, x903, x908]
        vreal x909 = access(evo_lapse, stencil_idx_0_0_0_VVV); // x909: Dependency! Liveness = 11; [evo_lapse, x129, x48, x49, x50, x51, x63, x898, x900, x902, x909]
        vreal x910 = access(evo_shiftU0, stencil_idx_0_0_0_VVV); // x910: Dependency! Liveness = 12; [evo_shiftU0, x129, x48, x49, x50, x51, x63, x898, x900, x902, x909, x910]
        vreal x911 = access(evo_shiftU1, stencil_idx_0_0_0_VVV); // x911: Dependency! Liveness = 13; [evo_shiftU1, x129, x48, x49, x50, x51, x63, x898, x900, x902, x909, x910, x911]
        vreal x912 = access(evo_shiftU2, stencil_idx_0_0_0_VVV); // x912: Dependency! Liveness = 14; [evo_shiftU2, x129, x48, x49, x50, x51, x63, x898, x900, x902, x909, x910, x911, x912]
        store(alp, stencil_idx_0_0_0_VVV, x909); // alp: Liveness = 12; [alp, x129, x48, x49, x50, x51, x63, x900, x909, x910, x911, x912]
        store(betax, stencil_idx_0_0_0_VVV, x910); // betax: Liveness = 11; [betax, x129, x48, x49, x50, x51, x63, x900, x910, x911, x912]
        store(betay, stencil_idx_0_0_0_VVV, x911); // betay: Liveness = 10; [betay, x129, x48, x49, x50, x51, x63, x900, x911, x912]
        store(betaz, stencil_idx_0_0_0_VVV, x912); // betaz: Liveness = 9; [betaz, x129, x48, x49, x50, x51, x63, x900, x912]
        vreal x913 = (x48 * x900); // x913: Dependency! Liveness = 7; [x129, x48, x49, x50, x51, x63, x900]
        store(gxx, stencil_idx_0_0_0_VVV, x913); // gxx: Liveness = 9; [gxx, x129, x48, x49, x50, x51, x63, x900, x913]
        vreal x914 = (x50 * x900); // x914: Dependency! Liveness = 6; [x129, x49, x50, x51, x63, x900]
        store(gxy, stencil_idx_0_0_0_VVV, x914); // gxy: Liveness = 9; [gxy, x129, x48, x49, x50, x51, x63, x900, x914]
        vreal x915 = (x51 * x900); // x915: Dependency! Liveness = 5; [x129, x49, x51, x63, x900]
        store(gxz, stencil_idx_0_0_0_VVV, x915); // gxz: Liveness = 9; [gxz, x129, x48, x49, x50, x51, x63, x900, x915]
        vreal x916 = (x129 * x900); // x916: Dependency! Liveness = 4; [x129, x49, x63, x900]
        store(gyy, stencil_idx_0_0_0_VVV, x916); // gyy: Liveness = 9; [gyy, x129, x48, x49, x50, x51, x63, x900, x916]
        vreal x917 = (x49 * x900); // x917: Dependency! Liveness = 3; [x49, x63, x900]
        store(gyz, stencil_idx_0_0_0_VVV, x917); // gyz: Liveness = 9; [gyz, x129, x48, x49, x50, x51, x63, x900, x917]
        vreal x918 = (x63 * x900); // x918: Dependency! Liveness = 2; [x63, x900]
        store(gzz, stencil_idx_0_0_0_VVV, x918); // gzz: Liveness = 9; [gzz, x129, x48, x49, x50, x51, x63, x900, x918]    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}