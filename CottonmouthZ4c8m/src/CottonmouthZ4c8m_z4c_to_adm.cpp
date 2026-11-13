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
        vreal x99 = access(gtDD00, stencil_idx_0_0_0_VVV); // x99: Dependency! Liveness = 14; [gtDD00, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x852, x872, x99]
        vreal x852 = access(chi, stencil_idx_0_0_0_VVV); // x852: Dependency! Liveness = 14; [chi, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x852, x872, x99]
        vreal x872 = pown<vreal>(x852, -1); // x872: Dependency! Liveness = 13; [x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x852, x872, x99]
        vreal x882 = (x872 * x99); // x882: Dependency! Liveness = 7; [x36, x37, x38, x39, x52, x872, x99]
        store(gxx, stencil_idx_0_0_0_VVV, x882); // gxx: 
        vreal x36 = access(gtDD01, stencil_idx_0_0_0_VVV); // x36: Dependency! Liveness = 13; [gtDD01, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x872, x99]
        vreal x899 = (x36 * x872); // x899: Dependency! Liveness = 6; [x36, x37, x38, x39, x52, x872]
        store(gxy, stencil_idx_0_0_0_VVV, x899); // gxy: 
        vreal x39 = access(gtDD11, stencil_idx_0_0_0_VVV); // x39: Dependency! Liveness = 13; [gtDD11, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x872, x99]
        vreal x903 = (x39 * x872); // x903: Dependency! Liveness = 5; [x37, x38, x39, x52, x872]
        store(gyy, stencil_idx_0_0_0_VVV, x903); // gyy: 
        vreal x38 = access(gtDD02, stencil_idx_0_0_0_VVV); // x38: Dependency! Liveness = 13; [gtDD02, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x872, x99]
        vreal x907 = (x38 * x872); // x907: Dependency! Liveness = 4; [x37, x38, x52, x872]
        store(gxz, stencil_idx_0_0_0_VVV, x907); // gxz: 
        vreal x37 = access(gtDD12, stencil_idx_0_0_0_VVV); // x37: Dependency! Liveness = 13; [gtDD12, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x872, x99]
        vreal x912 = (x37 * x872); // x912: Dependency! Liveness = 3; [x37, x52, x872]
        store(gyz, stencil_idx_0_0_0_VVV, x912); // gyz: 
        vreal x52 = access(gtDD22, stencil_idx_0_0_0_VVV); // x52: Dependency! Liveness = 13; [gtDD22, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x872, x99]
        vreal x916 = (x52 * x872); // x916: Dependency! Liveness = 2; [x52, x872]
        store(gzz, stencil_idx_0_0_0_VVV, x916); // gzz: 
        vreal x1171 = access(AtDD00, stencil_idx_0_0_0_VVV); // x1171: Dependency! Liveness = 13; [AtDD00, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x872, x99]
        vreal x1131 = access(trK, stencil_idx_0_0_0_VVV); // x1131: Dependency! Liveness = 9; [trK, x1131, x36, x37, x38, x39, x52, x872, x99]
        vreal x1447 = ((1.0 / 3.0) * x1131); // x1447: Dependency! Liveness = 12; [x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x872, x99]
        store(kxx, stencil_idx_0_0_0_VVV, (x872 * (x1171 + (x1447 * x99)))); // kxx: 
        vreal x1172 = access(AtDD01, stencil_idx_0_0_0_VVV); // x1172: Dependency! Liveness = 13; [AtDD01, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x872, x99]
        store(kxy, stencil_idx_0_0_0_VVV, (x872 * (x1172 + (x1447 * x36)))); // kxy: 
        vreal x1174 = access(AtDD11, stencil_idx_0_0_0_VVV); // x1174: Dependency! Liveness = 13; [AtDD11, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x872, x99]
        store(kyy, stencil_idx_0_0_0_VVV, (x872 * (x1174 + (x1447 * x39)))); // kyy: 
        vreal x1166 = access(AtDD02, stencil_idx_0_0_0_VVV); // x1166: Dependency! Liveness = 13; [AtDD02, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x872, x99]
        store(kxz, stencil_idx_0_0_0_VVV, (x872 * (x1166 + (x1447 * x38)))); // kxz: 
        vreal x1168 = access(AtDD12, stencil_idx_0_0_0_VVV); // x1168: Dependency! Liveness = 13; [AtDD12, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x872, x99]
        store(kyz, stencil_idx_0_0_0_VVV, (x872 * (x1168 + (x1447 * x37)))); // kyz: 
        vreal x1164 = access(AtDD22, stencil_idx_0_0_0_VVV); // x1164: Dependency! Liveness = 13; [AtDD22, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x872, x99]
        store(kzz, stencil_idx_0_0_0_VVV, (x872 * (x1164 + (x1447 * x52)))); // kzz: 
        vreal x1143 = access(evo_lapse, stencil_idx_0_0_0_VVV); // x1143: Dependency! Liveness = 10; [evo_lapse, x1131, x1143, x36, x37, x38, x39, x52, x872, x99]
        store(alp, stencil_idx_0_0_0_VVV, x1143); // alp: 
        vreal x1145 = access(evo_shiftU0, stencil_idx_0_0_0_VVV); // x1145: Dependency! Liveness = 11; [evo_shiftU0, x1131, x1143, x1145, x36, x37, x38, x39, x52, x872, x99]
        store(betax, stencil_idx_0_0_0_VVV, x1145); // betax: 
        vreal x1148 = access(evo_shiftU1, stencil_idx_0_0_0_VVV); // x1148: Dependency! Liveness = 12; [evo_shiftU1, x1131, x1143, x1145, x1148, x36, x37, x38, x39, x52, x872, x99]
        store(betay, stencil_idx_0_0_0_VVV, x1148); // betay: 
        vreal x1150 = access(evo_shiftU2, stencil_idx_0_0_0_VVV); // x1150: Dependency! Liveness = 13; [evo_shiftU2, x1131, x1143, x1145, x1148, x1150, x36, x37, x38, x39, x52, x872, x99]
        store(betaz, stencil_idx_0_0_0_VVV, x1150); // betaz:     
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}