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
        vreal x99 = access(gtDD00, stencil_idx_0_0_0_VVV); // x99: Dependency! Liveness = 14; [gtDD00, x1143, x1273, x1289, x1292, x1295, x36, x37, x38, x39, x52, x852, x877, x99]
        vreal x852 = access(chi, stencil_idx_0_0_0_VVV); // x852: Dependency! Liveness = 14; [chi, x1143, x1273, x1289, x1292, x1295, x36, x37, x38, x39, x52, x852, x877, x99]
        vreal x877 = pown<vreal>(x852, -1); // x877: Dependency! Liveness = 13; [x1143, x1273, x1289, x1292, x1295, x36, x37, x38, x39, x52, x852, x877, x99]
        vreal x894 = (x877 * x99); // x894: Dependency! Liveness = 7; [x36, x37, x38, x39, x52, x877, x99]
        store(gxx, stencil_idx_0_0_0_VVV, x894); // gxx: 
        vreal x36 = access(gtDD01, stencil_idx_0_0_0_VVV); // x36: Dependency! Liveness = 13; [gtDD01, x1143, x1273, x1289, x1292, x1295, x36, x37, x38, x39, x52, x877, x99]
        vreal x911 = (x36 * x877); // x911: Dependency! Liveness = 6; [x36, x37, x38, x39, x52, x877]
        store(gxy, stencil_idx_0_0_0_VVV, x911); // gxy: 
        vreal x39 = access(gtDD11, stencil_idx_0_0_0_VVV); // x39: Dependency! Liveness = 13; [gtDD11, x1143, x1273, x1289, x1292, x1295, x36, x37, x38, x39, x52, x877, x99]
        vreal x915 = (x39 * x877); // x915: Dependency! Liveness = 5; [x37, x38, x39, x52, x877]
        store(gyy, stencil_idx_0_0_0_VVV, x915); // gyy: 
        vreal x38 = access(gtDD02, stencil_idx_0_0_0_VVV); // x38: Dependency! Liveness = 13; [gtDD02, x1143, x1273, x1289, x1292, x1295, x36, x37, x38, x39, x52, x877, x99]
        vreal x919 = (x38 * x877); // x919: Dependency! Liveness = 4; [x37, x38, x52, x877]
        store(gxz, stencil_idx_0_0_0_VVV, x919); // gxz: 
        vreal x37 = access(gtDD12, stencil_idx_0_0_0_VVV); // x37: Dependency! Liveness = 13; [gtDD12, x1143, x1273, x1289, x1292, x1295, x36, x37, x38, x39, x52, x877, x99]
        vreal x924 = (x37 * x877); // x924: Dependency! Liveness = 3; [x37, x52, x877]
        store(gyz, stencil_idx_0_0_0_VVV, x924); // gyz: 
        vreal x52 = access(gtDD22, stencil_idx_0_0_0_VVV); // x52: Dependency! Liveness = 13; [gtDD22, x1143, x1273, x1289, x1292, x1295, x36, x37, x38, x39, x52, x877, x99]
        vreal x928 = (x52 * x877); // x928: Dependency! Liveness = 2; [x52, x877]
        store(gzz, stencil_idx_0_0_0_VVV, x928); // gzz: 
        vreal x1162 = access(AtDD00, stencil_idx_0_0_0_VVV); // x1162: Dependency! Liveness = 9; [AtDD00, x1143, x36, x37, x38, x39, x52, x877, x99]
        vreal x1143 = access(trK, stencil_idx_0_0_0_VVV); // x1143: Dependency! Liveness = 9; [trK, x1143, x36, x37, x38, x39, x52, x877, x99]
        vreal x1431 = ((1.0 / 3.0) * x1143); // x1431: Dependency! Liveness = 12; [x1143, x1273, x1289, x1292, x1295, x36, x37, x38, x39, x52, x877, x99]
        store(kxx, stencil_idx_0_0_0_VVV, (x877 * (x1162 + (x1431 * x99)))); // kxx: 
        vreal x1163 = access(AtDD01, stencil_idx_0_0_0_VVV); // x1163: Dependency! Liveness = 9; [AtDD01, x1143, x36, x37, x38, x39, x52, x877, x99]
        store(kxy, stencil_idx_0_0_0_VVV, (x877 * (x1163 + (x1431 * x36)))); // kxy: 
        vreal x1165 = access(AtDD11, stencil_idx_0_0_0_VVV); // x1165: Dependency! Liveness = 9; [AtDD11, x1143, x36, x37, x38, x39, x52, x877, x99]
        store(kyy, stencil_idx_0_0_0_VVV, (x877 * (x1165 + (x1431 * x39)))); // kyy: 
        vreal x1157 = access(AtDD02, stencil_idx_0_0_0_VVV); // x1157: Dependency! Liveness = 9; [AtDD02, x1143, x36, x37, x38, x39, x52, x877, x99]
        store(kxz, stencil_idx_0_0_0_VVV, (x877 * (x1157 + (x1431 * x38)))); // kxz: 
        vreal x1159 = access(AtDD12, stencil_idx_0_0_0_VVV); // x1159: Dependency! Liveness = 9; [AtDD12, x1143, x36, x37, x38, x39, x52, x877, x99]
        store(kyz, stencil_idx_0_0_0_VVV, (x877 * (x1159 + (x1431 * x37)))); // kyz: 
        vreal x1155 = access(AtDD22, stencil_idx_0_0_0_VVV); // x1155: Dependency! Liveness = 9; [AtDD22, x1143, x36, x37, x38, x39, x52, x877, x99]
        store(kzz, stencil_idx_0_0_0_VVV, (x877 * (x1155 + (x1431 * x52)))); // kzz: 
        vreal x1273 = access(evo_lapse, stencil_idx_0_0_0_VVV); // x1273: Dependency! Liveness = 10; [evo_lapse, x1143, x1273, x36, x37, x38, x39, x52, x877, x99]
        store(alp, stencil_idx_0_0_0_VVV, x1273); // alp: 
        vreal x1289 = access(evo_shiftU0, stencil_idx_0_0_0_VVV); // x1289: Dependency! Liveness = 11; [evo_shiftU0, x1143, x1273, x1289, x36, x37, x38, x39, x52, x877, x99]
        store(betax, stencil_idx_0_0_0_VVV, x1289); // betax: 
        vreal x1292 = access(evo_shiftU1, stencil_idx_0_0_0_VVV); // x1292: Dependency! Liveness = 12; [evo_shiftU1, x1143, x1273, x1289, x1292, x36, x37, x38, x39, x52, x877, x99]
        store(betay, stencil_idx_0_0_0_VVV, x1292); // betay: 
        vreal x1295 = access(evo_shiftU2, stencil_idx_0_0_0_VVV); // x1295: Dependency! Liveness = 13; [evo_shiftU2, x1143, x1273, x1289, x1292, x1295, x36, x37, x38, x39, x52, x877, x99]
        store(betaz, stencil_idx_0_0_0_VVV, x1295); // betaz:     
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}