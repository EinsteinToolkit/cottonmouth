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
void adm2bssn_pt1(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_adm2bssn_pt1;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("adm2bssn_pt1");
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
    #define dtbetax_layout VVV_layout
    #define dtbetay_layout VVV_layout
    #define dtbetaz_layout VVV_layout
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
    #define shift_BU0_layout VVV_layout
    #define shift_BU1_layout VVV_layout
    #define shift_BU2_layout VVV_layout
    #define trK_layout VVV_layout
    #define w_layout VVV_layout
    const auto DX = CCTK_DELTA_SPACE(0);
    const auto DY = CCTK_DELTA_SPACE(1);
    const auto DZ = CCTK_DELTA_SPACE(2);
    const auto DT = cctk_delta_time;
    const auto DXI = (1.0 / DX);
    const auto DYI = (1.0 / DY);
    const auto DZI = (1.0 / DZ);
    CCTK_ASSERT((cctk_nghostzones[0] >= 3));
    CCTK_ASSERT((cctk_nghostzones[1] >= 3));
    CCTK_ASSERT((cctk_nghostzones[2] >= 3));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // adm2bssn_pt1 loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x0 = access(betax, stencil_idx_0_0_0_VVV); // x0: Dependency! Liveness = 19; [betax, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x1 = ((1.0 / 60.0) * DXI); // x1: Dependency! Liveness = 20; [DXI, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6, x9]
        vreal x2 = (x0 * x1); // x2: Dependency! Liveness = 32; [x0, x1, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x22, x23, x24, x25, x26, x27, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x40, x42, x44, x46]
        vreal x3 = access(betay, stencil_idx_0_0_0_VVV); // x3: Dependency! Liveness = 19; [betay, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x4 = ((1.0 / 60.0) * DYI); // x4: Dependency! Liveness = 20; [DYI, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6, x9]
        vreal x5 = (x3 * x4); // x5: Dependency! Liveness = 33; [x0, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x22, x23, x24, x25, x26, x27, x29, x3, x30, x31, x32, x33, x34, x35, x36, x37, x38, x4, x40, x42, x44, x46]
        vreal x6 = access(betaz, stencil_idx_0_0_0_VVV); // x6: Dependency! Liveness = 19; [betaz, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x7 = ((1.0 / 60.0) * DZI); // x7: Dependency! Liveness = 20; [DZI, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6, x9]
        vreal x8 = (x6 * x7); // x8: Dependency! Liveness = 34; [x0, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x22, x23, x24, x25, x26, x27, x29, x3, x30, x31, x32, x33, x34, x35, x36, x37, x38, x40, x42, x44, x46, x6, x7]
        vreal x9 = access(alp, stencil_idx_0_0_0_VVV); // x9: Dependency! Liveness = 20; [alp, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6, x9]
        vreal x10 = ((4.0 / 3.0) * pown<vreal>(x9, -1)); // x10: Dependency! Liveness = 19; [x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6, x9]
        store(shift_BU0, stencil_idx_0_0_0_VVV, (-1 * x10 * ((-(access(dtbetax, stencil_idx_0_0_0_VVV))) + (x2 * ((-(stencil(betax, stencil_idx_m3_0_0_VVV))) + (-9 * stencil(betax, stencil_idx_2_0_0_VVV)) + (9 * stencil(betax, stencil_idx_m2_0_0_VVV)) + (45 * (((-(stencil(betax, stencil_idx_m1_0_0_VVV))) + stencil(betax, stencil_idx_1_0_0_VVV)))) + stencil(betax, stencil_idx_3_0_0_VVV))) + (x5 * ((-(stencil(betax, stencil_idx_0_m3_0_VVV))) + (-9 * stencil(betax, stencil_idx_0_2_0_VVV)) + (9 * stencil(betax, stencil_idx_0_m2_0_VVV)) + (45 * (((-(stencil(betax, stencil_idx_0_m1_0_VVV))) + stencil(betax, stencil_idx_0_1_0_VVV)))) + stencil(betax, stencil_idx_0_3_0_VVV))) + (x8 * ((-(stencil(betax, stencil_idx_0_0_m3_VVV))) + (-9 * stencil(betax, stencil_idx_0_0_2_VVV)) + (9 * stencil(betax, stencil_idx_0_0_m2_VVV)) + (45 * (((-(stencil(betax, stencil_idx_0_0_m1_VVV))) + stencil(betax, stencil_idx_0_0_1_VVV)))) + stencil(betax, stencil_idx_0_0_3_VVV)))))); // shift_BU0: Liveness = 24; [__dummy_stencil__betax__0__0__1_, __dummy_stencil__betax__0__0__2_, __dummy_stencil__betax__0__0__3_, __dummy_stencil__betax__0__0__m1_, __dummy_stencil__betax__0__0__m2_, __dummy_stencil__betax__0__0__m3_, __dummy_stencil__betax__0__1__0_, __dummy_stencil__betax__0__2__0_, __dummy_stencil__betax__0__3__0_, __dummy_stencil__betax__0__m1__0_, __dummy_stencil__betax__0__m2__0_, __dummy_stencil__betax__0__m3__0_, __dummy_stencil__betax__1__0__0_, __dummy_stencil__betax__2__0__0_, __dummy_stencil__betax__3__0__0_, __dummy_stencil__betax__m1__0__0_, __dummy_stencil__betax__m2__0__0_, __dummy_stencil__betax__m3__0__0_, dtbetax, shift_BU0, x10, x2, x5, x8]
        store(shift_BU1, stencil_idx_0_0_0_VVV, (-1 * x10 * ((-(access(dtbetay, stencil_idx_0_0_0_VVV))) + (x2 * ((-(stencil(betay, stencil_idx_m3_0_0_VVV))) + (-9 * stencil(betay, stencil_idx_2_0_0_VVV)) + (9 * stencil(betay, stencil_idx_m2_0_0_VVV)) + (45 * (((-(stencil(betay, stencil_idx_m1_0_0_VVV))) + stencil(betay, stencil_idx_1_0_0_VVV)))) + stencil(betay, stencil_idx_3_0_0_VVV))) + (x5 * ((-(stencil(betay, stencil_idx_0_m3_0_VVV))) + (-9 * stencil(betay, stencil_idx_0_2_0_VVV)) + (9 * stencil(betay, stencil_idx_0_m2_0_VVV)) + (45 * (((-(stencil(betay, stencil_idx_0_m1_0_VVV))) + stencil(betay, stencil_idx_0_1_0_VVV)))) + stencil(betay, stencil_idx_0_3_0_VVV))) + (x8 * ((-(stencil(betay, stencil_idx_0_0_m3_VVV))) + (-9 * stencil(betay, stencil_idx_0_0_2_VVV)) + (9 * stencil(betay, stencil_idx_0_0_m2_VVV)) + (45 * (((-(stencil(betay, stencil_idx_0_0_m1_VVV))) + stencil(betay, stencil_idx_0_0_1_VVV)))) + stencil(betay, stencil_idx_0_0_3_VVV)))))); // shift_BU1: Liveness = 24; [__dummy_stencil__betay__0__0__1_, __dummy_stencil__betay__0__0__2_, __dummy_stencil__betay__0__0__3_, __dummy_stencil__betay__0__0__m1_, __dummy_stencil__betay__0__0__m2_, __dummy_stencil__betay__0__0__m3_, __dummy_stencil__betay__0__1__0_, __dummy_stencil__betay__0__2__0_, __dummy_stencil__betay__0__3__0_, __dummy_stencil__betay__0__m1__0_, __dummy_stencil__betay__0__m2__0_, __dummy_stencil__betay__0__m3__0_, __dummy_stencil__betay__1__0__0_, __dummy_stencil__betay__2__0__0_, __dummy_stencil__betay__3__0__0_, __dummy_stencil__betay__m1__0__0_, __dummy_stencil__betay__m2__0__0_, __dummy_stencil__betay__m3__0__0_, dtbetay, shift_BU1, x10, x2, x5, x8]
        store(shift_BU2, stencil_idx_0_0_0_VVV, (-1 * x10 * ((-(access(dtbetaz, stencil_idx_0_0_0_VVV))) + (x2 * ((-(stencil(betaz, stencil_idx_m3_0_0_VVV))) + (-9 * stencil(betaz, stencil_idx_2_0_0_VVV)) + (9 * stencil(betaz, stencil_idx_m2_0_0_VVV)) + (45 * (((-(stencil(betaz, stencil_idx_m1_0_0_VVV))) + stencil(betaz, stencil_idx_1_0_0_VVV)))) + stencil(betaz, stencil_idx_3_0_0_VVV))) + (x5 * ((-(stencil(betaz, stencil_idx_0_m3_0_VVV))) + (-9 * stencil(betaz, stencil_idx_0_2_0_VVV)) + (9 * stencil(betaz, stencil_idx_0_m2_0_VVV)) + (45 * (((-(stencil(betaz, stencil_idx_0_m1_0_VVV))) + stencil(betaz, stencil_idx_0_1_0_VVV)))) + stencil(betaz, stencil_idx_0_3_0_VVV))) + (x8 * ((-(stencil(betaz, stencil_idx_0_0_m3_VVV))) + (-9 * stencil(betaz, stencil_idx_0_0_2_VVV)) + (9 * stencil(betaz, stencil_idx_0_0_m2_VVV)) + (45 * (((-(stencil(betaz, stencil_idx_0_0_m1_VVV))) + stencil(betaz, stencil_idx_0_0_1_VVV)))) + stencil(betaz, stencil_idx_0_0_3_VVV)))))); // shift_BU2: Liveness = 24; [__dummy_stencil__betaz__0__0__1_, __dummy_stencil__betaz__0__0__2_, __dummy_stencil__betaz__0__0__3_, __dummy_stencil__betaz__0__0__m1_, __dummy_stencil__betaz__0__0__m2_, __dummy_stencil__betaz__0__0__m3_, __dummy_stencil__betaz__0__1__0_, __dummy_stencil__betaz__0__2__0_, __dummy_stencil__betaz__0__3__0_, __dummy_stencil__betaz__0__m1__0_, __dummy_stencil__betaz__0__m2__0_, __dummy_stencil__betaz__0__m3__0_, __dummy_stencil__betaz__1__0__0_, __dummy_stencil__betaz__2__0__0_, __dummy_stencil__betaz__3__0__0_, __dummy_stencil__betaz__m1__0__0_, __dummy_stencil__betaz__m2__0__0_, __dummy_stencil__betaz__m3__0__0_, dtbetaz, shift_BU2, x10, x2, x5, x8]
        vreal x11 = access(gxx, stencil_idx_0_0_0_VVV); // x11: Dependency! Liveness = 19; [gxx, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x17 = access(gyy, stencil_idx_0_0_0_VVV); // x17: Dependency! Liveness = 19; [gyy, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x24 = access(kzz, stencil_idx_0_0_0_VVV); // x24: Dependency! Liveness = 19; [kzz, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x15 = access(gxy, stencil_idx_0_0_0_VVV); // x15: Dependency! Liveness = 19; [gxy, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x16 = pow2(x15); // x16: Dependency! Liveness = 2; [x15, x18]
        vreal x25 = (x24 * ((-(x16)) + (x11 * x17))); // x25: Dependency! Liveness = 33; [x0, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x22, x23, x24, x25, x26, x27, x29, x3, x30, x31, x32, x33, x34, x35, x36, x37, x38, x40, x42, x44, x46, x6]
        vreal x14 = access(gzz, stencil_idx_0_0_0_VVV); // x14: Dependency! Liveness = 19; [gzz, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x31 = access(kyy, stencil_idx_0_0_0_VVV); // x31: Dependency! Liveness = 19; [kyy, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x18 = access(gxz, stencil_idx_0_0_0_VVV); // x18: Dependency! Liveness = 19; [gxz, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x19 = pow2(x18); // x19: Dependency! Liveness = 1; [x18]
        vreal x32 = (x31 * ((-(x19)) + (x11 * x14))); // x32: Dependency! Liveness = 31; [x0, x11, x12, x13, x14, x15, x17, x18, x19, x20, x22, x23, x25, x26, x27, x29, x3, x30, x31, x32, x33, x34, x35, x36, x37, x38, x40, x42, x44, x46, x6]
        vreal x23 = access(kxx, stencil_idx_0_0_0_VVV); // x23: Dependency! Liveness = 19; [kxx, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x12 = access(gyz, stencil_idx_0_0_0_VVV); // x12: Dependency! Liveness = 19; [gyz, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x13 = pow2(x12); // x13: Dependency! Liveness = 3; [x12, x15, x18]
        vreal x37 = (x23 * ((-(x13)) + (x14 * x17))); // x37: Dependency! Liveness = 29; [x0, x11, x12, x13, x14, x15, x17, x18, x20, x22, x23, x25, x26, x27, x29, x3, x30, x32, x33, x34, x35, x36, x37, x38, x40, x42, x44, x46, x6]
        vreal x33 = access(kxz, stencil_idx_0_0_0_VVV); // x33: Dependency! Liveness = 19; [kxz, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x34 = (x33 * ((x12 * x15) + (-1 * x17 * x18))); // x34: Dependency! Liveness = 19; [x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x33, x38, x40, x42, x44, x46, x6]
        vreal x29 = access(kyz, stencil_idx_0_0_0_VVV); // x29: Dependency! Liveness = 19; [kyz, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x35 = access(kxy, stencil_idx_0_0_0_VVV); // x35: Dependency! Liveness = 19; [kxy, x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6]
        vreal x30 = ((x11 * x12) + (-1 * x15 * x18)); // x30: Dependency! Liveness = 19; [x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6, x9]
        vreal x36 = ((x14 * x15) + (-1 * x12 * x18)); // x36: Dependency! Liveness = 19; [x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6, x9]
        vreal x20 = ((x11 * (x13 + (-1 * x14 * x17))) + (x14 * x16) + (x17 * x19) + (-2 * x12 * x15 * x18)); // x20: Dependency! Liveness = 30; [x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x22, x23, x24, x25, x26, x27, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x40, x42, x44, x46]
        vreal x21 = (-(x20)); // x21: Dependency! Liveness = 19; [x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6, x9]
        vreal x22 = pow(static_cast<vreal>(x21), (-1.0 / 3.0)); // x22: Dependency! Liveness = 20; [x0, x11, x12, x14, x15, x17, x18, x20, x21, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6, x9]
        vreal x26 = pown<vreal>(x20, -1); // x26: Dependency! Liveness = 19; [x0, x11, x12, x14, x15, x17, x18, x20, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6, x9]
        vreal x27 = (x11 * x26); // x27: Dependency! Liveness = 8; [x11, x12, x14, x15, x17, x18, x22, x26]
        vreal x28 = ((1.0 / 3.0) * x27); // x28: Dependency! Liveness = 18; [x0, x11, x12, x14, x15, x17, x18, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6, x9]
        store(AtDD00, stencil_idx_0_0_0_VVV, (-1 * x22 * ((-(x23)) + (-1 * x28 * (x25 + x32 + x37)) + ((-2.0 / 3.0) * x27 * x34) + ((2.0 / 3.0) * x11 * x26 * ((x29 * x30) + (x35 * x36)))))); // AtDD00: Liveness = 15; [AtDD00, x11, x22, x23, x25, x26, x27, x28, x29, x30, x32, x34, x35, x36, x37]
        vreal x42 = (x17 * x26); // x42: Dependency! Liveness = 7; [x12, x14, x15, x17, x18, x22, x26]
        vreal x43 = ((1.0 / 3.0) * x42); // x43: Dependency! Liveness = 15; [x0, x11, x12, x14, x15, x17, x18, x22, x26, x3, x42, x44, x46, x6, x9]
        store(AtDD11, stencil_idx_0_0_0_VVV, (-1 * x22 * ((-(x31)) + (-1 * x43 * (x25 + x32 + x37)) + ((-2.0 / 3.0) * x34 * x42) + ((2.0 / 3.0) * x17 * x26 * ((x29 * x30) + (x35 * x36)))))); // AtDD11: Liveness = 18; [AtDD11, x11, x17, x22, x23, x25, x26, x27, x29, x30, x31, x32, x34, x35, x36, x37, x42, x43]
        vreal x46 = (x14 * x26); // x46: Dependency! Liveness = 6; [x12, x14, x15, x18, x22, x26]
        vreal x47 = ((1.0 / 3.0) * x46); // x47: Dependency! Liveness = 13; [x0, x11, x12, x14, x15, x17, x18, x22, x26, x3, x46, x6, x9]
        store(AtDD22, stencil_idx_0_0_0_VVV, (-1 * x22 * ((-(x24)) + (-1 * x47 * (x25 + x32 + x37)) + ((-2.0 / 3.0) * x34 * x46) + ((2.0 / 3.0) * x14 * x26 * ((x29 * x30) + (x35 * x36)))))); // AtDD22: Liveness = 21; [AtDD22, x11, x14, x17, x22, x23, x24, x25, x26, x27, x29, x30, x31, x32, x34, x35, x36, x37, x42, x46, x47]
        vreal x40 = (x18 * x26); // x40: Dependency! Liveness = 4; [x12, x15, x18, x26]
        vreal x41 = ((1.0 / 3.0) * x40); // x41: Dependency! Liveness = 16; [x0, x11, x12, x14, x15, x17, x18, x22, x26, x3, x40, x42, x44, x46, x6, x9]
        store(AtDD02, stencil_idx_0_0_0_VVV, (-1 * x22 * ((-(x33)) + (-1 * x41 * (x25 + x32 + x37)) + ((-2.0 / 3.0) * x34 * x40) + ((2.0 / 3.0) * x18 * x26 * ((x29 * x30) + (x35 * x36)))))); // AtDD02: Liveness = 24; [AtDD02, x11, x14, x17, x18, x22, x23, x24, x25, x26, x27, x29, x30, x31, x32, x33, x34, x35, x36, x37, x40, x41, x42, x46]
        vreal x38 = (x15 * x26); // x38: Dependency! Liveness = 4; [x12, x15, x18, x26]
        vreal x39 = ((1.0 / 3.0) * x38); // x39: Dependency! Liveness = 17; [x0, x11, x12, x14, x15, x17, x18, x22, x26, x3, x38, x40, x42, x44, x46, x6, x9]
        store(AtDD01, stencil_idx_0_0_0_VVV, (-1 * x22 * ((-(x35)) + (-1 * x39 * (x25 + x32 + x37)) + ((-2.0 / 3.0) * x34 * x38) + ((2.0 / 3.0) * x15 * x26 * ((x29 * x30) + (x35 * x36)))))); // AtDD01: Liveness = 26; [AtDD01, x11, x14, x15, x17, x18, x22, x23, x24, x25, x26, x27, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x42, x46]
        vreal x44 = (x12 * x26); // x44: Dependency! Liveness = 4; [x12, x15, x18, x26]
        vreal x45 = ((1.0 / 3.0) * x44); // x45: Dependency! Liveness = 14; [x0, x11, x12, x14, x15, x17, x18, x22, x26, x3, x44, x46, x6, x9]
        store(AtDD12, stencil_idx_0_0_0_VVV, (-1 * x22 * ((-(x29)) + (-1 * x45 * (x25 + x32 + x37)) + ((-2.0 / 3.0) * x34 * x44) + ((2.0 / 3.0) * x12 * x26 * ((x29 * x30) + (x35 * x36)))))); // AtDD12: Liveness = 28; [AtDD12, x11, x12, x14, x15, x17, x18, x22, x23, x24, x25, x26, x27, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x40, x42, x44, x45, x46]
        store(trK, stencil_idx_0_0_0_VVV, (x26 * ((-(x25)) + (-(x32)) + (-(x37)) + (-2 * x34) + (2 * x29 * x30) + (2 * x35 * x36)))); // trK: Liveness = 28; [trK, x0, x11, x12, x14, x15, x17, x18, x20, x22, x25, x26, x27, x29, x3, x30, x32, x33, x34, x35, x36, x37, x38, x40, x42, x44, x46, x6]
        store(w, stencil_idx_0_0_0_VVV, pow(static_cast<vreal>(x21), (-1.0 / 6.0))); // w: Liveness = 21; [w, x0, x11, x12, x14, x15, x17, x18, x20, x21, x22, x26, x27, x3, x38, x40, x42, x44, x46, x6, x9]
        store(evo_lapse, stencil_idx_0_0_0_VVV, x9); // evo_lapse: Liveness = 13; [evo_lapse, x0, x11, x12, x14, x15, x17, x18, x22, x26, x3, x6, x9]
        store(evo_shiftU0, stencil_idx_0_0_0_VVV, x0); // evo_shiftU0: Liveness = 12; [evo_shiftU0, x0, x11, x12, x14, x15, x17, x18, x22, x26, x3, x6]
        store(evo_shiftU1, stencil_idx_0_0_0_VVV, x3); // evo_shiftU1: Liveness = 11; [evo_shiftU1, x11, x12, x14, x15, x17, x18, x22, x26, x3, x6]
        store(evo_shiftU2, stencil_idx_0_0_0_VVV, x6); // evo_shiftU2: Liveness = 10; [evo_shiftU2, x11, x12, x14, x15, x17, x18, x22, x26, x6]
        store(gtDD00, stencil_idx_0_0_0_VVV, (x11 * x22)); // gtDD00: Liveness = 9; [gtDD00, x11, x12, x14, x15, x17, x18, x22, x26]
        store(gtDD11, stencil_idx_0_0_0_VVV, (x17 * x22)); // gtDD11: Liveness = 9; [gtDD11, x11, x12, x14, x15, x17, x18, x22, x26]
        store(gtDD22, stencil_idx_0_0_0_VVV, (x14 * x22)); // gtDD22: Liveness = 9; [gtDD22, x11, x12, x14, x15, x17, x18, x22, x26]
        store(gtDD01, stencil_idx_0_0_0_VVV, (x15 * x22)); // gtDD01: Liveness = 6; [gtDD01, x12, x15, x18, x22, x26]
        store(gtDD02, stencil_idx_0_0_0_VVV, (x18 * x22)); // gtDD02: Liveness = 6; [gtDD02, x12, x15, x18, x22, x26]
        store(gtDD12, stencil_idx_0_0_0_VVV, (x12 * x22)); // gtDD12: Liveness = 6; [gtDD12, x12, x15, x18, x22, x26]    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}