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
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    CCTK_ASSERT((cctk_nghostzones[0] >= 2));
    CCTK_ASSERT((cctk_nghostzones[1] >= 2));
    CCTK_ASSERT((cctk_nghostzones[2] >= 2));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // adm2bssn_pt1 loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x0 = access(betax, stencil_idx_0_0_0_VVV); // x0: Dependency! Liveness = 13; [betax, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x8, x9]
        vreal x1 = (DXI * x0); // x1: Dependency! Liveness = 19; [DXI, x0, x10, x11, x12, x13, x14, x15, x16, x17, x19, x20, x21, x25, x28, x32, x34, x8, x9]
        vreal x2 = access(betay, stencil_idx_0_0_0_VVV); // x2: Dependency! Liveness = 13; [betay, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x8, x9]
        vreal x3 = (DYI * x2); // x3: Dependency! Liveness = 20; [DYI, x0, x10, x11, x12, x13, x14, x15, x16, x17, x19, x2, x20, x21, x25, x28, x32, x34, x8, x9]
        vreal x4 = access(betaz, stencil_idx_0_0_0_VVV); // x4: Dependency! Liveness = 13; [betaz, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x8, x9]
        vreal x5 = (DZI * x4); // x5: Dependency! Liveness = 21; [DZI, x0, x10, x11, x12, x13, x14, x15, x16, x17, x19, x2, x20, x21, x25, x28, x32, x34, x4, x8, x9]
        vreal x6 = access(alp, stencil_idx_0_0_0_VVV); // x6: Dependency! Liveness = 14; [alp, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x6, x8, x9]
        vreal x7 = ((4.0 / 3.0) * pown<vreal>(x6, -1)); // x7: Dependency! Liveness = 13; [x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x6, x8, x9]
        store(shift_BU0, stencil_idx_0_0_0_VVV, (x7 * (access(dtbetax, stencil_idx_0_0_0_VVV) + (-1 * x1 * (((1.0 / 12.0) * (((-(stencil(betax, stencil_idx_2_0_0_VVV))) + stencil(betax, stencil_idx_m2_0_0_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betax, stencil_idx_m1_0_0_VVV))) + stencil(betax, stencil_idx_1_0_0_VVV)))))) + (-1 * x3 * (((1.0 / 12.0) * (((-(stencil(betax, stencil_idx_0_2_0_VVV))) + stencil(betax, stencil_idx_0_m2_0_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betax, stencil_idx_0_m1_0_VVV))) + stencil(betax, stencil_idx_0_1_0_VVV)))))) + (-1 * x5 * (((1.0 / 12.0) * (((-(stencil(betax, stencil_idx_0_0_2_VVV))) + stencil(betax, stencil_idx_0_0_m2_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betax, stencil_idx_0_0_m1_VVV))) + stencil(betax, stencil_idx_0_0_1_VVV))))))))); // shift_BU0: Liveness = 18; [__dummy_stencil__betax__0__0__1_, __dummy_stencil__betax__0__0__2_, __dummy_stencil__betax__0__0__m1_, __dummy_stencil__betax__0__0__m2_, __dummy_stencil__betax__0__1__0_, __dummy_stencil__betax__0__2__0_, __dummy_stencil__betax__0__m1__0_, __dummy_stencil__betax__0__m2__0_, __dummy_stencil__betax__1__0__0_, __dummy_stencil__betax__2__0__0_, __dummy_stencil__betax__m1__0__0_, __dummy_stencil__betax__m2__0__0_, dtbetax, shift_BU0, x1, x3, x5, x7]
        store(shift_BU1, stencil_idx_0_0_0_VVV, (x7 * (access(dtbetay, stencil_idx_0_0_0_VVV) + (-1 * x1 * (((1.0 / 12.0) * (((-(stencil(betay, stencil_idx_2_0_0_VVV))) + stencil(betay, stencil_idx_m2_0_0_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betay, stencil_idx_m1_0_0_VVV))) + stencil(betay, stencil_idx_1_0_0_VVV)))))) + (-1 * x3 * (((1.0 / 12.0) * (((-(stencil(betay, stencil_idx_0_2_0_VVV))) + stencil(betay, stencil_idx_0_m2_0_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betay, stencil_idx_0_m1_0_VVV))) + stencil(betay, stencil_idx_0_1_0_VVV)))))) + (-1 * x5 * (((1.0 / 12.0) * (((-(stencil(betay, stencil_idx_0_0_2_VVV))) + stencil(betay, stencil_idx_0_0_m2_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betay, stencil_idx_0_0_m1_VVV))) + stencil(betay, stencil_idx_0_0_1_VVV))))))))); // shift_BU1: Liveness = 18; [__dummy_stencil__betay__0__0__1_, __dummy_stencil__betay__0__0__2_, __dummy_stencil__betay__0__0__m1_, __dummy_stencil__betay__0__0__m2_, __dummy_stencil__betay__0__1__0_, __dummy_stencil__betay__0__2__0_, __dummy_stencil__betay__0__m1__0_, __dummy_stencil__betay__0__m2__0_, __dummy_stencil__betay__1__0__0_, __dummy_stencil__betay__2__0__0_, __dummy_stencil__betay__m1__0__0_, __dummy_stencil__betay__m2__0__0_, dtbetay, shift_BU1, x1, x3, x5, x7]
        store(shift_BU2, stencil_idx_0_0_0_VVV, (x7 * (access(dtbetaz, stencil_idx_0_0_0_VVV) + (-1 * x1 * (((1.0 / 12.0) * (((-(stencil(betaz, stencil_idx_2_0_0_VVV))) + stencil(betaz, stencil_idx_m2_0_0_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betaz, stencil_idx_m1_0_0_VVV))) + stencil(betaz, stencil_idx_1_0_0_VVV)))))) + (-1 * x3 * (((1.0 / 12.0) * (((-(stencil(betaz, stencil_idx_0_2_0_VVV))) + stencil(betaz, stencil_idx_0_m2_0_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betaz, stencil_idx_0_m1_0_VVV))) + stencil(betaz, stencil_idx_0_1_0_VVV)))))) + (-1 * x5 * (((1.0 / 12.0) * (((-(stencil(betaz, stencil_idx_0_0_2_VVV))) + stencil(betaz, stencil_idx_0_0_m2_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betaz, stencil_idx_0_0_m1_VVV))) + stencil(betaz, stencil_idx_0_0_1_VVV))))))))); // shift_BU2: Liveness = 18; [__dummy_stencil__betaz__0__0__1_, __dummy_stencil__betaz__0__0__2_, __dummy_stencil__betaz__0__0__m1_, __dummy_stencil__betaz__0__0__m2_, __dummy_stencil__betaz__0__1__0_, __dummy_stencil__betaz__0__2__0_, __dummy_stencil__betaz__0__m1__0_, __dummy_stencil__betaz__0__m2__0_, __dummy_stencil__betaz__1__0__0_, __dummy_stencil__betaz__2__0__0_, __dummy_stencil__betaz__m1__0__0_, __dummy_stencil__betaz__m2__0__0_, dtbetaz, shift_BU2, x1, x3, x5, x7]
        vreal x14 = access(gyy, stencil_idx_0_0_0_VVV); // x14: Dependency! Liveness = 13; [gyy, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x8, x9]
        vreal x21 = access(kzz, stencil_idx_0_0_0_VVV); // x21: Dependency! Liveness = 13; [kzz, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x8, x9]
        vreal x8 = access(gxx, stencil_idx_0_0_0_VVV); // x8: Dependency! Liveness = 14; [gxx, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x6, x8, x9]
        vreal x11 = access(gzz, stencil_idx_0_0_0_VVV); // x11: Dependency! Liveness = 13; [gzz, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x8, x9]
        vreal x12 = access(gxy, stencil_idx_0_0_0_VVV); // x12: Dependency! Liveness = 13; [gxy, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x8, x9]
        vreal x15 = access(gxz, stencil_idx_0_0_0_VVV); // x15: Dependency! Liveness = 13; [gxz, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x8, x9]
        vreal x9 = access(gyz, stencil_idx_0_0_0_VVV); // x9: Dependency! Liveness = 14; [gyz, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x6, x8, x9]
        vreal x10 = pow2(x9); // x10: Dependency! Liveness = 3; [x12, x15, x9]
        vreal x13 = pow2(x12); // x13: Dependency! Liveness = 3; [x12, x15, x9]
        vreal x16 = pow2(x15); // x16: Dependency! Liveness = 3; [x12, x15, x9]
        vreal x17 = ((x10 * x8) + (x11 * x13) + (x14 * x16) + (-1 * x11 * x14 * x8) + (-2 * x12 * x15 * x9)); // x17: Dependency! Liveness = 17; [x10, x11, x12, x13, x14, x15, x16, x17, x19, x20, x21, x25, x28, x32, x34, x8, x9]
        vreal x22 = pown<vreal>(x17, -1); // x22: Dependency! Liveness = 13; [x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x6, x8, x9]
        vreal x23 = (x21 * x22 * (x13 + (-1 * x14 * x8))); // x23: Dependency! Liveness = 21; [x0, x10, x11, x12, x13, x14, x15, x16, x17, x19, x2, x20, x21, x22, x25, x28, x32, x34, x4, x8, x9]
        vreal x25 = access(kyy, stencil_idx_0_0_0_VVV); // x25: Dependency! Liveness = 13; [kyy, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x8, x9]
        vreal x26 = (x22 * x25 * (x16 + (-1 * x11 * x8))); // x26: Dependency! Liveness = 19; [x0, x10, x11, x12, x14, x15, x16, x17, x19, x2, x20, x22, x25, x28, x32, x34, x4, x8, x9]
        vreal x20 = access(kxx, stencil_idx_0_0_0_VVV); // x20: Dependency! Liveness = 13; [kxx, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x8, x9]
        vreal x27 = (x20 * x22 * (x10 + (-1 * x11 * x14))); // x27: Dependency! Liveness = 17; [x0, x10, x11, x12, x14, x15, x17, x19, x2, x20, x22, x28, x32, x34, x4, x8, x9]
        vreal x28 = access(kyz, stencil_idx_0_0_0_VVV); // x28: Dependency! Liveness = 13; [kyz, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x8, x9]
        vreal x29 = (x28 * ((x8 * x9) + (-1 * x12 * x15))); // x29: Dependency! Liveness = 15; [x0, x11, x12, x14, x15, x17, x19, x2, x22, x28, x32, x34, x4, x8, x9]
        vreal x32 = access(kxz, stencil_idx_0_0_0_VVV); // x32: Dependency! Liveness = 13; [kxz, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x8, x9]
        vreal x33 = (x32 * ((x14 * x15) + (-1 * x12 * x9))); // x33: Dependency! Liveness = 14; [x0, x11, x12, x14, x15, x17, x19, x2, x22, x32, x34, x4, x8, x9]
        vreal x34 = access(kxy, stencil_idx_0_0_0_VVV); // x34: Dependency! Liveness = 13; [kxy, x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x8, x9]
        vreal x35 = (x34 * ((x11 * x12) + (-1 * x15 * x9))); // x35: Dependency! Liveness = 13; [x0, x11, x12, x14, x15, x17, x19, x2, x22, x34, x4, x8, x9]
        vreal x18 = (-(x17)); // x18: Dependency! Liveness = 13; [x0, x11, x12, x14, x15, x17, x19, x2, x22, x4, x6, x8, x9]
        vreal x19 = pow(static_cast<vreal>(x18), (-1.0 / 3.0)); // x19: Dependency! Liveness = 14; [x0, x11, x12, x14, x15, x17, x18, x19, x2, x22, x4, x6, x8, x9]
        vreal x30 = ((2.0 / 3.0) * x22); // x30: Dependency! Liveness = 7; [x11, x12, x14, x15, x22, x8, x9]
        vreal x31 = (x30 * x8); // x31: Dependency! Liveness = 9; [x11, x12, x14, x15, x19, x22, x30, x8, x9]
        vreal x24 = ((1.0 / 3.0) * x8); // x24: Dependency! Liveness = 6; [x11, x12, x14, x15, x8, x9]
        store(AtDD00, stencil_idx_0_0_0_VVV, (x19 * (x20 + (-1 * x23 * x24) + (-1 * x24 * x26) + (-1 * x24 * x27) + (-1 * x29 * x31) + (-1 * x31 * x33) + (-1 * x31 * x35)))); // AtDD00: Liveness = 11; [AtDD00, x19, x20, x23, x24, x26, x27, x29, x31, x33, x35]
        vreal x37 = (x12 * x30); // x37: Dependency! Liveness = 9; [x11, x12, x14, x15, x19, x22, x30, x8, x9]
        vreal x36 = ((1.0 / 3.0) * x12); // x36: Dependency! Liveness = 3; [x12, x15, x9]
        store(AtDD01, stencil_idx_0_0_0_VVV, (x19 * (x34 + (-1 * x23 * x36) + (-1 * x26 * x36) + (-1 * x27 * x36) + (-1 * x29 * x37) + (-1 * x33 * x37) + (-1 * x35 * x37)))); // AtDD01: Liveness = 12; [AtDD01, x19, x20, x23, x26, x27, x29, x33, x34, x35, x36, x37]
        vreal x39 = (x15 * x30); // x39: Dependency! Liveness = 9; [x11, x12, x14, x15, x19, x22, x30, x8, x9]
        vreal x38 = ((1.0 / 3.0) * x15); // x38: Dependency! Liveness = 2; [x15, x9]
        store(AtDD02, stencil_idx_0_0_0_VVV, (x19 * (x32 + (-1 * x23 * x38) + (-1 * x26 * x38) + (-1 * x27 * x38) + (-1 * x29 * x39) + (-1 * x33 * x39) + (-1 * x35 * x39)))); // AtDD02: Liveness = 13; [AtDD02, x19, x20, x23, x26, x27, x29, x32, x33, x34, x35, x38, x39]
        vreal x41 = (x14 * x30); // x41: Dependency! Liveness = 9; [x11, x12, x14, x15, x19, x22, x30, x8, x9]
        vreal x40 = ((1.0 / 3.0) * x14); // x40: Dependency! Liveness = 5; [x11, x12, x14, x15, x9]
        store(AtDD11, stencil_idx_0_0_0_VVV, (x19 * (x25 + (-1 * x23 * x40) + (-1 * x26 * x40) + (-1 * x27 * x40) + (-1 * x29 * x41) + (-1 * x33 * x41) + (-1 * x35 * x41)))); // AtDD11: Liveness = 14; [AtDD11, x19, x20, x23, x25, x26, x27, x29, x32, x33, x34, x35, x40, x41]
        vreal x43 = (x30 * x9); // x43: Dependency! Liveness = 9; [x11, x12, x14, x15, x19, x22, x30, x8, x9]
        vreal x42 = ((1.0 / 3.0) * x9); // x42: Dependency! Liveness = 1; [x9]
        store(AtDD12, stencil_idx_0_0_0_VVV, (x19 * (x28 + (-1 * x23 * x42) + (-1 * x26 * x42) + (-1 * x27 * x42) + (-1 * x29 * x43) + (-1 * x33 * x43) + (-1 * x35 * x43)))); // AtDD12: Liveness = 15; [AtDD12, x19, x20, x23, x25, x26, x27, x28, x29, x32, x33, x34, x35, x42, x43]
        vreal x45 = (x11 * x30); // x45: Dependency! Liveness = 9; [x11, x12, x14, x15, x19, x22, x30, x8, x9]
        vreal x44 = ((1.0 / 3.0) * x11); // x44: Dependency! Liveness = 4; [x11, x12, x15, x9]
        store(AtDD22, stencil_idx_0_0_0_VVV, (x19 * (x21 + (-1 * x23 * x44) + (-1 * x26 * x44) + (-1 * x27 * x44) + (-1 * x29 * x45) + (-1 * x33 * x45) + (-1 * x35 * x45)))); // AtDD22: Liveness = 16; [AtDD22, x19, x20, x21, x23, x25, x26, x27, x28, x29, x32, x33, x34, x35, x44, x45]
        vreal x46 = (2 * x22); // x46: Dependency! Liveness = 7; [x11, x12, x14, x15, x22, x8, x9]
        store(trK, stencil_idx_0_0_0_VVV, (x23 + x26 + x27 + (x29 * x46) + (x33 * x46) + (x35 * x46))); // trK: Liveness = 15; [trK, x19, x20, x21, x23, x25, x26, x27, x28, x29, x32, x33, x34, x35, x46]
        store(w, stencil_idx_0_0_0_VVV, pow(static_cast<vreal>(x18), (-1.0 / 6.0))); // w: Liveness = 15; [w, x0, x11, x12, x14, x15, x17, x18, x19, x2, x22, x4, x6, x8, x9]
        store(evo_lapse, stencil_idx_0_0_0_VVV, x6); // evo_lapse: Liveness = 13; [evo_lapse, x0, x11, x12, x14, x15, x19, x2, x22, x4, x6, x8, x9]
        store(evo_shiftU0, stencil_idx_0_0_0_VVV, x0); // evo_shiftU0: Liveness = 12; [evo_shiftU0, x0, x11, x12, x14, x15, x19, x2, x22, x4, x8, x9]
        store(evo_shiftU1, stencil_idx_0_0_0_VVV, x2); // evo_shiftU1: Liveness = 11; [evo_shiftU1, x11, x12, x14, x15, x19, x2, x22, x4, x8, x9]
        store(evo_shiftU2, stencil_idx_0_0_0_VVV, x4); // evo_shiftU2: Liveness = 10; [evo_shiftU2, x11, x12, x14, x15, x19, x22, x4, x8, x9]
        store(gtDD00, stencil_idx_0_0_0_VVV, (x19 * x8)); // gtDD00: Liveness = 9; [gtDD00, x11, x12, x14, x15, x19, x22, x8, x9]
        store(gtDD11, stencil_idx_0_0_0_VVV, (x14 * x19)); // gtDD11: Liveness = 9; [gtDD11, x11, x12, x14, x15, x19, x22, x8, x9]
        store(gtDD22, stencil_idx_0_0_0_VVV, (x11 * x19)); // gtDD22: Liveness = 9; [gtDD22, x11, x12, x14, x15, x19, x22, x8, x9]
        store(gtDD01, stencil_idx_0_0_0_VVV, (x12 * x19)); // gtDD01: Liveness = 9; [gtDD01, x11, x12, x14, x15, x19, x22, x8, x9]
        store(gtDD02, stencil_idx_0_0_0_VVV, (x15 * x19)); // gtDD02: Liveness = 9; [gtDD02, x11, x12, x14, x15, x19, x22, x8, x9]
        store(gtDD12, stencil_idx_0_0_0_VVV, (x19 * x9)); // gtDD12: Liveness = 9; [gtDD12, x11, x12, x14, x15, x19, x22, x8, x9]    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}