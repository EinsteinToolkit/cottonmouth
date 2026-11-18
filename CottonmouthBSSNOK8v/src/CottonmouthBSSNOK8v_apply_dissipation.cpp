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
void apply_dissipation(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_apply_dissipation;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("apply_dissipation");
    #endif
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define AtDD00_layout VVV_layout
    #define AtDD01_layout VVV_layout
    #define AtDD02_layout VVV_layout
    #define AtDD11_layout VVV_layout
    #define AtDD12_layout VVV_layout
    #define AtDD22_layout VVV_layout
    #define At_rhsDD00_layout VVV_layout
    #define At_rhsDD01_layout VVV_layout
    #define At_rhsDD02_layout VVV_layout
    #define At_rhsDD11_layout VVV_layout
    #define At_rhsDD12_layout VVV_layout
    #define At_rhsDD22_layout VVV_layout
    #define ConfConnectU0_layout VVV_layout
    #define ConfConnectU1_layout VVV_layout
    #define ConfConnectU2_layout VVV_layout
    #define ConfConnect_rhsU0_layout VVV_layout
    #define ConfConnect_rhsU1_layout VVV_layout
    #define ConfConnect_rhsU2_layout VVV_layout
    #define evo_lapse_layout VVV_layout
    #define evo_lapse_rhs_layout VVV_layout
    #define evo_shiftU0_layout VVV_layout
    #define evo_shiftU1_layout VVV_layout
    #define evo_shiftU2_layout VVV_layout
    #define evo_shift_rhsU0_layout VVV_layout
    #define evo_shift_rhsU1_layout VVV_layout
    #define evo_shift_rhsU2_layout VVV_layout
    #define gtDD00_layout VVV_layout
    #define gtDD01_layout VVV_layout
    #define gtDD02_layout VVV_layout
    #define gtDD11_layout VVV_layout
    #define gtDD12_layout VVV_layout
    #define gtDD22_layout VVV_layout
    #define gt_rhsDD00_layout VVV_layout
    #define gt_rhsDD01_layout VVV_layout
    #define gt_rhsDD02_layout VVV_layout
    #define gt_rhsDD11_layout VVV_layout
    #define gt_rhsDD12_layout VVV_layout
    #define gt_rhsDD22_layout VVV_layout
    #define shift_BU0_layout VVV_layout
    #define shift_BU1_layout VVV_layout
    #define shift_BU2_layout VVV_layout
    #define shift_B_rhsU0_layout VVV_layout
    #define shift_B_rhsU1_layout VVV_layout
    #define shift_B_rhsU2_layout VVV_layout
    #define trK_layout VVV_layout
    #define trK_rhs_layout VVV_layout
    #define w_layout VVV_layout
    #define w_rhs_layout VVV_layout
    const auto DX = CCTK_DELTA_SPACE(0);
    const auto DY = CCTK_DELTA_SPACE(1);
    const auto DZ = CCTK_DELTA_SPACE(2);
    const auto DT = cctk_delta_time;
    const auto DXI = (1.0 / DX);
    const auto DYI = (1.0 / DY);
    const auto DZI = (1.0 / DZ);
    CCTK_ASSERT((cctk_nghostzones[0] >= 5));
    CCTK_ASSERT((cctk_nghostzones[1] >= 5));
    CCTK_ASSERT((cctk_nghostzones[2] >= 5));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // apply_dissipation loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_m5_0_0_VVV(VVV_layout, p.I - 5*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_m5_0_VVV(VVV_layout, p.I - 5*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m5_VVV(VVV_layout, p.I - 5*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_5_VVV(VVV_layout, p.I + 5*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_0_5_0_VVV(VVV_layout, p.I + 5*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        const GF3D5index stencil_idx_5_0_0_VVV(VVV_layout, p.I + 5*p.DI[0]);
        vreal x226 = stencil(gtDD00, stencil_idx_m4_0_0_VVV); // x226: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m4__0__0_]
        vreal x231 = stencil(gtDD00, stencil_idx_4_0_0_VVV); // x231: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__4__0__0_]
        vreal x310 = ((x226 + x231)); // x310: Dependency! Liveness = 2; [x226, x231]
        vreal x227 = stencil(gtDD00, stencil_idx_m3_0_0_VVV); // x227: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m3__0__0_]
        vreal x230 = stencil(gtDD00, stencil_idx_3_0_0_VVV); // x230: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__3__0__0_]
        vreal x311 = ((x227 + x230)); // x311: Dependency! Liveness = 2; [x227, x230]
        vreal x228 = stencil(gtDD00, stencil_idx_m2_0_0_VVV); // x228: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m2__0__0_]
        vreal x229 = stencil(gtDD00, stencil_idx_2_0_0_VVV); // x229: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__2__0__0_]
        vreal x312 = ((x228 + x229)); // x312: Dependency! Liveness = 2; [x228, x229]
        vreal x232 = stencil(gtDD00, stencil_idx_m1_0_0_VVV); // x232: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m1__0__0_]
        vreal x233 = stencil(gtDD00, stencil_idx_1_0_0_VVV); // x233: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__1__0__0_]
        vreal x313 = ((x232 + x233)); // x313: Dependency! Liveness = 2; [x232, x233]
        vreal x132 = stencil(gtDD00, stencil_idx_0_m4_0_VVV); // x132: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m4__0_]
        vreal x137 = stencil(gtDD00, stencil_idx_0_4_0_VVV); // x137: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__4__0_]
        vreal x314 = ((x132 + x137)); // x314: Dependency! Liveness = 2; [x132, x137]
        vreal x133 = stencil(gtDD00, stencil_idx_0_m3_0_VVV); // x133: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m3__0_]
        vreal x136 = stencil(gtDD00, stencil_idx_0_3_0_VVV); // x136: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__3__0_]
        vreal x315 = ((x133 + x136)); // x315: Dependency! Liveness = 2; [x133, x136]
        vreal x134 = stencil(gtDD00, stencil_idx_0_m2_0_VVV); // x134: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m2__0_]
        vreal x135 = stencil(gtDD00, stencil_idx_0_2_0_VVV); // x135: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__2__0_]
        vreal x316 = ((x134 + x135)); // x316: Dependency! Liveness = 2; [x134, x135]
        vreal x138 = stencil(gtDD00, stencil_idx_0_m1_0_VVV); // x138: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m1__0_]
        vreal x139 = stencil(gtDD00, stencil_idx_0_1_0_VVV); // x139: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__1__0_]
        vreal x317 = ((x138 + x139)); // x317: Dependency! Liveness = 2; [x138, x139]
        vreal x117 = stencil(gtDD00, stencil_idx_0_0_m4_VVV); // x117: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m4_]
        vreal x122 = stencil(gtDD00, stencil_idx_0_0_4_VVV); // x122: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__4_]
        vreal x318 = ((x117 + x122)); // x318: Dependency! Liveness = 2; [x117, x122]
        vreal x118 = stencil(gtDD00, stencil_idx_0_0_m3_VVV); // x118: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m3_]
        vreal x121 = stencil(gtDD00, stencil_idx_0_0_3_VVV); // x121: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__3_]
        vreal x319 = ((x118 + x121)); // x319: Dependency! Liveness = 2; [x118, x121]
        vreal x119 = stencil(gtDD00, stencil_idx_0_0_m2_VVV); // x119: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m2_]
        vreal x120 = stencil(gtDD00, stencil_idx_0_0_2_VVV); // x120: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__2_]
        vreal x320 = ((x119 + x120)); // x320: Dependency! Liveness = 2; [x119, x120]
        vreal x123 = stencil(gtDD00, stencil_idx_0_0_m1_VVV); // x123: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m1_]
        vreal x124 = stencil(gtDD00, stencil_idx_0_0_1_VVV); // x124: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__1_]
        vreal x321 = ((x123 + x124)); // x321: Dependency! Liveness = 2; [x123, x124]
        vreal x309 = (-252 * stencil(gtDD00, stencil_idx_0_0_0_VVV)); // x309: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__0_]
        vreal x322 = ((1.0 / 1024.0) * dissipation_epsilon); // x322: Dependency! Liveness = 1; [dissipation_epsilon]
        store(gt_rhsDD00, stencil_idx_0_0_0_VVV, (access(gt_rhsDD00, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x309 + (-120 * x312) + (-10 * x310) + (45 * x311) + (210 * x313) + ((stencil(gtDD00, stencil_idx_m5_0_0_VVV) + stencil(gtDD00, stencil_idx_5_0_0_VVV))))) + (DYI * (x309 + (-120 * x316) + (-10 * x314) + (45 * x315) + (210 * x317) + ((stencil(gtDD00, stencil_idx_0_m5_0_VVV) + stencil(gtDD00, stencil_idx_0_5_0_VVV))))) + (DZI * (x309 + (-120 * x320) + (-10 * x318) + (45 * x319) + (210 * x321) + ((stencil(gtDD00, stencil_idx_0_0_m5_VVV) + stencil(gtDD00, stencil_idx_0_0_5_VVV)))))))));
        vreal x247 = stencil(gtDD01, stencil_idx_m4_0_0_VVV); // x247: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m4__0__0_]
        vreal x252 = stencil(gtDD01, stencil_idx_4_0_0_VVV); // x252: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__4__0__0_]
        vreal x324 = ((x247 + x252)); // x324: Dependency! Liveness = 2; [x247, x252]
        vreal x248 = stencil(gtDD01, stencil_idx_m3_0_0_VVV); // x248: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m3__0__0_]
        vreal x251 = stencil(gtDD01, stencil_idx_3_0_0_VVV); // x251: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__3__0__0_]
        vreal x325 = ((x248 + x251)); // x325: Dependency! Liveness = 2; [x248, x251]
        vreal x249 = stencil(gtDD01, stencil_idx_m2_0_0_VVV); // x249: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m2__0__0_]
        vreal x250 = stencil(gtDD01, stencil_idx_2_0_0_VVV); // x250: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__2__0__0_]
        vreal x326 = ((x249 + x250)); // x326: Dependency! Liveness = 2; [x249, x250]
        vreal x253 = stencil(gtDD01, stencil_idx_m1_0_0_VVV); // x253: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m1__0__0_]
        vreal x254 = stencil(gtDD01, stencil_idx_1_0_0_VVV); // x254: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__1__0__0_]
        vreal x327 = ((x253 + x254)); // x327: Dependency! Liveness = 2; [x253, x254]
        vreal x200 = stencil(gtDD01, stencil_idx_0_m4_0_VVV); // x200: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m4__0_]
        vreal x205 = stencil(gtDD01, stencil_idx_0_4_0_VVV); // x205: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__4__0_]
        vreal x328 = ((x200 + x205)); // x328: Dependency! Liveness = 2; [x200, x205]
        vreal x201 = stencil(gtDD01, stencil_idx_0_m3_0_VVV); // x201: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m3__0_]
        vreal x204 = stencil(gtDD01, stencil_idx_0_3_0_VVV); // x204: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__3__0_]
        vreal x329 = ((x201 + x204)); // x329: Dependency! Liveness = 2; [x201, x204]
        vreal x202 = stencil(gtDD01, stencil_idx_0_m2_0_VVV); // x202: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m2__0_]
        vreal x203 = stencil(gtDD01, stencil_idx_0_2_0_VVV); // x203: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__2__0_]
        vreal x330 = ((x202 + x203)); // x330: Dependency! Liveness = 2; [x202, x203]
        vreal x206 = stencil(gtDD01, stencil_idx_0_m1_0_VVV); // x206: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m1__0_]
        vreal x207 = stencil(gtDD01, stencil_idx_0_1_0_VVV); // x207: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__1__0_]
        vreal x331 = ((x206 + x207)); // x331: Dependency! Liveness = 2; [x206, x207]
        vreal x86 = stencil(gtDD01, stencil_idx_0_0_m4_VVV); // x86: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m4_]
        vreal x91 = stencil(gtDD01, stencil_idx_0_0_4_VVV); // x91: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__4_]
        vreal x332 = ((x86 + x91)); // x332: Dependency! Liveness = 2; [x86, x91]
        vreal x87 = stencil(gtDD01, stencil_idx_0_0_m3_VVV); // x87: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m3_]
        vreal x90 = stencil(gtDD01, stencil_idx_0_0_3_VVV); // x90: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__3_]
        vreal x333 = ((x87 + x90)); // x333: Dependency! Liveness = 2; [x87, x90]
        vreal x88 = stencil(gtDD01, stencil_idx_0_0_m2_VVV); // x88: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m2_]
        vreal x89 = stencil(gtDD01, stencil_idx_0_0_2_VVV); // x89: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__2_]
        vreal x334 = ((x88 + x89)); // x334: Dependency! Liveness = 2; [x88, x89]
        vreal x92 = stencil(gtDD01, stencil_idx_0_0_m1_VVV); // x92: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m1_]
        vreal x93 = stencil(gtDD01, stencil_idx_0_0_1_VVV); // x93: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__1_]
        vreal x335 = ((x92 + x93)); // x335: Dependency! Liveness = 2; [x92, x93]
        vreal x323 = (-252 * stencil(gtDD01, stencil_idx_0_0_0_VVV)); // x323: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__0_]
        store(gt_rhsDD01, stencil_idx_0_0_0_VVV, (access(gt_rhsDD01, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x323 + (-120 * x326) + (-10 * x324) + (45 * x325) + (210 * x327) + ((stencil(gtDD01, stencil_idx_m5_0_0_VVV) + stencil(gtDD01, stencil_idx_5_0_0_VVV))))) + (DYI * (x323 + (-120 * x330) + (-10 * x328) + (45 * x329) + (210 * x331) + ((stencil(gtDD01, stencil_idx_0_m5_0_VVV) + stencil(gtDD01, stencil_idx_0_5_0_VVV))))) + (DZI * (x323 + (-120 * x334) + (-10 * x332) + (45 * x333) + (210 * x335) + ((stencil(gtDD01, stencil_idx_0_0_m5_VVV) + stencil(gtDD01, stencil_idx_0_0_5_VVV)))))))));
        vreal x236 = stencil(gtDD02, stencil_idx_m4_0_0_VVV); // x236: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m4__0__0_]
        vreal x241 = stencil(gtDD02, stencil_idx_4_0_0_VVV); // x241: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__4__0__0_]
        vreal x337 = ((x236 + x241)); // x337: Dependency! Liveness = 2; [x236, x241]
        vreal x237 = stencil(gtDD02, stencil_idx_m3_0_0_VVV); // x237: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m3__0__0_]
        vreal x240 = stencil(gtDD02, stencil_idx_3_0_0_VVV); // x240: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__3__0__0_]
        vreal x338 = ((x237 + x240)); // x338: Dependency! Liveness = 2; [x237, x240]
        vreal x238 = stencil(gtDD02, stencil_idx_m2_0_0_VVV); // x238: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m2__0__0_]
        vreal x239 = stencil(gtDD02, stencil_idx_2_0_0_VVV); // x239: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__2__0__0_]
        vreal x339 = ((x238 + x239)); // x339: Dependency! Liveness = 2; [x238, x239]
        vreal x242 = stencil(gtDD02, stencil_idx_m1_0_0_VVV); // x242: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m1__0__0_]
        vreal x243 = stencil(gtDD02, stencil_idx_1_0_0_VVV); // x243: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__1__0__0_]
        vreal x340 = ((x242 + x243)); // x340: Dependency! Liveness = 2; [x242, x243]
        vreal x77 = stencil(gtDD02, stencil_idx_0_m4_0_VVV); // x77: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m4__0_]
        vreal x82 = stencil(gtDD02, stencil_idx_0_4_0_VVV); // x82: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__4__0_]
        vreal x341 = ((x77 + x82)); // x341: Dependency! Liveness = 2; [x77, x82]
        vreal x78 = stencil(gtDD02, stencil_idx_0_m3_0_VVV); // x78: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m3__0_]
        vreal x81 = stencil(gtDD02, stencil_idx_0_3_0_VVV); // x81: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__3__0_]
        vreal x342 = ((x78 + x81)); // x342: Dependency! Liveness = 2; [x78, x81]
        vreal x79 = stencil(gtDD02, stencil_idx_0_m2_0_VVV); // x79: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m2__0_]
        vreal x80 = stencil(gtDD02, stencil_idx_0_2_0_VVV); // x80: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__2__0_]
        vreal x343 = ((x79 + x80)); // x343: Dependency! Liveness = 2; [x79, x80]
        vreal x83 = stencil(gtDD02, stencil_idx_0_m1_0_VVV); // x83: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m1__0_]
        vreal x84 = stencil(gtDD02, stencil_idx_0_1_0_VVV); // x84: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__1__0_]
        vreal x344 = ((x83 + x84)); // x344: Dependency! Liveness = 2; [x83, x84]
        vreal x178 = stencil(gtDD02, stencil_idx_0_0_m4_VVV); // x178: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m4_]
        vreal x183 = stencil(gtDD02, stencil_idx_0_0_4_VVV); // x183: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__4_]
        vreal x345 = ((x178 + x183)); // x345: Dependency! Liveness = 2; [x178, x183]
        vreal x179 = stencil(gtDD02, stencil_idx_0_0_m3_VVV); // x179: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m3_]
        vreal x182 = stencil(gtDD02, stencil_idx_0_0_3_VVV); // x182: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__3_]
        vreal x346 = ((x179 + x182)); // x346: Dependency! Liveness = 2; [x179, x182]
        vreal x180 = stencil(gtDD02, stencil_idx_0_0_m2_VVV); // x180: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m2_]
        vreal x181 = stencil(gtDD02, stencil_idx_0_0_2_VVV); // x181: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__2_]
        vreal x347 = ((x180 + x181)); // x347: Dependency! Liveness = 2; [x180, x181]
        vreal x184 = stencil(gtDD02, stencil_idx_0_0_m1_VVV); // x184: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m1_]
        vreal x185 = stencil(gtDD02, stencil_idx_0_0_1_VVV); // x185: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__1_]
        vreal x348 = ((x184 + x185)); // x348: Dependency! Liveness = 2; [x184, x185]
        vreal x336 = (-252 * stencil(gtDD02, stencil_idx_0_0_0_VVV)); // x336: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__0_]
        store(gt_rhsDD02, stencil_idx_0_0_0_VVV, (access(gt_rhsDD02, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x336 + (-120 * x339) + (-10 * x337) + (45 * x338) + (210 * x340) + ((stencil(gtDD02, stencil_idx_m5_0_0_VVV) + stencil(gtDD02, stencil_idx_5_0_0_VVV))))) + (DYI * (x336 + (-120 * x343) + (-10 * x341) + (45 * x342) + (210 * x344) + ((stencil(gtDD02, stencil_idx_0_m5_0_VVV) + stencil(gtDD02, stencil_idx_0_5_0_VVV))))) + (DZI * (x336 + (-120 * x347) + (-10 * x345) + (45 * x346) + (210 * x348) + ((stencil(gtDD02, stencil_idx_0_0_m5_VVV) + stencil(gtDD02, stencil_idx_0_0_5_VVV)))))))));
        vreal x142 = stencil(gtDD11, stencil_idx_m4_0_0_VVV); // x142: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m4__0__0_]
        vreal x147 = stencil(gtDD11, stencil_idx_4_0_0_VVV); // x147: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__4__0__0_]
        vreal x350 = ((x142 + x147)); // x350: Dependency! Liveness = 2; [x142, x147]
        vreal x143 = stencil(gtDD11, stencil_idx_m3_0_0_VVV); // x143: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m3__0__0_]
        vreal x146 = stencil(gtDD11, stencil_idx_3_0_0_VVV); // x146: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__3__0__0_]
        vreal x351 = ((x143 + x146)); // x351: Dependency! Liveness = 2; [x143, x146]
        vreal x144 = stencil(gtDD11, stencil_idx_m2_0_0_VVV); // x144: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m2__0__0_]
        vreal x145 = stencil(gtDD11, stencil_idx_2_0_0_VVV); // x145: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__2__0__0_]
        vreal x352 = ((x144 + x145)); // x352: Dependency! Liveness = 2; [x144, x145]
        vreal x148 = stencil(gtDD11, stencil_idx_m1_0_0_VVV); // x148: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m1__0__0_]
        vreal x149 = stencil(gtDD11, stencil_idx_1_0_0_VVV); // x149: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__1__0__0_]
        vreal x353 = ((x148 + x149)); // x353: Dependency! Liveness = 2; [x148, x149]
        vreal x190 = stencil(gtDD11, stencil_idx_0_m4_0_VVV); // x190: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m4__0_]
        vreal x195 = stencil(gtDD11, stencil_idx_0_4_0_VVV); // x195: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__4__0_]
        vreal x354 = ((x190 + x195)); // x354: Dependency! Liveness = 2; [x190, x195]
        vreal x191 = stencil(gtDD11, stencil_idx_0_m3_0_VVV); // x191: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m3__0_]
        vreal x194 = stencil(gtDD11, stencil_idx_0_3_0_VVV); // x194: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__3__0_]
        vreal x355 = ((x191 + x194)); // x355: Dependency! Liveness = 2; [x191, x194]
        vreal x192 = stencil(gtDD11, stencil_idx_0_m2_0_VVV); // x192: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m2__0_]
        vreal x193 = stencil(gtDD11, stencil_idx_0_2_0_VVV); // x193: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__2__0_]
        vreal x356 = ((x192 + x193)); // x356: Dependency! Liveness = 2; [x192, x193]
        vreal x196 = stencil(gtDD11, stencil_idx_0_m1_0_VVV); // x196: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m1__0_]
        vreal x197 = stencil(gtDD11, stencil_idx_0_1_0_VVV); // x197: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__1__0_]
        vreal x357 = ((x196 + x197)); // x357: Dependency! Liveness = 2; [x196, x197]
        vreal x66 = stencil(gtDD11, stencil_idx_0_0_m4_VVV); // x66: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m4_]
        vreal x71 = stencil(gtDD11, stencil_idx_0_0_4_VVV); // x71: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__4_]
        vreal x358 = ((x66 + x71)); // x358: Dependency! Liveness = 2; [x66, x71]
        vreal x67 = stencil(gtDD11, stencil_idx_0_0_m3_VVV); // x67: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m3_]
        vreal x70 = stencil(gtDD11, stencil_idx_0_0_3_VVV); // x70: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__3_]
        vreal x359 = ((x67 + x70)); // x359: Dependency! Liveness = 2; [x67, x70]
        vreal x68 = stencil(gtDD11, stencil_idx_0_0_m2_VVV); // x68: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m2_]
        vreal x69 = stencil(gtDD11, stencil_idx_0_0_2_VVV); // x69: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__2_]
        vreal x360 = ((x68 + x69)); // x360: Dependency! Liveness = 2; [x68, x69]
        vreal x72 = stencil(gtDD11, stencil_idx_0_0_m1_VVV); // x72: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m1_]
        vreal x73 = stencil(gtDD11, stencil_idx_0_0_1_VVV); // x73: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__1_]
        vreal x361 = ((x72 + x73)); // x361: Dependency! Liveness = 2; [x72, x73]
        vreal x349 = (-252 * stencil(gtDD11, stencil_idx_0_0_0_VVV)); // x349: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__0_]
        store(gt_rhsDD11, stencil_idx_0_0_0_VVV, (access(gt_rhsDD11, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x349 + (-120 * x352) + (-10 * x350) + (45 * x351) + (210 * x353) + ((stencil(gtDD11, stencil_idx_m5_0_0_VVV) + stencil(gtDD11, stencil_idx_5_0_0_VVV))))) + (DYI * (x349 + (-120 * x356) + (-10 * x354) + (45 * x355) + (210 * x357) + ((stencil(gtDD11, stencil_idx_0_m5_0_VVV) + stencil(gtDD11, stencil_idx_0_5_0_VVV))))) + (DZI * (x349 + (-120 * x360) + (-10 * x358) + (45 * x359) + (210 * x361) + ((stencil(gtDD11, stencil_idx_0_0_m5_VVV) + stencil(gtDD11, stencil_idx_0_0_5_VVV)))))))));
        vreal x100 = stencil(gtDD12, stencil_idx_4_0_0_VVV); // x100: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__4__0__0_]
        vreal x95 = stencil(gtDD12, stencil_idx_m4_0_0_VVV); // x95: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m4__0__0_]
        vreal x363 = ((x100 + x95)); // x363: Dependency! Liveness = 2; [x100, x95]
        vreal x96 = stencil(gtDD12, stencil_idx_m3_0_0_VVV); // x96: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m3__0__0_]
        vreal x99 = stencil(gtDD12, stencil_idx_3_0_0_VVV); // x99: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__3__0__0_]
        vreal x364 = ((x96 + x99)); // x364: Dependency! Liveness = 2; [x96, x99]
        vreal x97 = stencil(gtDD12, stencil_idx_m2_0_0_VVV); // x97: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m2__0__0_]
        vreal x98 = stencil(gtDD12, stencil_idx_2_0_0_VVV); // x98: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__2__0__0_]
        vreal x365 = ((x97 + x98)); // x365: Dependency! Liveness = 2; [x97, x98]
        vreal x101 = stencil(gtDD12, stencil_idx_m1_0_0_VVV); // x101: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m1__0__0_]
        vreal x102 = stencil(gtDD12, stencil_idx_1_0_0_VVV); // x102: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__1__0__0_]
        vreal x366 = ((x101 + x102)); // x366: Dependency! Liveness = 2; [x101, x102]
        vreal x212 = stencil(gtDD12, stencil_idx_0_m4_0_VVV); // x212: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m4__0_]
        vreal x217 = stencil(gtDD12, stencil_idx_0_4_0_VVV); // x217: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__4__0_]
        vreal x367 = ((x212 + x217)); // x367: Dependency! Liveness = 2; [x212, x217]
        vreal x213 = stencil(gtDD12, stencil_idx_0_m3_0_VVV); // x213: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m3__0_]
        vreal x216 = stencil(gtDD12, stencil_idx_0_3_0_VVV); // x216: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__3__0_]
        vreal x368 = ((x213 + x216)); // x368: Dependency! Liveness = 2; [x213, x216]
        vreal x214 = stencil(gtDD12, stencil_idx_0_m2_0_VVV); // x214: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m2__0_]
        vreal x215 = stencil(gtDD12, stencil_idx_0_2_0_VVV); // x215: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__2__0_]
        vreal x369 = ((x214 + x215)); // x369: Dependency! Liveness = 2; [x214, x215]
        vreal x218 = stencil(gtDD12, stencil_idx_0_m1_0_VVV); // x218: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m1__0_]
        vreal x219 = stencil(gtDD12, stencil_idx_0_1_0_VVV); // x219: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__1__0_]
        vreal x370 = ((x218 + x219)); // x370: Dependency! Liveness = 2; [x218, x219]
        vreal x167 = stencil(gtDD12, stencil_idx_0_0_m4_VVV); // x167: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m4_]
        vreal x172 = stencil(gtDD12, stencil_idx_0_0_4_VVV); // x172: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__4_]
        vreal x371 = ((x167 + x172)); // x371: Dependency! Liveness = 2; [x167, x172]
        vreal x168 = stencil(gtDD12, stencil_idx_0_0_m3_VVV); // x168: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m3_]
        vreal x171 = stencil(gtDD12, stencil_idx_0_0_3_VVV); // x171: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__3_]
        vreal x372 = ((x168 + x171)); // x372: Dependency! Liveness = 2; [x168, x171]
        vreal x169 = stencil(gtDD12, stencil_idx_0_0_m2_VVV); // x169: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m2_]
        vreal x170 = stencil(gtDD12, stencil_idx_0_0_2_VVV); // x170: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__2_]
        vreal x373 = ((x169 + x170)); // x373: Dependency! Liveness = 2; [x169, x170]
        vreal x173 = stencil(gtDD12, stencil_idx_0_0_m1_VVV); // x173: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m1_]
        vreal x174 = stencil(gtDD12, stencil_idx_0_0_1_VVV); // x174: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__1_]
        vreal x374 = ((x173 + x174)); // x374: Dependency! Liveness = 2; [x173, x174]
        vreal x362 = (-252 * stencil(gtDD12, stencil_idx_0_0_0_VVV)); // x362: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__0_]
        store(gt_rhsDD12, stencil_idx_0_0_0_VVV, (access(gt_rhsDD12, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x362 + (-120 * x365) + (-10 * x363) + (45 * x364) + (210 * x366) + ((stencil(gtDD12, stencil_idx_m5_0_0_VVV) + stencil(gtDD12, stencil_idx_5_0_0_VVV))))) + (DYI * (x362 + (-120 * x369) + (-10 * x367) + (45 * x368) + (210 * x370) + ((stencil(gtDD12, stencil_idx_0_m5_0_VVV) + stencil(gtDD12, stencil_idx_0_5_0_VVV))))) + (DZI * (x362 + (-120 * x373) + (-10 * x371) + (45 * x372) + (210 * x374) + ((stencil(gtDD12, stencil_idx_0_0_m5_VVV) + stencil(gtDD12, stencil_idx_0_0_5_VVV)))))))));
        vreal x107 = stencil(gtDD22, stencil_idx_m4_0_0_VVV); // x107: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m4__0__0_]
        vreal x112 = stencil(gtDD22, stencil_idx_4_0_0_VVV); // x112: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__4__0__0_]
        vreal x376 = ((x107 + x112)); // x376: Dependency! Liveness = 2; [x107, x112]
        vreal x108 = stencil(gtDD22, stencil_idx_m3_0_0_VVV); // x108: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m3__0__0_]
        vreal x111 = stencil(gtDD22, stencil_idx_3_0_0_VVV); // x111: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__3__0__0_]
        vreal x377 = ((x108 + x111)); // x377: Dependency! Liveness = 2; [x108, x111]
        vreal x109 = stencil(gtDD22, stencil_idx_m2_0_0_VVV); // x109: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m2__0__0_]
        vreal x110 = stencil(gtDD22, stencil_idx_2_0_0_VVV); // x110: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__2__0__0_]
        vreal x378 = ((x109 + x110)); // x378: Dependency! Liveness = 2; [x109, x110]
        vreal x113 = stencil(gtDD22, stencil_idx_m1_0_0_VVV); // x113: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m1__0__0_]
        vreal x114 = stencil(gtDD22, stencil_idx_1_0_0_VVV); // x114: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__1__0__0_]
        vreal x379 = ((x113 + x114)); // x379: Dependency! Liveness = 2; [x113, x114]
        vreal x53 = stencil(gtDD22, stencil_idx_0_m4_0_VVV); // x53: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m4__0_]
        vreal x58 = stencil(gtDD22, stencil_idx_0_4_0_VVV); // x58: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__4__0_]
        vreal x380 = ((x53 + x58)); // x380: Dependency! Liveness = 2; [x53, x58]
        vreal x54 = stencil(gtDD22, stencil_idx_0_m3_0_VVV); // x54: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m3__0_]
        vreal x57 = stencil(gtDD22, stencil_idx_0_3_0_VVV); // x57: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__3__0_]
        vreal x381 = ((x54 + x57)); // x381: Dependency! Liveness = 2; [x54, x57]
        vreal x55 = stencil(gtDD22, stencil_idx_0_m2_0_VVV); // x55: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m2__0_]
        vreal x56 = stencil(gtDD22, stencil_idx_0_2_0_VVV); // x56: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__2__0_]
        vreal x382 = ((x55 + x56)); // x382: Dependency! Liveness = 2; [x55, x56]
        vreal x59 = stencil(gtDD22, stencil_idx_0_m1_0_VVV); // x59: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m1__0_]
        vreal x60 = stencil(gtDD22, stencil_idx_0_1_0_VVV); // x60: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__1__0_]
        vreal x383 = ((x59 + x60)); // x383: Dependency! Liveness = 2; [x59, x60]
        vreal x157 = stencil(gtDD22, stencil_idx_0_0_m4_VVV); // x157: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m4_]
        vreal x162 = stencil(gtDD22, stencil_idx_0_0_4_VVV); // x162: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__4_]
        vreal x384 = ((x157 + x162)); // x384: Dependency! Liveness = 2; [x157, x162]
        vreal x158 = stencil(gtDD22, stencil_idx_0_0_m3_VVV); // x158: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m3_]
        vreal x161 = stencil(gtDD22, stencil_idx_0_0_3_VVV); // x161: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__3_]
        vreal x385 = ((x158 + x161)); // x385: Dependency! Liveness = 2; [x158, x161]
        vreal x159 = stencil(gtDD22, stencil_idx_0_0_m2_VVV); // x159: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m2_]
        vreal x160 = stencil(gtDD22, stencil_idx_0_0_2_VVV); // x160: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__2_]
        vreal x386 = ((x159 + x160)); // x386: Dependency! Liveness = 2; [x159, x160]
        vreal x163 = stencil(gtDD22, stencil_idx_0_0_m1_VVV); // x163: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m1_]
        vreal x164 = stencil(gtDD22, stencil_idx_0_0_1_VVV); // x164: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__1_]
        vreal x387 = ((x163 + x164)); // x387: Dependency! Liveness = 2; [x163, x164]
        vreal x375 = (-252 * stencil(gtDD22, stencil_idx_0_0_0_VVV)); // x375: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__0_]
        store(gt_rhsDD22, stencil_idx_0_0_0_VVV, (access(gt_rhsDD22, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x375 + (-120 * x378) + (-10 * x376) + (45 * x377) + (210 * x379) + ((stencil(gtDD22, stencil_idx_m5_0_0_VVV) + stencil(gtDD22, stencil_idx_5_0_0_VVV))))) + (DYI * (x375 + (-120 * x382) + (-10 * x380) + (45 * x381) + (210 * x383) + ((stencil(gtDD22, stencil_idx_0_m5_0_VVV) + stencil(gtDD22, stencil_idx_0_5_0_VVV))))) + (DZI * (x375 + (-120 * x386) + (-10 * x384) + (45 * x385) + (210 * x387) + ((stencil(gtDD22, stencil_idx_0_0_m5_VVV) + stencil(gtDD22, stencil_idx_0_0_5_VVV)))))))));    
    });
    // apply_dissipation loop 1
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_m5_0_0_VVV(VVV_layout, p.I - 5*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_m5_0_VVV(VVV_layout, p.I - 5*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m5_VVV(VVV_layout, p.I - 5*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_5_VVV(VVV_layout, p.I + 5*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_0_5_0_VVV(VVV_layout, p.I + 5*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        const GF3D5index stencil_idx_5_0_0_VVV(VVV_layout, p.I + 5*p.DI[0]);
        vreal x389 = stencil(w, stencil_idx_m4_0_0_VVV); // x389: Dependency! Liveness = 1; [__dummy_stencil__w__m4__0__0_]
        vreal x390 = stencil(w, stencil_idx_4_0_0_VVV); // x390: Dependency! Liveness = 1; [__dummy_stencil__w__4__0__0_]
        vreal x391 = ((x389 + x390)); // x391: Dependency! Liveness = 2; [x389, x390]
        vreal x392 = stencil(w, stencil_idx_m3_0_0_VVV); // x392: Dependency! Liveness = 1; [__dummy_stencil__w__m3__0__0_]
        vreal x393 = stencil(w, stencil_idx_3_0_0_VVV); // x393: Dependency! Liveness = 1; [__dummy_stencil__w__3__0__0_]
        vreal x394 = ((x392 + x393)); // x394: Dependency! Liveness = 2; [x392, x393]
        vreal x395 = stencil(w, stencil_idx_m2_0_0_VVV); // x395: Dependency! Liveness = 1; [__dummy_stencil__w__m2__0__0_]
        vreal x396 = stencil(w, stencil_idx_2_0_0_VVV); // x396: Dependency! Liveness = 1; [__dummy_stencil__w__2__0__0_]
        vreal x397 = ((x395 + x396)); // x397: Dependency! Liveness = 2; [x395, x396]
        vreal x398 = stencil(w, stencil_idx_m1_0_0_VVV); // x398: Dependency! Liveness = 1; [__dummy_stencil__w__m1__0__0_]
        vreal x399 = stencil(w, stencil_idx_1_0_0_VVV); // x399: Dependency! Liveness = 1; [__dummy_stencil__w__1__0__0_]
        vreal x400 = ((x398 + x399)); // x400: Dependency! Liveness = 2; [x398, x399]
        vreal x401 = stencil(w, stencil_idx_0_m4_0_VVV); // x401: Dependency! Liveness = 1; [__dummy_stencil__w__0__m4__0_]
        vreal x402 = stencil(w, stencil_idx_0_4_0_VVV); // x402: Dependency! Liveness = 1; [__dummy_stencil__w__0__4__0_]
        vreal x403 = ((x401 + x402)); // x403: Dependency! Liveness = 2; [x401, x402]
        vreal x404 = stencil(w, stencil_idx_0_m3_0_VVV); // x404: Dependency! Liveness = 1; [__dummy_stencil__w__0__m3__0_]
        vreal x405 = stencil(w, stencil_idx_0_3_0_VVV); // x405: Dependency! Liveness = 1; [__dummy_stencil__w__0__3__0_]
        vreal x406 = ((x404 + x405)); // x406: Dependency! Liveness = 2; [x404, x405]
        vreal x407 = stencil(w, stencil_idx_0_m2_0_VVV); // x407: Dependency! Liveness = 1; [__dummy_stencil__w__0__m2__0_]
        vreal x408 = stencil(w, stencil_idx_0_2_0_VVV); // x408: Dependency! Liveness = 1; [__dummy_stencil__w__0__2__0_]
        vreal x409 = ((x407 + x408)); // x409: Dependency! Liveness = 2; [x407, x408]
        vreal x410 = stencil(w, stencil_idx_0_m1_0_VVV); // x410: Dependency! Liveness = 1; [__dummy_stencil__w__0__m1__0_]
        vreal x411 = stencil(w, stencil_idx_0_1_0_VVV); // x411: Dependency! Liveness = 1; [__dummy_stencil__w__0__1__0_]
        vreal x412 = ((x410 + x411)); // x412: Dependency! Liveness = 2; [x410, x411]
        vreal x413 = stencil(w, stencil_idx_0_0_m4_VVV); // x413: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__m4_]
        vreal x414 = stencil(w, stencil_idx_0_0_4_VVV); // x414: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__4_]
        vreal x415 = ((x413 + x414)); // x415: Dependency! Liveness = 2; [x413, x414]
        vreal x416 = stencil(w, stencil_idx_0_0_m3_VVV); // x416: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__m3_]
        vreal x417 = stencil(w, stencil_idx_0_0_3_VVV); // x417: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__3_]
        vreal x418 = ((x416 + x417)); // x418: Dependency! Liveness = 2; [x416, x417]
        vreal x419 = stencil(w, stencil_idx_0_0_m2_VVV); // x419: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__m2_]
        vreal x420 = stencil(w, stencil_idx_0_0_2_VVV); // x420: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__2_]
        vreal x421 = ((x419 + x420)); // x421: Dependency! Liveness = 2; [x419, x420]
        vreal x422 = stencil(w, stencil_idx_0_0_m1_VVV); // x422: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__m1_]
        vreal x423 = stencil(w, stencil_idx_0_0_1_VVV); // x423: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__1_]
        vreal x424 = ((x422 + x423)); // x424: Dependency! Liveness = 2; [x422, x423]
        vreal x388 = (-252 * stencil(w, stencil_idx_0_0_0_VVV)); // x388: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__0_]
        vreal x322 = ((1.0 / 1024.0) * dissipation_epsilon); // x322: Dependency! Liveness = 1; [dissipation_epsilon]
        store(w_rhs, stencil_idx_0_0_0_VVV, (access(w_rhs, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x388 + (-120 * x397) + (-10 * x391) + (45 * x394) + (210 * x400) + ((stencil(w, stencil_idx_m5_0_0_VVV) + stencil(w, stencil_idx_5_0_0_VVV))))) + (DYI * (x388 + (-120 * x409) + (-10 * x403) + (45 * x406) + (210 * x412) + ((stencil(w, stencil_idx_0_m5_0_VVV) + stencil(w, stencil_idx_0_5_0_VVV))))) + (DZI * (x388 + (-120 * x421) + (-10 * x415) + (45 * x418) + (210 * x424) + ((stencil(w, stencil_idx_0_0_m5_VVV) + stencil(w, stencil_idx_0_0_5_VVV)))))))));    
    });
    // apply_dissipation loop 2
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_m5_0_0_VVV(VVV_layout, p.I - 5*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_m5_0_VVV(VVV_layout, p.I - 5*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m5_VVV(VVV_layout, p.I - 5*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_5_VVV(VVV_layout, p.I + 5*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_0_5_0_VVV(VVV_layout, p.I + 5*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        const GF3D5index stencil_idx_5_0_0_VVV(VVV_layout, p.I + 5*p.DI[0]);
        vreal x434 = stencil(trK, stencil_idx_0_m4_0_VVV); // x434: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m4__0_]
        vreal x435 = stencil(trK, stencil_idx_0_4_0_VVV); // x435: Dependency! Liveness = 1; [__dummy_stencil__trK__0__4__0_]
        vreal x436 = stencil(trK, stencil_idx_0_m3_0_VVV); // x436: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m3__0_]
        vreal x437 = stencil(trK, stencil_idx_0_3_0_VVV); // x437: Dependency! Liveness = 1; [__dummy_stencil__trK__0__3__0_]
        vreal x438 = stencil(trK, stencil_idx_0_m2_0_VVV); // x438: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m2__0_]
        vreal x439 = stencil(trK, stencil_idx_0_2_0_VVV); // x439: Dependency! Liveness = 1; [__dummy_stencil__trK__0__2__0_]
        vreal x440 = stencil(trK, stencil_idx_0_m1_0_VVV); // x440: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m1__0_]
        vreal x441 = stencil(trK, stencil_idx_0_1_0_VVV); // x441: Dependency! Liveness = 1; [__dummy_stencil__trK__0__1__0_]
        vreal x442 = stencil(trK, stencil_idx_0_0_m4_VVV); // x442: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m4_]
        vreal x443 = stencil(trK, stencil_idx_0_0_4_VVV); // x443: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__4_]
        vreal x444 = stencil(trK, stencil_idx_0_0_m3_VVV); // x444: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m3_]
        vreal x445 = stencil(trK, stencil_idx_0_0_3_VVV); // x445: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__3_]
        vreal x446 = stencil(trK, stencil_idx_0_0_m2_VVV); // x446: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m2_]
        vreal x447 = stencil(trK, stencil_idx_0_0_2_VVV); // x447: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__2_]
        vreal x448 = stencil(trK, stencil_idx_0_0_m1_VVV); // x448: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m1_]
        vreal x449 = stencil(trK, stencil_idx_0_0_1_VVV); // x449: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__1_]
        vreal x426 = stencil(trK, stencil_idx_m4_0_0_VVV); // x426: Dependency! Liveness = 1; [__dummy_stencil__trK__m4__0__0_]
        vreal x427 = stencil(trK, stencil_idx_4_0_0_VVV); // x427: Dependency! Liveness = 1; [__dummy_stencil__trK__4__0__0_]
        vreal x428 = stencil(trK, stencil_idx_m3_0_0_VVV); // x428: Dependency! Liveness = 1; [__dummy_stencil__trK__m3__0__0_]
        vreal x429 = stencil(trK, stencil_idx_3_0_0_VVV); // x429: Dependency! Liveness = 1; [__dummy_stencil__trK__3__0__0_]
        vreal x430 = stencil(trK, stencil_idx_m2_0_0_VVV); // x430: Dependency! Liveness = 1; [__dummy_stencil__trK__m2__0__0_]
        vreal x431 = stencil(trK, stencil_idx_2_0_0_VVV); // x431: Dependency! Liveness = 1; [__dummy_stencil__trK__2__0__0_]
        vreal x432 = stencil(trK, stencil_idx_m1_0_0_VVV); // x432: Dependency! Liveness = 1; [__dummy_stencil__trK__m1__0__0_]
        vreal x433 = stencil(trK, stencil_idx_1_0_0_VVV); // x433: Dependency! Liveness = 1; [__dummy_stencil__trK__1__0__0_]
        vreal x425 = (-252 * stencil(trK, stencil_idx_0_0_0_VVV)); // x425: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__0_]
        vreal x322 = ((1.0 / 1024.0) * dissipation_epsilon); // x322: Dependency! Liveness = 1; [dissipation_epsilon]
        store(trK_rhs, stencil_idx_0_0_0_VVV, (access(trK_rhs, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x425 + (-120 * ((x430 + x431))) + (-10 * ((x426 + x427))) + (45 * ((x428 + x429))) + (210 * ((x432 + x433))) + ((stencil(trK, stencil_idx_m5_0_0_VVV) + stencil(trK, stencil_idx_5_0_0_VVV))))) + (DYI * (x425 + (-120 * ((x438 + x439))) + (-10 * ((x434 + x435))) + (45 * ((x436 + x437))) + (210 * ((x440 + x441))) + ((stencil(trK, stencil_idx_0_m5_0_VVV) + stencil(trK, stencil_idx_0_5_0_VVV))))) + (DZI * (x425 + (-120 * ((x446 + x447))) + (-10 * ((x442 + x443))) + (45 * ((x444 + x445))) + (210 * ((x448 + x449))) + ((stencil(trK, stencil_idx_0_0_m5_VVV) + stencil(trK, stencil_idx_0_0_5_VVV)))))))));    
    });
    // apply_dissipation loop 3
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_m5_0_0_VVV(VVV_layout, p.I - 5*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_m5_0_VVV(VVV_layout, p.I - 5*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m5_VVV(VVV_layout, p.I - 5*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_5_VVV(VVV_layout, p.I + 5*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_0_5_0_VVV(VVV_layout, p.I + 5*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        const GF3D5index stencil_idx_5_0_0_VVV(VVV_layout, p.I + 5*p.DI[0]);
        vreal x459 = stencil(AtDD00, stencil_idx_0_m4_0_VVV); // x459: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m4__0_]
        vreal x460 = stencil(AtDD00, stencil_idx_0_4_0_VVV); // x460: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__4__0_]
        vreal x461 = stencil(AtDD00, stencil_idx_0_m3_0_VVV); // x461: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m3__0_]
        vreal x462 = stencil(AtDD00, stencil_idx_0_3_0_VVV); // x462: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__3__0_]
        vreal x463 = stencil(AtDD00, stencil_idx_0_m2_0_VVV); // x463: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m2__0_]
        vreal x464 = stencil(AtDD00, stencil_idx_0_2_0_VVV); // x464: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__2__0_]
        vreal x465 = stencil(AtDD00, stencil_idx_0_m1_0_VVV); // x465: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m1__0_]
        vreal x466 = stencil(AtDD00, stencil_idx_0_1_0_VVV); // x466: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__1__0_]
        vreal x467 = stencil(AtDD00, stencil_idx_0_0_m4_VVV); // x467: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m4_]
        vreal x468 = stencil(AtDD00, stencil_idx_0_0_4_VVV); // x468: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__4_]
        vreal x469 = stencil(AtDD00, stencil_idx_0_0_m3_VVV); // x469: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m3_]
        vreal x470 = stencil(AtDD00, stencil_idx_0_0_3_VVV); // x470: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__3_]
        vreal x471 = stencil(AtDD00, stencil_idx_0_0_m2_VVV); // x471: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m2_]
        vreal x472 = stencil(AtDD00, stencil_idx_0_0_2_VVV); // x472: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__2_]
        vreal x473 = stencil(AtDD00, stencil_idx_0_0_m1_VVV); // x473: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m1_]
        vreal x474 = stencil(AtDD00, stencil_idx_0_0_1_VVV); // x474: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__1_]
        vreal x451 = stencil(AtDD00, stencil_idx_m4_0_0_VVV); // x451: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m4__0__0_]
        vreal x452 = stencil(AtDD00, stencil_idx_4_0_0_VVV); // x452: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__4__0__0_]
        vreal x453 = stencil(AtDD00, stencil_idx_m3_0_0_VVV); // x453: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m3__0__0_]
        vreal x454 = stencil(AtDD00, stencil_idx_3_0_0_VVV); // x454: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__3__0__0_]
        vreal x455 = stencil(AtDD00, stencil_idx_m2_0_0_VVV); // x455: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m2__0__0_]
        vreal x456 = stencil(AtDD00, stencil_idx_2_0_0_VVV); // x456: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__2__0__0_]
        vreal x457 = stencil(AtDD00, stencil_idx_m1_0_0_VVV); // x457: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m1__0__0_]
        vreal x458 = stencil(AtDD00, stencil_idx_1_0_0_VVV); // x458: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__1__0__0_]
        vreal x450 = (-252 * stencil(AtDD00, stencil_idx_0_0_0_VVV)); // x450: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__0_]
        vreal x322 = ((1.0 / 1024.0) * dissipation_epsilon); // x322: Dependency! Liveness = 1; [dissipation_epsilon]
        store(At_rhsDD00, stencil_idx_0_0_0_VVV, (access(At_rhsDD00, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x450 + (-120 * ((x455 + x456))) + (-10 * ((x451 + x452))) + (45 * ((x453 + x454))) + (210 * ((x457 + x458))) + ((stencil(AtDD00, stencil_idx_m5_0_0_VVV) + stencil(AtDD00, stencil_idx_5_0_0_VVV))))) + (DYI * (x450 + (-120 * ((x463 + x464))) + (-10 * ((x459 + x460))) + (45 * ((x461 + x462))) + (210 * ((x465 + x466))) + ((stencil(AtDD00, stencil_idx_0_m5_0_VVV) + stencil(AtDD00, stencil_idx_0_5_0_VVV))))) + (DZI * (x450 + (-120 * ((x471 + x472))) + (-10 * ((x467 + x468))) + (45 * ((x469 + x470))) + (210 * ((x473 + x474))) + ((stencil(AtDD00, stencil_idx_0_0_m5_VVV) + stencil(AtDD00, stencil_idx_0_0_5_VVV)))))))));
        vreal x484 = stencil(AtDD01, stencil_idx_0_m4_0_VVV); // x484: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m4__0_]
        vreal x485 = stencil(AtDD01, stencil_idx_0_4_0_VVV); // x485: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__4__0_]
        vreal x486 = stencil(AtDD01, stencil_idx_0_m3_0_VVV); // x486: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m3__0_]
        vreal x487 = stencil(AtDD01, stencil_idx_0_3_0_VVV); // x487: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__3__0_]
        vreal x488 = stencil(AtDD01, stencil_idx_0_m2_0_VVV); // x488: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m2__0_]
        vreal x489 = stencil(AtDD01, stencil_idx_0_2_0_VVV); // x489: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__2__0_]
        vreal x490 = stencil(AtDD01, stencil_idx_0_m1_0_VVV); // x490: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m1__0_]
        vreal x491 = stencil(AtDD01, stencil_idx_0_1_0_VVV); // x491: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__1__0_]
        vreal x492 = stencil(AtDD01, stencil_idx_0_0_m4_VVV); // x492: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m4_]
        vreal x493 = stencil(AtDD01, stencil_idx_0_0_4_VVV); // x493: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__4_]
        vreal x494 = stencil(AtDD01, stencil_idx_0_0_m3_VVV); // x494: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m3_]
        vreal x495 = stencil(AtDD01, stencil_idx_0_0_3_VVV); // x495: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__3_]
        vreal x496 = stencil(AtDD01, stencil_idx_0_0_m2_VVV); // x496: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m2_]
        vreal x497 = stencil(AtDD01, stencil_idx_0_0_2_VVV); // x497: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__2_]
        vreal x498 = stencil(AtDD01, stencil_idx_0_0_m1_VVV); // x498: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m1_]
        vreal x499 = stencil(AtDD01, stencil_idx_0_0_1_VVV); // x499: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__1_]
        vreal x476 = stencil(AtDD01, stencil_idx_m4_0_0_VVV); // x476: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m4__0__0_]
        vreal x477 = stencil(AtDD01, stencil_idx_4_0_0_VVV); // x477: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__4__0__0_]
        vreal x478 = stencil(AtDD01, stencil_idx_m3_0_0_VVV); // x478: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m3__0__0_]
        vreal x479 = stencil(AtDD01, stencil_idx_3_0_0_VVV); // x479: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__3__0__0_]
        vreal x480 = stencil(AtDD01, stencil_idx_m2_0_0_VVV); // x480: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m2__0__0_]
        vreal x481 = stencil(AtDD01, stencil_idx_2_0_0_VVV); // x481: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__2__0__0_]
        vreal x482 = stencil(AtDD01, stencil_idx_m1_0_0_VVV); // x482: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m1__0__0_]
        vreal x483 = stencil(AtDD01, stencil_idx_1_0_0_VVV); // x483: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__1__0__0_]
        vreal x475 = (-252 * stencil(AtDD01, stencil_idx_0_0_0_VVV)); // x475: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__0_]
        store(At_rhsDD01, stencil_idx_0_0_0_VVV, (access(At_rhsDD01, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x475 + (-120 * ((x480 + x481))) + (-10 * ((x476 + x477))) + (45 * ((x478 + x479))) + (210 * ((x482 + x483))) + ((stencil(AtDD01, stencil_idx_m5_0_0_VVV) + stencil(AtDD01, stencil_idx_5_0_0_VVV))))) + (DYI * (x475 + (-120 * ((x488 + x489))) + (-10 * ((x484 + x485))) + (45 * ((x486 + x487))) + (210 * ((x490 + x491))) + ((stencil(AtDD01, stencil_idx_0_m5_0_VVV) + stencil(AtDD01, stencil_idx_0_5_0_VVV))))) + (DZI * (x475 + (-120 * ((x496 + x497))) + (-10 * ((x492 + x493))) + (45 * ((x494 + x495))) + (210 * ((x498 + x499))) + ((stencil(AtDD01, stencil_idx_0_0_m5_VVV) + stencil(AtDD01, stencil_idx_0_0_5_VVV)))))))));
        vreal x509 = stencil(AtDD02, stencil_idx_0_m4_0_VVV); // x509: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m4__0_]
        vreal x510 = stencil(AtDD02, stencil_idx_0_4_0_VVV); // x510: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__4__0_]
        vreal x511 = stencil(AtDD02, stencil_idx_0_m3_0_VVV); // x511: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m3__0_]
        vreal x512 = stencil(AtDD02, stencil_idx_0_3_0_VVV); // x512: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__3__0_]
        vreal x513 = stencil(AtDD02, stencil_idx_0_m2_0_VVV); // x513: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m2__0_]
        vreal x514 = stencil(AtDD02, stencil_idx_0_2_0_VVV); // x514: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__2__0_]
        vreal x515 = stencil(AtDD02, stencil_idx_0_m1_0_VVV); // x515: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m1__0_]
        vreal x516 = stencil(AtDD02, stencil_idx_0_1_0_VVV); // x516: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__1__0_]
        vreal x517 = stencil(AtDD02, stencil_idx_0_0_m4_VVV); // x517: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m4_]
        vreal x518 = stencil(AtDD02, stencil_idx_0_0_4_VVV); // x518: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__4_]
        vreal x519 = stencil(AtDD02, stencil_idx_0_0_m3_VVV); // x519: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m3_]
        vreal x520 = stencil(AtDD02, stencil_idx_0_0_3_VVV); // x520: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__3_]
        vreal x521 = stencil(AtDD02, stencil_idx_0_0_m2_VVV); // x521: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m2_]
        vreal x522 = stencil(AtDD02, stencil_idx_0_0_2_VVV); // x522: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__2_]
        vreal x523 = stencil(AtDD02, stencil_idx_0_0_m1_VVV); // x523: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m1_]
        vreal x524 = stencil(AtDD02, stencil_idx_0_0_1_VVV); // x524: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__1_]
        vreal x501 = stencil(AtDD02, stencil_idx_m4_0_0_VVV); // x501: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m4__0__0_]
        vreal x502 = stencil(AtDD02, stencil_idx_4_0_0_VVV); // x502: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__4__0__0_]
        vreal x503 = stencil(AtDD02, stencil_idx_m3_0_0_VVV); // x503: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m3__0__0_]
        vreal x504 = stencil(AtDD02, stencil_idx_3_0_0_VVV); // x504: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__3__0__0_]
        vreal x505 = stencil(AtDD02, stencil_idx_m2_0_0_VVV); // x505: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m2__0__0_]
        vreal x506 = stencil(AtDD02, stencil_idx_2_0_0_VVV); // x506: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__2__0__0_]
        vreal x507 = stencil(AtDD02, stencil_idx_m1_0_0_VVV); // x507: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m1__0__0_]
        vreal x508 = stencil(AtDD02, stencil_idx_1_0_0_VVV); // x508: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__1__0__0_]
        vreal x500 = (-252 * stencil(AtDD02, stencil_idx_0_0_0_VVV)); // x500: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__0_]
        store(At_rhsDD02, stencil_idx_0_0_0_VVV, (access(At_rhsDD02, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x500 + (-120 * ((x505 + x506))) + (-10 * ((x501 + x502))) + (45 * ((x503 + x504))) + (210 * ((x507 + x508))) + ((stencil(AtDD02, stencil_idx_m5_0_0_VVV) + stencil(AtDD02, stencil_idx_5_0_0_VVV))))) + (DYI * (x500 + (-120 * ((x513 + x514))) + (-10 * ((x509 + x510))) + (45 * ((x511 + x512))) + (210 * ((x515 + x516))) + ((stencil(AtDD02, stencil_idx_0_m5_0_VVV) + stencil(AtDD02, stencil_idx_0_5_0_VVV))))) + (DZI * (x500 + (-120 * ((x521 + x522))) + (-10 * ((x517 + x518))) + (45 * ((x519 + x520))) + (210 * ((x523 + x524))) + ((stencil(AtDD02, stencil_idx_0_0_m5_VVV) + stencil(AtDD02, stencil_idx_0_0_5_VVV)))))))));
        vreal x534 = stencil(AtDD11, stencil_idx_0_m4_0_VVV); // x534: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m4__0_]
        vreal x535 = stencil(AtDD11, stencil_idx_0_4_0_VVV); // x535: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__4__0_]
        vreal x536 = stencil(AtDD11, stencil_idx_0_m3_0_VVV); // x536: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m3__0_]
        vreal x537 = stencil(AtDD11, stencil_idx_0_3_0_VVV); // x537: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__3__0_]
        vreal x538 = stencil(AtDD11, stencil_idx_0_m2_0_VVV); // x538: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m2__0_]
        vreal x539 = stencil(AtDD11, stencil_idx_0_2_0_VVV); // x539: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__2__0_]
        vreal x540 = stencil(AtDD11, stencil_idx_0_m1_0_VVV); // x540: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m1__0_]
        vreal x541 = stencil(AtDD11, stencil_idx_0_1_0_VVV); // x541: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__1__0_]
        vreal x542 = stencil(AtDD11, stencil_idx_0_0_m4_VVV); // x542: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m4_]
        vreal x543 = stencil(AtDD11, stencil_idx_0_0_4_VVV); // x543: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__4_]
        vreal x544 = stencil(AtDD11, stencil_idx_0_0_m3_VVV); // x544: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m3_]
        vreal x545 = stencil(AtDD11, stencil_idx_0_0_3_VVV); // x545: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__3_]
        vreal x546 = stencil(AtDD11, stencil_idx_0_0_m2_VVV); // x546: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m2_]
        vreal x547 = stencil(AtDD11, stencil_idx_0_0_2_VVV); // x547: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__2_]
        vreal x548 = stencil(AtDD11, stencil_idx_0_0_m1_VVV); // x548: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m1_]
        vreal x549 = stencil(AtDD11, stencil_idx_0_0_1_VVV); // x549: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__1_]
        vreal x526 = stencil(AtDD11, stencil_idx_m4_0_0_VVV); // x526: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m4__0__0_]
        vreal x527 = stencil(AtDD11, stencil_idx_4_0_0_VVV); // x527: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__4__0__0_]
        vreal x528 = stencil(AtDD11, stencil_idx_m3_0_0_VVV); // x528: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m3__0__0_]
        vreal x529 = stencil(AtDD11, stencil_idx_3_0_0_VVV); // x529: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__3__0__0_]
        vreal x530 = stencil(AtDD11, stencil_idx_m2_0_0_VVV); // x530: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m2__0__0_]
        vreal x531 = stencil(AtDD11, stencil_idx_2_0_0_VVV); // x531: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__2__0__0_]
        vreal x532 = stencil(AtDD11, stencil_idx_m1_0_0_VVV); // x532: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m1__0__0_]
        vreal x533 = stencil(AtDD11, stencil_idx_1_0_0_VVV); // x533: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__1__0__0_]
        vreal x525 = (-252 * stencil(AtDD11, stencil_idx_0_0_0_VVV)); // x525: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__0_]
        store(At_rhsDD11, stencil_idx_0_0_0_VVV, (access(At_rhsDD11, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x525 + (-120 * ((x530 + x531))) + (-10 * ((x526 + x527))) + (45 * ((x528 + x529))) + (210 * ((x532 + x533))) + ((stencil(AtDD11, stencil_idx_m5_0_0_VVV) + stencil(AtDD11, stencil_idx_5_0_0_VVV))))) + (DYI * (x525 + (-120 * ((x538 + x539))) + (-10 * ((x534 + x535))) + (45 * ((x536 + x537))) + (210 * ((x540 + x541))) + ((stencil(AtDD11, stencil_idx_0_m5_0_VVV) + stencil(AtDD11, stencil_idx_0_5_0_VVV))))) + (DZI * (x525 + (-120 * ((x546 + x547))) + (-10 * ((x542 + x543))) + (45 * ((x544 + x545))) + (210 * ((x548 + x549))) + ((stencil(AtDD11, stencil_idx_0_0_m5_VVV) + stencil(AtDD11, stencil_idx_0_0_5_VVV)))))))));
        vreal x559 = stencil(AtDD12, stencil_idx_0_m4_0_VVV); // x559: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m4__0_]
        vreal x560 = stencil(AtDD12, stencil_idx_0_4_0_VVV); // x560: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__4__0_]
        vreal x561 = stencil(AtDD12, stencil_idx_0_m3_0_VVV); // x561: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m3__0_]
        vreal x562 = stencil(AtDD12, stencil_idx_0_3_0_VVV); // x562: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__3__0_]
        vreal x563 = stencil(AtDD12, stencil_idx_0_m2_0_VVV); // x563: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m2__0_]
        vreal x564 = stencil(AtDD12, stencil_idx_0_2_0_VVV); // x564: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__2__0_]
        vreal x565 = stencil(AtDD12, stencil_idx_0_m1_0_VVV); // x565: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m1__0_]
        vreal x566 = stencil(AtDD12, stencil_idx_0_1_0_VVV); // x566: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__1__0_]
        vreal x567 = stencil(AtDD12, stencil_idx_0_0_m4_VVV); // x567: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m4_]
        vreal x568 = stencil(AtDD12, stencil_idx_0_0_4_VVV); // x568: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__4_]
        vreal x569 = stencil(AtDD12, stencil_idx_0_0_m3_VVV); // x569: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m3_]
        vreal x570 = stencil(AtDD12, stencil_idx_0_0_3_VVV); // x570: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__3_]
        vreal x571 = stencil(AtDD12, stencil_idx_0_0_m2_VVV); // x571: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m2_]
        vreal x572 = stencil(AtDD12, stencil_idx_0_0_2_VVV); // x572: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__2_]
        vreal x573 = stencil(AtDD12, stencil_idx_0_0_m1_VVV); // x573: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m1_]
        vreal x574 = stencil(AtDD12, stencil_idx_0_0_1_VVV); // x574: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__1_]
        vreal x551 = stencil(AtDD12, stencil_idx_m4_0_0_VVV); // x551: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m4__0__0_]
        vreal x552 = stencil(AtDD12, stencil_idx_4_0_0_VVV); // x552: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__4__0__0_]
        vreal x553 = stencil(AtDD12, stencil_idx_m3_0_0_VVV); // x553: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m3__0__0_]
        vreal x554 = stencil(AtDD12, stencil_idx_3_0_0_VVV); // x554: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__3__0__0_]
        vreal x555 = stencil(AtDD12, stencil_idx_m2_0_0_VVV); // x555: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m2__0__0_]
        vreal x556 = stencil(AtDD12, stencil_idx_2_0_0_VVV); // x556: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__2__0__0_]
        vreal x557 = stencil(AtDD12, stencil_idx_m1_0_0_VVV); // x557: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m1__0__0_]
        vreal x558 = stencil(AtDD12, stencil_idx_1_0_0_VVV); // x558: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__1__0__0_]
        vreal x550 = (-252 * stencil(AtDD12, stencil_idx_0_0_0_VVV)); // x550: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__0_]
        store(At_rhsDD12, stencil_idx_0_0_0_VVV, (access(At_rhsDD12, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x550 + (-120 * ((x555 + x556))) + (-10 * ((x551 + x552))) + (45 * ((x553 + x554))) + (210 * ((x557 + x558))) + ((stencil(AtDD12, stencil_idx_m5_0_0_VVV) + stencil(AtDD12, stencil_idx_5_0_0_VVV))))) + (DYI * (x550 + (-120 * ((x563 + x564))) + (-10 * ((x559 + x560))) + (45 * ((x561 + x562))) + (210 * ((x565 + x566))) + ((stencil(AtDD12, stencil_idx_0_m5_0_VVV) + stencil(AtDD12, stencil_idx_0_5_0_VVV))))) + (DZI * (x550 + (-120 * ((x571 + x572))) + (-10 * ((x567 + x568))) + (45 * ((x569 + x570))) + (210 * ((x573 + x574))) + ((stencil(AtDD12, stencil_idx_0_0_m5_VVV) + stencil(AtDD12, stencil_idx_0_0_5_VVV)))))))));
        vreal x584 = stencil(AtDD22, stencil_idx_0_m4_0_VVV); // x584: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m4__0_]
        vreal x585 = stencil(AtDD22, stencil_idx_0_4_0_VVV); // x585: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__4__0_]
        vreal x586 = stencil(AtDD22, stencil_idx_0_m3_0_VVV); // x586: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m3__0_]
        vreal x587 = stencil(AtDD22, stencil_idx_0_3_0_VVV); // x587: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__3__0_]
        vreal x588 = stencil(AtDD22, stencil_idx_0_m2_0_VVV); // x588: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m2__0_]
        vreal x589 = stencil(AtDD22, stencil_idx_0_2_0_VVV); // x589: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__2__0_]
        vreal x590 = stencil(AtDD22, stencil_idx_0_m1_0_VVV); // x590: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m1__0_]
        vreal x591 = stencil(AtDD22, stencil_idx_0_1_0_VVV); // x591: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__1__0_]
        vreal x592 = stencil(AtDD22, stencil_idx_0_0_m4_VVV); // x592: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m4_]
        vreal x593 = stencil(AtDD22, stencil_idx_0_0_4_VVV); // x593: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__4_]
        vreal x594 = stencil(AtDD22, stencil_idx_0_0_m3_VVV); // x594: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m3_]
        vreal x595 = stencil(AtDD22, stencil_idx_0_0_3_VVV); // x595: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__3_]
        vreal x596 = stencil(AtDD22, stencil_idx_0_0_m2_VVV); // x596: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m2_]
        vreal x597 = stencil(AtDD22, stencil_idx_0_0_2_VVV); // x597: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__2_]
        vreal x598 = stencil(AtDD22, stencil_idx_0_0_m1_VVV); // x598: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m1_]
        vreal x599 = stencil(AtDD22, stencil_idx_0_0_1_VVV); // x599: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__1_]
        vreal x576 = stencil(AtDD22, stencil_idx_m4_0_0_VVV); // x576: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m4__0__0_]
        vreal x577 = stencil(AtDD22, stencil_idx_4_0_0_VVV); // x577: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__4__0__0_]
        vreal x578 = stencil(AtDD22, stencil_idx_m3_0_0_VVV); // x578: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m3__0__0_]
        vreal x579 = stencil(AtDD22, stencil_idx_3_0_0_VVV); // x579: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__3__0__0_]
        vreal x580 = stencil(AtDD22, stencil_idx_m2_0_0_VVV); // x580: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m2__0__0_]
        vreal x581 = stencil(AtDD22, stencil_idx_2_0_0_VVV); // x581: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__2__0__0_]
        vreal x582 = stencil(AtDD22, stencil_idx_m1_0_0_VVV); // x582: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m1__0__0_]
        vreal x583 = stencil(AtDD22, stencil_idx_1_0_0_VVV); // x583: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__1__0__0_]
        vreal x575 = (-252 * stencil(AtDD22, stencil_idx_0_0_0_VVV)); // x575: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__0_]
        store(At_rhsDD22, stencil_idx_0_0_0_VVV, (access(At_rhsDD22, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x575 + (-120 * ((x580 + x581))) + (-10 * ((x576 + x577))) + (45 * ((x578 + x579))) + (210 * ((x582 + x583))) + ((stencil(AtDD22, stencil_idx_m5_0_0_VVV) + stencil(AtDD22, stencil_idx_5_0_0_VVV))))) + (DYI * (x575 + (-120 * ((x588 + x589))) + (-10 * ((x584 + x585))) + (45 * ((x586 + x587))) + (210 * ((x590 + x591))) + ((stencil(AtDD22, stencil_idx_0_m5_0_VVV) + stencil(AtDD22, stencil_idx_0_5_0_VVV))))) + (DZI * (x575 + (-120 * ((x596 + x597))) + (-10 * ((x592 + x593))) + (45 * ((x594 + x595))) + (210 * ((x598 + x599))) + ((stencil(AtDD22, stencil_idx_0_0_m5_VVV) + stencil(AtDD22, stencil_idx_0_0_5_VVV)))))))));    
    });
    // apply_dissipation loop 4
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_m5_0_0_VVV(VVV_layout, p.I - 5*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_m5_0_VVV(VVV_layout, p.I - 5*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m5_VVV(VVV_layout, p.I - 5*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_5_VVV(VVV_layout, p.I + 5*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_0_5_0_VVV(VVV_layout, p.I + 5*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        const GF3D5index stencil_idx_5_0_0_VVV(VVV_layout, p.I + 5*p.DI[0]);
        vreal x609 = stencil(ConfConnectU0, stencil_idx_0_m4_0_VVV); // x609: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__m4__0_]
        vreal x610 = stencil(ConfConnectU0, stencil_idx_0_4_0_VVV); // x610: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__4__0_]
        vreal x611 = stencil(ConfConnectU0, stencil_idx_0_m3_0_VVV); // x611: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__m3__0_]
        vreal x612 = stencil(ConfConnectU0, stencil_idx_0_3_0_VVV); // x612: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__3__0_]
        vreal x613 = stencil(ConfConnectU0, stencil_idx_0_m2_0_VVV); // x613: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__m2__0_]
        vreal x614 = stencil(ConfConnectU0, stencil_idx_0_2_0_VVV); // x614: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__2__0_]
        vreal x615 = stencil(ConfConnectU0, stencil_idx_0_m1_0_VVV); // x615: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__m1__0_]
        vreal x616 = stencil(ConfConnectU0, stencil_idx_0_1_0_VVV); // x616: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__1__0_]
        vreal x617 = stencil(ConfConnectU0, stencil_idx_0_0_m4_VVV); // x617: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__m4_]
        vreal x618 = stencil(ConfConnectU0, stencil_idx_0_0_4_VVV); // x618: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__4_]
        vreal x619 = stencil(ConfConnectU0, stencil_idx_0_0_m3_VVV); // x619: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__m3_]
        vreal x620 = stencil(ConfConnectU0, stencil_idx_0_0_3_VVV); // x620: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__3_]
        vreal x621 = stencil(ConfConnectU0, stencil_idx_0_0_m2_VVV); // x621: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__m2_]
        vreal x622 = stencil(ConfConnectU0, stencil_idx_0_0_2_VVV); // x622: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__2_]
        vreal x623 = stencil(ConfConnectU0, stencil_idx_0_0_m1_VVV); // x623: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__m1_]
        vreal x624 = stencil(ConfConnectU0, stencil_idx_0_0_1_VVV); // x624: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__1_]
        vreal x601 = stencil(ConfConnectU0, stencil_idx_m4_0_0_VVV); // x601: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__m4__0__0_]
        vreal x602 = stencil(ConfConnectU0, stencil_idx_4_0_0_VVV); // x602: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__4__0__0_]
        vreal x603 = stencil(ConfConnectU0, stencil_idx_m3_0_0_VVV); // x603: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__m3__0__0_]
        vreal x604 = stencil(ConfConnectU0, stencil_idx_3_0_0_VVV); // x604: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__3__0__0_]
        vreal x605 = stencil(ConfConnectU0, stencil_idx_m2_0_0_VVV); // x605: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__m2__0__0_]
        vreal x606 = stencil(ConfConnectU0, stencil_idx_2_0_0_VVV); // x606: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__2__0__0_]
        vreal x607 = stencil(ConfConnectU0, stencil_idx_m1_0_0_VVV); // x607: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__m1__0__0_]
        vreal x608 = stencil(ConfConnectU0, stencil_idx_1_0_0_VVV); // x608: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__1__0__0_]
        vreal x600 = (-252 * stencil(ConfConnectU0, stencil_idx_0_0_0_VVV)); // x600: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__0_]
        vreal x322 = ((1.0 / 1024.0) * dissipation_epsilon); // x322: Dependency! Liveness = 1; [dissipation_epsilon]
        store(ConfConnect_rhsU0, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU0, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x600 + (-120 * ((x605 + x606))) + (-10 * ((x601 + x602))) + (45 * ((x603 + x604))) + (210 * ((x607 + x608))) + ((stencil(ConfConnectU0, stencil_idx_m5_0_0_VVV) + stencil(ConfConnectU0, stencil_idx_5_0_0_VVV))))) + (DYI * (x600 + (-120 * ((x613 + x614))) + (-10 * ((x609 + x610))) + (45 * ((x611 + x612))) + (210 * ((x615 + x616))) + ((stencil(ConfConnectU0, stencil_idx_0_m5_0_VVV) + stencil(ConfConnectU0, stencil_idx_0_5_0_VVV))))) + (DZI * (x600 + (-120 * ((x621 + x622))) + (-10 * ((x617 + x618))) + (45 * ((x619 + x620))) + (210 * ((x623 + x624))) + ((stencil(ConfConnectU0, stencil_idx_0_0_m5_VVV) + stencil(ConfConnectU0, stencil_idx_0_0_5_VVV)))))))));
        vreal x634 = stencil(ConfConnectU1, stencil_idx_0_m4_0_VVV); // x634: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__m4__0_]
        vreal x635 = stencil(ConfConnectU1, stencil_idx_0_4_0_VVV); // x635: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__4__0_]
        vreal x636 = stencil(ConfConnectU1, stencil_idx_0_m3_0_VVV); // x636: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__m3__0_]
        vreal x637 = stencil(ConfConnectU1, stencil_idx_0_3_0_VVV); // x637: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__3__0_]
        vreal x638 = stencil(ConfConnectU1, stencil_idx_0_m2_0_VVV); // x638: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__m2__0_]
        vreal x639 = stencil(ConfConnectU1, stencil_idx_0_2_0_VVV); // x639: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__2__0_]
        vreal x640 = stencil(ConfConnectU1, stencil_idx_0_m1_0_VVV); // x640: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__m1__0_]
        vreal x641 = stencil(ConfConnectU1, stencil_idx_0_1_0_VVV); // x641: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__1__0_]
        vreal x642 = stencil(ConfConnectU1, stencil_idx_0_0_m4_VVV); // x642: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__m4_]
        vreal x643 = stencil(ConfConnectU1, stencil_idx_0_0_4_VVV); // x643: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__4_]
        vreal x644 = stencil(ConfConnectU1, stencil_idx_0_0_m3_VVV); // x644: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__m3_]
        vreal x645 = stencil(ConfConnectU1, stencil_idx_0_0_3_VVV); // x645: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__3_]
        vreal x646 = stencil(ConfConnectU1, stencil_idx_0_0_m2_VVV); // x646: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__m2_]
        vreal x647 = stencil(ConfConnectU1, stencil_idx_0_0_2_VVV); // x647: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__2_]
        vreal x648 = stencil(ConfConnectU1, stencil_idx_0_0_m1_VVV); // x648: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__m1_]
        vreal x649 = stencil(ConfConnectU1, stencil_idx_0_0_1_VVV); // x649: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__1_]
        vreal x626 = stencil(ConfConnectU1, stencil_idx_m4_0_0_VVV); // x626: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__m4__0__0_]
        vreal x627 = stencil(ConfConnectU1, stencil_idx_4_0_0_VVV); // x627: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__4__0__0_]
        vreal x628 = stencil(ConfConnectU1, stencil_idx_m3_0_0_VVV); // x628: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__m3__0__0_]
        vreal x629 = stencil(ConfConnectU1, stencil_idx_3_0_0_VVV); // x629: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__3__0__0_]
        vreal x630 = stencil(ConfConnectU1, stencil_idx_m2_0_0_VVV); // x630: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__m2__0__0_]
        vreal x631 = stencil(ConfConnectU1, stencil_idx_2_0_0_VVV); // x631: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__2__0__0_]
        vreal x632 = stencil(ConfConnectU1, stencil_idx_m1_0_0_VVV); // x632: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__m1__0__0_]
        vreal x633 = stencil(ConfConnectU1, stencil_idx_1_0_0_VVV); // x633: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__1__0__0_]
        vreal x625 = (-252 * stencil(ConfConnectU1, stencil_idx_0_0_0_VVV)); // x625: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__0_]
        store(ConfConnect_rhsU1, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU1, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x625 + (-120 * ((x630 + x631))) + (-10 * ((x626 + x627))) + (45 * ((x628 + x629))) + (210 * ((x632 + x633))) + ((stencil(ConfConnectU1, stencil_idx_m5_0_0_VVV) + stencil(ConfConnectU1, stencil_idx_5_0_0_VVV))))) + (DYI * (x625 + (-120 * ((x638 + x639))) + (-10 * ((x634 + x635))) + (45 * ((x636 + x637))) + (210 * ((x640 + x641))) + ((stencil(ConfConnectU1, stencil_idx_0_m5_0_VVV) + stencil(ConfConnectU1, stencil_idx_0_5_0_VVV))))) + (DZI * (x625 + (-120 * ((x646 + x647))) + (-10 * ((x642 + x643))) + (45 * ((x644 + x645))) + (210 * ((x648 + x649))) + ((stencil(ConfConnectU1, stencil_idx_0_0_m5_VVV) + stencil(ConfConnectU1, stencil_idx_0_0_5_VVV)))))))));
        vreal x659 = stencil(ConfConnectU2, stencil_idx_0_m4_0_VVV); // x659: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__m4__0_]
        vreal x660 = stencil(ConfConnectU2, stencil_idx_0_4_0_VVV); // x660: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__4__0_]
        vreal x661 = stencil(ConfConnectU2, stencil_idx_0_m3_0_VVV); // x661: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__m3__0_]
        vreal x662 = stencil(ConfConnectU2, stencil_idx_0_3_0_VVV); // x662: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__3__0_]
        vreal x663 = stencil(ConfConnectU2, stencil_idx_0_m2_0_VVV); // x663: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__m2__0_]
        vreal x664 = stencil(ConfConnectU2, stencil_idx_0_2_0_VVV); // x664: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__2__0_]
        vreal x665 = stencil(ConfConnectU2, stencil_idx_0_m1_0_VVV); // x665: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__m1__0_]
        vreal x666 = stencil(ConfConnectU2, stencil_idx_0_1_0_VVV); // x666: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__1__0_]
        vreal x667 = stencil(ConfConnectU2, stencil_idx_0_0_m4_VVV); // x667: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__m4_]
        vreal x668 = stencil(ConfConnectU2, stencil_idx_0_0_4_VVV); // x668: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__4_]
        vreal x669 = stencil(ConfConnectU2, stencil_idx_0_0_m3_VVV); // x669: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__m3_]
        vreal x670 = stencil(ConfConnectU2, stencil_idx_0_0_3_VVV); // x670: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__3_]
        vreal x671 = stencil(ConfConnectU2, stencil_idx_0_0_m2_VVV); // x671: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__m2_]
        vreal x672 = stencil(ConfConnectU2, stencil_idx_0_0_2_VVV); // x672: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__2_]
        vreal x673 = stencil(ConfConnectU2, stencil_idx_0_0_m1_VVV); // x673: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__m1_]
        vreal x674 = stencil(ConfConnectU2, stencil_idx_0_0_1_VVV); // x674: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__1_]
        vreal x651 = stencil(ConfConnectU2, stencil_idx_m4_0_0_VVV); // x651: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__m4__0__0_]
        vreal x652 = stencil(ConfConnectU2, stencil_idx_4_0_0_VVV); // x652: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__4__0__0_]
        vreal x653 = stencil(ConfConnectU2, stencil_idx_m3_0_0_VVV); // x653: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__m3__0__0_]
        vreal x654 = stencil(ConfConnectU2, stencil_idx_3_0_0_VVV); // x654: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__3__0__0_]
        vreal x655 = stencil(ConfConnectU2, stencil_idx_m2_0_0_VVV); // x655: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__m2__0__0_]
        vreal x656 = stencil(ConfConnectU2, stencil_idx_2_0_0_VVV); // x656: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__2__0__0_]
        vreal x657 = stencil(ConfConnectU2, stencil_idx_m1_0_0_VVV); // x657: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__m1__0__0_]
        vreal x658 = stencil(ConfConnectU2, stencil_idx_1_0_0_VVV); // x658: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__1__0__0_]
        vreal x650 = (-252 * stencil(ConfConnectU2, stencil_idx_0_0_0_VVV)); // x650: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__0_]
        store(ConfConnect_rhsU2, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU2, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x650 + (-120 * ((x655 + x656))) + (-10 * ((x651 + x652))) + (45 * ((x653 + x654))) + (210 * ((x657 + x658))) + ((stencil(ConfConnectU2, stencil_idx_m5_0_0_VVV) + stencil(ConfConnectU2, stencil_idx_5_0_0_VVV))))) + (DYI * (x650 + (-120 * ((x663 + x664))) + (-10 * ((x659 + x660))) + (45 * ((x661 + x662))) + (210 * ((x665 + x666))) + ((stencil(ConfConnectU2, stencil_idx_0_m5_0_VVV) + stencil(ConfConnectU2, stencil_idx_0_5_0_VVV))))) + (DZI * (x650 + (-120 * ((x671 + x672))) + (-10 * ((x667 + x668))) + (45 * ((x669 + x670))) + (210 * ((x673 + x674))) + ((stencil(ConfConnectU2, stencil_idx_0_0_m5_VVV) + stencil(ConfConnectU2, stencil_idx_0_0_5_VVV)))))))));    
    });
    // apply_dissipation loop 5
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_m5_0_0_VVV(VVV_layout, p.I - 5*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_m5_0_VVV(VVV_layout, p.I - 5*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m5_VVV(VVV_layout, p.I - 5*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_5_VVV(VVV_layout, p.I + 5*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_0_5_0_VVV(VVV_layout, p.I + 5*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        const GF3D5index stencil_idx_5_0_0_VVV(VVV_layout, p.I + 5*p.DI[0]);
        vreal x676 = stencil(evo_lapse, stencil_idx_m4_0_0_VVV); // x676: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m4__0__0_]
        vreal x677 = stencil(evo_lapse, stencil_idx_4_0_0_VVV); // x677: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__4__0__0_]
        vreal x678 = ((x676 + x677)); // x678: Dependency! Liveness = 2; [x676, x677]
        vreal x679 = stencil(evo_lapse, stencil_idx_m3_0_0_VVV); // x679: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m3__0__0_]
        vreal x680 = stencil(evo_lapse, stencil_idx_3_0_0_VVV); // x680: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__3__0__0_]
        vreal x681 = ((x679 + x680)); // x681: Dependency! Liveness = 2; [x679, x680]
        vreal x682 = stencil(evo_lapse, stencil_idx_m2_0_0_VVV); // x682: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m2__0__0_]
        vreal x683 = stencil(evo_lapse, stencil_idx_2_0_0_VVV); // x683: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__2__0__0_]
        vreal x684 = ((x682 + x683)); // x684: Dependency! Liveness = 2; [x682, x683]
        vreal x685 = stencil(evo_lapse, stencil_idx_m1_0_0_VVV); // x685: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m1__0__0_]
        vreal x686 = stencil(evo_lapse, stencil_idx_1_0_0_VVV); // x686: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__1__0__0_]
        vreal x687 = ((x685 + x686)); // x687: Dependency! Liveness = 2; [x685, x686]
        vreal x688 = stencil(evo_lapse, stencil_idx_0_m4_0_VVV); // x688: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m4__0_]
        vreal x689 = stencil(evo_lapse, stencil_idx_0_4_0_VVV); // x689: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__4__0_]
        vreal x690 = ((x688 + x689)); // x690: Dependency! Liveness = 2; [x688, x689]
        vreal x691 = stencil(evo_lapse, stencil_idx_0_m3_0_VVV); // x691: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m3__0_]
        vreal x692 = stencil(evo_lapse, stencil_idx_0_3_0_VVV); // x692: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__3__0_]
        vreal x693 = ((x691 + x692)); // x693: Dependency! Liveness = 2; [x691, x692]
        vreal x694 = stencil(evo_lapse, stencil_idx_0_m2_0_VVV); // x694: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m2__0_]
        vreal x695 = stencil(evo_lapse, stencil_idx_0_2_0_VVV); // x695: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__2__0_]
        vreal x696 = ((x694 + x695)); // x696: Dependency! Liveness = 2; [x694, x695]
        vreal x697 = stencil(evo_lapse, stencil_idx_0_m1_0_VVV); // x697: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m1__0_]
        vreal x698 = stencil(evo_lapse, stencil_idx_0_1_0_VVV); // x698: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__1__0_]
        vreal x699 = ((x697 + x698)); // x699: Dependency! Liveness = 2; [x697, x698]
        vreal x700 = stencil(evo_lapse, stencil_idx_0_0_m4_VVV); // x700: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m4_]
        vreal x701 = stencil(evo_lapse, stencil_idx_0_0_4_VVV); // x701: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__4_]
        vreal x702 = ((x700 + x701)); // x702: Dependency! Liveness = 2; [x700, x701]
        vreal x703 = stencil(evo_lapse, stencil_idx_0_0_m3_VVV); // x703: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m3_]
        vreal x704 = stencil(evo_lapse, stencil_idx_0_0_3_VVV); // x704: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__3_]
        vreal x705 = ((x703 + x704)); // x705: Dependency! Liveness = 2; [x703, x704]
        vreal x706 = stencil(evo_lapse, stencil_idx_0_0_m2_VVV); // x706: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m2_]
        vreal x707 = stencil(evo_lapse, stencil_idx_0_0_2_VVV); // x707: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__2_]
        vreal x708 = ((x706 + x707)); // x708: Dependency! Liveness = 2; [x706, x707]
        vreal x709 = stencil(evo_lapse, stencil_idx_0_0_m1_VVV); // x709: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m1_]
        vreal x710 = stencil(evo_lapse, stencil_idx_0_0_1_VVV); // x710: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__1_]
        vreal x711 = ((x709 + x710)); // x711: Dependency! Liveness = 2; [x709, x710]
        vreal x675 = (-252 * stencil(evo_lapse, stencil_idx_0_0_0_VVV)); // x675: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__0_]
        vreal x322 = ((1.0 / 1024.0) * dissipation_epsilon); // x322: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_lapse_rhs, stencil_idx_0_0_0_VVV, (access(evo_lapse_rhs, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x675 + (-120 * x684) + (-10 * x678) + (45 * x681) + (210 * x687) + ((stencil(evo_lapse, stencil_idx_m5_0_0_VVV) + stencil(evo_lapse, stencil_idx_5_0_0_VVV))))) + (DYI * (x675 + (-120 * x696) + (-10 * x690) + (45 * x693) + (210 * x699) + ((stencil(evo_lapse, stencil_idx_0_m5_0_VVV) + stencil(evo_lapse, stencil_idx_0_5_0_VVV))))) + (DZI * (x675 + (-120 * x708) + (-10 * x702) + (45 * x705) + (210 * x711) + ((stencil(evo_lapse, stencil_idx_0_0_m5_VVV) + stencil(evo_lapse, stencil_idx_0_0_5_VVV)))))))));    
    });
    // apply_dissipation loop 6
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_m5_0_0_VVV(VVV_layout, p.I - 5*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_m5_0_VVV(VVV_layout, p.I - 5*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m5_VVV(VVV_layout, p.I - 5*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_5_VVV(VVV_layout, p.I + 5*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_0_5_0_VVV(VVV_layout, p.I + 5*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        const GF3D5index stencil_idx_5_0_0_VVV(VVV_layout, p.I + 5*p.DI[0]);
        vreal x713 = stencil(evo_shiftU0, stencil_idx_m4_0_0_VVV); // x713: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m4__0__0_]
        vreal x714 = stencil(evo_shiftU0, stencil_idx_4_0_0_VVV); // x714: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__4__0__0_]
        vreal x715 = ((x713 + x714)); // x715: Dependency! Liveness = 2; [x713, x714]
        vreal x716 = stencil(evo_shiftU0, stencil_idx_m3_0_0_VVV); // x716: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m3__0__0_]
        vreal x717 = stencil(evo_shiftU0, stencil_idx_3_0_0_VVV); // x717: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__3__0__0_]
        vreal x718 = ((x716 + x717)); // x718: Dependency! Liveness = 2; [x716, x717]
        vreal x719 = stencil(evo_shiftU0, stencil_idx_m2_0_0_VVV); // x719: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m2__0__0_]
        vreal x720 = stencil(evo_shiftU0, stencil_idx_2_0_0_VVV); // x720: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__2__0__0_]
        vreal x721 = ((x719 + x720)); // x721: Dependency! Liveness = 2; [x719, x720]
        vreal x722 = stencil(evo_shiftU0, stencil_idx_m1_0_0_VVV); // x722: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m1__0__0_]
        vreal x723 = stencil(evo_shiftU0, stencil_idx_1_0_0_VVV); // x723: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__1__0__0_]
        vreal x724 = ((x722 + x723)); // x724: Dependency! Liveness = 2; [x722, x723]
        vreal x725 = stencil(evo_shiftU0, stencil_idx_0_m4_0_VVV); // x725: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m4__0_]
        vreal x726 = stencil(evo_shiftU0, stencil_idx_0_4_0_VVV); // x726: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__4__0_]
        vreal x727 = ((x725 + x726)); // x727: Dependency! Liveness = 2; [x725, x726]
        vreal x728 = stencil(evo_shiftU0, stencil_idx_0_m3_0_VVV); // x728: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m3__0_]
        vreal x729 = stencil(evo_shiftU0, stencil_idx_0_3_0_VVV); // x729: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__3__0_]
        vreal x730 = ((x728 + x729)); // x730: Dependency! Liveness = 2; [x728, x729]
        vreal x731 = stencil(evo_shiftU0, stencil_idx_0_m2_0_VVV); // x731: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m2__0_]
        vreal x732 = stencil(evo_shiftU0, stencil_idx_0_2_0_VVV); // x732: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__2__0_]
        vreal x733 = ((x731 + x732)); // x733: Dependency! Liveness = 2; [x731, x732]
        vreal x734 = stencil(evo_shiftU0, stencil_idx_0_m1_0_VVV); // x734: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m1__0_]
        vreal x735 = stencil(evo_shiftU0, stencil_idx_0_1_0_VVV); // x735: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__1__0_]
        vreal x736 = ((x734 + x735)); // x736: Dependency! Liveness = 2; [x734, x735]
        vreal x737 = stencil(evo_shiftU0, stencil_idx_0_0_m4_VVV); // x737: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m4_]
        vreal x738 = stencil(evo_shiftU0, stencil_idx_0_0_4_VVV); // x738: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__4_]
        vreal x739 = ((x737 + x738)); // x739: Dependency! Liveness = 2; [x737, x738]
        vreal x740 = stencil(evo_shiftU0, stencil_idx_0_0_m3_VVV); // x740: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m3_]
        vreal x741 = stencil(evo_shiftU0, stencil_idx_0_0_3_VVV); // x741: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__3_]
        vreal x742 = ((x740 + x741)); // x742: Dependency! Liveness = 2; [x740, x741]
        vreal x743 = stencil(evo_shiftU0, stencil_idx_0_0_m2_VVV); // x743: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m2_]
        vreal x744 = stencil(evo_shiftU0, stencil_idx_0_0_2_VVV); // x744: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__2_]
        vreal x745 = ((x743 + x744)); // x745: Dependency! Liveness = 2; [x743, x744]
        vreal x746 = stencil(evo_shiftU0, stencil_idx_0_0_m1_VVV); // x746: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m1_]
        vreal x747 = stencil(evo_shiftU0, stencil_idx_0_0_1_VVV); // x747: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__1_]
        vreal x748 = ((x746 + x747)); // x748: Dependency! Liveness = 2; [x746, x747]
        vreal x712 = (-252 * stencil(evo_shiftU0, stencil_idx_0_0_0_VVV)); // x712: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__0_]
        vreal x322 = ((1.0 / 1024.0) * dissipation_epsilon); // x322: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_shift_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU0, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x712 + (-120 * x721) + (-10 * x715) + (45 * x718) + (210 * x724) + ((stencil(evo_shiftU0, stencil_idx_m5_0_0_VVV) + stencil(evo_shiftU0, stencil_idx_5_0_0_VVV))))) + (DYI * (x712 + (-120 * x733) + (-10 * x727) + (45 * x730) + (210 * x736) + ((stencil(evo_shiftU0, stencil_idx_0_m5_0_VVV) + stencil(evo_shiftU0, stencil_idx_0_5_0_VVV))))) + (DZI * (x712 + (-120 * x745) + (-10 * x739) + (45 * x742) + (210 * x748) + ((stencil(evo_shiftU0, stencil_idx_0_0_m5_VVV) + stencil(evo_shiftU0, stencil_idx_0_0_5_VVV)))))))));
        vreal x750 = stencil(evo_shiftU1, stencil_idx_m4_0_0_VVV); // x750: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m4__0__0_]
        vreal x751 = stencil(evo_shiftU1, stencil_idx_4_0_0_VVV); // x751: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__4__0__0_]
        vreal x752 = ((x750 + x751)); // x752: Dependency! Liveness = 2; [x750, x751]
        vreal x753 = stencil(evo_shiftU1, stencil_idx_m3_0_0_VVV); // x753: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m3__0__0_]
        vreal x754 = stencil(evo_shiftU1, stencil_idx_3_0_0_VVV); // x754: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__3__0__0_]
        vreal x755 = ((x753 + x754)); // x755: Dependency! Liveness = 2; [x753, x754]
        vreal x756 = stencil(evo_shiftU1, stencil_idx_m2_0_0_VVV); // x756: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m2__0__0_]
        vreal x757 = stencil(evo_shiftU1, stencil_idx_2_0_0_VVV); // x757: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__2__0__0_]
        vreal x758 = ((x756 + x757)); // x758: Dependency! Liveness = 2; [x756, x757]
        vreal x759 = stencil(evo_shiftU1, stencil_idx_m1_0_0_VVV); // x759: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m1__0__0_]
        vreal x760 = stencil(evo_shiftU1, stencil_idx_1_0_0_VVV); // x760: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__1__0__0_]
        vreal x761 = ((x759 + x760)); // x761: Dependency! Liveness = 2; [x759, x760]
        vreal x762 = stencil(evo_shiftU1, stencil_idx_0_m4_0_VVV); // x762: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m4__0_]
        vreal x763 = stencil(evo_shiftU1, stencil_idx_0_4_0_VVV); // x763: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__4__0_]
        vreal x764 = ((x762 + x763)); // x764: Dependency! Liveness = 2; [x762, x763]
        vreal x765 = stencil(evo_shiftU1, stencil_idx_0_m3_0_VVV); // x765: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m3__0_]
        vreal x766 = stencil(evo_shiftU1, stencil_idx_0_3_0_VVV); // x766: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__3__0_]
        vreal x767 = ((x765 + x766)); // x767: Dependency! Liveness = 2; [x765, x766]
        vreal x768 = stencil(evo_shiftU1, stencil_idx_0_m2_0_VVV); // x768: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m2__0_]
        vreal x769 = stencil(evo_shiftU1, stencil_idx_0_2_0_VVV); // x769: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__2__0_]
        vreal x770 = ((x768 + x769)); // x770: Dependency! Liveness = 2; [x768, x769]
        vreal x771 = stencil(evo_shiftU1, stencil_idx_0_m1_0_VVV); // x771: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m1__0_]
        vreal x772 = stencil(evo_shiftU1, stencil_idx_0_1_0_VVV); // x772: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__1__0_]
        vreal x773 = ((x771 + x772)); // x773: Dependency! Liveness = 2; [x771, x772]
        vreal x774 = stencil(evo_shiftU1, stencil_idx_0_0_m4_VVV); // x774: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m4_]
        vreal x775 = stencil(evo_shiftU1, stencil_idx_0_0_4_VVV); // x775: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__4_]
        vreal x776 = ((x774 + x775)); // x776: Dependency! Liveness = 2; [x774, x775]
        vreal x777 = stencil(evo_shiftU1, stencil_idx_0_0_m3_VVV); // x777: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m3_]
        vreal x778 = stencil(evo_shiftU1, stencil_idx_0_0_3_VVV); // x778: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__3_]
        vreal x779 = ((x777 + x778)); // x779: Dependency! Liveness = 2; [x777, x778]
        vreal x780 = stencil(evo_shiftU1, stencil_idx_0_0_m2_VVV); // x780: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m2_]
        vreal x781 = stencil(evo_shiftU1, stencil_idx_0_0_2_VVV); // x781: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__2_]
        vreal x782 = ((x780 + x781)); // x782: Dependency! Liveness = 2; [x780, x781]
        vreal x783 = stencil(evo_shiftU1, stencil_idx_0_0_m1_VVV); // x783: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m1_]
        vreal x784 = stencil(evo_shiftU1, stencil_idx_0_0_1_VVV); // x784: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__1_]
        vreal x785 = ((x783 + x784)); // x785: Dependency! Liveness = 2; [x783, x784]
        vreal x749 = (-252 * stencil(evo_shiftU1, stencil_idx_0_0_0_VVV)); // x749: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__0_]
        store(evo_shift_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU1, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x749 + (-120 * x758) + (-10 * x752) + (45 * x755) + (210 * x761) + ((stencil(evo_shiftU1, stencil_idx_m5_0_0_VVV) + stencil(evo_shiftU1, stencil_idx_5_0_0_VVV))))) + (DYI * (x749 + (-120 * x770) + (-10 * x764) + (45 * x767) + (210 * x773) + ((stencil(evo_shiftU1, stencil_idx_0_m5_0_VVV) + stencil(evo_shiftU1, stencil_idx_0_5_0_VVV))))) + (DZI * (x749 + (-120 * x782) + (-10 * x776) + (45 * x779) + (210 * x785) + ((stencil(evo_shiftU1, stencil_idx_0_0_m5_VVV) + stencil(evo_shiftU1, stencil_idx_0_0_5_VVV)))))))));
        vreal x787 = stencil(evo_shiftU2, stencil_idx_m4_0_0_VVV); // x787: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m4__0__0_]
        vreal x788 = stencil(evo_shiftU2, stencil_idx_4_0_0_VVV); // x788: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__4__0__0_]
        vreal x789 = ((x787 + x788)); // x789: Dependency! Liveness = 2; [x787, x788]
        vreal x790 = stencil(evo_shiftU2, stencil_idx_m3_0_0_VVV); // x790: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m3__0__0_]
        vreal x791 = stencil(evo_shiftU2, stencil_idx_3_0_0_VVV); // x791: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__3__0__0_]
        vreal x792 = ((x790 + x791)); // x792: Dependency! Liveness = 2; [x790, x791]
        vreal x793 = stencil(evo_shiftU2, stencil_idx_m2_0_0_VVV); // x793: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m2__0__0_]
        vreal x794 = stencil(evo_shiftU2, stencil_idx_2_0_0_VVV); // x794: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__2__0__0_]
        vreal x795 = ((x793 + x794)); // x795: Dependency! Liveness = 2; [x793, x794]
        vreal x796 = stencil(evo_shiftU2, stencil_idx_m1_0_0_VVV); // x796: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m1__0__0_]
        vreal x797 = stencil(evo_shiftU2, stencil_idx_1_0_0_VVV); // x797: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__1__0__0_]
        vreal x798 = ((x796 + x797)); // x798: Dependency! Liveness = 2; [x796, x797]
        vreal x799 = stencil(evo_shiftU2, stencil_idx_0_m4_0_VVV); // x799: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m4__0_]
        vreal x800 = stencil(evo_shiftU2, stencil_idx_0_4_0_VVV); // x800: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__4__0_]
        vreal x801 = ((x799 + x800)); // x801: Dependency! Liveness = 2; [x799, x800]
        vreal x802 = stencil(evo_shiftU2, stencil_idx_0_m3_0_VVV); // x802: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m3__0_]
        vreal x803 = stencil(evo_shiftU2, stencil_idx_0_3_0_VVV); // x803: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__3__0_]
        vreal x804 = ((x802 + x803)); // x804: Dependency! Liveness = 2; [x802, x803]
        vreal x805 = stencil(evo_shiftU2, stencil_idx_0_m2_0_VVV); // x805: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m2__0_]
        vreal x806 = stencil(evo_shiftU2, stencil_idx_0_2_0_VVV); // x806: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__2__0_]
        vreal x807 = ((x805 + x806)); // x807: Dependency! Liveness = 2; [x805, x806]
        vreal x808 = stencil(evo_shiftU2, stencil_idx_0_m1_0_VVV); // x808: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m1__0_]
        vreal x809 = stencil(evo_shiftU2, stencil_idx_0_1_0_VVV); // x809: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__1__0_]
        vreal x810 = ((x808 + x809)); // x810: Dependency! Liveness = 2; [x808, x809]
        vreal x811 = stencil(evo_shiftU2, stencil_idx_0_0_m4_VVV); // x811: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m4_]
        vreal x812 = stencil(evo_shiftU2, stencil_idx_0_0_4_VVV); // x812: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__4_]
        vreal x813 = ((x811 + x812)); // x813: Dependency! Liveness = 2; [x811, x812]
        vreal x814 = stencil(evo_shiftU2, stencil_idx_0_0_m3_VVV); // x814: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m3_]
        vreal x815 = stencil(evo_shiftU2, stencil_idx_0_0_3_VVV); // x815: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__3_]
        vreal x816 = ((x814 + x815)); // x816: Dependency! Liveness = 2; [x814, x815]
        vreal x817 = stencil(evo_shiftU2, stencil_idx_0_0_m2_VVV); // x817: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m2_]
        vreal x818 = stencil(evo_shiftU2, stencil_idx_0_0_2_VVV); // x818: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__2_]
        vreal x819 = ((x817 + x818)); // x819: Dependency! Liveness = 2; [x817, x818]
        vreal x820 = stencil(evo_shiftU2, stencil_idx_0_0_m1_VVV); // x820: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m1_]
        vreal x821 = stencil(evo_shiftU2, stencil_idx_0_0_1_VVV); // x821: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__1_]
        vreal x822 = ((x820 + x821)); // x822: Dependency! Liveness = 2; [x820, x821]
        vreal x786 = (-252 * stencil(evo_shiftU2, stencil_idx_0_0_0_VVV)); // x786: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__0_]
        store(evo_shift_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU2, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x786 + (-120 * x795) + (-10 * x789) + (45 * x792) + (210 * x798) + ((stencil(evo_shiftU2, stencil_idx_m5_0_0_VVV) + stencil(evo_shiftU2, stencil_idx_5_0_0_VVV))))) + (DYI * (x786 + (-120 * x807) + (-10 * x801) + (45 * x804) + (210 * x810) + ((stencil(evo_shiftU2, stencil_idx_0_m5_0_VVV) + stencil(evo_shiftU2, stencil_idx_0_5_0_VVV))))) + (DZI * (x786 + (-120 * x819) + (-10 * x813) + (45 * x816) + (210 * x822) + ((stencil(evo_shiftU2, stencil_idx_0_0_m5_VVV) + stencil(evo_shiftU2, stencil_idx_0_0_5_VVV)))))))));    
    });
    // apply_dissipation loop 7
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_m5_0_0_VVV(VVV_layout, p.I - 5*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_m5_0_VVV(VVV_layout, p.I - 5*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m5_VVV(VVV_layout, p.I - 5*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_5_VVV(VVV_layout, p.I + 5*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_0_5_0_VVV(VVV_layout, p.I + 5*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        const GF3D5index stencil_idx_5_0_0_VVV(VVV_layout, p.I + 5*p.DI[0]);
        vreal x832 = stencil(shift_BU0, stencil_idx_0_m4_0_VVV); // x832: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__m4__0_]
        vreal x833 = stencil(shift_BU0, stencil_idx_0_4_0_VVV); // x833: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__4__0_]
        vreal x834 = stencil(shift_BU0, stencil_idx_0_m3_0_VVV); // x834: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__m3__0_]
        vreal x835 = stencil(shift_BU0, stencil_idx_0_3_0_VVV); // x835: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__3__0_]
        vreal x836 = stencil(shift_BU0, stencil_idx_0_m2_0_VVV); // x836: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__m2__0_]
        vreal x837 = stencil(shift_BU0, stencil_idx_0_2_0_VVV); // x837: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__2__0_]
        vreal x838 = stencil(shift_BU0, stencil_idx_0_m1_0_VVV); // x838: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__m1__0_]
        vreal x839 = stencil(shift_BU0, stencil_idx_0_1_0_VVV); // x839: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__1__0_]
        vreal x840 = stencil(shift_BU0, stencil_idx_0_0_m4_VVV); // x840: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__m4_]
        vreal x841 = stencil(shift_BU0, stencil_idx_0_0_4_VVV); // x841: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__4_]
        vreal x842 = stencil(shift_BU0, stencil_idx_0_0_m3_VVV); // x842: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__m3_]
        vreal x843 = stencil(shift_BU0, stencil_idx_0_0_3_VVV); // x843: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__3_]
        vreal x844 = stencil(shift_BU0, stencil_idx_0_0_m2_VVV); // x844: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__m2_]
        vreal x845 = stencil(shift_BU0, stencil_idx_0_0_2_VVV); // x845: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__2_]
        vreal x846 = stencil(shift_BU0, stencil_idx_0_0_m1_VVV); // x846: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__m1_]
        vreal x847 = stencil(shift_BU0, stencil_idx_0_0_1_VVV); // x847: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__1_]
        vreal x824 = stencil(shift_BU0, stencil_idx_m4_0_0_VVV); // x824: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__m4__0__0_]
        vreal x825 = stencil(shift_BU0, stencil_idx_4_0_0_VVV); // x825: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__4__0__0_]
        vreal x826 = stencil(shift_BU0, stencil_idx_m3_0_0_VVV); // x826: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__m3__0__0_]
        vreal x827 = stencil(shift_BU0, stencil_idx_3_0_0_VVV); // x827: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__3__0__0_]
        vreal x828 = stencil(shift_BU0, stencil_idx_m2_0_0_VVV); // x828: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__m2__0__0_]
        vreal x829 = stencil(shift_BU0, stencil_idx_2_0_0_VVV); // x829: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__2__0__0_]
        vreal x830 = stencil(shift_BU0, stencil_idx_m1_0_0_VVV); // x830: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__m1__0__0_]
        vreal x831 = stencil(shift_BU0, stencil_idx_1_0_0_VVV); // x831: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__1__0__0_]
        vreal x823 = (-252 * stencil(shift_BU0, stencil_idx_0_0_0_VVV)); // x823: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__0_]
        vreal x322 = ((1.0 / 1024.0) * dissipation_epsilon); // x322: Dependency! Liveness = 1; [dissipation_epsilon]
        store(shift_B_rhsU0, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU0, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x823 + (-120 * ((x828 + x829))) + (-10 * ((x824 + x825))) + (45 * ((x826 + x827))) + (210 * ((x830 + x831))) + ((stencil(shift_BU0, stencil_idx_m5_0_0_VVV) + stencil(shift_BU0, stencil_idx_5_0_0_VVV))))) + (DYI * (x823 + (-120 * ((x836 + x837))) + (-10 * ((x832 + x833))) + (45 * ((x834 + x835))) + (210 * ((x838 + x839))) + ((stencil(shift_BU0, stencil_idx_0_m5_0_VVV) + stencil(shift_BU0, stencil_idx_0_5_0_VVV))))) + (DZI * (x823 + (-120 * ((x844 + x845))) + (-10 * ((x840 + x841))) + (45 * ((x842 + x843))) + (210 * ((x846 + x847))) + ((stencil(shift_BU0, stencil_idx_0_0_m5_VVV) + stencil(shift_BU0, stencil_idx_0_0_5_VVV)))))))));
        vreal x857 = stencil(shift_BU1, stencil_idx_0_m4_0_VVV); // x857: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__m4__0_]
        vreal x858 = stencil(shift_BU1, stencil_idx_0_4_0_VVV); // x858: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__4__0_]
        vreal x859 = stencil(shift_BU1, stencil_idx_0_m3_0_VVV); // x859: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__m3__0_]
        vreal x860 = stencil(shift_BU1, stencil_idx_0_3_0_VVV); // x860: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__3__0_]
        vreal x861 = stencil(shift_BU1, stencil_idx_0_m2_0_VVV); // x861: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__m2__0_]
        vreal x862 = stencil(shift_BU1, stencil_idx_0_2_0_VVV); // x862: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__2__0_]
        vreal x863 = stencil(shift_BU1, stencil_idx_0_m1_0_VVV); // x863: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__m1__0_]
        vreal x864 = stencil(shift_BU1, stencil_idx_0_1_0_VVV); // x864: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__1__0_]
        vreal x865 = stencil(shift_BU1, stencil_idx_0_0_m4_VVV); // x865: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__m4_]
        vreal x866 = stencil(shift_BU1, stencil_idx_0_0_4_VVV); // x866: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__4_]
        vreal x867 = stencil(shift_BU1, stencil_idx_0_0_m3_VVV); // x867: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__m3_]
        vreal x868 = stencil(shift_BU1, stencil_idx_0_0_3_VVV); // x868: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__3_]
        vreal x869 = stencil(shift_BU1, stencil_idx_0_0_m2_VVV); // x869: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__m2_]
        vreal x870 = stencil(shift_BU1, stencil_idx_0_0_2_VVV); // x870: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__2_]
        vreal x871 = stencil(shift_BU1, stencil_idx_0_0_m1_VVV); // x871: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__m1_]
        vreal x872 = stencil(shift_BU1, stencil_idx_0_0_1_VVV); // x872: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__1_]
        vreal x849 = stencil(shift_BU1, stencil_idx_m4_0_0_VVV); // x849: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__m4__0__0_]
        vreal x850 = stencil(shift_BU1, stencil_idx_4_0_0_VVV); // x850: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__4__0__0_]
        vreal x851 = stencil(shift_BU1, stencil_idx_m3_0_0_VVV); // x851: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__m3__0__0_]
        vreal x852 = stencil(shift_BU1, stencil_idx_3_0_0_VVV); // x852: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__3__0__0_]
        vreal x853 = stencil(shift_BU1, stencil_idx_m2_0_0_VVV); // x853: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__m2__0__0_]
        vreal x854 = stencil(shift_BU1, stencil_idx_2_0_0_VVV); // x854: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__2__0__0_]
        vreal x855 = stencil(shift_BU1, stencil_idx_m1_0_0_VVV); // x855: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__m1__0__0_]
        vreal x856 = stencil(shift_BU1, stencil_idx_1_0_0_VVV); // x856: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__1__0__0_]
        vreal x848 = (-252 * stencil(shift_BU1, stencil_idx_0_0_0_VVV)); // x848: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__0_]
        store(shift_B_rhsU1, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU1, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x848 + (-120 * ((x853 + x854))) + (-10 * ((x849 + x850))) + (45 * ((x851 + x852))) + (210 * ((x855 + x856))) + ((stencil(shift_BU1, stencil_idx_m5_0_0_VVV) + stencil(shift_BU1, stencil_idx_5_0_0_VVV))))) + (DYI * (x848 + (-120 * ((x861 + x862))) + (-10 * ((x857 + x858))) + (45 * ((x859 + x860))) + (210 * ((x863 + x864))) + ((stencil(shift_BU1, stencil_idx_0_m5_0_VVV) + stencil(shift_BU1, stencil_idx_0_5_0_VVV))))) + (DZI * (x848 + (-120 * ((x869 + x870))) + (-10 * ((x865 + x866))) + (45 * ((x867 + x868))) + (210 * ((x871 + x872))) + ((stencil(shift_BU1, stencil_idx_0_0_m5_VVV) + stencil(shift_BU1, stencil_idx_0_0_5_VVV)))))))));
        vreal x882 = stencil(shift_BU2, stencil_idx_0_m4_0_VVV); // x882: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__m4__0_]
        vreal x883 = stencil(shift_BU2, stencil_idx_0_4_0_VVV); // x883: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__4__0_]
        vreal x884 = stencil(shift_BU2, stencil_idx_0_m3_0_VVV); // x884: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__m3__0_]
        vreal x885 = stencil(shift_BU2, stencil_idx_0_3_0_VVV); // x885: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__3__0_]
        vreal x886 = stencil(shift_BU2, stencil_idx_0_m2_0_VVV); // x886: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__m2__0_]
        vreal x887 = stencil(shift_BU2, stencil_idx_0_2_0_VVV); // x887: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__2__0_]
        vreal x888 = stencil(shift_BU2, stencil_idx_0_m1_0_VVV); // x888: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__m1__0_]
        vreal x889 = stencil(shift_BU2, stencil_idx_0_1_0_VVV); // x889: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__1__0_]
        vreal x890 = stencil(shift_BU2, stencil_idx_0_0_m4_VVV); // x890: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__m4_]
        vreal x891 = stencil(shift_BU2, stencil_idx_0_0_4_VVV); // x891: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__4_]
        vreal x892 = stencil(shift_BU2, stencil_idx_0_0_m3_VVV); // x892: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__m3_]
        vreal x893 = stencil(shift_BU2, stencil_idx_0_0_3_VVV); // x893: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__3_]
        vreal x894 = stencil(shift_BU2, stencil_idx_0_0_m2_VVV); // x894: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__m2_]
        vreal x895 = stencil(shift_BU2, stencil_idx_0_0_2_VVV); // x895: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__2_]
        vreal x896 = stencil(shift_BU2, stencil_idx_0_0_m1_VVV); // x896: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__m1_]
        vreal x897 = stencil(shift_BU2, stencil_idx_0_0_1_VVV); // x897: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__1_]
        vreal x874 = stencil(shift_BU2, stencil_idx_m4_0_0_VVV); // x874: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__m4__0__0_]
        vreal x875 = stencil(shift_BU2, stencil_idx_4_0_0_VVV); // x875: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__4__0__0_]
        vreal x876 = stencil(shift_BU2, stencil_idx_m3_0_0_VVV); // x876: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__m3__0__0_]
        vreal x877 = stencil(shift_BU2, stencil_idx_3_0_0_VVV); // x877: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__3__0__0_]
        vreal x878 = stencil(shift_BU2, stencil_idx_m2_0_0_VVV); // x878: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__m2__0__0_]
        vreal x879 = stencil(shift_BU2, stencil_idx_2_0_0_VVV); // x879: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__2__0__0_]
        vreal x880 = stencil(shift_BU2, stencil_idx_m1_0_0_VVV); // x880: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__m1__0__0_]
        vreal x881 = stencil(shift_BU2, stencil_idx_1_0_0_VVV); // x881: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__1__0__0_]
        vreal x873 = (-252 * stencil(shift_BU2, stencil_idx_0_0_0_VVV)); // x873: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__0_]
        store(shift_B_rhsU2, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU2, stencil_idx_0_0_0_VVV) + (x322 * ((DXI * (x873 + (-120 * ((x878 + x879))) + (-10 * ((x874 + x875))) + (45 * ((x876 + x877))) + (210 * ((x880 + x881))) + ((stencil(shift_BU2, stencil_idx_m5_0_0_VVV) + stencil(shift_BU2, stencil_idx_5_0_0_VVV))))) + (DYI * (x873 + (-120 * ((x886 + x887))) + (-10 * ((x882 + x883))) + (45 * ((x884 + x885))) + (210 * ((x888 + x889))) + ((stencil(shift_BU2, stencil_idx_0_m5_0_VVV) + stencil(shift_BU2, stencil_idx_0_5_0_VVV))))) + (DZI * (x873 + (-120 * ((x894 + x895))) + (-10 * ((x890 + x891))) + (45 * ((x892 + x893))) + (210 * ((x896 + x897))) + ((stencil(shift_BU2, stencil_idx_0_0_m5_VVV) + stencil(shift_BU2, stencil_idx_0_0_5_VVV)))))))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}