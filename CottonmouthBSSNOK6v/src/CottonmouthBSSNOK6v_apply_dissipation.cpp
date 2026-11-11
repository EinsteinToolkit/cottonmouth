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
    CCTK_ASSERT((cctk_nghostzones[0] >= 4));
    CCTK_ASSERT((cctk_nghostzones[1] >= 4));
    CCTK_ASSERT((cctk_nghostzones[2] >= 4));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // apply_dissipation loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        vreal x196 = stencil(gtDD00, stencil_idx_3_0_0_VVV); // x196: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__3__0__0_]
        vreal x197 = stencil(gtDD00, stencil_idx_m3_0_0_VVV); // x197: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m3__0__0_]
        vreal x274 = ((x196 + x197)); // x274: Dependency! Liveness = 2; [x196, x197]
        vreal x198 = stencil(gtDD00, stencil_idx_m2_0_0_VVV); // x198: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m2__0__0_]
        vreal x199 = stencil(gtDD00, stencil_idx_2_0_0_VVV); // x199: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__2__0__0_]
        vreal x275 = ((x198 + x199)); // x275: Dependency! Liveness = 2; [x198, x199]
        vreal x200 = stencil(gtDD00, stencil_idx_m1_0_0_VVV); // x200: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m1__0__0_]
        vreal x201 = stencil(gtDD00, stencil_idx_1_0_0_VVV); // x201: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__1__0__0_]
        vreal x276 = ((x200 + x201)); // x276: Dependency! Liveness = 2; [x200, x201]
        vreal x118 = stencil(gtDD00, stencil_idx_0_3_0_VVV); // x118: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__3__0_]
        vreal x119 = stencil(gtDD00, stencil_idx_0_m3_0_VVV); // x119: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m3__0_]
        vreal x277 = ((x118 + x119)); // x277: Dependency! Liveness = 2; [x118, x119]
        vreal x120 = stencil(gtDD00, stencil_idx_0_m2_0_VVV); // x120: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m2__0_]
        vreal x121 = stencil(gtDD00, stencil_idx_0_2_0_VVV); // x121: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__2__0_]
        vreal x278 = ((x120 + x121)); // x278: Dependency! Liveness = 2; [x120, x121]
        vreal x122 = stencil(gtDD00, stencil_idx_0_m1_0_VVV); // x122: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m1__0_]
        vreal x123 = stencil(gtDD00, stencil_idx_0_1_0_VVV); // x123: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__1__0_]
        vreal x279 = ((x122 + x123)); // x279: Dependency! Liveness = 2; [x122, x123]
        vreal x105 = stencil(gtDD00, stencil_idx_0_0_3_VVV); // x105: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__3_]
        vreal x106 = stencil(gtDD00, stencil_idx_0_0_m3_VVV); // x106: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m3_]
        vreal x280 = ((x105 + x106)); // x280: Dependency! Liveness = 2; [x105, x106]
        vreal x107 = stencil(gtDD00, stencil_idx_0_0_m2_VVV); // x107: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m2_]
        vreal x108 = stencil(gtDD00, stencil_idx_0_0_2_VVV); // x108: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__2_]
        vreal x281 = ((x107 + x108)); // x281: Dependency! Liveness = 2; [x107, x108]
        vreal x109 = stencil(gtDD00, stencil_idx_0_0_m1_VVV); // x109: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m1_]
        vreal x110 = stencil(gtDD00, stencil_idx_0_0_1_VVV); // x110: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__1_]
        vreal x282 = ((x109 + x110)); // x282: Dependency! Liveness = 2; [x109, x110]
        vreal x273 = (70 * stencil(gtDD00, stencil_idx_0_0_0_VVV)); // x273: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__0_]
        vreal x283 = ((1.0 / 256.0) * dissipation_epsilon); // x283: Dependency! Liveness = 1; [dissipation_epsilon]
        store(gt_rhsDD00, stencil_idx_0_0_0_VVV, (access(gt_rhsDD00, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x273 + (-56 * x276) + (-8 * x274) + (28 * x275) + ((stencil(gtDD00, stencil_idx_m4_0_0_VVV) + stencil(gtDD00, stencil_idx_4_0_0_VVV))))) + (DYI * (x273 + (-56 * x279) + (-8 * x277) + (28 * x278) + ((stencil(gtDD00, stencil_idx_0_m4_0_VVV) + stencil(gtDD00, stencil_idx_0_4_0_VVV))))) + (DZI * (x273 + (-56 * x282) + (-8 * x280) + (28 * x281) + ((stencil(gtDD00, stencil_idx_0_0_m4_VVV) + stencil(gtDD00, stencil_idx_0_0_4_VVV)))))))));
        vreal x213 = stencil(gtDD01, stencil_idx_3_0_0_VVV); // x213: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__3__0__0_]
        vreal x214 = stencil(gtDD01, stencil_idx_m3_0_0_VVV); // x214: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m3__0__0_]
        vreal x285 = ((x213 + x214)); // x285: Dependency! Liveness = 2; [x213, x214]
        vreal x215 = stencil(gtDD01, stencil_idx_m2_0_0_VVV); // x215: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m2__0__0_]
        vreal x216 = stencil(gtDD01, stencil_idx_2_0_0_VVV); // x216: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__2__0__0_]
        vreal x286 = ((x215 + x216)); // x286: Dependency! Liveness = 2; [x215, x216]
        vreal x217 = stencil(gtDD01, stencil_idx_m1_0_0_VVV); // x217: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m1__0__0_]
        vreal x218 = stencil(gtDD01, stencil_idx_1_0_0_VVV); // x218: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__1__0__0_]
        vreal x287 = ((x217 + x218)); // x287: Dependency! Liveness = 2; [x217, x218]
        vreal x174 = stencil(gtDD01, stencil_idx_0_3_0_VVV); // x174: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__3__0_]
        vreal x175 = stencil(gtDD01, stencil_idx_0_m3_0_VVV); // x175: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m3__0_]
        vreal x288 = ((x174 + x175)); // x288: Dependency! Liveness = 2; [x174, x175]
        vreal x176 = stencil(gtDD01, stencil_idx_0_m2_0_VVV); // x176: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m2__0_]
        vreal x177 = stencil(gtDD01, stencil_idx_0_2_0_VVV); // x177: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__2__0_]
        vreal x289 = ((x176 + x177)); // x289: Dependency! Liveness = 2; [x176, x177]
        vreal x178 = stencil(gtDD01, stencil_idx_0_m1_0_VVV); // x178: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m1__0_]
        vreal x179 = stencil(gtDD01, stencil_idx_0_1_0_VVV); // x179: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__1__0_]
        vreal x290 = ((x178 + x179)); // x290: Dependency! Liveness = 2; [x178, x179]
        vreal x80 = stencil(gtDD01, stencil_idx_0_0_3_VVV); // x80: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__3_]
        vreal x81 = stencil(gtDD01, stencil_idx_0_0_m3_VVV); // x81: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m3_]
        vreal x291 = ((x80 + x81)); // x291: Dependency! Liveness = 2; [x80, x81]
        vreal x82 = stencil(gtDD01, stencil_idx_0_0_m2_VVV); // x82: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m2_]
        vreal x83 = stencil(gtDD01, stencil_idx_0_0_2_VVV); // x83: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__2_]
        vreal x292 = ((x82 + x83)); // x292: Dependency! Liveness = 2; [x82, x83]
        vreal x84 = stencil(gtDD01, stencil_idx_0_0_m1_VVV); // x84: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m1_]
        vreal x85 = stencil(gtDD01, stencil_idx_0_0_1_VVV); // x85: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__1_]
        vreal x293 = ((x84 + x85)); // x293: Dependency! Liveness = 2; [x84, x85]
        vreal x284 = (70 * stencil(gtDD01, stencil_idx_0_0_0_VVV)); // x284: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__0_]
        store(gt_rhsDD01, stencil_idx_0_0_0_VVV, (access(gt_rhsDD01, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x284 + (-56 * x287) + (-8 * x285) + (28 * x286) + ((stencil(gtDD01, stencil_idx_m4_0_0_VVV) + stencil(gtDD01, stencil_idx_4_0_0_VVV))))) + (DYI * (x284 + (-56 * x290) + (-8 * x288) + (28 * x289) + ((stencil(gtDD01, stencil_idx_0_m4_0_VVV) + stencil(gtDD01, stencil_idx_0_4_0_VVV))))) + (DZI * (x284 + (-56 * x293) + (-8 * x291) + (28 * x292) + ((stencil(gtDD01, stencil_idx_0_0_m4_VVV) + stencil(gtDD01, stencil_idx_0_0_4_VVV)))))))));
        vreal x204 = stencil(gtDD02, stencil_idx_3_0_0_VVV); // x204: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__3__0__0_]
        vreal x205 = stencil(gtDD02, stencil_idx_m3_0_0_VVV); // x205: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m3__0__0_]
        vreal x295 = ((x204 + x205)); // x295: Dependency! Liveness = 2; [x204, x205]
        vreal x206 = stencil(gtDD02, stencil_idx_m2_0_0_VVV); // x206: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m2__0__0_]
        vreal x207 = stencil(gtDD02, stencil_idx_2_0_0_VVV); // x207: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__2__0__0_]
        vreal x296 = ((x206 + x207)); // x296: Dependency! Liveness = 2; [x206, x207]
        vreal x208 = stencil(gtDD02, stencil_idx_m1_0_0_VVV); // x208: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m1__0__0_]
        vreal x209 = stencil(gtDD02, stencil_idx_1_0_0_VVV); // x209: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__1__0__0_]
        vreal x297 = ((x208 + x209)); // x297: Dependency! Liveness = 2; [x208, x209]
        vreal x73 = stencil(gtDD02, stencil_idx_0_3_0_VVV); // x73: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__3__0_]
        vreal x74 = stencil(gtDD02, stencil_idx_0_m3_0_VVV); // x74: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m3__0_]
        vreal x298 = ((x73 + x74)); // x298: Dependency! Liveness = 2; [x73, x74]
        vreal x75 = stencil(gtDD02, stencil_idx_0_m2_0_VVV); // x75: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m2__0_]
        vreal x76 = stencil(gtDD02, stencil_idx_0_2_0_VVV); // x76: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__2__0_]
        vreal x299 = ((x75 + x76)); // x299: Dependency! Liveness = 2; [x75, x76]
        vreal x77 = stencil(gtDD02, stencil_idx_0_m1_0_VVV); // x77: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m1__0_]
        vreal x78 = stencil(gtDD02, stencil_idx_0_1_0_VVV); // x78: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__1__0_]
        vreal x300 = ((x77 + x78)); // x300: Dependency! Liveness = 2; [x77, x78]
        vreal x156 = stencil(gtDD02, stencil_idx_0_0_3_VVV); // x156: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__3_]
        vreal x157 = stencil(gtDD02, stencil_idx_0_0_m3_VVV); // x157: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m3_]
        vreal x301 = ((x156 + x157)); // x301: Dependency! Liveness = 2; [x156, x157]
        vreal x158 = stencil(gtDD02, stencil_idx_0_0_m2_VVV); // x158: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m2_]
        vreal x159 = stencil(gtDD02, stencil_idx_0_0_2_VVV); // x159: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__2_]
        vreal x302 = ((x158 + x159)); // x302: Dependency! Liveness = 2; [x158, x159]
        vreal x160 = stencil(gtDD02, stencil_idx_0_0_m1_VVV); // x160: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m1_]
        vreal x161 = stencil(gtDD02, stencil_idx_0_0_1_VVV); // x161: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__1_]
        vreal x303 = ((x160 + x161)); // x303: Dependency! Liveness = 2; [x160, x161]
        vreal x294 = (70 * stencil(gtDD02, stencil_idx_0_0_0_VVV)); // x294: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__0_]
        store(gt_rhsDD02, stencil_idx_0_0_0_VVV, (access(gt_rhsDD02, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x294 + (-56 * x297) + (-8 * x295) + (28 * x296) + ((stencil(gtDD02, stencil_idx_m4_0_0_VVV) + stencil(gtDD02, stencil_idx_4_0_0_VVV))))) + (DYI * (x294 + (-56 * x300) + (-8 * x298) + (28 * x299) + ((stencil(gtDD02, stencil_idx_0_m4_0_VVV) + stencil(gtDD02, stencil_idx_0_4_0_VVV))))) + (DZI * (x294 + (-56 * x303) + (-8 * x301) + (28 * x302) + ((stencil(gtDD02, stencil_idx_0_0_m4_VVV) + stencil(gtDD02, stencil_idx_0_0_4_VVV)))))))));
        vreal x126 = stencil(gtDD11, stencil_idx_3_0_0_VVV); // x126: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__3__0__0_]
        vreal x127 = stencil(gtDD11, stencil_idx_m3_0_0_VVV); // x127: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m3__0__0_]
        vreal x305 = ((x126 + x127)); // x305: Dependency! Liveness = 2; [x126, x127]
        vreal x128 = stencil(gtDD11, stencil_idx_m2_0_0_VVV); // x128: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m2__0__0_]
        vreal x129 = stencil(gtDD11, stencil_idx_2_0_0_VVV); // x129: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__2__0__0_]
        vreal x306 = ((x128 + x129)); // x306: Dependency! Liveness = 2; [x128, x129]
        vreal x130 = stencil(gtDD11, stencil_idx_m1_0_0_VVV); // x130: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m1__0__0_]
        vreal x131 = stencil(gtDD11, stencil_idx_1_0_0_VVV); // x131: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__1__0__0_]
        vreal x307 = ((x130 + x131)); // x307: Dependency! Liveness = 2; [x130, x131]
        vreal x166 = stencil(gtDD11, stencil_idx_0_3_0_VVV); // x166: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__3__0_]
        vreal x167 = stencil(gtDD11, stencil_idx_0_m3_0_VVV); // x167: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m3__0_]
        vreal x308 = ((x166 + x167)); // x308: Dependency! Liveness = 2; [x166, x167]
        vreal x168 = stencil(gtDD11, stencil_idx_0_m2_0_VVV); // x168: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m2__0_]
        vreal x169 = stencil(gtDD11, stencil_idx_0_2_0_VVV); // x169: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__2__0_]
        vreal x309 = ((x168 + x169)); // x309: Dependency! Liveness = 2; [x168, x169]
        vreal x170 = stencil(gtDD11, stencil_idx_0_m1_0_VVV); // x170: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m1__0_]
        vreal x171 = stencil(gtDD11, stencil_idx_0_1_0_VVV); // x171: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__1__0_]
        vreal x310 = ((x170 + x171)); // x310: Dependency! Liveness = 2; [x170, x171]
        vreal x64 = stencil(gtDD11, stencil_idx_0_0_3_VVV); // x64: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__3_]
        vreal x65 = stencil(gtDD11, stencil_idx_0_0_m3_VVV); // x65: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m3_]
        vreal x311 = ((x64 + x65)); // x311: Dependency! Liveness = 2; [x64, x65]
        vreal x66 = stencil(gtDD11, stencil_idx_0_0_m2_VVV); // x66: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m2_]
        vreal x67 = stencil(gtDD11, stencil_idx_0_0_2_VVV); // x67: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__2_]
        vreal x312 = ((x66 + x67)); // x312: Dependency! Liveness = 2; [x66, x67]
        vreal x68 = stencil(gtDD11, stencil_idx_0_0_m1_VVV); // x68: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m1_]
        vreal x69 = stencil(gtDD11, stencil_idx_0_0_1_VVV); // x69: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__1_]
        vreal x313 = ((x68 + x69)); // x313: Dependency! Liveness = 2; [x68, x69]
        vreal x304 = (70 * stencil(gtDD11, stencil_idx_0_0_0_VVV)); // x304: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__0_]
        store(gt_rhsDD11, stencil_idx_0_0_0_VVV, (access(gt_rhsDD11, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x304 + (-56 * x307) + (-8 * x305) + (28 * x306) + ((stencil(gtDD11, stencil_idx_m4_0_0_VVV) + stencil(gtDD11, stencil_idx_4_0_0_VVV))))) + (DYI * (x304 + (-56 * x310) + (-8 * x308) + (28 * x309) + ((stencil(gtDD11, stencil_idx_0_m4_0_VVV) + stencil(gtDD11, stencil_idx_0_4_0_VVV))))) + (DZI * (x304 + (-56 * x313) + (-8 * x311) + (28 * x312) + ((stencil(gtDD11, stencil_idx_0_0_m4_VVV) + stencil(gtDD11, stencil_idx_0_0_4_VVV)))))))));
        vreal x87 = stencil(gtDD12, stencil_idx_3_0_0_VVV); // x87: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__3__0__0_]
        vreal x88 = stencil(gtDD12, stencil_idx_m3_0_0_VVV); // x88: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m3__0__0_]
        vreal x315 = ((x87 + x88)); // x315: Dependency! Liveness = 2; [x87, x88]
        vreal x89 = stencil(gtDD12, stencil_idx_m2_0_0_VVV); // x89: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m2__0__0_]
        vreal x90 = stencil(gtDD12, stencil_idx_2_0_0_VVV); // x90: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__2__0__0_]
        vreal x316 = ((x89 + x90)); // x316: Dependency! Liveness = 2; [x89, x90]
        vreal x91 = stencil(gtDD12, stencil_idx_m1_0_0_VVV); // x91: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m1__0__0_]
        vreal x92 = stencil(gtDD12, stencil_idx_1_0_0_VVV); // x92: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__1__0__0_]
        vreal x317 = ((x91 + x92)); // x317: Dependency! Liveness = 2; [x91, x92]
        vreal x184 = stencil(gtDD12, stencil_idx_0_3_0_VVV); // x184: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__3__0_]
        vreal x185 = stencil(gtDD12, stencil_idx_0_m3_0_VVV); // x185: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m3__0_]
        vreal x318 = ((x184 + x185)); // x318: Dependency! Liveness = 2; [x184, x185]
        vreal x186 = stencil(gtDD12, stencil_idx_0_m2_0_VVV); // x186: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m2__0_]
        vreal x187 = stencil(gtDD12, stencil_idx_0_2_0_VVV); // x187: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__2__0_]
        vreal x319 = ((x186 + x187)); // x319: Dependency! Liveness = 2; [x186, x187]
        vreal x188 = stencil(gtDD12, stencil_idx_0_m1_0_VVV); // x188: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m1__0_]
        vreal x189 = stencil(gtDD12, stencil_idx_0_1_0_VVV); // x189: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__1__0_]
        vreal x320 = ((x188 + x189)); // x320: Dependency! Liveness = 2; [x188, x189]
        vreal x147 = stencil(gtDD12, stencil_idx_0_0_3_VVV); // x147: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__3_]
        vreal x148 = stencil(gtDD12, stencil_idx_0_0_m3_VVV); // x148: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m3_]
        vreal x321 = ((x147 + x148)); // x321: Dependency! Liveness = 2; [x147, x148]
        vreal x149 = stencil(gtDD12, stencil_idx_0_0_m2_VVV); // x149: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m2_]
        vreal x150 = stencil(gtDD12, stencil_idx_0_0_2_VVV); // x150: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__2_]
        vreal x322 = ((x149 + x150)); // x322: Dependency! Liveness = 2; [x149, x150]
        vreal x151 = stencil(gtDD12, stencil_idx_0_0_m1_VVV); // x151: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m1_]
        vreal x152 = stencil(gtDD12, stencil_idx_0_0_1_VVV); // x152: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__1_]
        vreal x323 = ((x151 + x152)); // x323: Dependency! Liveness = 2; [x151, x152]
        vreal x314 = (70 * stencil(gtDD12, stencil_idx_0_0_0_VVV)); // x314: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__0_]
        store(gt_rhsDD12, stencil_idx_0_0_0_VVV, (access(gt_rhsDD12, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x314 + (-56 * x317) + (-8 * x315) + (28 * x316) + ((stencil(gtDD12, stencil_idx_m4_0_0_VVV) + stencil(gtDD12, stencil_idx_4_0_0_VVV))))) + (DYI * (x314 + (-56 * x320) + (-8 * x318) + (28 * x319) + ((stencil(gtDD12, stencil_idx_0_m4_0_VVV) + stencil(gtDD12, stencil_idx_0_4_0_VVV))))) + (DZI * (x314 + (-56 * x323) + (-8 * x321) + (28 * x322) + ((stencil(gtDD12, stencil_idx_0_0_m4_VVV) + stencil(gtDD12, stencil_idx_0_0_4_VVV)))))))));
        vreal x97 = stencil(gtDD22, stencil_idx_3_0_0_VVV); // x97: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__3__0__0_]
        vreal x98 = stencil(gtDD22, stencil_idx_m3_0_0_VVV); // x98: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m3__0__0_]
        vreal x325 = ((x97 + x98)); // x325: Dependency! Liveness = 2; [x97, x98]
        vreal x100 = stencil(gtDD22, stencil_idx_2_0_0_VVV); // x100: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__2__0__0_]
        vreal x99 = stencil(gtDD22, stencil_idx_m2_0_0_VVV); // x99: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m2__0__0_]
        vreal x326 = ((x100 + x99)); // x326: Dependency! Liveness = 2; [x100, x99]
        vreal x101 = stencil(gtDD22, stencil_idx_m1_0_0_VVV); // x101: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m1__0__0_]
        vreal x102 = stencil(gtDD22, stencil_idx_1_0_0_VVV); // x102: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__1__0__0_]
        vreal x327 = ((x101 + x102)); // x327: Dependency! Liveness = 2; [x101, x102]
        vreal x53 = stencil(gtDD22, stencil_idx_0_3_0_VVV); // x53: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__3__0_]
        vreal x54 = stencil(gtDD22, stencil_idx_0_m3_0_VVV); // x54: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m3__0_]
        vreal x328 = ((x53 + x54)); // x328: Dependency! Liveness = 2; [x53, x54]
        vreal x55 = stencil(gtDD22, stencil_idx_0_m2_0_VVV); // x55: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m2__0_]
        vreal x56 = stencil(gtDD22, stencil_idx_0_2_0_VVV); // x56: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__2__0_]
        vreal x329 = ((x55 + x56)); // x329: Dependency! Liveness = 2; [x55, x56]
        vreal x57 = stencil(gtDD22, stencil_idx_0_m1_0_VVV); // x57: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m1__0_]
        vreal x58 = stencil(gtDD22, stencil_idx_0_1_0_VVV); // x58: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__1__0_]
        vreal x330 = ((x57 + x58)); // x330: Dependency! Liveness = 2; [x57, x58]
        vreal x139 = stencil(gtDD22, stencil_idx_0_0_3_VVV); // x139: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__3_]
        vreal x140 = stencil(gtDD22, stencil_idx_0_0_m3_VVV); // x140: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m3_]
        vreal x331 = ((x139 + x140)); // x331: Dependency! Liveness = 2; [x139, x140]
        vreal x141 = stencil(gtDD22, stencil_idx_0_0_m2_VVV); // x141: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m2_]
        vreal x142 = stencil(gtDD22, stencil_idx_0_0_2_VVV); // x142: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__2_]
        vreal x332 = ((x141 + x142)); // x332: Dependency! Liveness = 2; [x141, x142]
        vreal x143 = stencil(gtDD22, stencil_idx_0_0_m1_VVV); // x143: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m1_]
        vreal x144 = stencil(gtDD22, stencil_idx_0_0_1_VVV); // x144: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__1_]
        vreal x333 = ((x143 + x144)); // x333: Dependency! Liveness = 2; [x143, x144]
        vreal x324 = (70 * stencil(gtDD22, stencil_idx_0_0_0_VVV)); // x324: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__0_]
        store(gt_rhsDD22, stencil_idx_0_0_0_VVV, (access(gt_rhsDD22, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x324 + (-56 * x327) + (-8 * x325) + (28 * x326) + ((stencil(gtDD22, stencil_idx_m4_0_0_VVV) + stencil(gtDD22, stencil_idx_4_0_0_VVV))))) + (DYI * (x324 + (-56 * x330) + (-8 * x328) + (28 * x329) + ((stencil(gtDD22, stencil_idx_0_m4_0_VVV) + stencil(gtDD22, stencil_idx_0_4_0_VVV))))) + (DZI * (x324 + (-56 * x333) + (-8 * x331) + (28 * x332) + ((stencil(gtDD22, stencil_idx_0_0_m4_VVV) + stencil(gtDD22, stencil_idx_0_0_4_VVV)))))))));    
    });
    // apply_dissipation loop 1
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        vreal x335 = stencil(w, stencil_idx_m3_0_0_VVV); // x335: Dependency! Liveness = 1; [__dummy_stencil__w__m3__0__0_]
        vreal x336 = stencil(w, stencil_idx_3_0_0_VVV); // x336: Dependency! Liveness = 1; [__dummy_stencil__w__3__0__0_]
        vreal x337 = ((x335 + x336)); // x337: Dependency! Liveness = 2; [x335, x336]
        vreal x338 = stencil(w, stencil_idx_m2_0_0_VVV); // x338: Dependency! Liveness = 1; [__dummy_stencil__w__m2__0__0_]
        vreal x339 = stencil(w, stencil_idx_2_0_0_VVV); // x339: Dependency! Liveness = 1; [__dummy_stencil__w__2__0__0_]
        vreal x340 = ((x338 + x339)); // x340: Dependency! Liveness = 2; [x338, x339]
        vreal x341 = stencil(w, stencil_idx_m1_0_0_VVV); // x341: Dependency! Liveness = 1; [__dummy_stencil__w__m1__0__0_]
        vreal x342 = stencil(w, stencil_idx_1_0_0_VVV); // x342: Dependency! Liveness = 1; [__dummy_stencil__w__1__0__0_]
        vreal x343 = ((x341 + x342)); // x343: Dependency! Liveness = 2; [x341, x342]
        vreal x344 = stencil(w, stencil_idx_0_m3_0_VVV); // x344: Dependency! Liveness = 1; [__dummy_stencil__w__0__m3__0_]
        vreal x345 = stencil(w, stencil_idx_0_3_0_VVV); // x345: Dependency! Liveness = 1; [__dummy_stencil__w__0__3__0_]
        vreal x346 = ((x344 + x345)); // x346: Dependency! Liveness = 2; [x344, x345]
        vreal x347 = stencil(w, stencil_idx_0_m2_0_VVV); // x347: Dependency! Liveness = 1; [__dummy_stencil__w__0__m2__0_]
        vreal x348 = stencil(w, stencil_idx_0_2_0_VVV); // x348: Dependency! Liveness = 1; [__dummy_stencil__w__0__2__0_]
        vreal x349 = ((x347 + x348)); // x349: Dependency! Liveness = 2; [x347, x348]
        vreal x350 = stencil(w, stencil_idx_0_m1_0_VVV); // x350: Dependency! Liveness = 1; [__dummy_stencil__w__0__m1__0_]
        vreal x351 = stencil(w, stencil_idx_0_1_0_VVV); // x351: Dependency! Liveness = 1; [__dummy_stencil__w__0__1__0_]
        vreal x352 = ((x350 + x351)); // x352: Dependency! Liveness = 2; [x350, x351]
        vreal x353 = stencil(w, stencil_idx_0_0_m3_VVV); // x353: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__m3_]
        vreal x354 = stencil(w, stencil_idx_0_0_3_VVV); // x354: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__3_]
        vreal x355 = ((x353 + x354)); // x355: Dependency! Liveness = 2; [x353, x354]
        vreal x356 = stencil(w, stencil_idx_0_0_m2_VVV); // x356: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__m2_]
        vreal x357 = stencil(w, stencil_idx_0_0_2_VVV); // x357: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__2_]
        vreal x358 = ((x356 + x357)); // x358: Dependency! Liveness = 2; [x356, x357]
        vreal x359 = stencil(w, stencil_idx_0_0_m1_VVV); // x359: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__m1_]
        vreal x360 = stencil(w, stencil_idx_0_0_1_VVV); // x360: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__1_]
        vreal x361 = ((x359 + x360)); // x361: Dependency! Liveness = 2; [x359, x360]
        vreal x334 = (70 * stencil(w, stencil_idx_0_0_0_VVV)); // x334: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__0_]
        vreal x283 = ((1.0 / 256.0) * dissipation_epsilon); // x283: Dependency! Liveness = 1; [dissipation_epsilon]
        store(w_rhs, stencil_idx_0_0_0_VVV, (access(w_rhs, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x334 + (-56 * x343) + (-8 * x337) + (28 * x340) + ((stencil(w, stencil_idx_m4_0_0_VVV) + stencil(w, stencil_idx_4_0_0_VVV))))) + (DYI * (x334 + (-56 * x352) + (-8 * x346) + (28 * x349) + ((stencil(w, stencil_idx_0_m4_0_VVV) + stencil(w, stencil_idx_0_4_0_VVV))))) + (DZI * (x334 + (-56 * x361) + (-8 * x355) + (28 * x358) + ((stencil(w, stencil_idx_0_0_m4_VVV) + stencil(w, stencil_idx_0_0_4_VVV)))))))));    
    });
    // apply_dissipation loop 2
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        vreal x369 = stencil(trK, stencil_idx_0_m3_0_VVV); // x369: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m3__0_]
        vreal x370 = stencil(trK, stencil_idx_0_3_0_VVV); // x370: Dependency! Liveness = 1; [__dummy_stencil__trK__0__3__0_]
        vreal x371 = stencil(trK, stencil_idx_0_m2_0_VVV); // x371: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m2__0_]
        vreal x372 = stencil(trK, stencil_idx_0_2_0_VVV); // x372: Dependency! Liveness = 1; [__dummy_stencil__trK__0__2__0_]
        vreal x373 = stencil(trK, stencil_idx_0_m1_0_VVV); // x373: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m1__0_]
        vreal x374 = stencil(trK, stencil_idx_0_1_0_VVV); // x374: Dependency! Liveness = 1; [__dummy_stencil__trK__0__1__0_]
        vreal x375 = stencil(trK, stencil_idx_0_0_m3_VVV); // x375: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m3_]
        vreal x376 = stencil(trK, stencil_idx_0_0_3_VVV); // x376: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__3_]
        vreal x377 = stencil(trK, stencil_idx_0_0_m2_VVV); // x377: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m2_]
        vreal x378 = stencil(trK, stencil_idx_0_0_2_VVV); // x378: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__2_]
        vreal x379 = stencil(trK, stencil_idx_0_0_m1_VVV); // x379: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m1_]
        vreal x380 = stencil(trK, stencil_idx_0_0_1_VVV); // x380: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__1_]
        vreal x363 = stencil(trK, stencil_idx_m3_0_0_VVV); // x363: Dependency! Liveness = 1; [__dummy_stencil__trK__m3__0__0_]
        vreal x364 = stencil(trK, stencil_idx_3_0_0_VVV); // x364: Dependency! Liveness = 1; [__dummy_stencil__trK__3__0__0_]
        vreal x365 = stencil(trK, stencil_idx_m2_0_0_VVV); // x365: Dependency! Liveness = 1; [__dummy_stencil__trK__m2__0__0_]
        vreal x366 = stencil(trK, stencil_idx_2_0_0_VVV); // x366: Dependency! Liveness = 1; [__dummy_stencil__trK__2__0__0_]
        vreal x367 = stencil(trK, stencil_idx_m1_0_0_VVV); // x367: Dependency! Liveness = 1; [__dummy_stencil__trK__m1__0__0_]
        vreal x368 = stencil(trK, stencil_idx_1_0_0_VVV); // x368: Dependency! Liveness = 1; [__dummy_stencil__trK__1__0__0_]
        vreal x362 = (70 * stencil(trK, stencil_idx_0_0_0_VVV)); // x362: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__0_]
        vreal x283 = ((1.0 / 256.0) * dissipation_epsilon); // x283: Dependency! Liveness = 1; [dissipation_epsilon]
        store(trK_rhs, stencil_idx_0_0_0_VVV, (access(trK_rhs, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x362 + (-56 * ((x367 + x368))) + (-8 * ((x363 + x364))) + (28 * ((x365 + x366))) + ((stencil(trK, stencil_idx_m4_0_0_VVV) + stencil(trK, stencil_idx_4_0_0_VVV))))) + (DYI * (x362 + (-56 * ((x373 + x374))) + (-8 * ((x369 + x370))) + (28 * ((x371 + x372))) + ((stencil(trK, stencil_idx_0_m4_0_VVV) + stencil(trK, stencil_idx_0_4_0_VVV))))) + (DZI * (x362 + (-56 * ((x379 + x380))) + (-8 * ((x375 + x376))) + (28 * ((x377 + x378))) + ((stencil(trK, stencil_idx_0_0_m4_VVV) + stencil(trK, stencil_idx_0_0_4_VVV)))))))));    
    });
    // apply_dissipation loop 3
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        vreal x388 = stencil(AtDD00, stencil_idx_0_m3_0_VVV); // x388: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m3__0_]
        vreal x389 = stencil(AtDD00, stencil_idx_0_3_0_VVV); // x389: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__3__0_]
        vreal x390 = stencil(AtDD00, stencil_idx_0_m2_0_VVV); // x390: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m2__0_]
        vreal x391 = stencil(AtDD00, stencil_idx_0_2_0_VVV); // x391: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__2__0_]
        vreal x392 = stencil(AtDD00, stencil_idx_0_m1_0_VVV); // x392: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m1__0_]
        vreal x393 = stencil(AtDD00, stencil_idx_0_1_0_VVV); // x393: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__1__0_]
        vreal x394 = stencil(AtDD00, stencil_idx_0_0_m3_VVV); // x394: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m3_]
        vreal x395 = stencil(AtDD00, stencil_idx_0_0_3_VVV); // x395: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__3_]
        vreal x396 = stencil(AtDD00, stencil_idx_0_0_m2_VVV); // x396: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m2_]
        vreal x397 = stencil(AtDD00, stencil_idx_0_0_2_VVV); // x397: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__2_]
        vreal x398 = stencil(AtDD00, stencil_idx_0_0_m1_VVV); // x398: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m1_]
        vreal x399 = stencil(AtDD00, stencil_idx_0_0_1_VVV); // x399: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__1_]
        vreal x382 = stencil(AtDD00, stencil_idx_m3_0_0_VVV); // x382: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m3__0__0_]
        vreal x383 = stencil(AtDD00, stencil_idx_3_0_0_VVV); // x383: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__3__0__0_]
        vreal x384 = stencil(AtDD00, stencil_idx_m2_0_0_VVV); // x384: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m2__0__0_]
        vreal x385 = stencil(AtDD00, stencil_idx_2_0_0_VVV); // x385: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__2__0__0_]
        vreal x386 = stencil(AtDD00, stencil_idx_m1_0_0_VVV); // x386: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m1__0__0_]
        vreal x387 = stencil(AtDD00, stencil_idx_1_0_0_VVV); // x387: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__1__0__0_]
        vreal x381 = (70 * stencil(AtDD00, stencil_idx_0_0_0_VVV)); // x381: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__0_]
        vreal x283 = ((1.0 / 256.0) * dissipation_epsilon); // x283: Dependency! Liveness = 1; [dissipation_epsilon]
        store(At_rhsDD00, stencil_idx_0_0_0_VVV, (access(At_rhsDD00, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x381 + (-56 * ((x386 + x387))) + (-8 * ((x382 + x383))) + (28 * ((x384 + x385))) + ((stencil(AtDD00, stencil_idx_m4_0_0_VVV) + stencil(AtDD00, stencil_idx_4_0_0_VVV))))) + (DYI * (x381 + (-56 * ((x392 + x393))) + (-8 * ((x388 + x389))) + (28 * ((x390 + x391))) + ((stencil(AtDD00, stencil_idx_0_m4_0_VVV) + stencil(AtDD00, stencil_idx_0_4_0_VVV))))) + (DZI * (x381 + (-56 * ((x398 + x399))) + (-8 * ((x394 + x395))) + (28 * ((x396 + x397))) + ((stencil(AtDD00, stencil_idx_0_0_m4_VVV) + stencil(AtDD00, stencil_idx_0_0_4_VVV)))))))));
        vreal x407 = stencil(AtDD01, stencil_idx_0_m3_0_VVV); // x407: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m3__0_]
        vreal x408 = stencil(AtDD01, stencil_idx_0_3_0_VVV); // x408: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__3__0_]
        vreal x409 = stencil(AtDD01, stencil_idx_0_m2_0_VVV); // x409: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m2__0_]
        vreal x410 = stencil(AtDD01, stencil_idx_0_2_0_VVV); // x410: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__2__0_]
        vreal x411 = stencil(AtDD01, stencil_idx_0_m1_0_VVV); // x411: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m1__0_]
        vreal x412 = stencil(AtDD01, stencil_idx_0_1_0_VVV); // x412: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__1__0_]
        vreal x413 = stencil(AtDD01, stencil_idx_0_0_m3_VVV); // x413: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m3_]
        vreal x414 = stencil(AtDD01, stencil_idx_0_0_3_VVV); // x414: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__3_]
        vreal x415 = stencil(AtDD01, stencil_idx_0_0_m2_VVV); // x415: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m2_]
        vreal x416 = stencil(AtDD01, stencil_idx_0_0_2_VVV); // x416: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__2_]
        vreal x417 = stencil(AtDD01, stencil_idx_0_0_m1_VVV); // x417: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m1_]
        vreal x418 = stencil(AtDD01, stencil_idx_0_0_1_VVV); // x418: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__1_]
        vreal x401 = stencil(AtDD01, stencil_idx_m3_0_0_VVV); // x401: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m3__0__0_]
        vreal x402 = stencil(AtDD01, stencil_idx_3_0_0_VVV); // x402: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__3__0__0_]
        vreal x403 = stencil(AtDD01, stencil_idx_m2_0_0_VVV); // x403: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m2__0__0_]
        vreal x404 = stencil(AtDD01, stencil_idx_2_0_0_VVV); // x404: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__2__0__0_]
        vreal x405 = stencil(AtDD01, stencil_idx_m1_0_0_VVV); // x405: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m1__0__0_]
        vreal x406 = stencil(AtDD01, stencil_idx_1_0_0_VVV); // x406: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__1__0__0_]
        vreal x400 = (70 * stencil(AtDD01, stencil_idx_0_0_0_VVV)); // x400: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__0_]
        store(At_rhsDD01, stencil_idx_0_0_0_VVV, (access(At_rhsDD01, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x400 + (-56 * ((x405 + x406))) + (-8 * ((x401 + x402))) + (28 * ((x403 + x404))) + ((stencil(AtDD01, stencil_idx_m4_0_0_VVV) + stencil(AtDD01, stencil_idx_4_0_0_VVV))))) + (DYI * (x400 + (-56 * ((x411 + x412))) + (-8 * ((x407 + x408))) + (28 * ((x409 + x410))) + ((stencil(AtDD01, stencil_idx_0_m4_0_VVV) + stencil(AtDD01, stencil_idx_0_4_0_VVV))))) + (DZI * (x400 + (-56 * ((x417 + x418))) + (-8 * ((x413 + x414))) + (28 * ((x415 + x416))) + ((stencil(AtDD01, stencil_idx_0_0_m4_VVV) + stencil(AtDD01, stencil_idx_0_0_4_VVV)))))))));
        vreal x426 = stencil(AtDD02, stencil_idx_0_m3_0_VVV); // x426: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m3__0_]
        vreal x427 = stencil(AtDD02, stencil_idx_0_3_0_VVV); // x427: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__3__0_]
        vreal x428 = stencil(AtDD02, stencil_idx_0_m2_0_VVV); // x428: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m2__0_]
        vreal x429 = stencil(AtDD02, stencil_idx_0_2_0_VVV); // x429: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__2__0_]
        vreal x430 = stencil(AtDD02, stencil_idx_0_m1_0_VVV); // x430: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m1__0_]
        vreal x431 = stencil(AtDD02, stencil_idx_0_1_0_VVV); // x431: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__1__0_]
        vreal x432 = stencil(AtDD02, stencil_idx_0_0_m3_VVV); // x432: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m3_]
        vreal x433 = stencil(AtDD02, stencil_idx_0_0_3_VVV); // x433: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__3_]
        vreal x434 = stencil(AtDD02, stencil_idx_0_0_m2_VVV); // x434: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m2_]
        vreal x435 = stencil(AtDD02, stencil_idx_0_0_2_VVV); // x435: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__2_]
        vreal x436 = stencil(AtDD02, stencil_idx_0_0_m1_VVV); // x436: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m1_]
        vreal x437 = stencil(AtDD02, stencil_idx_0_0_1_VVV); // x437: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__1_]
        vreal x420 = stencil(AtDD02, stencil_idx_m3_0_0_VVV); // x420: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m3__0__0_]
        vreal x421 = stencil(AtDD02, stencil_idx_3_0_0_VVV); // x421: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__3__0__0_]
        vreal x422 = stencil(AtDD02, stencil_idx_m2_0_0_VVV); // x422: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m2__0__0_]
        vreal x423 = stencil(AtDD02, stencil_idx_2_0_0_VVV); // x423: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__2__0__0_]
        vreal x424 = stencil(AtDD02, stencil_idx_m1_0_0_VVV); // x424: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m1__0__0_]
        vreal x425 = stencil(AtDD02, stencil_idx_1_0_0_VVV); // x425: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__1__0__0_]
        vreal x419 = (70 * stencil(AtDD02, stencil_idx_0_0_0_VVV)); // x419: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__0_]
        store(At_rhsDD02, stencil_idx_0_0_0_VVV, (access(At_rhsDD02, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x419 + (-56 * ((x424 + x425))) + (-8 * ((x420 + x421))) + (28 * ((x422 + x423))) + ((stencil(AtDD02, stencil_idx_m4_0_0_VVV) + stencil(AtDD02, stencil_idx_4_0_0_VVV))))) + (DYI * (x419 + (-56 * ((x430 + x431))) + (-8 * ((x426 + x427))) + (28 * ((x428 + x429))) + ((stencil(AtDD02, stencil_idx_0_m4_0_VVV) + stencil(AtDD02, stencil_idx_0_4_0_VVV))))) + (DZI * (x419 + (-56 * ((x436 + x437))) + (-8 * ((x432 + x433))) + (28 * ((x434 + x435))) + ((stencil(AtDD02, stencil_idx_0_0_m4_VVV) + stencil(AtDD02, stencil_idx_0_0_4_VVV)))))))));
        vreal x445 = stencil(AtDD11, stencil_idx_0_m3_0_VVV); // x445: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m3__0_]
        vreal x446 = stencil(AtDD11, stencil_idx_0_3_0_VVV); // x446: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__3__0_]
        vreal x447 = stencil(AtDD11, stencil_idx_0_m2_0_VVV); // x447: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m2__0_]
        vreal x448 = stencil(AtDD11, stencil_idx_0_2_0_VVV); // x448: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__2__0_]
        vreal x449 = stencil(AtDD11, stencil_idx_0_m1_0_VVV); // x449: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m1__0_]
        vreal x450 = stencil(AtDD11, stencil_idx_0_1_0_VVV); // x450: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__1__0_]
        vreal x451 = stencil(AtDD11, stencil_idx_0_0_m3_VVV); // x451: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m3_]
        vreal x452 = stencil(AtDD11, stencil_idx_0_0_3_VVV); // x452: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__3_]
        vreal x453 = stencil(AtDD11, stencil_idx_0_0_m2_VVV); // x453: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m2_]
        vreal x454 = stencil(AtDD11, stencil_idx_0_0_2_VVV); // x454: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__2_]
        vreal x455 = stencil(AtDD11, stencil_idx_0_0_m1_VVV); // x455: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m1_]
        vreal x456 = stencil(AtDD11, stencil_idx_0_0_1_VVV); // x456: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__1_]
        vreal x439 = stencil(AtDD11, stencil_idx_m3_0_0_VVV); // x439: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m3__0__0_]
        vreal x440 = stencil(AtDD11, stencil_idx_3_0_0_VVV); // x440: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__3__0__0_]
        vreal x441 = stencil(AtDD11, stencil_idx_m2_0_0_VVV); // x441: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m2__0__0_]
        vreal x442 = stencil(AtDD11, stencil_idx_2_0_0_VVV); // x442: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__2__0__0_]
        vreal x443 = stencil(AtDD11, stencil_idx_m1_0_0_VVV); // x443: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m1__0__0_]
        vreal x444 = stencil(AtDD11, stencil_idx_1_0_0_VVV); // x444: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__1__0__0_]
        vreal x438 = (70 * stencil(AtDD11, stencil_idx_0_0_0_VVV)); // x438: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__0_]
        store(At_rhsDD11, stencil_idx_0_0_0_VVV, (access(At_rhsDD11, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x438 + (-56 * ((x443 + x444))) + (-8 * ((x439 + x440))) + (28 * ((x441 + x442))) + ((stencil(AtDD11, stencil_idx_m4_0_0_VVV) + stencil(AtDD11, stencil_idx_4_0_0_VVV))))) + (DYI * (x438 + (-56 * ((x449 + x450))) + (-8 * ((x445 + x446))) + (28 * ((x447 + x448))) + ((stencil(AtDD11, stencil_idx_0_m4_0_VVV) + stencil(AtDD11, stencil_idx_0_4_0_VVV))))) + (DZI * (x438 + (-56 * ((x455 + x456))) + (-8 * ((x451 + x452))) + (28 * ((x453 + x454))) + ((stencil(AtDD11, stencil_idx_0_0_m4_VVV) + stencil(AtDD11, stencil_idx_0_0_4_VVV)))))))));
        vreal x464 = stencil(AtDD12, stencil_idx_0_m3_0_VVV); // x464: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m3__0_]
        vreal x465 = stencil(AtDD12, stencil_idx_0_3_0_VVV); // x465: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__3__0_]
        vreal x466 = stencil(AtDD12, stencil_idx_0_m2_0_VVV); // x466: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m2__0_]
        vreal x467 = stencil(AtDD12, stencil_idx_0_2_0_VVV); // x467: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__2__0_]
        vreal x468 = stencil(AtDD12, stencil_idx_0_m1_0_VVV); // x468: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m1__0_]
        vreal x469 = stencil(AtDD12, stencil_idx_0_1_0_VVV); // x469: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__1__0_]
        vreal x470 = stencil(AtDD12, stencil_idx_0_0_m3_VVV); // x470: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m3_]
        vreal x471 = stencil(AtDD12, stencil_idx_0_0_3_VVV); // x471: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__3_]
        vreal x472 = stencil(AtDD12, stencil_idx_0_0_m2_VVV); // x472: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m2_]
        vreal x473 = stencil(AtDD12, stencil_idx_0_0_2_VVV); // x473: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__2_]
        vreal x474 = stencil(AtDD12, stencil_idx_0_0_m1_VVV); // x474: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m1_]
        vreal x475 = stencil(AtDD12, stencil_idx_0_0_1_VVV); // x475: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__1_]
        vreal x458 = stencil(AtDD12, stencil_idx_m3_0_0_VVV); // x458: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m3__0__0_]
        vreal x459 = stencil(AtDD12, stencil_idx_3_0_0_VVV); // x459: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__3__0__0_]
        vreal x460 = stencil(AtDD12, stencil_idx_m2_0_0_VVV); // x460: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m2__0__0_]
        vreal x461 = stencil(AtDD12, stencil_idx_2_0_0_VVV); // x461: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__2__0__0_]
        vreal x462 = stencil(AtDD12, stencil_idx_m1_0_0_VVV); // x462: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m1__0__0_]
        vreal x463 = stencil(AtDD12, stencil_idx_1_0_0_VVV); // x463: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__1__0__0_]
        vreal x457 = (70 * stencil(AtDD12, stencil_idx_0_0_0_VVV)); // x457: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__0_]
        store(At_rhsDD12, stencil_idx_0_0_0_VVV, (access(At_rhsDD12, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x457 + (-56 * ((x462 + x463))) + (-8 * ((x458 + x459))) + (28 * ((x460 + x461))) + ((stencil(AtDD12, stencil_idx_m4_0_0_VVV) + stencil(AtDD12, stencil_idx_4_0_0_VVV))))) + (DYI * (x457 + (-56 * ((x468 + x469))) + (-8 * ((x464 + x465))) + (28 * ((x466 + x467))) + ((stencil(AtDD12, stencil_idx_0_m4_0_VVV) + stencil(AtDD12, stencil_idx_0_4_0_VVV))))) + (DZI * (x457 + (-56 * ((x474 + x475))) + (-8 * ((x470 + x471))) + (28 * ((x472 + x473))) + ((stencil(AtDD12, stencil_idx_0_0_m4_VVV) + stencil(AtDD12, stencil_idx_0_0_4_VVV)))))))));
        vreal x483 = stencil(AtDD22, stencil_idx_0_m3_0_VVV); // x483: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m3__0_]
        vreal x484 = stencil(AtDD22, stencil_idx_0_3_0_VVV); // x484: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__3__0_]
        vreal x485 = stencil(AtDD22, stencil_idx_0_m2_0_VVV); // x485: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m2__0_]
        vreal x486 = stencil(AtDD22, stencil_idx_0_2_0_VVV); // x486: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__2__0_]
        vreal x487 = stencil(AtDD22, stencil_idx_0_m1_0_VVV); // x487: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m1__0_]
        vreal x488 = stencil(AtDD22, stencil_idx_0_1_0_VVV); // x488: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__1__0_]
        vreal x489 = stencil(AtDD22, stencil_idx_0_0_m3_VVV); // x489: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m3_]
        vreal x490 = stencil(AtDD22, stencil_idx_0_0_3_VVV); // x490: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__3_]
        vreal x491 = stencil(AtDD22, stencil_idx_0_0_m2_VVV); // x491: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m2_]
        vreal x492 = stencil(AtDD22, stencil_idx_0_0_2_VVV); // x492: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__2_]
        vreal x493 = stencil(AtDD22, stencil_idx_0_0_m1_VVV); // x493: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m1_]
        vreal x494 = stencil(AtDD22, stencil_idx_0_0_1_VVV); // x494: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__1_]
        vreal x477 = stencil(AtDD22, stencil_idx_m3_0_0_VVV); // x477: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m3__0__0_]
        vreal x478 = stencil(AtDD22, stencil_idx_3_0_0_VVV); // x478: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__3__0__0_]
        vreal x479 = stencil(AtDD22, stencil_idx_m2_0_0_VVV); // x479: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m2__0__0_]
        vreal x480 = stencil(AtDD22, stencil_idx_2_0_0_VVV); // x480: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__2__0__0_]
        vreal x481 = stencil(AtDD22, stencil_idx_m1_0_0_VVV); // x481: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m1__0__0_]
        vreal x482 = stencil(AtDD22, stencil_idx_1_0_0_VVV); // x482: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__1__0__0_]
        vreal x476 = (70 * stencil(AtDD22, stencil_idx_0_0_0_VVV)); // x476: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__0_]
        store(At_rhsDD22, stencil_idx_0_0_0_VVV, (access(At_rhsDD22, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x476 + (-56 * ((x481 + x482))) + (-8 * ((x477 + x478))) + (28 * ((x479 + x480))) + ((stencil(AtDD22, stencil_idx_m4_0_0_VVV) + stencil(AtDD22, stencil_idx_4_0_0_VVV))))) + (DYI * (x476 + (-56 * ((x487 + x488))) + (-8 * ((x483 + x484))) + (28 * ((x485 + x486))) + ((stencil(AtDD22, stencil_idx_0_m4_0_VVV) + stencil(AtDD22, stencil_idx_0_4_0_VVV))))) + (DZI * (x476 + (-56 * ((x493 + x494))) + (-8 * ((x489 + x490))) + (28 * ((x491 + x492))) + ((stencil(AtDD22, stencil_idx_0_0_m4_VVV) + stencil(AtDD22, stencil_idx_0_0_4_VVV)))))))));    
    });
    // apply_dissipation loop 4
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        vreal x502 = stencil(ConfConnectU0, stencil_idx_0_m3_0_VVV); // x502: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__m3__0_]
        vreal x503 = stencil(ConfConnectU0, stencil_idx_0_3_0_VVV); // x503: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__3__0_]
        vreal x504 = stencil(ConfConnectU0, stencil_idx_0_m2_0_VVV); // x504: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__m2__0_]
        vreal x505 = stencil(ConfConnectU0, stencil_idx_0_2_0_VVV); // x505: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__2__0_]
        vreal x506 = stencil(ConfConnectU0, stencil_idx_0_m1_0_VVV); // x506: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__m1__0_]
        vreal x507 = stencil(ConfConnectU0, stencil_idx_0_1_0_VVV); // x507: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__1__0_]
        vreal x508 = stencil(ConfConnectU0, stencil_idx_0_0_m3_VVV); // x508: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__m3_]
        vreal x509 = stencil(ConfConnectU0, stencil_idx_0_0_3_VVV); // x509: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__3_]
        vreal x510 = stencil(ConfConnectU0, stencil_idx_0_0_m2_VVV); // x510: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__m2_]
        vreal x511 = stencil(ConfConnectU0, stencil_idx_0_0_2_VVV); // x511: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__2_]
        vreal x512 = stencil(ConfConnectU0, stencil_idx_0_0_m1_VVV); // x512: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__m1_]
        vreal x513 = stencil(ConfConnectU0, stencil_idx_0_0_1_VVV); // x513: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__1_]
        vreal x496 = stencil(ConfConnectU0, stencil_idx_m3_0_0_VVV); // x496: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__m3__0__0_]
        vreal x497 = stencil(ConfConnectU0, stencil_idx_3_0_0_VVV); // x497: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__3__0__0_]
        vreal x498 = stencil(ConfConnectU0, stencil_idx_m2_0_0_VVV); // x498: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__m2__0__0_]
        vreal x499 = stencil(ConfConnectU0, stencil_idx_2_0_0_VVV); // x499: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__2__0__0_]
        vreal x500 = stencil(ConfConnectU0, stencil_idx_m1_0_0_VVV); // x500: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__m1__0__0_]
        vreal x501 = stencil(ConfConnectU0, stencil_idx_1_0_0_VVV); // x501: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__1__0__0_]
        vreal x495 = (70 * stencil(ConfConnectU0, stencil_idx_0_0_0_VVV)); // x495: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__0_]
        vreal x283 = ((1.0 / 256.0) * dissipation_epsilon); // x283: Dependency! Liveness = 1; [dissipation_epsilon]
        store(ConfConnect_rhsU0, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU0, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x495 + (-56 * ((x500 + x501))) + (-8 * ((x496 + x497))) + (28 * ((x498 + x499))) + ((stencil(ConfConnectU0, stencil_idx_m4_0_0_VVV) + stencil(ConfConnectU0, stencil_idx_4_0_0_VVV))))) + (DYI * (x495 + (-56 * ((x506 + x507))) + (-8 * ((x502 + x503))) + (28 * ((x504 + x505))) + ((stencil(ConfConnectU0, stencil_idx_0_m4_0_VVV) + stencil(ConfConnectU0, stencil_idx_0_4_0_VVV))))) + (DZI * (x495 + (-56 * ((x512 + x513))) + (-8 * ((x508 + x509))) + (28 * ((x510 + x511))) + ((stencil(ConfConnectU0, stencil_idx_0_0_m4_VVV) + stencil(ConfConnectU0, stencil_idx_0_0_4_VVV)))))))));
        vreal x521 = stencil(ConfConnectU1, stencil_idx_0_m3_0_VVV); // x521: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__m3__0_]
        vreal x522 = stencil(ConfConnectU1, stencil_idx_0_3_0_VVV); // x522: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__3__0_]
        vreal x523 = stencil(ConfConnectU1, stencil_idx_0_m2_0_VVV); // x523: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__m2__0_]
        vreal x524 = stencil(ConfConnectU1, stencil_idx_0_2_0_VVV); // x524: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__2__0_]
        vreal x525 = stencil(ConfConnectU1, stencil_idx_0_m1_0_VVV); // x525: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__m1__0_]
        vreal x526 = stencil(ConfConnectU1, stencil_idx_0_1_0_VVV); // x526: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__1__0_]
        vreal x527 = stencil(ConfConnectU1, stencil_idx_0_0_m3_VVV); // x527: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__m3_]
        vreal x528 = stencil(ConfConnectU1, stencil_idx_0_0_3_VVV); // x528: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__3_]
        vreal x529 = stencil(ConfConnectU1, stencil_idx_0_0_m2_VVV); // x529: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__m2_]
        vreal x530 = stencil(ConfConnectU1, stencil_idx_0_0_2_VVV); // x530: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__2_]
        vreal x531 = stencil(ConfConnectU1, stencil_idx_0_0_m1_VVV); // x531: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__m1_]
        vreal x532 = stencil(ConfConnectU1, stencil_idx_0_0_1_VVV); // x532: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__1_]
        vreal x515 = stencil(ConfConnectU1, stencil_idx_m3_0_0_VVV); // x515: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__m3__0__0_]
        vreal x516 = stencil(ConfConnectU1, stencil_idx_3_0_0_VVV); // x516: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__3__0__0_]
        vreal x517 = stencil(ConfConnectU1, stencil_idx_m2_0_0_VVV); // x517: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__m2__0__0_]
        vreal x518 = stencil(ConfConnectU1, stencil_idx_2_0_0_VVV); // x518: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__2__0__0_]
        vreal x519 = stencil(ConfConnectU1, stencil_idx_m1_0_0_VVV); // x519: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__m1__0__0_]
        vreal x520 = stencil(ConfConnectU1, stencil_idx_1_0_0_VVV); // x520: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__1__0__0_]
        vreal x514 = (70 * stencil(ConfConnectU1, stencil_idx_0_0_0_VVV)); // x514: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__0_]
        store(ConfConnect_rhsU1, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU1, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x514 + (-56 * ((x519 + x520))) + (-8 * ((x515 + x516))) + (28 * ((x517 + x518))) + ((stencil(ConfConnectU1, stencil_idx_m4_0_0_VVV) + stencil(ConfConnectU1, stencil_idx_4_0_0_VVV))))) + (DYI * (x514 + (-56 * ((x525 + x526))) + (-8 * ((x521 + x522))) + (28 * ((x523 + x524))) + ((stencil(ConfConnectU1, stencil_idx_0_m4_0_VVV) + stencil(ConfConnectU1, stencil_idx_0_4_0_VVV))))) + (DZI * (x514 + (-56 * ((x531 + x532))) + (-8 * ((x527 + x528))) + (28 * ((x529 + x530))) + ((stencil(ConfConnectU1, stencil_idx_0_0_m4_VVV) + stencil(ConfConnectU1, stencil_idx_0_0_4_VVV)))))))));
        vreal x540 = stencil(ConfConnectU2, stencil_idx_0_m3_0_VVV); // x540: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__m3__0_]
        vreal x541 = stencil(ConfConnectU2, stencil_idx_0_3_0_VVV); // x541: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__3__0_]
        vreal x542 = stencil(ConfConnectU2, stencil_idx_0_m2_0_VVV); // x542: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__m2__0_]
        vreal x543 = stencil(ConfConnectU2, stencil_idx_0_2_0_VVV); // x543: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__2__0_]
        vreal x544 = stencil(ConfConnectU2, stencil_idx_0_m1_0_VVV); // x544: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__m1__0_]
        vreal x545 = stencil(ConfConnectU2, stencil_idx_0_1_0_VVV); // x545: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__1__0_]
        vreal x546 = stencil(ConfConnectU2, stencil_idx_0_0_m3_VVV); // x546: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__m3_]
        vreal x547 = stencil(ConfConnectU2, stencil_idx_0_0_3_VVV); // x547: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__3_]
        vreal x548 = stencil(ConfConnectU2, stencil_idx_0_0_m2_VVV); // x548: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__m2_]
        vreal x549 = stencil(ConfConnectU2, stencil_idx_0_0_2_VVV); // x549: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__2_]
        vreal x550 = stencil(ConfConnectU2, stencil_idx_0_0_m1_VVV); // x550: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__m1_]
        vreal x551 = stencil(ConfConnectU2, stencil_idx_0_0_1_VVV); // x551: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__1_]
        vreal x534 = stencil(ConfConnectU2, stencil_idx_m3_0_0_VVV); // x534: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__m3__0__0_]
        vreal x535 = stencil(ConfConnectU2, stencil_idx_3_0_0_VVV); // x535: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__3__0__0_]
        vreal x536 = stencil(ConfConnectU2, stencil_idx_m2_0_0_VVV); // x536: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__m2__0__0_]
        vreal x537 = stencil(ConfConnectU2, stencil_idx_2_0_0_VVV); // x537: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__2__0__0_]
        vreal x538 = stencil(ConfConnectU2, stencil_idx_m1_0_0_VVV); // x538: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__m1__0__0_]
        vreal x539 = stencil(ConfConnectU2, stencil_idx_1_0_0_VVV); // x539: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__1__0__0_]
        vreal x533 = (70 * stencil(ConfConnectU2, stencil_idx_0_0_0_VVV)); // x533: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__0_]
        store(ConfConnect_rhsU2, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU2, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x533 + (-56 * ((x538 + x539))) + (-8 * ((x534 + x535))) + (28 * ((x536 + x537))) + ((stencil(ConfConnectU2, stencil_idx_m4_0_0_VVV) + stencil(ConfConnectU2, stencil_idx_4_0_0_VVV))))) + (DYI * (x533 + (-56 * ((x544 + x545))) + (-8 * ((x540 + x541))) + (28 * ((x542 + x543))) + ((stencil(ConfConnectU2, stencil_idx_0_m4_0_VVV) + stencil(ConfConnectU2, stencil_idx_0_4_0_VVV))))) + (DZI * (x533 + (-56 * ((x550 + x551))) + (-8 * ((x546 + x547))) + (28 * ((x548 + x549))) + ((stencil(ConfConnectU2, stencil_idx_0_0_m4_VVV) + stencil(ConfConnectU2, stencil_idx_0_0_4_VVV)))))))));    
    });
    // apply_dissipation loop 5
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        vreal x553 = stencil(evo_lapse, stencil_idx_m3_0_0_VVV); // x553: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m3__0__0_]
        vreal x554 = stencil(evo_lapse, stencil_idx_3_0_0_VVV); // x554: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__3__0__0_]
        vreal x555 = ((x553 + x554)); // x555: Dependency! Liveness = 2; [x553, x554]
        vreal x556 = stencil(evo_lapse, stencil_idx_m2_0_0_VVV); // x556: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m2__0__0_]
        vreal x557 = stencil(evo_lapse, stencil_idx_2_0_0_VVV); // x557: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__2__0__0_]
        vreal x558 = ((x556 + x557)); // x558: Dependency! Liveness = 2; [x556, x557]
        vreal x559 = stencil(evo_lapse, stencil_idx_m1_0_0_VVV); // x559: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m1__0__0_]
        vreal x560 = stencil(evo_lapse, stencil_idx_1_0_0_VVV); // x560: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__1__0__0_]
        vreal x561 = ((x559 + x560)); // x561: Dependency! Liveness = 2; [x559, x560]
        vreal x562 = stencil(evo_lapse, stencil_idx_0_m3_0_VVV); // x562: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m3__0_]
        vreal x563 = stencil(evo_lapse, stencil_idx_0_3_0_VVV); // x563: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__3__0_]
        vreal x564 = ((x562 + x563)); // x564: Dependency! Liveness = 2; [x562, x563]
        vreal x565 = stencil(evo_lapse, stencil_idx_0_m2_0_VVV); // x565: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m2__0_]
        vreal x566 = stencil(evo_lapse, stencil_idx_0_2_0_VVV); // x566: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__2__0_]
        vreal x567 = ((x565 + x566)); // x567: Dependency! Liveness = 2; [x565, x566]
        vreal x568 = stencil(evo_lapse, stencil_idx_0_m1_0_VVV); // x568: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m1__0_]
        vreal x569 = stencil(evo_lapse, stencil_idx_0_1_0_VVV); // x569: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__1__0_]
        vreal x570 = ((x568 + x569)); // x570: Dependency! Liveness = 2; [x568, x569]
        vreal x571 = stencil(evo_lapse, stencil_idx_0_0_m3_VVV); // x571: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m3_]
        vreal x572 = stencil(evo_lapse, stencil_idx_0_0_3_VVV); // x572: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__3_]
        vreal x573 = ((x571 + x572)); // x573: Dependency! Liveness = 2; [x571, x572]
        vreal x574 = stencil(evo_lapse, stencil_idx_0_0_m2_VVV); // x574: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m2_]
        vreal x575 = stencil(evo_lapse, stencil_idx_0_0_2_VVV); // x575: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__2_]
        vreal x576 = ((x574 + x575)); // x576: Dependency! Liveness = 2; [x574, x575]
        vreal x577 = stencil(evo_lapse, stencil_idx_0_0_m1_VVV); // x577: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m1_]
        vreal x578 = stencil(evo_lapse, stencil_idx_0_0_1_VVV); // x578: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__1_]
        vreal x579 = ((x577 + x578)); // x579: Dependency! Liveness = 2; [x577, x578]
        vreal x552 = (70 * stencil(evo_lapse, stencil_idx_0_0_0_VVV)); // x552: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__0_]
        vreal x283 = ((1.0 / 256.0) * dissipation_epsilon); // x283: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_lapse_rhs, stencil_idx_0_0_0_VVV, (access(evo_lapse_rhs, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x552 + (-56 * x561) + (-8 * x555) + (28 * x558) + ((stencil(evo_lapse, stencil_idx_m4_0_0_VVV) + stencil(evo_lapse, stencil_idx_4_0_0_VVV))))) + (DYI * (x552 + (-56 * x570) + (-8 * x564) + (28 * x567) + ((stencil(evo_lapse, stencil_idx_0_m4_0_VVV) + stencil(evo_lapse, stencil_idx_0_4_0_VVV))))) + (DZI * (x552 + (-56 * x579) + (-8 * x573) + (28 * x576) + ((stencil(evo_lapse, stencil_idx_0_0_m4_VVV) + stencil(evo_lapse, stencil_idx_0_0_4_VVV)))))))));    
    });
    // apply_dissipation loop 6
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        vreal x581 = stencil(evo_shiftU0, stencil_idx_m3_0_0_VVV); // x581: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m3__0__0_]
        vreal x582 = stencil(evo_shiftU0, stencil_idx_3_0_0_VVV); // x582: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__3__0__0_]
        vreal x583 = ((x581 + x582)); // x583: Dependency! Liveness = 2; [x581, x582]
        vreal x584 = stencil(evo_shiftU0, stencil_idx_m2_0_0_VVV); // x584: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m2__0__0_]
        vreal x585 = stencil(evo_shiftU0, stencil_idx_2_0_0_VVV); // x585: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__2__0__0_]
        vreal x586 = ((x584 + x585)); // x586: Dependency! Liveness = 2; [x584, x585]
        vreal x587 = stencil(evo_shiftU0, stencil_idx_m1_0_0_VVV); // x587: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m1__0__0_]
        vreal x588 = stencil(evo_shiftU0, stencil_idx_1_0_0_VVV); // x588: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__1__0__0_]
        vreal x589 = ((x587 + x588)); // x589: Dependency! Liveness = 2; [x587, x588]
        vreal x590 = stencil(evo_shiftU0, stencil_idx_0_m3_0_VVV); // x590: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m3__0_]
        vreal x591 = stencil(evo_shiftU0, stencil_idx_0_3_0_VVV); // x591: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__3__0_]
        vreal x592 = ((x590 + x591)); // x592: Dependency! Liveness = 2; [x590, x591]
        vreal x593 = stencil(evo_shiftU0, stencil_idx_0_m2_0_VVV); // x593: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m2__0_]
        vreal x594 = stencil(evo_shiftU0, stencil_idx_0_2_0_VVV); // x594: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__2__0_]
        vreal x595 = ((x593 + x594)); // x595: Dependency! Liveness = 2; [x593, x594]
        vreal x596 = stencil(evo_shiftU0, stencil_idx_0_m1_0_VVV); // x596: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m1__0_]
        vreal x597 = stencil(evo_shiftU0, stencil_idx_0_1_0_VVV); // x597: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__1__0_]
        vreal x598 = ((x596 + x597)); // x598: Dependency! Liveness = 2; [x596, x597]
        vreal x599 = stencil(evo_shiftU0, stencil_idx_0_0_m3_VVV); // x599: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m3_]
        vreal x600 = stencil(evo_shiftU0, stencil_idx_0_0_3_VVV); // x600: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__3_]
        vreal x601 = ((x599 + x600)); // x601: Dependency! Liveness = 2; [x599, x600]
        vreal x602 = stencil(evo_shiftU0, stencil_idx_0_0_m2_VVV); // x602: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m2_]
        vreal x603 = stencil(evo_shiftU0, stencil_idx_0_0_2_VVV); // x603: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__2_]
        vreal x604 = ((x602 + x603)); // x604: Dependency! Liveness = 2; [x602, x603]
        vreal x605 = stencil(evo_shiftU0, stencil_idx_0_0_m1_VVV); // x605: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m1_]
        vreal x606 = stencil(evo_shiftU0, stencil_idx_0_0_1_VVV); // x606: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__1_]
        vreal x607 = ((x605 + x606)); // x607: Dependency! Liveness = 2; [x605, x606]
        vreal x580 = (70 * stencil(evo_shiftU0, stencil_idx_0_0_0_VVV)); // x580: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__0_]
        vreal x283 = ((1.0 / 256.0) * dissipation_epsilon); // x283: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_shift_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU0, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x580 + (-56 * x589) + (-8 * x583) + (28 * x586) + ((stencil(evo_shiftU0, stencil_idx_m4_0_0_VVV) + stencil(evo_shiftU0, stencil_idx_4_0_0_VVV))))) + (DYI * (x580 + (-56 * x598) + (-8 * x592) + (28 * x595) + ((stencil(evo_shiftU0, stencil_idx_0_m4_0_VVV) + stencil(evo_shiftU0, stencil_idx_0_4_0_VVV))))) + (DZI * (x580 + (-56 * x607) + (-8 * x601) + (28 * x604) + ((stencil(evo_shiftU0, stencil_idx_0_0_m4_VVV) + stencil(evo_shiftU0, stencil_idx_0_0_4_VVV)))))))));
        vreal x609 = stencil(evo_shiftU1, stencil_idx_m3_0_0_VVV); // x609: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m3__0__0_]
        vreal x610 = stencil(evo_shiftU1, stencil_idx_3_0_0_VVV); // x610: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__3__0__0_]
        vreal x611 = ((x609 + x610)); // x611: Dependency! Liveness = 2; [x609, x610]
        vreal x612 = stencil(evo_shiftU1, stencil_idx_m2_0_0_VVV); // x612: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m2__0__0_]
        vreal x613 = stencil(evo_shiftU1, stencil_idx_2_0_0_VVV); // x613: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__2__0__0_]
        vreal x614 = ((x612 + x613)); // x614: Dependency! Liveness = 2; [x612, x613]
        vreal x615 = stencil(evo_shiftU1, stencil_idx_m1_0_0_VVV); // x615: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m1__0__0_]
        vreal x616 = stencil(evo_shiftU1, stencil_idx_1_0_0_VVV); // x616: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__1__0__0_]
        vreal x617 = ((x615 + x616)); // x617: Dependency! Liveness = 2; [x615, x616]
        vreal x618 = stencil(evo_shiftU1, stencil_idx_0_m3_0_VVV); // x618: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m3__0_]
        vreal x619 = stencil(evo_shiftU1, stencil_idx_0_3_0_VVV); // x619: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__3__0_]
        vreal x620 = ((x618 + x619)); // x620: Dependency! Liveness = 2; [x618, x619]
        vreal x621 = stencil(evo_shiftU1, stencil_idx_0_m2_0_VVV); // x621: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m2__0_]
        vreal x622 = stencil(evo_shiftU1, stencil_idx_0_2_0_VVV); // x622: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__2__0_]
        vreal x623 = ((x621 + x622)); // x623: Dependency! Liveness = 2; [x621, x622]
        vreal x624 = stencil(evo_shiftU1, stencil_idx_0_m1_0_VVV); // x624: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m1__0_]
        vreal x625 = stencil(evo_shiftU1, stencil_idx_0_1_0_VVV); // x625: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__1__0_]
        vreal x626 = ((x624 + x625)); // x626: Dependency! Liveness = 2; [x624, x625]
        vreal x627 = stencil(evo_shiftU1, stencil_idx_0_0_m3_VVV); // x627: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m3_]
        vreal x628 = stencil(evo_shiftU1, stencil_idx_0_0_3_VVV); // x628: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__3_]
        vreal x629 = ((x627 + x628)); // x629: Dependency! Liveness = 2; [x627, x628]
        vreal x630 = stencil(evo_shiftU1, stencil_idx_0_0_m2_VVV); // x630: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m2_]
        vreal x631 = stencil(evo_shiftU1, stencil_idx_0_0_2_VVV); // x631: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__2_]
        vreal x632 = ((x630 + x631)); // x632: Dependency! Liveness = 2; [x630, x631]
        vreal x633 = stencil(evo_shiftU1, stencil_idx_0_0_m1_VVV); // x633: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m1_]
        vreal x634 = stencil(evo_shiftU1, stencil_idx_0_0_1_VVV); // x634: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__1_]
        vreal x635 = ((x633 + x634)); // x635: Dependency! Liveness = 2; [x633, x634]
        vreal x608 = (70 * stencil(evo_shiftU1, stencil_idx_0_0_0_VVV)); // x608: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__0_]
        store(evo_shift_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU1, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x608 + (-56 * x617) + (-8 * x611) + (28 * x614) + ((stencil(evo_shiftU1, stencil_idx_m4_0_0_VVV) + stencil(evo_shiftU1, stencil_idx_4_0_0_VVV))))) + (DYI * (x608 + (-56 * x626) + (-8 * x620) + (28 * x623) + ((stencil(evo_shiftU1, stencil_idx_0_m4_0_VVV) + stencil(evo_shiftU1, stencil_idx_0_4_0_VVV))))) + (DZI * (x608 + (-56 * x635) + (-8 * x629) + (28 * x632) + ((stencil(evo_shiftU1, stencil_idx_0_0_m4_VVV) + stencil(evo_shiftU1, stencil_idx_0_0_4_VVV)))))))));
        vreal x637 = stencil(evo_shiftU2, stencil_idx_m3_0_0_VVV); // x637: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m3__0__0_]
        vreal x638 = stencil(evo_shiftU2, stencil_idx_3_0_0_VVV); // x638: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__3__0__0_]
        vreal x639 = ((x637 + x638)); // x639: Dependency! Liveness = 2; [x637, x638]
        vreal x640 = stencil(evo_shiftU2, stencil_idx_m2_0_0_VVV); // x640: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m2__0__0_]
        vreal x641 = stencil(evo_shiftU2, stencil_idx_2_0_0_VVV); // x641: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__2__0__0_]
        vreal x642 = ((x640 + x641)); // x642: Dependency! Liveness = 2; [x640, x641]
        vreal x643 = stencil(evo_shiftU2, stencil_idx_m1_0_0_VVV); // x643: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m1__0__0_]
        vreal x644 = stencil(evo_shiftU2, stencil_idx_1_0_0_VVV); // x644: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__1__0__0_]
        vreal x645 = ((x643 + x644)); // x645: Dependency! Liveness = 2; [x643, x644]
        vreal x646 = stencil(evo_shiftU2, stencil_idx_0_m3_0_VVV); // x646: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m3__0_]
        vreal x647 = stencil(evo_shiftU2, stencil_idx_0_3_0_VVV); // x647: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__3__0_]
        vreal x648 = ((x646 + x647)); // x648: Dependency! Liveness = 2; [x646, x647]
        vreal x649 = stencil(evo_shiftU2, stencil_idx_0_m2_0_VVV); // x649: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m2__0_]
        vreal x650 = stencil(evo_shiftU2, stencil_idx_0_2_0_VVV); // x650: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__2__0_]
        vreal x651 = ((x649 + x650)); // x651: Dependency! Liveness = 2; [x649, x650]
        vreal x652 = stencil(evo_shiftU2, stencil_idx_0_m1_0_VVV); // x652: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m1__0_]
        vreal x653 = stencil(evo_shiftU2, stencil_idx_0_1_0_VVV); // x653: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__1__0_]
        vreal x654 = ((x652 + x653)); // x654: Dependency! Liveness = 2; [x652, x653]
        vreal x655 = stencil(evo_shiftU2, stencil_idx_0_0_m3_VVV); // x655: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m3_]
        vreal x656 = stencil(evo_shiftU2, stencil_idx_0_0_3_VVV); // x656: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__3_]
        vreal x657 = ((x655 + x656)); // x657: Dependency! Liveness = 2; [x655, x656]
        vreal x658 = stencil(evo_shiftU2, stencil_idx_0_0_m2_VVV); // x658: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m2_]
        vreal x659 = stencil(evo_shiftU2, stencil_idx_0_0_2_VVV); // x659: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__2_]
        vreal x660 = ((x658 + x659)); // x660: Dependency! Liveness = 2; [x658, x659]
        vreal x661 = stencil(evo_shiftU2, stencil_idx_0_0_m1_VVV); // x661: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m1_]
        vreal x662 = stencil(evo_shiftU2, stencil_idx_0_0_1_VVV); // x662: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__1_]
        vreal x663 = ((x661 + x662)); // x663: Dependency! Liveness = 2; [x661, x662]
        vreal x636 = (70 * stencil(evo_shiftU2, stencil_idx_0_0_0_VVV)); // x636: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__0_]
        store(evo_shift_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU2, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x636 + (-56 * x645) + (-8 * x639) + (28 * x642) + ((stencil(evo_shiftU2, stencil_idx_m4_0_0_VVV) + stencil(evo_shiftU2, stencil_idx_4_0_0_VVV))))) + (DYI * (x636 + (-56 * x654) + (-8 * x648) + (28 * x651) + ((stencil(evo_shiftU2, stencil_idx_0_m4_0_VVV) + stencil(evo_shiftU2, stencil_idx_0_4_0_VVV))))) + (DZI * (x636 + (-56 * x663) + (-8 * x657) + (28 * x660) + ((stencil(evo_shiftU2, stencil_idx_0_0_m4_VVV) + stencil(evo_shiftU2, stencil_idx_0_0_4_VVV)))))))));    
    });
    // apply_dissipation loop 7
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_m4_0_0_VVV(VVV_layout, p.I - 4*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_m4_0_VVV(VVV_layout, p.I - 4*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m4_VVV(VVV_layout, p.I - 4*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_4_VVV(VVV_layout, p.I + 4*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_0_4_0_VVV(VVV_layout, p.I + 4*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        const GF3D5index stencil_idx_4_0_0_VVV(VVV_layout, p.I + 4*p.DI[0]);
        vreal x671 = stencil(shift_BU0, stencil_idx_0_m3_0_VVV); // x671: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__m3__0_]
        vreal x672 = stencil(shift_BU0, stencil_idx_0_3_0_VVV); // x672: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__3__0_]
        vreal x673 = stencil(shift_BU0, stencil_idx_0_m2_0_VVV); // x673: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__m2__0_]
        vreal x674 = stencil(shift_BU0, stencil_idx_0_2_0_VVV); // x674: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__2__0_]
        vreal x675 = stencil(shift_BU0, stencil_idx_0_m1_0_VVV); // x675: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__m1__0_]
        vreal x676 = stencil(shift_BU0, stencil_idx_0_1_0_VVV); // x676: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__1__0_]
        vreal x677 = stencil(shift_BU0, stencil_idx_0_0_m3_VVV); // x677: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__m3_]
        vreal x678 = stencil(shift_BU0, stencil_idx_0_0_3_VVV); // x678: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__3_]
        vreal x679 = stencil(shift_BU0, stencil_idx_0_0_m2_VVV); // x679: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__m2_]
        vreal x680 = stencil(shift_BU0, stencil_idx_0_0_2_VVV); // x680: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__2_]
        vreal x681 = stencil(shift_BU0, stencil_idx_0_0_m1_VVV); // x681: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__m1_]
        vreal x682 = stencil(shift_BU0, stencil_idx_0_0_1_VVV); // x682: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__1_]
        vreal x665 = stencil(shift_BU0, stencil_idx_m3_0_0_VVV); // x665: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__m3__0__0_]
        vreal x666 = stencil(shift_BU0, stencil_idx_3_0_0_VVV); // x666: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__3__0__0_]
        vreal x667 = stencil(shift_BU0, stencil_idx_m2_0_0_VVV); // x667: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__m2__0__0_]
        vreal x668 = stencil(shift_BU0, stencil_idx_2_0_0_VVV); // x668: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__2__0__0_]
        vreal x669 = stencil(shift_BU0, stencil_idx_m1_0_0_VVV); // x669: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__m1__0__0_]
        vreal x670 = stencil(shift_BU0, stencil_idx_1_0_0_VVV); // x670: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__1__0__0_]
        vreal x664 = (70 * stencil(shift_BU0, stencil_idx_0_0_0_VVV)); // x664: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__0_]
        vreal x283 = ((1.0 / 256.0) * dissipation_epsilon); // x283: Dependency! Liveness = 1; [dissipation_epsilon]
        store(shift_B_rhsU0, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU0, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x664 + (-56 * ((x669 + x670))) + (-8 * ((x665 + x666))) + (28 * ((x667 + x668))) + ((stencil(shift_BU0, stencil_idx_m4_0_0_VVV) + stencil(shift_BU0, stencil_idx_4_0_0_VVV))))) + (DYI * (x664 + (-56 * ((x675 + x676))) + (-8 * ((x671 + x672))) + (28 * ((x673 + x674))) + ((stencil(shift_BU0, stencil_idx_0_m4_0_VVV) + stencil(shift_BU0, stencil_idx_0_4_0_VVV))))) + (DZI * (x664 + (-56 * ((x681 + x682))) + (-8 * ((x677 + x678))) + (28 * ((x679 + x680))) + ((stencil(shift_BU0, stencil_idx_0_0_m4_VVV) + stencil(shift_BU0, stencil_idx_0_0_4_VVV)))))))));
        vreal x690 = stencil(shift_BU1, stencil_idx_0_m3_0_VVV); // x690: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__m3__0_]
        vreal x691 = stencil(shift_BU1, stencil_idx_0_3_0_VVV); // x691: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__3__0_]
        vreal x692 = stencil(shift_BU1, stencil_idx_0_m2_0_VVV); // x692: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__m2__0_]
        vreal x693 = stencil(shift_BU1, stencil_idx_0_2_0_VVV); // x693: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__2__0_]
        vreal x694 = stencil(shift_BU1, stencil_idx_0_m1_0_VVV); // x694: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__m1__0_]
        vreal x695 = stencil(shift_BU1, stencil_idx_0_1_0_VVV); // x695: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__1__0_]
        vreal x696 = stencil(shift_BU1, stencil_idx_0_0_m3_VVV); // x696: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__m3_]
        vreal x697 = stencil(shift_BU1, stencil_idx_0_0_3_VVV); // x697: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__3_]
        vreal x698 = stencil(shift_BU1, stencil_idx_0_0_m2_VVV); // x698: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__m2_]
        vreal x699 = stencil(shift_BU1, stencil_idx_0_0_2_VVV); // x699: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__2_]
        vreal x700 = stencil(shift_BU1, stencil_idx_0_0_m1_VVV); // x700: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__m1_]
        vreal x701 = stencil(shift_BU1, stencil_idx_0_0_1_VVV); // x701: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__1_]
        vreal x684 = stencil(shift_BU1, stencil_idx_m3_0_0_VVV); // x684: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__m3__0__0_]
        vreal x685 = stencil(shift_BU1, stencil_idx_3_0_0_VVV); // x685: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__3__0__0_]
        vreal x686 = stencil(shift_BU1, stencil_idx_m2_0_0_VVV); // x686: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__m2__0__0_]
        vreal x687 = stencil(shift_BU1, stencil_idx_2_0_0_VVV); // x687: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__2__0__0_]
        vreal x688 = stencil(shift_BU1, stencil_idx_m1_0_0_VVV); // x688: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__m1__0__0_]
        vreal x689 = stencil(shift_BU1, stencil_idx_1_0_0_VVV); // x689: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__1__0__0_]
        vreal x683 = (70 * stencil(shift_BU1, stencil_idx_0_0_0_VVV)); // x683: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__0_]
        store(shift_B_rhsU1, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU1, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x683 + (-56 * ((x688 + x689))) + (-8 * ((x684 + x685))) + (28 * ((x686 + x687))) + ((stencil(shift_BU1, stencil_idx_m4_0_0_VVV) + stencil(shift_BU1, stencil_idx_4_0_0_VVV))))) + (DYI * (x683 + (-56 * ((x694 + x695))) + (-8 * ((x690 + x691))) + (28 * ((x692 + x693))) + ((stencil(shift_BU1, stencil_idx_0_m4_0_VVV) + stencil(shift_BU1, stencil_idx_0_4_0_VVV))))) + (DZI * (x683 + (-56 * ((x700 + x701))) + (-8 * ((x696 + x697))) + (28 * ((x698 + x699))) + ((stencil(shift_BU1, stencil_idx_0_0_m4_VVV) + stencil(shift_BU1, stencil_idx_0_0_4_VVV)))))))));
        vreal x709 = stencil(shift_BU2, stencil_idx_0_m3_0_VVV); // x709: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__m3__0_]
        vreal x710 = stencil(shift_BU2, stencil_idx_0_3_0_VVV); // x710: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__3__0_]
        vreal x711 = stencil(shift_BU2, stencil_idx_0_m2_0_VVV); // x711: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__m2__0_]
        vreal x712 = stencil(shift_BU2, stencil_idx_0_2_0_VVV); // x712: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__2__0_]
        vreal x713 = stencil(shift_BU2, stencil_idx_0_m1_0_VVV); // x713: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__m1__0_]
        vreal x714 = stencil(shift_BU2, stencil_idx_0_1_0_VVV); // x714: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__1__0_]
        vreal x715 = stencil(shift_BU2, stencil_idx_0_0_m3_VVV); // x715: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__m3_]
        vreal x716 = stencil(shift_BU2, stencil_idx_0_0_3_VVV); // x716: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__3_]
        vreal x717 = stencil(shift_BU2, stencil_idx_0_0_m2_VVV); // x717: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__m2_]
        vreal x718 = stencil(shift_BU2, stencil_idx_0_0_2_VVV); // x718: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__2_]
        vreal x719 = stencil(shift_BU2, stencil_idx_0_0_m1_VVV); // x719: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__m1_]
        vreal x720 = stencil(shift_BU2, stencil_idx_0_0_1_VVV); // x720: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__1_]
        vreal x703 = stencil(shift_BU2, stencil_idx_m3_0_0_VVV); // x703: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__m3__0__0_]
        vreal x704 = stencil(shift_BU2, stencil_idx_3_0_0_VVV); // x704: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__3__0__0_]
        vreal x705 = stencil(shift_BU2, stencil_idx_m2_0_0_VVV); // x705: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__m2__0__0_]
        vreal x706 = stencil(shift_BU2, stencil_idx_2_0_0_VVV); // x706: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__2__0__0_]
        vreal x707 = stencil(shift_BU2, stencil_idx_m1_0_0_VVV); // x707: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__m1__0__0_]
        vreal x708 = stencil(shift_BU2, stencil_idx_1_0_0_VVV); // x708: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__1__0__0_]
        vreal x702 = (70 * stencil(shift_BU2, stencil_idx_0_0_0_VVV)); // x702: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__0_]
        store(shift_B_rhsU2, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU2, stencil_idx_0_0_0_VVV) + (x283 * ((DXI * (x702 + (-56 * ((x707 + x708))) + (-8 * ((x703 + x704))) + (28 * ((x705 + x706))) + ((stencil(shift_BU2, stencil_idx_m4_0_0_VVV) + stencil(shift_BU2, stencil_idx_4_0_0_VVV))))) + (DYI * (x702 + (-56 * ((x713 + x714))) + (-8 * ((x709 + x710))) + (28 * ((x711 + x712))) + ((stencil(shift_BU2, stencil_idx_0_m4_0_VVV) + stencil(shift_BU2, stencil_idx_0_4_0_VVV))))) + (DZI * (x702 + (-56 * ((x719 + x720))) + (-8 * ((x715 + x716))) + (28 * ((x717 + x718))) + ((stencil(shift_BU2, stencil_idx_0_0_m4_VVV) + stencil(shift_BU2, stencil_idx_0_0_4_VVV)))))))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}