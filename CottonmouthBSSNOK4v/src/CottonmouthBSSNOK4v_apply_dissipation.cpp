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
    CCTK_ASSERT((cctk_nghostzones[0] >= 3));
    CCTK_ASSERT((cctk_nghostzones[1] >= 3));
    CCTK_ASSERT((cctk_nghostzones[2] >= 3));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // apply_dissipation loop 0
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
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        vreal x166 = stencil(gtDD00, stencil_idx_m2_0_0_VVV); // x166: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m2__0__0_]
        vreal x167 = stencil(gtDD00, stencil_idx_2_0_0_VVV); // x167: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__2__0__0_]
        vreal x238 = ((x166 + x167)); // x238: Dependency! Liveness = 2; [x166, x167]
        vreal x168 = stencil(gtDD00, stencil_idx_m1_0_0_VVV); // x168: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m1__0__0_]
        vreal x169 = stencil(gtDD00, stencil_idx_1_0_0_VVV); // x169: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__1__0__0_]
        vreal x239 = ((x168 + x169)); // x239: Dependency! Liveness = 2; [x168, x169]
        vreal x104 = stencil(gtDD00, stencil_idx_0_m2_0_VVV); // x104: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m2__0_]
        vreal x105 = stencil(gtDD00, stencil_idx_0_2_0_VVV); // x105: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__2__0_]
        vreal x240 = ((x104 + x105)); // x240: Dependency! Liveness = 2; [x104, x105]
        vreal x106 = stencil(gtDD00, stencil_idx_0_m1_0_VVV); // x106: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m1__0_]
        vreal x107 = stencil(gtDD00, stencil_idx_0_1_0_VVV); // x107: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__1__0_]
        vreal x241 = ((x106 + x107)); // x241: Dependency! Liveness = 2; [x106, x107]
        vreal x93 = stencil(gtDD00, stencil_idx_0_0_m2_VVV); // x93: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m2_]
        vreal x94 = stencil(gtDD00, stencil_idx_0_0_2_VVV); // x94: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__2_]
        vreal x242 = ((x93 + x94)); // x242: Dependency! Liveness = 2; [x93, x94]
        vreal x95 = stencil(gtDD00, stencil_idx_0_0_m1_VVV); // x95: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m1_]
        vreal x96 = stencil(gtDD00, stencil_idx_0_0_1_VVV); // x96: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__1_]
        vreal x243 = ((x95 + x96)); // x243: Dependency! Liveness = 2; [x95, x96]
        vreal x237 = (-20 * stencil(gtDD00, stencil_idx_0_0_0_VVV)); // x237: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__0_]
        vreal x244 = ((1.0 / 64.0) * dissipation_epsilon); // x244: Dependency! Liveness = 1; [dissipation_epsilon]
        store(gt_rhsDD00, stencil_idx_0_0_0_VVV, (access(gt_rhsDD00, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x237 + (-6 * x238) + (15 * x239) + ((stencil(gtDD00, stencil_idx_m3_0_0_VVV) + stencil(gtDD00, stencil_idx_3_0_0_VVV))))) + (DYI * (x237 + (-6 * x240) + (15 * x241) + ((stencil(gtDD00, stencil_idx_0_m3_0_VVV) + stencil(gtDD00, stencil_idx_0_3_0_VVV))))) + (DZI * (x237 + (-6 * x242) + (15 * x243) + ((stencil(gtDD00, stencil_idx_0_0_m3_VVV) + stencil(gtDD00, stencil_idx_0_0_3_VVV)))))))));
        vreal x179 = stencil(gtDD01, stencil_idx_m2_0_0_VVV); // x179: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m2__0__0_]
        vreal x180 = stencil(gtDD01, stencil_idx_2_0_0_VVV); // x180: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__2__0__0_]
        vreal x246 = ((x179 + x180)); // x246: Dependency! Liveness = 2; [x179, x180]
        vreal x181 = stencil(gtDD01, stencil_idx_m1_0_0_VVV); // x181: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m1__0__0_]
        vreal x182 = stencil(gtDD01, stencil_idx_1_0_0_VVV); // x182: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__1__0__0_]
        vreal x247 = ((x181 + x182)); // x247: Dependency! Liveness = 2; [x181, x182]
        vreal x148 = stencil(gtDD01, stencil_idx_0_m2_0_VVV); // x148: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m2__0_]
        vreal x149 = stencil(gtDD01, stencil_idx_0_2_0_VVV); // x149: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__2__0_]
        vreal x248 = ((x148 + x149)); // x248: Dependency! Liveness = 2; [x148, x149]
        vreal x150 = stencil(gtDD01, stencil_idx_0_m1_0_VVV); // x150: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m1__0_]
        vreal x151 = stencil(gtDD01, stencil_idx_0_1_0_VVV); // x151: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__1__0_]
        vreal x249 = ((x150 + x151)); // x249: Dependency! Liveness = 2; [x150, x151]
        vreal x74 = stencil(gtDD01, stencil_idx_0_0_m2_VVV); // x74: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m2_]
        vreal x75 = stencil(gtDD01, stencil_idx_0_0_2_VVV); // x75: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__2_]
        vreal x250 = ((x74 + x75)); // x250: Dependency! Liveness = 2; [x74, x75]
        vreal x76 = stencil(gtDD01, stencil_idx_0_0_m1_VVV); // x76: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m1_]
        vreal x77 = stencil(gtDD01, stencil_idx_0_0_1_VVV); // x77: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__1_]
        vreal x251 = ((x76 + x77)); // x251: Dependency! Liveness = 2; [x76, x77]
        vreal x245 = (-20 * stencil(gtDD01, stencil_idx_0_0_0_VVV)); // x245: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__0_]
        store(gt_rhsDD01, stencil_idx_0_0_0_VVV, (access(gt_rhsDD01, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x245 + (-6 * x246) + (15 * x247) + ((stencil(gtDD01, stencil_idx_m3_0_0_VVV) + stencil(gtDD01, stencil_idx_3_0_0_VVV))))) + (DYI * (x245 + (-6 * x248) + (15 * x249) + ((stencil(gtDD01, stencil_idx_0_m3_0_VVV) + stencil(gtDD01, stencil_idx_0_3_0_VVV))))) + (DZI * (x245 + (-6 * x250) + (15 * x251) + ((stencil(gtDD01, stencil_idx_0_0_m3_VVV) + stencil(gtDD01, stencil_idx_0_0_3_VVV)))))))));
        vreal x172 = stencil(gtDD02, stencil_idx_m2_0_0_VVV); // x172: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m2__0__0_]
        vreal x173 = stencil(gtDD02, stencil_idx_2_0_0_VVV); // x173: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__2__0__0_]
        vreal x253 = ((x172 + x173)); // x253: Dependency! Liveness = 2; [x172, x173]
        vreal x174 = stencil(gtDD02, stencil_idx_m1_0_0_VVV); // x174: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m1__0__0_]
        vreal x175 = stencil(gtDD02, stencil_idx_1_0_0_VVV); // x175: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__1__0__0_]
        vreal x254 = ((x174 + x175)); // x254: Dependency! Liveness = 2; [x174, x175]
        vreal x69 = stencil(gtDD02, stencil_idx_0_m2_0_VVV); // x69: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m2__0_]
        vreal x70 = stencil(gtDD02, stencil_idx_0_2_0_VVV); // x70: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__2__0_]
        vreal x255 = ((x69 + x70)); // x255: Dependency! Liveness = 2; [x69, x70]
        vreal x71 = stencil(gtDD02, stencil_idx_0_m1_0_VVV); // x71: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m1__0_]
        vreal x72 = stencil(gtDD02, stencil_idx_0_1_0_VVV); // x72: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__1__0_]
        vreal x256 = ((x71 + x72)); // x256: Dependency! Liveness = 2; [x71, x72]
        vreal x134 = stencil(gtDD02, stencil_idx_0_0_m2_VVV); // x134: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m2_]
        vreal x135 = stencil(gtDD02, stencil_idx_0_0_2_VVV); // x135: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__2_]
        vreal x257 = ((x134 + x135)); // x257: Dependency! Liveness = 2; [x134, x135]
        vreal x136 = stencil(gtDD02, stencil_idx_0_0_m1_VVV); // x136: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m1_]
        vreal x137 = stencil(gtDD02, stencil_idx_0_0_1_VVV); // x137: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__1_]
        vreal x258 = ((x136 + x137)); // x258: Dependency! Liveness = 2; [x136, x137]
        vreal x252 = (-20 * stencil(gtDD02, stencil_idx_0_0_0_VVV)); // x252: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__0_]
        store(gt_rhsDD02, stencil_idx_0_0_0_VVV, (access(gt_rhsDD02, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x252 + (-6 * x253) + (15 * x254) + ((stencil(gtDD02, stencil_idx_m3_0_0_VVV) + stencil(gtDD02, stencil_idx_3_0_0_VVV))))) + (DYI * (x252 + (-6 * x255) + (15 * x256) + ((stencil(gtDD02, stencil_idx_0_m3_0_VVV) + stencil(gtDD02, stencil_idx_0_3_0_VVV))))) + (DZI * (x252 + (-6 * x257) + (15 * x258) + ((stencil(gtDD02, stencil_idx_0_0_m3_VVV) + stencil(gtDD02, stencil_idx_0_0_3_VVV)))))))));
        vreal x110 = stencil(gtDD11, stencil_idx_m2_0_0_VVV); // x110: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m2__0__0_]
        vreal x111 = stencil(gtDD11, stencil_idx_2_0_0_VVV); // x111: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__2__0__0_]
        vreal x260 = ((x110 + x111)); // x260: Dependency! Liveness = 2; [x110, x111]
        vreal x112 = stencil(gtDD11, stencil_idx_m1_0_0_VVV); // x112: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m1__0__0_]
        vreal x113 = stencil(gtDD11, stencil_idx_1_0_0_VVV); // x113: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__1__0__0_]
        vreal x261 = ((x112 + x113)); // x261: Dependency! Liveness = 2; [x112, x113]
        vreal x142 = stencil(gtDD11, stencil_idx_0_m2_0_VVV); // x142: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m2__0_]
        vreal x143 = stencil(gtDD11, stencil_idx_0_2_0_VVV); // x143: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__2__0_]
        vreal x262 = ((x142 + x143)); // x262: Dependency! Liveness = 2; [x142, x143]
        vreal x144 = stencil(gtDD11, stencil_idx_0_m1_0_VVV); // x144: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m1__0_]
        vreal x145 = stencil(gtDD11, stencil_idx_0_1_0_VVV); // x145: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__1__0_]
        vreal x263 = ((x144 + x145)); // x263: Dependency! Liveness = 2; [x144, x145]
        vreal x62 = stencil(gtDD11, stencil_idx_0_0_m2_VVV); // x62: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m2_]
        vreal x63 = stencil(gtDD11, stencil_idx_0_0_2_VVV); // x63: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__2_]
        vreal x264 = ((x62 + x63)); // x264: Dependency! Liveness = 2; [x62, x63]
        vreal x64 = stencil(gtDD11, stencil_idx_0_0_m1_VVV); // x64: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m1_]
        vreal x65 = stencil(gtDD11, stencil_idx_0_0_1_VVV); // x65: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__1_]
        vreal x265 = ((x64 + x65)); // x265: Dependency! Liveness = 2; [x64, x65]
        vreal x259 = (-20 * stencil(gtDD11, stencil_idx_0_0_0_VVV)); // x259: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__0_]
        store(gt_rhsDD11, stencil_idx_0_0_0_VVV, (access(gt_rhsDD11, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x259 + (-6 * x260) + (15 * x261) + ((stencil(gtDD11, stencil_idx_m3_0_0_VVV) + stencil(gtDD11, stencil_idx_3_0_0_VVV))))) + (DYI * (x259 + (-6 * x262) + (15 * x263) + ((stencil(gtDD11, stencil_idx_0_m3_0_VVV) + stencil(gtDD11, stencil_idx_0_3_0_VVV))))) + (DZI * (x259 + (-6 * x264) + (15 * x265) + ((stencil(gtDD11, stencil_idx_0_0_m3_VVV) + stencil(gtDD11, stencil_idx_0_0_3_VVV)))))))));
        vreal x79 = stencil(gtDD12, stencil_idx_m2_0_0_VVV); // x79: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m2__0__0_]
        vreal x80 = stencil(gtDD12, stencil_idx_2_0_0_VVV); // x80: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__2__0__0_]
        vreal x267 = ((x79 + x80)); // x267: Dependency! Liveness = 2; [x79, x80]
        vreal x81 = stencil(gtDD12, stencil_idx_m1_0_0_VVV); // x81: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m1__0__0_]
        vreal x82 = stencil(gtDD12, stencil_idx_1_0_0_VVV); // x82: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__1__0__0_]
        vreal x268 = ((x81 + x82)); // x268: Dependency! Liveness = 2; [x81, x82]
        vreal x156 = stencil(gtDD12, stencil_idx_0_m2_0_VVV); // x156: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m2__0_]
        vreal x157 = stencil(gtDD12, stencil_idx_0_2_0_VVV); // x157: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__2__0_]
        vreal x269 = ((x156 + x157)); // x269: Dependency! Liveness = 2; [x156, x157]
        vreal x158 = stencil(gtDD12, stencil_idx_0_m1_0_VVV); // x158: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m1__0_]
        vreal x159 = stencil(gtDD12, stencil_idx_0_1_0_VVV); // x159: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__1__0_]
        vreal x270 = ((x158 + x159)); // x270: Dependency! Liveness = 2; [x158, x159]
        vreal x127 = stencil(gtDD12, stencil_idx_0_0_m2_VVV); // x127: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m2_]
        vreal x128 = stencil(gtDD12, stencil_idx_0_0_2_VVV); // x128: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__2_]
        vreal x271 = ((x127 + x128)); // x271: Dependency! Liveness = 2; [x127, x128]
        vreal x129 = stencil(gtDD12, stencil_idx_0_0_m1_VVV); // x129: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m1_]
        vreal x130 = stencil(gtDD12, stencil_idx_0_0_1_VVV); // x130: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__1_]
        vreal x272 = ((x129 + x130)); // x272: Dependency! Liveness = 2; [x129, x130]
        vreal x266 = (-20 * stencil(gtDD12, stencil_idx_0_0_0_VVV)); // x266: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__0_]
        store(gt_rhsDD12, stencil_idx_0_0_0_VVV, (access(gt_rhsDD12, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x266 + (-6 * x267) + (15 * x268) + ((stencil(gtDD12, stencil_idx_m3_0_0_VVV) + stencil(gtDD12, stencil_idx_3_0_0_VVV))))) + (DYI * (x266 + (-6 * x269) + (15 * x270) + ((stencil(gtDD12, stencil_idx_0_m3_0_VVV) + stencil(gtDD12, stencil_idx_0_3_0_VVV))))) + (DZI * (x266 + (-6 * x271) + (15 * x272) + ((stencil(gtDD12, stencil_idx_0_0_m3_VVV) + stencil(gtDD12, stencil_idx_0_0_3_VVV)))))))));
        vreal x87 = stencil(gtDD22, stencil_idx_m2_0_0_VVV); // x87: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m2__0__0_]
        vreal x88 = stencil(gtDD22, stencil_idx_2_0_0_VVV); // x88: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__2__0__0_]
        vreal x274 = ((x87 + x88)); // x274: Dependency! Liveness = 2; [x87, x88]
        vreal x89 = stencil(gtDD22, stencil_idx_m1_0_0_VVV); // x89: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m1__0__0_]
        vreal x90 = stencil(gtDD22, stencil_idx_1_0_0_VVV); // x90: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__1__0__0_]
        vreal x275 = ((x89 + x90)); // x275: Dependency! Liveness = 2; [x89, x90]
        vreal x53 = stencil(gtDD22, stencil_idx_0_m2_0_VVV); // x53: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m2__0_]
        vreal x54 = stencil(gtDD22, stencil_idx_0_2_0_VVV); // x54: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__2__0_]
        vreal x276 = ((x53 + x54)); // x276: Dependency! Liveness = 2; [x53, x54]
        vreal x55 = stencil(gtDD22, stencil_idx_0_m1_0_VVV); // x55: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m1__0_]
        vreal x56 = stencil(gtDD22, stencil_idx_0_1_0_VVV); // x56: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__1__0_]
        vreal x277 = ((x55 + x56)); // x277: Dependency! Liveness = 2; [x55, x56]
        vreal x121 = stencil(gtDD22, stencil_idx_0_0_m2_VVV); // x121: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m2_]
        vreal x122 = stencil(gtDD22, stencil_idx_0_0_2_VVV); // x122: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__2_]
        vreal x278 = ((x121 + x122)); // x278: Dependency! Liveness = 2; [x121, x122]
        vreal x123 = stencil(gtDD22, stencil_idx_0_0_m1_VVV); // x123: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m1_]
        vreal x124 = stencil(gtDD22, stencil_idx_0_0_1_VVV); // x124: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__1_]
        vreal x279 = ((x123 + x124)); // x279: Dependency! Liveness = 2; [x123, x124]
        vreal x273 = (-20 * stencil(gtDD22, stencil_idx_0_0_0_VVV)); // x273: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__0_]
        store(gt_rhsDD22, stencil_idx_0_0_0_VVV, (access(gt_rhsDD22, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x273 + (-6 * x274) + (15 * x275) + ((stencil(gtDD22, stencil_idx_m3_0_0_VVV) + stencil(gtDD22, stencil_idx_3_0_0_VVV))))) + (DYI * (x273 + (-6 * x276) + (15 * x277) + ((stencil(gtDD22, stencil_idx_0_m3_0_VVV) + stencil(gtDD22, stencil_idx_0_3_0_VVV))))) + (DZI * (x273 + (-6 * x278) + (15 * x279) + ((stencil(gtDD22, stencil_idx_0_0_m3_VVV) + stencil(gtDD22, stencil_idx_0_0_3_VVV)))))))));    
    });
    // apply_dissipation loop 1
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
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        vreal x281 = stencil(w, stencil_idx_m2_0_0_VVV); // x281: Dependency! Liveness = 1; [__dummy_stencil__w__m2__0__0_]
        vreal x282 = stencil(w, stencil_idx_2_0_0_VVV); // x282: Dependency! Liveness = 1; [__dummy_stencil__w__2__0__0_]
        vreal x283 = ((x281 + x282)); // x283: Dependency! Liveness = 2; [x281, x282]
        vreal x284 = stencil(w, stencil_idx_m1_0_0_VVV); // x284: Dependency! Liveness = 1; [__dummy_stencil__w__m1__0__0_]
        vreal x285 = stencil(w, stencil_idx_1_0_0_VVV); // x285: Dependency! Liveness = 1; [__dummy_stencil__w__1__0__0_]
        vreal x286 = ((x284 + x285)); // x286: Dependency! Liveness = 2; [x284, x285]
        vreal x287 = stencil(w, stencil_idx_0_m2_0_VVV); // x287: Dependency! Liveness = 1; [__dummy_stencil__w__0__m2__0_]
        vreal x288 = stencil(w, stencil_idx_0_2_0_VVV); // x288: Dependency! Liveness = 1; [__dummy_stencil__w__0__2__0_]
        vreal x289 = ((x287 + x288)); // x289: Dependency! Liveness = 2; [x287, x288]
        vreal x290 = stencil(w, stencil_idx_0_m1_0_VVV); // x290: Dependency! Liveness = 1; [__dummy_stencil__w__0__m1__0_]
        vreal x291 = stencil(w, stencil_idx_0_1_0_VVV); // x291: Dependency! Liveness = 1; [__dummy_stencil__w__0__1__0_]
        vreal x292 = ((x290 + x291)); // x292: Dependency! Liveness = 2; [x290, x291]
        vreal x293 = stencil(w, stencil_idx_0_0_m2_VVV); // x293: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__m2_]
        vreal x294 = stencil(w, stencil_idx_0_0_2_VVV); // x294: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__2_]
        vreal x295 = ((x293 + x294)); // x295: Dependency! Liveness = 2; [x293, x294]
        vreal x296 = stencil(w, stencil_idx_0_0_m1_VVV); // x296: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__m1_]
        vreal x297 = stencil(w, stencil_idx_0_0_1_VVV); // x297: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__1_]
        vreal x298 = ((x296 + x297)); // x298: Dependency! Liveness = 2; [x296, x297]
        vreal x280 = (-20 * stencil(w, stencil_idx_0_0_0_VVV)); // x280: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__0_]
        vreal x244 = ((1.0 / 64.0) * dissipation_epsilon); // x244: Dependency! Liveness = 1; [dissipation_epsilon]
        store(w_rhs, stencil_idx_0_0_0_VVV, (access(w_rhs, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x280 + (-6 * x283) + (15 * x286) + ((stencil(w, stencil_idx_m3_0_0_VVV) + stencil(w, stencil_idx_3_0_0_VVV))))) + (DYI * (x280 + (-6 * x289) + (15 * x292) + ((stencil(w, stencil_idx_0_m3_0_VVV) + stencil(w, stencil_idx_0_3_0_VVV))))) + (DZI * (x280 + (-6 * x295) + (15 * x298) + ((stencil(w, stencil_idx_0_0_m3_VVV) + stencil(w, stencil_idx_0_0_3_VVV)))))))));    
    });
    // apply_dissipation loop 2
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
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        vreal x304 = stencil(trK, stencil_idx_0_m2_0_VVV); // x304: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m2__0_]
        vreal x305 = stencil(trK, stencil_idx_0_2_0_VVV); // x305: Dependency! Liveness = 1; [__dummy_stencil__trK__0__2__0_]
        vreal x306 = stencil(trK, stencil_idx_0_m1_0_VVV); // x306: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m1__0_]
        vreal x307 = stencil(trK, stencil_idx_0_1_0_VVV); // x307: Dependency! Liveness = 1; [__dummy_stencil__trK__0__1__0_]
        vreal x308 = stencil(trK, stencil_idx_0_0_m2_VVV); // x308: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m2_]
        vreal x309 = stencil(trK, stencil_idx_0_0_2_VVV); // x309: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__2_]
        vreal x310 = stencil(trK, stencil_idx_0_0_m1_VVV); // x310: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m1_]
        vreal x311 = stencil(trK, stencil_idx_0_0_1_VVV); // x311: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__1_]
        vreal x300 = stencil(trK, stencil_idx_m2_0_0_VVV); // x300: Dependency! Liveness = 1; [__dummy_stencil__trK__m2__0__0_]
        vreal x301 = stencil(trK, stencil_idx_2_0_0_VVV); // x301: Dependency! Liveness = 1; [__dummy_stencil__trK__2__0__0_]
        vreal x302 = stencil(trK, stencil_idx_m1_0_0_VVV); // x302: Dependency! Liveness = 1; [__dummy_stencil__trK__m1__0__0_]
        vreal x303 = stencil(trK, stencil_idx_1_0_0_VVV); // x303: Dependency! Liveness = 1; [__dummy_stencil__trK__1__0__0_]
        vreal x299 = (-20 * stencil(trK, stencil_idx_0_0_0_VVV)); // x299: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__0_]
        vreal x244 = ((1.0 / 64.0) * dissipation_epsilon); // x244: Dependency! Liveness = 1; [dissipation_epsilon]
        store(trK_rhs, stencil_idx_0_0_0_VVV, (access(trK_rhs, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x299 + (-6 * ((x300 + x301))) + (15 * ((x302 + x303))) + ((stencil(trK, stencil_idx_m3_0_0_VVV) + stencil(trK, stencil_idx_3_0_0_VVV))))) + (DYI * (x299 + (-6 * ((x304 + x305))) + (15 * ((x306 + x307))) + ((stencil(trK, stencil_idx_0_m3_0_VVV) + stencil(trK, stencil_idx_0_3_0_VVV))))) + (DZI * (x299 + (-6 * ((x308 + x309))) + (15 * ((x310 + x311))) + ((stencil(trK, stencil_idx_0_0_m3_VVV) + stencil(trK, stencil_idx_0_0_3_VVV)))))))));    
    });
    // apply_dissipation loop 3
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
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        vreal x317 = stencil(AtDD00, stencil_idx_0_m2_0_VVV); // x317: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m2__0_]
        vreal x318 = stencil(AtDD00, stencil_idx_0_2_0_VVV); // x318: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__2__0_]
        vreal x319 = stencil(AtDD00, stencil_idx_0_m1_0_VVV); // x319: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m1__0_]
        vreal x320 = stencil(AtDD00, stencil_idx_0_1_0_VVV); // x320: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__1__0_]
        vreal x321 = stencil(AtDD00, stencil_idx_0_0_m2_VVV); // x321: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m2_]
        vreal x322 = stencil(AtDD00, stencil_idx_0_0_2_VVV); // x322: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__2_]
        vreal x323 = stencil(AtDD00, stencil_idx_0_0_m1_VVV); // x323: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m1_]
        vreal x324 = stencil(AtDD00, stencil_idx_0_0_1_VVV); // x324: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__1_]
        vreal x313 = stencil(AtDD00, stencil_idx_m2_0_0_VVV); // x313: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m2__0__0_]
        vreal x314 = stencil(AtDD00, stencil_idx_2_0_0_VVV); // x314: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__2__0__0_]
        vreal x315 = stencil(AtDD00, stencil_idx_m1_0_0_VVV); // x315: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m1__0__0_]
        vreal x316 = stencil(AtDD00, stencil_idx_1_0_0_VVV); // x316: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__1__0__0_]
        vreal x312 = (-20 * stencil(AtDD00, stencil_idx_0_0_0_VVV)); // x312: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__0_]
        vreal x244 = ((1.0 / 64.0) * dissipation_epsilon); // x244: Dependency! Liveness = 1; [dissipation_epsilon]
        store(At_rhsDD00, stencil_idx_0_0_0_VVV, (access(At_rhsDD00, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x312 + (-6 * ((x313 + x314))) + (15 * ((x315 + x316))) + ((stencil(AtDD00, stencil_idx_m3_0_0_VVV) + stencil(AtDD00, stencil_idx_3_0_0_VVV))))) + (DYI * (x312 + (-6 * ((x317 + x318))) + (15 * ((x319 + x320))) + ((stencil(AtDD00, stencil_idx_0_m3_0_VVV) + stencil(AtDD00, stencil_idx_0_3_0_VVV))))) + (DZI * (x312 + (-6 * ((x321 + x322))) + (15 * ((x323 + x324))) + ((stencil(AtDD00, stencil_idx_0_0_m3_VVV) + stencil(AtDD00, stencil_idx_0_0_3_VVV)))))))));
        vreal x330 = stencil(AtDD01, stencil_idx_0_m2_0_VVV); // x330: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m2__0_]
        vreal x331 = stencil(AtDD01, stencil_idx_0_2_0_VVV); // x331: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__2__0_]
        vreal x332 = stencil(AtDD01, stencil_idx_0_m1_0_VVV); // x332: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m1__0_]
        vreal x333 = stencil(AtDD01, stencil_idx_0_1_0_VVV); // x333: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__1__0_]
        vreal x334 = stencil(AtDD01, stencil_idx_0_0_m2_VVV); // x334: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m2_]
        vreal x335 = stencil(AtDD01, stencil_idx_0_0_2_VVV); // x335: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__2_]
        vreal x336 = stencil(AtDD01, stencil_idx_0_0_m1_VVV); // x336: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m1_]
        vreal x337 = stencil(AtDD01, stencil_idx_0_0_1_VVV); // x337: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__1_]
        vreal x326 = stencil(AtDD01, stencil_idx_m2_0_0_VVV); // x326: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m2__0__0_]
        vreal x327 = stencil(AtDD01, stencil_idx_2_0_0_VVV); // x327: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__2__0__0_]
        vreal x328 = stencil(AtDD01, stencil_idx_m1_0_0_VVV); // x328: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m1__0__0_]
        vreal x329 = stencil(AtDD01, stencil_idx_1_0_0_VVV); // x329: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__1__0__0_]
        vreal x325 = (-20 * stencil(AtDD01, stencil_idx_0_0_0_VVV)); // x325: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__0_]
        store(At_rhsDD01, stencil_idx_0_0_0_VVV, (access(At_rhsDD01, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x325 + (-6 * ((x326 + x327))) + (15 * ((x328 + x329))) + ((stencil(AtDD01, stencil_idx_m3_0_0_VVV) + stencil(AtDD01, stencil_idx_3_0_0_VVV))))) + (DYI * (x325 + (-6 * ((x330 + x331))) + (15 * ((x332 + x333))) + ((stencil(AtDD01, stencil_idx_0_m3_0_VVV) + stencil(AtDD01, stencil_idx_0_3_0_VVV))))) + (DZI * (x325 + (-6 * ((x334 + x335))) + (15 * ((x336 + x337))) + ((stencil(AtDD01, stencil_idx_0_0_m3_VVV) + stencil(AtDD01, stencil_idx_0_0_3_VVV)))))))));
        vreal x343 = stencil(AtDD02, stencil_idx_0_m2_0_VVV); // x343: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m2__0_]
        vreal x344 = stencil(AtDD02, stencil_idx_0_2_0_VVV); // x344: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__2__0_]
        vreal x345 = stencil(AtDD02, stencil_idx_0_m1_0_VVV); // x345: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m1__0_]
        vreal x346 = stencil(AtDD02, stencil_idx_0_1_0_VVV); // x346: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__1__0_]
        vreal x347 = stencil(AtDD02, stencil_idx_0_0_m2_VVV); // x347: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m2_]
        vreal x348 = stencil(AtDD02, stencil_idx_0_0_2_VVV); // x348: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__2_]
        vreal x349 = stencil(AtDD02, stencil_idx_0_0_m1_VVV); // x349: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m1_]
        vreal x350 = stencil(AtDD02, stencil_idx_0_0_1_VVV); // x350: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__1_]
        vreal x339 = stencil(AtDD02, stencil_idx_m2_0_0_VVV); // x339: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m2__0__0_]
        vreal x340 = stencil(AtDD02, stencil_idx_2_0_0_VVV); // x340: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__2__0__0_]
        vreal x341 = stencil(AtDD02, stencil_idx_m1_0_0_VVV); // x341: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m1__0__0_]
        vreal x342 = stencil(AtDD02, stencil_idx_1_0_0_VVV); // x342: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__1__0__0_]
        vreal x338 = (-20 * stencil(AtDD02, stencil_idx_0_0_0_VVV)); // x338: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__0_]
        store(At_rhsDD02, stencil_idx_0_0_0_VVV, (access(At_rhsDD02, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x338 + (-6 * ((x339 + x340))) + (15 * ((x341 + x342))) + ((stencil(AtDD02, stencil_idx_m3_0_0_VVV) + stencil(AtDD02, stencil_idx_3_0_0_VVV))))) + (DYI * (x338 + (-6 * ((x343 + x344))) + (15 * ((x345 + x346))) + ((stencil(AtDD02, stencil_idx_0_m3_0_VVV) + stencil(AtDD02, stencil_idx_0_3_0_VVV))))) + (DZI * (x338 + (-6 * ((x347 + x348))) + (15 * ((x349 + x350))) + ((stencil(AtDD02, stencil_idx_0_0_m3_VVV) + stencil(AtDD02, stencil_idx_0_0_3_VVV)))))))));
        vreal x356 = stencil(AtDD11, stencil_idx_0_m2_0_VVV); // x356: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m2__0_]
        vreal x357 = stencil(AtDD11, stencil_idx_0_2_0_VVV); // x357: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__2__0_]
        vreal x358 = stencil(AtDD11, stencil_idx_0_m1_0_VVV); // x358: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m1__0_]
        vreal x359 = stencil(AtDD11, stencil_idx_0_1_0_VVV); // x359: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__1__0_]
        vreal x360 = stencil(AtDD11, stencil_idx_0_0_m2_VVV); // x360: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m2_]
        vreal x361 = stencil(AtDD11, stencil_idx_0_0_2_VVV); // x361: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__2_]
        vreal x362 = stencil(AtDD11, stencil_idx_0_0_m1_VVV); // x362: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m1_]
        vreal x363 = stencil(AtDD11, stencil_idx_0_0_1_VVV); // x363: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__1_]
        vreal x352 = stencil(AtDD11, stencil_idx_m2_0_0_VVV); // x352: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m2__0__0_]
        vreal x353 = stencil(AtDD11, stencil_idx_2_0_0_VVV); // x353: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__2__0__0_]
        vreal x354 = stencil(AtDD11, stencil_idx_m1_0_0_VVV); // x354: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m1__0__0_]
        vreal x355 = stencil(AtDD11, stencil_idx_1_0_0_VVV); // x355: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__1__0__0_]
        vreal x351 = (-20 * stencil(AtDD11, stencil_idx_0_0_0_VVV)); // x351: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__0_]
        store(At_rhsDD11, stencil_idx_0_0_0_VVV, (access(At_rhsDD11, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x351 + (-6 * ((x352 + x353))) + (15 * ((x354 + x355))) + ((stencil(AtDD11, stencil_idx_m3_0_0_VVV) + stencil(AtDD11, stencil_idx_3_0_0_VVV))))) + (DYI * (x351 + (-6 * ((x356 + x357))) + (15 * ((x358 + x359))) + ((stencil(AtDD11, stencil_idx_0_m3_0_VVV) + stencil(AtDD11, stencil_idx_0_3_0_VVV))))) + (DZI * (x351 + (-6 * ((x360 + x361))) + (15 * ((x362 + x363))) + ((stencil(AtDD11, stencil_idx_0_0_m3_VVV) + stencil(AtDD11, stencil_idx_0_0_3_VVV)))))))));
        vreal x369 = stencil(AtDD12, stencil_idx_0_m2_0_VVV); // x369: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m2__0_]
        vreal x370 = stencil(AtDD12, stencil_idx_0_2_0_VVV); // x370: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__2__0_]
        vreal x371 = stencil(AtDD12, stencil_idx_0_m1_0_VVV); // x371: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m1__0_]
        vreal x372 = stencil(AtDD12, stencil_idx_0_1_0_VVV); // x372: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__1__0_]
        vreal x373 = stencil(AtDD12, stencil_idx_0_0_m2_VVV); // x373: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m2_]
        vreal x374 = stencil(AtDD12, stencil_idx_0_0_2_VVV); // x374: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__2_]
        vreal x375 = stencil(AtDD12, stencil_idx_0_0_m1_VVV); // x375: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m1_]
        vreal x376 = stencil(AtDD12, stencil_idx_0_0_1_VVV); // x376: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__1_]
        vreal x365 = stencil(AtDD12, stencil_idx_m2_0_0_VVV); // x365: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m2__0__0_]
        vreal x366 = stencil(AtDD12, stencil_idx_2_0_0_VVV); // x366: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__2__0__0_]
        vreal x367 = stencil(AtDD12, stencil_idx_m1_0_0_VVV); // x367: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m1__0__0_]
        vreal x368 = stencil(AtDD12, stencil_idx_1_0_0_VVV); // x368: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__1__0__0_]
        vreal x364 = (-20 * stencil(AtDD12, stencil_idx_0_0_0_VVV)); // x364: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__0_]
        store(At_rhsDD12, stencil_idx_0_0_0_VVV, (access(At_rhsDD12, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x364 + (-6 * ((x365 + x366))) + (15 * ((x367 + x368))) + ((stencil(AtDD12, stencil_idx_m3_0_0_VVV) + stencil(AtDD12, stencil_idx_3_0_0_VVV))))) + (DYI * (x364 + (-6 * ((x369 + x370))) + (15 * ((x371 + x372))) + ((stencil(AtDD12, stencil_idx_0_m3_0_VVV) + stencil(AtDD12, stencil_idx_0_3_0_VVV))))) + (DZI * (x364 + (-6 * ((x373 + x374))) + (15 * ((x375 + x376))) + ((stencil(AtDD12, stencil_idx_0_0_m3_VVV) + stencil(AtDD12, stencil_idx_0_0_3_VVV)))))))));
        vreal x382 = stencil(AtDD22, stencil_idx_0_m2_0_VVV); // x382: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m2__0_]
        vreal x383 = stencil(AtDD22, stencil_idx_0_2_0_VVV); // x383: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__2__0_]
        vreal x384 = stencil(AtDD22, stencil_idx_0_m1_0_VVV); // x384: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m1__0_]
        vreal x385 = stencil(AtDD22, stencil_idx_0_1_0_VVV); // x385: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__1__0_]
        vreal x386 = stencil(AtDD22, stencil_idx_0_0_m2_VVV); // x386: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m2_]
        vreal x387 = stencil(AtDD22, stencil_idx_0_0_2_VVV); // x387: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__2_]
        vreal x388 = stencil(AtDD22, stencil_idx_0_0_m1_VVV); // x388: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m1_]
        vreal x389 = stencil(AtDD22, stencil_idx_0_0_1_VVV); // x389: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__1_]
        vreal x378 = stencil(AtDD22, stencil_idx_m2_0_0_VVV); // x378: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m2__0__0_]
        vreal x379 = stencil(AtDD22, stencil_idx_2_0_0_VVV); // x379: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__2__0__0_]
        vreal x380 = stencil(AtDD22, stencil_idx_m1_0_0_VVV); // x380: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m1__0__0_]
        vreal x381 = stencil(AtDD22, stencil_idx_1_0_0_VVV); // x381: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__1__0__0_]
        vreal x377 = (-20 * stencil(AtDD22, stencil_idx_0_0_0_VVV)); // x377: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__0_]
        store(At_rhsDD22, stencil_idx_0_0_0_VVV, (access(At_rhsDD22, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x377 + (-6 * ((x378 + x379))) + (15 * ((x380 + x381))) + ((stencil(AtDD22, stencil_idx_m3_0_0_VVV) + stencil(AtDD22, stencil_idx_3_0_0_VVV))))) + (DYI * (x377 + (-6 * ((x382 + x383))) + (15 * ((x384 + x385))) + ((stencil(AtDD22, stencil_idx_0_m3_0_VVV) + stencil(AtDD22, stencil_idx_0_3_0_VVV))))) + (DZI * (x377 + (-6 * ((x386 + x387))) + (15 * ((x388 + x389))) + ((stencil(AtDD22, stencil_idx_0_0_m3_VVV) + stencil(AtDD22, stencil_idx_0_0_3_VVV)))))))));    
    });
    // apply_dissipation loop 4
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
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        vreal x395 = stencil(ConfConnectU0, stencil_idx_0_m2_0_VVV); // x395: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__m2__0_]
        vreal x396 = stencil(ConfConnectU0, stencil_idx_0_2_0_VVV); // x396: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__2__0_]
        vreal x397 = stencil(ConfConnectU0, stencil_idx_0_m1_0_VVV); // x397: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__m1__0_]
        vreal x398 = stencil(ConfConnectU0, stencil_idx_0_1_0_VVV); // x398: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__1__0_]
        vreal x399 = stencil(ConfConnectU0, stencil_idx_0_0_m2_VVV); // x399: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__m2_]
        vreal x400 = stencil(ConfConnectU0, stencil_idx_0_0_2_VVV); // x400: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__2_]
        vreal x401 = stencil(ConfConnectU0, stencil_idx_0_0_m1_VVV); // x401: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__m1_]
        vreal x402 = stencil(ConfConnectU0, stencil_idx_0_0_1_VVV); // x402: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__1_]
        vreal x391 = stencil(ConfConnectU0, stencil_idx_m2_0_0_VVV); // x391: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__m2__0__0_]
        vreal x392 = stencil(ConfConnectU0, stencil_idx_2_0_0_VVV); // x392: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__2__0__0_]
        vreal x393 = stencil(ConfConnectU0, stencil_idx_m1_0_0_VVV); // x393: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__m1__0__0_]
        vreal x394 = stencil(ConfConnectU0, stencil_idx_1_0_0_VVV); // x394: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__1__0__0_]
        vreal x390 = (-20 * stencil(ConfConnectU0, stencil_idx_0_0_0_VVV)); // x390: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__0_]
        vreal x244 = ((1.0 / 64.0) * dissipation_epsilon); // x244: Dependency! Liveness = 1; [dissipation_epsilon]
        store(ConfConnect_rhsU0, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU0, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x390 + (-6 * ((x391 + x392))) + (15 * ((x393 + x394))) + ((stencil(ConfConnectU0, stencil_idx_m3_0_0_VVV) + stencil(ConfConnectU0, stencil_idx_3_0_0_VVV))))) + (DYI * (x390 + (-6 * ((x395 + x396))) + (15 * ((x397 + x398))) + ((stencil(ConfConnectU0, stencil_idx_0_m3_0_VVV) + stencil(ConfConnectU0, stencil_idx_0_3_0_VVV))))) + (DZI * (x390 + (-6 * ((x399 + x400))) + (15 * ((x401 + x402))) + ((stencil(ConfConnectU0, stencil_idx_0_0_m3_VVV) + stencil(ConfConnectU0, stencil_idx_0_0_3_VVV)))))))));
        vreal x408 = stencil(ConfConnectU1, stencil_idx_0_m2_0_VVV); // x408: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__m2__0_]
        vreal x409 = stencil(ConfConnectU1, stencil_idx_0_2_0_VVV); // x409: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__2__0_]
        vreal x410 = stencil(ConfConnectU1, stencil_idx_0_m1_0_VVV); // x410: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__m1__0_]
        vreal x411 = stencil(ConfConnectU1, stencil_idx_0_1_0_VVV); // x411: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__1__0_]
        vreal x412 = stencil(ConfConnectU1, stencil_idx_0_0_m2_VVV); // x412: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__m2_]
        vreal x413 = stencil(ConfConnectU1, stencil_idx_0_0_2_VVV); // x413: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__2_]
        vreal x414 = stencil(ConfConnectU1, stencil_idx_0_0_m1_VVV); // x414: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__m1_]
        vreal x415 = stencil(ConfConnectU1, stencil_idx_0_0_1_VVV); // x415: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__1_]
        vreal x404 = stencil(ConfConnectU1, stencil_idx_m2_0_0_VVV); // x404: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__m2__0__0_]
        vreal x405 = stencil(ConfConnectU1, stencil_idx_2_0_0_VVV); // x405: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__2__0__0_]
        vreal x406 = stencil(ConfConnectU1, stencil_idx_m1_0_0_VVV); // x406: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__m1__0__0_]
        vreal x407 = stencil(ConfConnectU1, stencil_idx_1_0_0_VVV); // x407: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__1__0__0_]
        vreal x403 = (-20 * stencil(ConfConnectU1, stencil_idx_0_0_0_VVV)); // x403: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__0_]
        store(ConfConnect_rhsU1, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU1, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x403 + (-6 * ((x404 + x405))) + (15 * ((x406 + x407))) + ((stencil(ConfConnectU1, stencil_idx_m3_0_0_VVV) + stencil(ConfConnectU1, stencil_idx_3_0_0_VVV))))) + (DYI * (x403 + (-6 * ((x408 + x409))) + (15 * ((x410 + x411))) + ((stencil(ConfConnectU1, stencil_idx_0_m3_0_VVV) + stencil(ConfConnectU1, stencil_idx_0_3_0_VVV))))) + (DZI * (x403 + (-6 * ((x412 + x413))) + (15 * ((x414 + x415))) + ((stencil(ConfConnectU1, stencil_idx_0_0_m3_VVV) + stencil(ConfConnectU1, stencil_idx_0_0_3_VVV)))))))));
        vreal x421 = stencil(ConfConnectU2, stencil_idx_0_m2_0_VVV); // x421: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__m2__0_]
        vreal x422 = stencil(ConfConnectU2, stencil_idx_0_2_0_VVV); // x422: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__2__0_]
        vreal x423 = stencil(ConfConnectU2, stencil_idx_0_m1_0_VVV); // x423: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__m1__0_]
        vreal x424 = stencil(ConfConnectU2, stencil_idx_0_1_0_VVV); // x424: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__1__0_]
        vreal x425 = stencil(ConfConnectU2, stencil_idx_0_0_m2_VVV); // x425: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__m2_]
        vreal x426 = stencil(ConfConnectU2, stencil_idx_0_0_2_VVV); // x426: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__2_]
        vreal x427 = stencil(ConfConnectU2, stencil_idx_0_0_m1_VVV); // x427: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__m1_]
        vreal x428 = stencil(ConfConnectU2, stencil_idx_0_0_1_VVV); // x428: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__1_]
        vreal x417 = stencil(ConfConnectU2, stencil_idx_m2_0_0_VVV); // x417: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__m2__0__0_]
        vreal x418 = stencil(ConfConnectU2, stencil_idx_2_0_0_VVV); // x418: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__2__0__0_]
        vreal x419 = stencil(ConfConnectU2, stencil_idx_m1_0_0_VVV); // x419: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__m1__0__0_]
        vreal x420 = stencil(ConfConnectU2, stencil_idx_1_0_0_VVV); // x420: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__1__0__0_]
        vreal x416 = (-20 * stencil(ConfConnectU2, stencil_idx_0_0_0_VVV)); // x416: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__0_]
        store(ConfConnect_rhsU2, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU2, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x416 + (-6 * ((x417 + x418))) + (15 * ((x419 + x420))) + ((stencil(ConfConnectU2, stencil_idx_m3_0_0_VVV) + stencil(ConfConnectU2, stencil_idx_3_0_0_VVV))))) + (DYI * (x416 + (-6 * ((x421 + x422))) + (15 * ((x423 + x424))) + ((stencil(ConfConnectU2, stencil_idx_0_m3_0_VVV) + stencil(ConfConnectU2, stencil_idx_0_3_0_VVV))))) + (DZI * (x416 + (-6 * ((x425 + x426))) + (15 * ((x427 + x428))) + ((stencil(ConfConnectU2, stencil_idx_0_0_m3_VVV) + stencil(ConfConnectU2, stencil_idx_0_0_3_VVV)))))))));    
    });
    // apply_dissipation loop 5
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
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        vreal x430 = stencil(evo_lapse, stencil_idx_m2_0_0_VVV); // x430: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m2__0__0_]
        vreal x431 = stencil(evo_lapse, stencil_idx_2_0_0_VVV); // x431: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__2__0__0_]
        vreal x432 = ((x430 + x431)); // x432: Dependency! Liveness = 2; [x430, x431]
        vreal x433 = stencil(evo_lapse, stencil_idx_m1_0_0_VVV); // x433: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m1__0__0_]
        vreal x434 = stencil(evo_lapse, stencil_idx_1_0_0_VVV); // x434: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__1__0__0_]
        vreal x435 = ((x433 + x434)); // x435: Dependency! Liveness = 2; [x433, x434]
        vreal x436 = stencil(evo_lapse, stencil_idx_0_m2_0_VVV); // x436: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m2__0_]
        vreal x437 = stencil(evo_lapse, stencil_idx_0_2_0_VVV); // x437: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__2__0_]
        vreal x438 = ((x436 + x437)); // x438: Dependency! Liveness = 2; [x436, x437]
        vreal x439 = stencil(evo_lapse, stencil_idx_0_m1_0_VVV); // x439: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m1__0_]
        vreal x440 = stencil(evo_lapse, stencil_idx_0_1_0_VVV); // x440: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__1__0_]
        vreal x441 = ((x439 + x440)); // x441: Dependency! Liveness = 2; [x439, x440]
        vreal x442 = stencil(evo_lapse, stencil_idx_0_0_m2_VVV); // x442: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m2_]
        vreal x443 = stencil(evo_lapse, stencil_idx_0_0_2_VVV); // x443: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__2_]
        vreal x444 = ((x442 + x443)); // x444: Dependency! Liveness = 2; [x442, x443]
        vreal x445 = stencil(evo_lapse, stencil_idx_0_0_m1_VVV); // x445: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m1_]
        vreal x446 = stencil(evo_lapse, stencil_idx_0_0_1_VVV); // x446: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__1_]
        vreal x447 = ((x445 + x446)); // x447: Dependency! Liveness = 2; [x445, x446]
        vreal x429 = (-20 * stencil(evo_lapse, stencil_idx_0_0_0_VVV)); // x429: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__0_]
        vreal x244 = ((1.0 / 64.0) * dissipation_epsilon); // x244: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_lapse_rhs, stencil_idx_0_0_0_VVV, (access(evo_lapse_rhs, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x429 + (-6 * x432) + (15 * x435) + ((stencil(evo_lapse, stencil_idx_m3_0_0_VVV) + stencil(evo_lapse, stencil_idx_3_0_0_VVV))))) + (DYI * (x429 + (-6 * x438) + (15 * x441) + ((stencil(evo_lapse, stencil_idx_0_m3_0_VVV) + stencil(evo_lapse, stencil_idx_0_3_0_VVV))))) + (DZI * (x429 + (-6 * x444) + (15 * x447) + ((stencil(evo_lapse, stencil_idx_0_0_m3_VVV) + stencil(evo_lapse, stencil_idx_0_0_3_VVV)))))))));    
    });
    // apply_dissipation loop 6
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
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        vreal x449 = stencil(evo_shiftU0, stencil_idx_m2_0_0_VVV); // x449: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m2__0__0_]
        vreal x450 = stencil(evo_shiftU0, stencil_idx_2_0_0_VVV); // x450: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__2__0__0_]
        vreal x451 = ((x449 + x450)); // x451: Dependency! Liveness = 2; [x449, x450]
        vreal x452 = stencil(evo_shiftU0, stencil_idx_m1_0_0_VVV); // x452: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m1__0__0_]
        vreal x453 = stencil(evo_shiftU0, stencil_idx_1_0_0_VVV); // x453: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__1__0__0_]
        vreal x454 = ((x452 + x453)); // x454: Dependency! Liveness = 2; [x452, x453]
        vreal x455 = stencil(evo_shiftU0, stencil_idx_0_m2_0_VVV); // x455: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m2__0_]
        vreal x456 = stencil(evo_shiftU0, stencil_idx_0_2_0_VVV); // x456: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__2__0_]
        vreal x457 = ((x455 + x456)); // x457: Dependency! Liveness = 2; [x455, x456]
        vreal x458 = stencil(evo_shiftU0, stencil_idx_0_m1_0_VVV); // x458: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m1__0_]
        vreal x459 = stencil(evo_shiftU0, stencil_idx_0_1_0_VVV); // x459: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__1__0_]
        vreal x460 = ((x458 + x459)); // x460: Dependency! Liveness = 2; [x458, x459]
        vreal x461 = stencil(evo_shiftU0, stencil_idx_0_0_m2_VVV); // x461: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m2_]
        vreal x462 = stencil(evo_shiftU0, stencil_idx_0_0_2_VVV); // x462: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__2_]
        vreal x463 = ((x461 + x462)); // x463: Dependency! Liveness = 2; [x461, x462]
        vreal x464 = stencil(evo_shiftU0, stencil_idx_0_0_m1_VVV); // x464: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m1_]
        vreal x465 = stencil(evo_shiftU0, stencil_idx_0_0_1_VVV); // x465: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__1_]
        vreal x466 = ((x464 + x465)); // x466: Dependency! Liveness = 2; [x464, x465]
        vreal x448 = (-20 * stencil(evo_shiftU0, stencil_idx_0_0_0_VVV)); // x448: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__0_]
        vreal x244 = ((1.0 / 64.0) * dissipation_epsilon); // x244: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_shift_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU0, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x448 + (-6 * x451) + (15 * x454) + ((stencil(evo_shiftU0, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU0, stencil_idx_3_0_0_VVV))))) + (DYI * (x448 + (-6 * x457) + (15 * x460) + ((stencil(evo_shiftU0, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU0, stencil_idx_0_3_0_VVV))))) + (DZI * (x448 + (-6 * x463) + (15 * x466) + ((stencil(evo_shiftU0, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU0, stencil_idx_0_0_3_VVV)))))))));
        vreal x468 = stencil(evo_shiftU1, stencil_idx_m2_0_0_VVV); // x468: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m2__0__0_]
        vreal x469 = stencil(evo_shiftU1, stencil_idx_2_0_0_VVV); // x469: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__2__0__0_]
        vreal x470 = ((x468 + x469)); // x470: Dependency! Liveness = 2; [x468, x469]
        vreal x471 = stencil(evo_shiftU1, stencil_idx_m1_0_0_VVV); // x471: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m1__0__0_]
        vreal x472 = stencil(evo_shiftU1, stencil_idx_1_0_0_VVV); // x472: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__1__0__0_]
        vreal x473 = ((x471 + x472)); // x473: Dependency! Liveness = 2; [x471, x472]
        vreal x474 = stencil(evo_shiftU1, stencil_idx_0_m2_0_VVV); // x474: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m2__0_]
        vreal x475 = stencil(evo_shiftU1, stencil_idx_0_2_0_VVV); // x475: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__2__0_]
        vreal x476 = ((x474 + x475)); // x476: Dependency! Liveness = 2; [x474, x475]
        vreal x477 = stencil(evo_shiftU1, stencil_idx_0_m1_0_VVV); // x477: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m1__0_]
        vreal x478 = stencil(evo_shiftU1, stencil_idx_0_1_0_VVV); // x478: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__1__0_]
        vreal x479 = ((x477 + x478)); // x479: Dependency! Liveness = 2; [x477, x478]
        vreal x480 = stencil(evo_shiftU1, stencil_idx_0_0_m2_VVV); // x480: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m2_]
        vreal x481 = stencil(evo_shiftU1, stencil_idx_0_0_2_VVV); // x481: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__2_]
        vreal x482 = ((x480 + x481)); // x482: Dependency! Liveness = 2; [x480, x481]
        vreal x483 = stencil(evo_shiftU1, stencil_idx_0_0_m1_VVV); // x483: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m1_]
        vreal x484 = stencil(evo_shiftU1, stencil_idx_0_0_1_VVV); // x484: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__1_]
        vreal x485 = ((x483 + x484)); // x485: Dependency! Liveness = 2; [x483, x484]
        vreal x467 = (-20 * stencil(evo_shiftU1, stencil_idx_0_0_0_VVV)); // x467: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__0_]
        store(evo_shift_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU1, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x467 + (-6 * x470) + (15 * x473) + ((stencil(evo_shiftU1, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU1, stencil_idx_3_0_0_VVV))))) + (DYI * (x467 + (-6 * x476) + (15 * x479) + ((stencil(evo_shiftU1, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU1, stencil_idx_0_3_0_VVV))))) + (DZI * (x467 + (-6 * x482) + (15 * x485) + ((stencil(evo_shiftU1, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU1, stencil_idx_0_0_3_VVV)))))))));
        vreal x487 = stencil(evo_shiftU2, stencil_idx_m2_0_0_VVV); // x487: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m2__0__0_]
        vreal x488 = stencil(evo_shiftU2, stencil_idx_2_0_0_VVV); // x488: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__2__0__0_]
        vreal x489 = ((x487 + x488)); // x489: Dependency! Liveness = 2; [x487, x488]
        vreal x490 = stencil(evo_shiftU2, stencil_idx_m1_0_0_VVV); // x490: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m1__0__0_]
        vreal x491 = stencil(evo_shiftU2, stencil_idx_1_0_0_VVV); // x491: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__1__0__0_]
        vreal x492 = ((x490 + x491)); // x492: Dependency! Liveness = 2; [x490, x491]
        vreal x493 = stencil(evo_shiftU2, stencil_idx_0_m2_0_VVV); // x493: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m2__0_]
        vreal x494 = stencil(evo_shiftU2, stencil_idx_0_2_0_VVV); // x494: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__2__0_]
        vreal x495 = ((x493 + x494)); // x495: Dependency! Liveness = 2; [x493, x494]
        vreal x496 = stencil(evo_shiftU2, stencil_idx_0_m1_0_VVV); // x496: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m1__0_]
        vreal x497 = stencil(evo_shiftU2, stencil_idx_0_1_0_VVV); // x497: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__1__0_]
        vreal x498 = ((x496 + x497)); // x498: Dependency! Liveness = 2; [x496, x497]
        vreal x499 = stencil(evo_shiftU2, stencil_idx_0_0_m2_VVV); // x499: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m2_]
        vreal x500 = stencil(evo_shiftU2, stencil_idx_0_0_2_VVV); // x500: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__2_]
        vreal x501 = ((x499 + x500)); // x501: Dependency! Liveness = 2; [x499, x500]
        vreal x502 = stencil(evo_shiftU2, stencil_idx_0_0_m1_VVV); // x502: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m1_]
        vreal x503 = stencil(evo_shiftU2, stencil_idx_0_0_1_VVV); // x503: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__1_]
        vreal x504 = ((x502 + x503)); // x504: Dependency! Liveness = 2; [x502, x503]
        vreal x486 = (-20 * stencil(evo_shiftU2, stencil_idx_0_0_0_VVV)); // x486: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__0_]
        store(evo_shift_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU2, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x486 + (-6 * x489) + (15 * x492) + ((stencil(evo_shiftU2, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU2, stencil_idx_3_0_0_VVV))))) + (DYI * (x486 + (-6 * x495) + (15 * x498) + ((stencil(evo_shiftU2, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU2, stencil_idx_0_3_0_VVV))))) + (DZI * (x486 + (-6 * x501) + (15 * x504) + ((stencil(evo_shiftU2, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU2, stencil_idx_0_0_3_VVV)))))))));    
    });
    // apply_dissipation loop 7
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
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        vreal x510 = stencil(shift_BU0, stencil_idx_0_m2_0_VVV); // x510: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__m2__0_]
        vreal x511 = stencil(shift_BU0, stencil_idx_0_2_0_VVV); // x511: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__2__0_]
        vreal x512 = stencil(shift_BU0, stencil_idx_0_m1_0_VVV); // x512: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__m1__0_]
        vreal x513 = stencil(shift_BU0, stencil_idx_0_1_0_VVV); // x513: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__1__0_]
        vreal x514 = stencil(shift_BU0, stencil_idx_0_0_m2_VVV); // x514: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__m2_]
        vreal x515 = stencil(shift_BU0, stencil_idx_0_0_2_VVV); // x515: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__2_]
        vreal x516 = stencil(shift_BU0, stencil_idx_0_0_m1_VVV); // x516: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__m1_]
        vreal x517 = stencil(shift_BU0, stencil_idx_0_0_1_VVV); // x517: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__1_]
        vreal x506 = stencil(shift_BU0, stencil_idx_m2_0_0_VVV); // x506: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__m2__0__0_]
        vreal x507 = stencil(shift_BU0, stencil_idx_2_0_0_VVV); // x507: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__2__0__0_]
        vreal x508 = stencil(shift_BU0, stencil_idx_m1_0_0_VVV); // x508: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__m1__0__0_]
        vreal x509 = stencil(shift_BU0, stencil_idx_1_0_0_VVV); // x509: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__1__0__0_]
        vreal x505 = (-20 * stencil(shift_BU0, stencil_idx_0_0_0_VVV)); // x505: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__0_]
        vreal x244 = ((1.0 / 64.0) * dissipation_epsilon); // x244: Dependency! Liveness = 1; [dissipation_epsilon]
        store(shift_B_rhsU0, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU0, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x505 + (-6 * ((x506 + x507))) + (15 * ((x508 + x509))) + ((stencil(shift_BU0, stencil_idx_m3_0_0_VVV) + stencil(shift_BU0, stencil_idx_3_0_0_VVV))))) + (DYI * (x505 + (-6 * ((x510 + x511))) + (15 * ((x512 + x513))) + ((stencil(shift_BU0, stencil_idx_0_m3_0_VVV) + stencil(shift_BU0, stencil_idx_0_3_0_VVV))))) + (DZI * (x505 + (-6 * ((x514 + x515))) + (15 * ((x516 + x517))) + ((stencil(shift_BU0, stencil_idx_0_0_m3_VVV) + stencil(shift_BU0, stencil_idx_0_0_3_VVV)))))))));
        vreal x523 = stencil(shift_BU1, stencil_idx_0_m2_0_VVV); // x523: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__m2__0_]
        vreal x524 = stencil(shift_BU1, stencil_idx_0_2_0_VVV); // x524: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__2__0_]
        vreal x525 = stencil(shift_BU1, stencil_idx_0_m1_0_VVV); // x525: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__m1__0_]
        vreal x526 = stencil(shift_BU1, stencil_idx_0_1_0_VVV); // x526: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__1__0_]
        vreal x527 = stencil(shift_BU1, stencil_idx_0_0_m2_VVV); // x527: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__m2_]
        vreal x528 = stencil(shift_BU1, stencil_idx_0_0_2_VVV); // x528: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__2_]
        vreal x529 = stencil(shift_BU1, stencil_idx_0_0_m1_VVV); // x529: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__m1_]
        vreal x530 = stencil(shift_BU1, stencil_idx_0_0_1_VVV); // x530: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__1_]
        vreal x519 = stencil(shift_BU1, stencil_idx_m2_0_0_VVV); // x519: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__m2__0__0_]
        vreal x520 = stencil(shift_BU1, stencil_idx_2_0_0_VVV); // x520: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__2__0__0_]
        vreal x521 = stencil(shift_BU1, stencil_idx_m1_0_0_VVV); // x521: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__m1__0__0_]
        vreal x522 = stencil(shift_BU1, stencil_idx_1_0_0_VVV); // x522: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__1__0__0_]
        vreal x518 = (-20 * stencil(shift_BU1, stencil_idx_0_0_0_VVV)); // x518: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__0_]
        store(shift_B_rhsU1, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU1, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x518 + (-6 * ((x519 + x520))) + (15 * ((x521 + x522))) + ((stencil(shift_BU1, stencil_idx_m3_0_0_VVV) + stencil(shift_BU1, stencil_idx_3_0_0_VVV))))) + (DYI * (x518 + (-6 * ((x523 + x524))) + (15 * ((x525 + x526))) + ((stencil(shift_BU1, stencil_idx_0_m3_0_VVV) + stencil(shift_BU1, stencil_idx_0_3_0_VVV))))) + (DZI * (x518 + (-6 * ((x527 + x528))) + (15 * ((x529 + x530))) + ((stencil(shift_BU1, stencil_idx_0_0_m3_VVV) + stencil(shift_BU1, stencil_idx_0_0_3_VVV)))))))));
        vreal x536 = stencil(shift_BU2, stencil_idx_0_m2_0_VVV); // x536: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__m2__0_]
        vreal x537 = stencil(shift_BU2, stencil_idx_0_2_0_VVV); // x537: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__2__0_]
        vreal x538 = stencil(shift_BU2, stencil_idx_0_m1_0_VVV); // x538: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__m1__0_]
        vreal x539 = stencil(shift_BU2, stencil_idx_0_1_0_VVV); // x539: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__1__0_]
        vreal x540 = stencil(shift_BU2, stencil_idx_0_0_m2_VVV); // x540: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__m2_]
        vreal x541 = stencil(shift_BU2, stencil_idx_0_0_2_VVV); // x541: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__2_]
        vreal x542 = stencil(shift_BU2, stencil_idx_0_0_m1_VVV); // x542: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__m1_]
        vreal x543 = stencil(shift_BU2, stencil_idx_0_0_1_VVV); // x543: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__1_]
        vreal x532 = stencil(shift_BU2, stencil_idx_m2_0_0_VVV); // x532: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__m2__0__0_]
        vreal x533 = stencil(shift_BU2, stencil_idx_2_0_0_VVV); // x533: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__2__0__0_]
        vreal x534 = stencil(shift_BU2, stencil_idx_m1_0_0_VVV); // x534: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__m1__0__0_]
        vreal x535 = stencil(shift_BU2, stencil_idx_1_0_0_VVV); // x535: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__1__0__0_]
        vreal x531 = (-20 * stencil(shift_BU2, stencil_idx_0_0_0_VVV)); // x531: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__0_]
        store(shift_B_rhsU2, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU2, stencil_idx_0_0_0_VVV) + (x244 * ((DXI * (x531 + (-6 * ((x532 + x533))) + (15 * ((x534 + x535))) + ((stencil(shift_BU2, stencil_idx_m3_0_0_VVV) + stencil(shift_BU2, stencil_idx_3_0_0_VVV))))) + (DYI * (x531 + (-6 * ((x536 + x537))) + (15 * ((x538 + x539))) + ((stencil(shift_BU2, stencil_idx_0_m3_0_VVV) + stencil(shift_BU2, stencil_idx_0_3_0_VVV))))) + (DZI * (x531 + (-6 * ((x540 + x541))) + (15 * ((x542 + x543))) + ((stencil(shift_BU2, stencil_idx_0_0_m3_VVV) + stencil(shift_BU2, stencil_idx_0_0_3_VVV)))))))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}