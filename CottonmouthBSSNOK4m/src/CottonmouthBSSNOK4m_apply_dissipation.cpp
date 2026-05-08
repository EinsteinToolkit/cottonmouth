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
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
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
        vreal x162 = stencil(gtDD00, stencil_idx_m2_0_0_VVV); // x162: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m2__0__0_]
        vreal x163 = stencil(gtDD00, stencil_idx_2_0_0_VVV); // x163: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__2__0__0_]
        vreal x265 = ((x162 + x163)); // x265: Dependency! Liveness = 2; [x162, x163]
        vreal x164 = stencil(gtDD00, stencil_idx_m1_0_0_VVV); // x164: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m1__0__0_]
        vreal x165 = stencil(gtDD00, stencil_idx_1_0_0_VVV); // x165: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__1__0__0_]
        vreal x266 = ((x164 + x165)); // x266: Dependency! Liveness = 2; [x164, x165]
        vreal x121 = stencil(gtDD00, stencil_idx_0_m2_0_VVV); // x121: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m2__0_]
        vreal x122 = stencil(gtDD00, stencil_idx_0_2_0_VVV); // x122: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__2__0_]
        vreal x267 = ((x121 + x122)); // x267: Dependency! Liveness = 2; [x121, x122]
        vreal x123 = stencil(gtDD00, stencil_idx_0_m1_0_VVV); // x123: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m1__0_]
        vreal x124 = stencil(gtDD00, stencil_idx_0_1_0_VVV); // x124: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__1__0_]
        vreal x268 = ((x123 + x124)); // x268: Dependency! Liveness = 2; [x123, x124]
        vreal x104 = stencil(gtDD00, stencil_idx_0_0_m2_VVV); // x104: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m2_]
        vreal x105 = stencil(gtDD00, stencil_idx_0_0_2_VVV); // x105: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__2_]
        vreal x269 = ((x104 + x105)); // x269: Dependency! Liveness = 2; [x104, x105]
        vreal x106 = stencil(gtDD00, stencil_idx_0_0_m1_VVV); // x106: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m1_]
        vreal x107 = stencil(gtDD00, stencil_idx_0_0_1_VVV); // x107: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__1_]
        vreal x270 = ((x106 + x107)); // x270: Dependency! Liveness = 2; [x106, x107]
        vreal x264 = ((-5.0 / 16.0) * stencil(gtDD00, stencil_idx_0_0_0_VVV)); // x264: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__0_]
        store(gt_rhsDD00, stencil_idx_0_0_0_VVV, (access(gt_rhsDD00, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x264 + ((-3.0 / 32.0) * x265) + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_m3_0_0_VVV) + stencil(gtDD00, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x266))) + (DYI * (x264 + ((-3.0 / 32.0) * x267) + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_0_m3_0_VVV) + stencil(gtDD00, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x268))) + (DZI * (x264 + ((-3.0 / 32.0) * x269) + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_0_0_m3_VVV) + stencil(gtDD00, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x270)))))));
        vreal x169 = stencil(gtDD01, stencil_idx_m2_0_0_VVV); // x169: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m2__0__0_]
        vreal x170 = stencil(gtDD01, stencil_idx_2_0_0_VVV); // x170: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__2__0__0_]
        vreal x272 = ((x169 + x170)); // x272: Dependency! Liveness = 2; [x169, x170]
        vreal x171 = stencil(gtDD01, stencil_idx_m1_0_0_VVV); // x171: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m1__0__0_]
        vreal x172 = stencil(gtDD01, stencil_idx_1_0_0_VVV); // x172: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__1__0__0_]
        vreal x273 = ((x171 + x172)); // x273: Dependency! Liveness = 2; [x171, x172]
        vreal x140 = stencil(gtDD01, stencil_idx_0_m2_0_VVV); // x140: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m2__0_]
        vreal x141 = stencil(gtDD01, stencil_idx_0_2_0_VVV); // x141: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__2__0_]
        vreal x274 = ((x140 + x141)); // x274: Dependency! Liveness = 2; [x140, x141]
        vreal x142 = stencil(gtDD01, stencil_idx_0_m1_0_VVV); // x142: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m1__0_]
        vreal x143 = stencil(gtDD01, stencil_idx_0_1_0_VVV); // x143: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__1__0_]
        vreal x275 = ((x142 + x143)); // x275: Dependency! Liveness = 2; [x142, x143]
        vreal x81 = stencil(gtDD01, stencil_idx_0_0_m2_VVV); // x81: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m2_]
        vreal x82 = stencil(gtDD01, stencil_idx_0_0_2_VVV); // x82: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__2_]
        vreal x276 = ((x81 + x82)); // x276: Dependency! Liveness = 2; [x81, x82]
        vreal x83 = stencil(gtDD01, stencil_idx_0_0_m1_VVV); // x83: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m1_]
        vreal x84 = stencil(gtDD01, stencil_idx_0_0_1_VVV); // x84: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__1_]
        vreal x277 = ((x83 + x84)); // x277: Dependency! Liveness = 2; [x83, x84]
        vreal x271 = ((-5.0 / 16.0) * stencil(gtDD01, stencil_idx_0_0_0_VVV)); // x271: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__0_]
        store(gt_rhsDD01, stencil_idx_0_0_0_VVV, (access(gt_rhsDD01, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x271 + ((-3.0 / 32.0) * x272) + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_m3_0_0_VVV) + stencil(gtDD01, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x273))) + (DYI * (x271 + ((-3.0 / 32.0) * x274) + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_0_m3_0_VVV) + stencil(gtDD01, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x275))) + (DZI * (x271 + ((-3.0 / 32.0) * x276) + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_0_0_m3_VVV) + stencil(gtDD01, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x277)))))));
        vreal x176 = stencil(gtDD02, stencil_idx_m2_0_0_VVV); // x176: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m2__0__0_]
        vreal x177 = stencil(gtDD02, stencil_idx_2_0_0_VVV); // x177: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__2__0__0_]
        vreal x279 = ((x176 + x177)); // x279: Dependency! Liveness = 2; [x176, x177]
        vreal x178 = stencil(gtDD02, stencil_idx_m1_0_0_VVV); // x178: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m1__0__0_]
        vreal x179 = stencil(gtDD02, stencil_idx_1_0_0_VVV); // x179: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__1__0__0_]
        vreal x280 = ((x178 + x179)); // x280: Dependency! Liveness = 2; [x178, x179]
        vreal x76 = stencil(gtDD02, stencil_idx_0_m2_0_VVV); // x76: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m2__0_]
        vreal x77 = stencil(gtDD02, stencil_idx_0_2_0_VVV); // x77: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__2__0_]
        vreal x281 = ((x76 + x77)); // x281: Dependency! Liveness = 2; [x76, x77]
        vreal x78 = stencil(gtDD02, stencil_idx_0_m1_0_VVV); // x78: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m1__0_]
        vreal x79 = stencil(gtDD02, stencil_idx_0_1_0_VVV); // x79: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__1__0_]
        vreal x282 = ((x78 + x79)); // x282: Dependency! Liveness = 2; [x78, x79]
        vreal x193 = stencil(gtDD02, stencil_idx_0_0_m2_VVV); // x193: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m2_]
        vreal x194 = stencil(gtDD02, stencil_idx_0_0_2_VVV); // x194: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__2_]
        vreal x283 = ((x193 + x194)); // x283: Dependency! Liveness = 2; [x193, x194]
        vreal x195 = stencil(gtDD02, stencil_idx_0_0_m1_VVV); // x195: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m1_]
        vreal x196 = stencil(gtDD02, stencil_idx_0_0_1_VVV); // x196: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__1_]
        vreal x284 = ((x195 + x196)); // x284: Dependency! Liveness = 2; [x195, x196]
        vreal x278 = ((-5.0 / 16.0) * stencil(gtDD02, stencil_idx_0_0_0_VVV)); // x278: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__0_]
        store(gt_rhsDD02, stencil_idx_0_0_0_VVV, (access(gt_rhsDD02, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x278 + ((-3.0 / 32.0) * x279) + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_m3_0_0_VVV) + stencil(gtDD02, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x280))) + (DYI * (x278 + ((-3.0 / 32.0) * x281) + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_0_m3_0_VVV) + stencil(gtDD02, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x282))) + (DZI * (x278 + ((-3.0 / 32.0) * x283) + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_0_0_m3_VVV) + stencil(gtDD02, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x284)))))));
        vreal x115 = stencil(gtDD11, stencil_idx_m2_0_0_VVV); // x115: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m2__0__0_]
        vreal x116 = stencil(gtDD11, stencil_idx_2_0_0_VVV); // x116: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__2__0__0_]
        vreal x286 = ((x115 + x116)); // x286: Dependency! Liveness = 2; [x115, x116]
        vreal x117 = stencil(gtDD11, stencil_idx_m1_0_0_VVV); // x117: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m1__0__0_]
        vreal x118 = stencil(gtDD11, stencil_idx_1_0_0_VVV); // x118: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__1__0__0_]
        vreal x287 = ((x117 + x118)); // x287: Dependency! Liveness = 2; [x117, x118]
        vreal x132 = stencil(gtDD11, stencil_idx_0_m2_0_VVV); // x132: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m2__0_]
        vreal x133 = stencil(gtDD11, stencil_idx_0_2_0_VVV); // x133: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__2__0_]
        vreal x288 = ((x132 + x133)); // x288: Dependency! Liveness = 2; [x132, x133]
        vreal x134 = stencil(gtDD11, stencil_idx_0_m1_0_VVV); // x134: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m1__0_]
        vreal x135 = stencil(gtDD11, stencil_idx_0_1_0_VVV); // x135: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__1__0_]
        vreal x289 = ((x134 + x135)); // x289: Dependency! Liveness = 2; [x134, x135]
        vreal x66 = stencil(gtDD11, stencil_idx_0_0_m2_VVV); // x66: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m2_]
        vreal x67 = stencil(gtDD11, stencil_idx_0_0_2_VVV); // x67: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__2_]
        vreal x290 = ((x66 + x67)); // x290: Dependency! Liveness = 2; [x66, x67]
        vreal x68 = stencil(gtDD11, stencil_idx_0_0_m1_VVV); // x68: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m1_]
        vreal x69 = stencil(gtDD11, stencil_idx_0_0_1_VVV); // x69: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__1_]
        vreal x291 = ((x68 + x69)); // x291: Dependency! Liveness = 2; [x68, x69]
        vreal x285 = ((-5.0 / 16.0) * stencil(gtDD11, stencil_idx_0_0_0_VVV)); // x285: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__0_]
        store(gt_rhsDD11, stencil_idx_0_0_0_VVV, (access(gt_rhsDD11, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x285 + ((-3.0 / 32.0) * x286) + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_m3_0_0_VVV) + stencil(gtDD11, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x287))) + (DYI * (x285 + ((-3.0 / 32.0) * x288) + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_0_m3_0_VVV) + stencil(gtDD11, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x289))) + (DZI * (x285 + ((-3.0 / 32.0) * x290) + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_0_0_m3_VVV) + stencil(gtDD11, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x291)))))));
        vreal x86 = stencil(gtDD12, stencil_idx_m2_0_0_VVV); // x86: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m2__0__0_]
        vreal x87 = stencil(gtDD12, stencil_idx_2_0_0_VVV); // x87: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__2__0__0_]
        vreal x293 = ((x86 + x87)); // x293: Dependency! Liveness = 2; [x86, x87]
        vreal x88 = stencil(gtDD12, stencil_idx_m1_0_0_VVV); // x88: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m1__0__0_]
        vreal x89 = stencil(gtDD12, stencil_idx_1_0_0_VVV); // x89: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__1__0__0_]
        vreal x294 = ((x88 + x89)); // x294: Dependency! Liveness = 2; [x88, x89]
        vreal x150 = stencil(gtDD12, stencil_idx_0_m2_0_VVV); // x150: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m2__0_]
        vreal x151 = stencil(gtDD12, stencil_idx_0_2_0_VVV); // x151: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__2__0_]
        vreal x295 = ((x150 + x151)); // x295: Dependency! Liveness = 2; [x150, x151]
        vreal x152 = stencil(gtDD12, stencil_idx_0_m1_0_VVV); // x152: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m1__0_]
        vreal x153 = stencil(gtDD12, stencil_idx_0_1_0_VVV); // x153: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__1__0_]
        vreal x296 = ((x152 + x153)); // x296: Dependency! Liveness = 2; [x152, x153]
        vreal x200 = stencil(gtDD12, stencil_idx_0_0_m2_VVV); // x200: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m2_]
        vreal x201 = stencil(gtDD12, stencil_idx_0_0_2_VVV); // x201: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__2_]
        vreal x297 = ((x200 + x201)); // x297: Dependency! Liveness = 2; [x200, x201]
        vreal x202 = stencil(gtDD12, stencil_idx_0_0_m1_VVV); // x202: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m1_]
        vreal x203 = stencil(gtDD12, stencil_idx_0_0_1_VVV); // x203: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__1_]
        vreal x298 = ((x202 + x203)); // x298: Dependency! Liveness = 2; [x202, x203]
        vreal x292 = ((-5.0 / 16.0) * stencil(gtDD12, stencil_idx_0_0_0_VVV)); // x292: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__0_]
        store(gt_rhsDD12, stencil_idx_0_0_0_VVV, (access(gt_rhsDD12, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x292 + ((-3.0 / 32.0) * x293) + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_m3_0_0_VVV) + stencil(gtDD12, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x294))) + (DYI * (x292 + ((-3.0 / 32.0) * x295) + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_0_m3_0_VVV) + stencil(gtDD12, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x296))) + (DZI * (x292 + ((-3.0 / 32.0) * x297) + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_0_0_m3_VVV) + stencil(gtDD12, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x298)))))));
        vreal x98 = stencil(gtDD22, stencil_idx_m2_0_0_VVV); // x98: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m2__0__0_]
        vreal x99 = stencil(gtDD22, stencil_idx_2_0_0_VVV); // x99: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__2__0__0_]
        vreal x300 = ((x98 + x99)); // x300: Dependency! Liveness = 2; [x98, x99]
        vreal x100 = stencil(gtDD22, stencil_idx_m1_0_0_VVV); // x100: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m1__0__0_]
        vreal x101 = stencil(gtDD22, stencil_idx_1_0_0_VVV); // x101: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__1__0__0_]
        vreal x301 = ((x100 + x101)); // x301: Dependency! Liveness = 2; [x100, x101]
        vreal x56 = stencil(gtDD22, stencil_idx_0_m2_0_VVV); // x56: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m2__0_]
        vreal x57 = stencil(gtDD22, stencil_idx_0_2_0_VVV); // x57: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__2__0_]
        vreal x302 = ((x56 + x57)); // x302: Dependency! Liveness = 2; [x56, x57]
        vreal x58 = stencil(gtDD22, stencil_idx_0_m1_0_VVV); // x58: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m1__0_]
        vreal x59 = stencil(gtDD22, stencil_idx_0_1_0_VVV); // x59: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__1__0_]
        vreal x303 = ((x58 + x59)); // x303: Dependency! Liveness = 2; [x58, x59]
        vreal x187 = stencil(gtDD22, stencil_idx_0_0_m2_VVV); // x187: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m2_]
        vreal x188 = stencil(gtDD22, stencil_idx_0_0_2_VVV); // x188: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__2_]
        vreal x304 = ((x187 + x188)); // x304: Dependency! Liveness = 2; [x187, x188]
        vreal x189 = stencil(gtDD22, stencil_idx_0_0_m1_VVV); // x189: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m1_]
        vreal x190 = stencil(gtDD22, stencil_idx_0_0_1_VVV); // x190: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__1_]
        vreal x305 = ((x189 + x190)); // x305: Dependency! Liveness = 2; [x189, x190]
        vreal x299 = ((-5.0 / 16.0) * stencil(gtDD22, stencil_idx_0_0_0_VVV)); // x299: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__0_]
        store(gt_rhsDD22, stencil_idx_0_0_0_VVV, (access(gt_rhsDD22, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x299 + ((-3.0 / 32.0) * x300) + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_m3_0_0_VVV) + stencil(gtDD22, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x301))) + (DYI * (x299 + ((-3.0 / 32.0) * x302) + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_0_m3_0_VVV) + stencil(gtDD22, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x303))) + (DZI * (x299 + ((-3.0 / 32.0) * x304) + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_0_0_m3_VVV) + stencil(gtDD22, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x305)))))));    
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
        vreal x307 = stencil(w, stencil_idx_m2_0_0_VVV); // x307: Dependency! Liveness = 1; [__dummy_stencil__w__m2__0__0_]
        vreal x308 = stencil(w, stencil_idx_2_0_0_VVV); // x308: Dependency! Liveness = 1; [__dummy_stencil__w__2__0__0_]
        vreal x309 = ((x307 + x308)); // x309: Dependency! Liveness = 2; [x307, x308]
        vreal x310 = stencil(w, stencil_idx_m1_0_0_VVV); // x310: Dependency! Liveness = 1; [__dummy_stencil__w__m1__0__0_]
        vreal x311 = stencil(w, stencil_idx_1_0_0_VVV); // x311: Dependency! Liveness = 1; [__dummy_stencil__w__1__0__0_]
        vreal x312 = ((x310 + x311)); // x312: Dependency! Liveness = 2; [x310, x311]
        vreal x313 = stencil(w, stencil_idx_0_m2_0_VVV); // x313: Dependency! Liveness = 1; [__dummy_stencil__w__0__m2__0_]
        vreal x314 = stencil(w, stencil_idx_0_2_0_VVV); // x314: Dependency! Liveness = 1; [__dummy_stencil__w__0__2__0_]
        vreal x315 = ((x313 + x314)); // x315: Dependency! Liveness = 2; [x313, x314]
        vreal x316 = stencil(w, stencil_idx_0_m1_0_VVV); // x316: Dependency! Liveness = 1; [__dummy_stencil__w__0__m1__0_]
        vreal x317 = stencil(w, stencil_idx_0_1_0_VVV); // x317: Dependency! Liveness = 1; [__dummy_stencil__w__0__1__0_]
        vreal x318 = ((x316 + x317)); // x318: Dependency! Liveness = 2; [x316, x317]
        vreal x319 = stencil(w, stencil_idx_0_0_m2_VVV); // x319: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__m2_]
        vreal x320 = stencil(w, stencil_idx_0_0_2_VVV); // x320: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__2_]
        vreal x321 = ((x319 + x320)); // x321: Dependency! Liveness = 2; [x319, x320]
        vreal x322 = stencil(w, stencil_idx_0_0_m1_VVV); // x322: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__m1_]
        vreal x323 = stencil(w, stencil_idx_0_0_1_VVV); // x323: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__1_]
        vreal x324 = ((x322 + x323)); // x324: Dependency! Liveness = 2; [x322, x323]
        vreal x306 = ((-5.0 / 16.0) * stencil(w, stencil_idx_0_0_0_VVV)); // x306: Dependency! Liveness = 1; [__dummy_stencil__w__0__0__0_]
        store(w_rhs, stencil_idx_0_0_0_VVV, (access(w_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x306 + ((-3.0 / 32.0) * x309) + ((1.0 / 64.0) * ((stencil(w, stencil_idx_m3_0_0_VVV) + stencil(w, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x312))) + (DYI * (x306 + ((-3.0 / 32.0) * x315) + ((1.0 / 64.0) * ((stencil(w, stencil_idx_0_m3_0_VVV) + stencil(w, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x318))) + (DZI * (x306 + ((-3.0 / 32.0) * x321) + ((1.0 / 64.0) * ((stencil(w, stencil_idx_0_0_m3_VVV) + stencil(w, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x324)))))));    
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
        vreal x330 = stencil(trK, stencil_idx_0_m2_0_VVV); // x330: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m2__0_]
        vreal x331 = stencil(trK, stencil_idx_0_2_0_VVV); // x331: Dependency! Liveness = 1; [__dummy_stencil__trK__0__2__0_]
        vreal x332 = stencil(trK, stencil_idx_0_m1_0_VVV); // x332: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m1__0_]
        vreal x333 = stencil(trK, stencil_idx_0_1_0_VVV); // x333: Dependency! Liveness = 1; [__dummy_stencil__trK__0__1__0_]
        vreal x334 = stencil(trK, stencil_idx_0_0_m2_VVV); // x334: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m2_]
        vreal x335 = stencil(trK, stencil_idx_0_0_2_VVV); // x335: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__2_]
        vreal x336 = stencil(trK, stencil_idx_0_0_m1_VVV); // x336: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m1_]
        vreal x337 = stencil(trK, stencil_idx_0_0_1_VVV); // x337: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__1_]
        vreal x326 = stencil(trK, stencil_idx_m2_0_0_VVV); // x326: Dependency! Liveness = 1; [__dummy_stencil__trK__m2__0__0_]
        vreal x327 = stencil(trK, stencil_idx_2_0_0_VVV); // x327: Dependency! Liveness = 1; [__dummy_stencil__trK__2__0__0_]
        vreal x328 = stencil(trK, stencil_idx_m1_0_0_VVV); // x328: Dependency! Liveness = 1; [__dummy_stencil__trK__m1__0__0_]
        vreal x329 = stencil(trK, stencil_idx_1_0_0_VVV); // x329: Dependency! Liveness = 1; [__dummy_stencil__trK__1__0__0_]
        vreal x325 = ((-5.0 / 16.0) * stencil(trK, stencil_idx_0_0_0_VVV)); // x325: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__0_]
        store(trK_rhs, stencil_idx_0_0_0_VVV, (access(trK_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x325 + ((-3.0 / 32.0) * ((x326 + x327))) + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_m3_0_0_VVV) + stencil(trK, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x328 + x329))))) + (DYI * (x325 + ((-3.0 / 32.0) * ((x330 + x331))) + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_0_m3_0_VVV) + stencil(trK, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x332 + x333))))) + (DZI * (x325 + ((-3.0 / 32.0) * ((x334 + x335))) + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_0_0_m3_VVV) + stencil(trK, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x336 + x337)))))))));    
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
        vreal x343 = stencil(AtDD00, stencil_idx_0_m2_0_VVV); // x343: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m2__0_]
        vreal x344 = stencil(AtDD00, stencil_idx_0_2_0_VVV); // x344: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__2__0_]
        vreal x345 = stencil(AtDD00, stencil_idx_0_m1_0_VVV); // x345: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m1__0_]
        vreal x346 = stencil(AtDD00, stencil_idx_0_1_0_VVV); // x346: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__1__0_]
        vreal x347 = stencil(AtDD00, stencil_idx_0_0_m2_VVV); // x347: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m2_]
        vreal x348 = stencil(AtDD00, stencil_idx_0_0_2_VVV); // x348: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__2_]
        vreal x349 = stencil(AtDD00, stencil_idx_0_0_m1_VVV); // x349: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m1_]
        vreal x350 = stencil(AtDD00, stencil_idx_0_0_1_VVV); // x350: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__1_]
        vreal x339 = stencil(AtDD00, stencil_idx_m2_0_0_VVV); // x339: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m2__0__0_]
        vreal x340 = stencil(AtDD00, stencil_idx_2_0_0_VVV); // x340: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__2__0__0_]
        vreal x341 = stencil(AtDD00, stencil_idx_m1_0_0_VVV); // x341: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m1__0__0_]
        vreal x342 = stencil(AtDD00, stencil_idx_1_0_0_VVV); // x342: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__1__0__0_]
        vreal x338 = ((-5.0 / 16.0) * stencil(AtDD00, stencil_idx_0_0_0_VVV)); // x338: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__0_]
        store(At_rhsDD00, stencil_idx_0_0_0_VVV, (access(At_rhsDD00, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x338 + ((-3.0 / 32.0) * ((x339 + x340))) + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_m3_0_0_VVV) + stencil(AtDD00, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x341 + x342))))) + (DYI * (x338 + ((-3.0 / 32.0) * ((x343 + x344))) + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_0_m3_0_VVV) + stencil(AtDD00, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x345 + x346))))) + (DZI * (x338 + ((-3.0 / 32.0) * ((x347 + x348))) + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_0_0_m3_VVV) + stencil(AtDD00, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x349 + x350)))))))));
        vreal x356 = stencil(AtDD01, stencil_idx_0_m2_0_VVV); // x356: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m2__0_]
        vreal x357 = stencil(AtDD01, stencil_idx_0_2_0_VVV); // x357: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__2__0_]
        vreal x358 = stencil(AtDD01, stencil_idx_0_m1_0_VVV); // x358: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m1__0_]
        vreal x359 = stencil(AtDD01, stencil_idx_0_1_0_VVV); // x359: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__1__0_]
        vreal x360 = stencil(AtDD01, stencil_idx_0_0_m2_VVV); // x360: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m2_]
        vreal x361 = stencil(AtDD01, stencil_idx_0_0_2_VVV); // x361: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__2_]
        vreal x362 = stencil(AtDD01, stencil_idx_0_0_m1_VVV); // x362: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m1_]
        vreal x363 = stencil(AtDD01, stencil_idx_0_0_1_VVV); // x363: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__1_]
        vreal x352 = stencil(AtDD01, stencil_idx_m2_0_0_VVV); // x352: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m2__0__0_]
        vreal x353 = stencil(AtDD01, stencil_idx_2_0_0_VVV); // x353: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__2__0__0_]
        vreal x354 = stencil(AtDD01, stencil_idx_m1_0_0_VVV); // x354: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m1__0__0_]
        vreal x355 = stencil(AtDD01, stencil_idx_1_0_0_VVV); // x355: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__1__0__0_]
        vreal x351 = ((-5.0 / 16.0) * stencil(AtDD01, stencil_idx_0_0_0_VVV)); // x351: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__0_]
        store(At_rhsDD01, stencil_idx_0_0_0_VVV, (access(At_rhsDD01, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x351 + ((-3.0 / 32.0) * ((x352 + x353))) + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_m3_0_0_VVV) + stencil(AtDD01, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x354 + x355))))) + (DYI * (x351 + ((-3.0 / 32.0) * ((x356 + x357))) + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_0_m3_0_VVV) + stencil(AtDD01, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x358 + x359))))) + (DZI * (x351 + ((-3.0 / 32.0) * ((x360 + x361))) + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_0_0_m3_VVV) + stencil(AtDD01, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x362 + x363)))))))));
        vreal x369 = stencil(AtDD02, stencil_idx_0_m2_0_VVV); // x369: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m2__0_]
        vreal x370 = stencil(AtDD02, stencil_idx_0_2_0_VVV); // x370: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__2__0_]
        vreal x371 = stencil(AtDD02, stencil_idx_0_m1_0_VVV); // x371: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m1__0_]
        vreal x372 = stencil(AtDD02, stencil_idx_0_1_0_VVV); // x372: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__1__0_]
        vreal x373 = stencil(AtDD02, stencil_idx_0_0_m2_VVV); // x373: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m2_]
        vreal x374 = stencil(AtDD02, stencil_idx_0_0_2_VVV); // x374: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__2_]
        vreal x375 = stencil(AtDD02, stencil_idx_0_0_m1_VVV); // x375: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m1_]
        vreal x376 = stencil(AtDD02, stencil_idx_0_0_1_VVV); // x376: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__1_]
        vreal x365 = stencil(AtDD02, stencil_idx_m2_0_0_VVV); // x365: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m2__0__0_]
        vreal x366 = stencil(AtDD02, stencil_idx_2_0_0_VVV); // x366: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__2__0__0_]
        vreal x367 = stencil(AtDD02, stencil_idx_m1_0_0_VVV); // x367: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m1__0__0_]
        vreal x368 = stencil(AtDD02, stencil_idx_1_0_0_VVV); // x368: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__1__0__0_]
        vreal x364 = ((-5.0 / 16.0) * stencil(AtDD02, stencil_idx_0_0_0_VVV)); // x364: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__0_]
        store(At_rhsDD02, stencil_idx_0_0_0_VVV, (access(At_rhsDD02, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x364 + ((-3.0 / 32.0) * ((x365 + x366))) + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_m3_0_0_VVV) + stencil(AtDD02, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x367 + x368))))) + (DYI * (x364 + ((-3.0 / 32.0) * ((x369 + x370))) + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_0_m3_0_VVV) + stencil(AtDD02, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x371 + x372))))) + (DZI * (x364 + ((-3.0 / 32.0) * ((x373 + x374))) + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_0_0_m3_VVV) + stencil(AtDD02, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x375 + x376)))))))));
        vreal x382 = stencil(AtDD11, stencil_idx_0_m2_0_VVV); // x382: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m2__0_]
        vreal x383 = stencil(AtDD11, stencil_idx_0_2_0_VVV); // x383: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__2__0_]
        vreal x384 = stencil(AtDD11, stencil_idx_0_m1_0_VVV); // x384: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m1__0_]
        vreal x385 = stencil(AtDD11, stencil_idx_0_1_0_VVV); // x385: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__1__0_]
        vreal x386 = stencil(AtDD11, stencil_idx_0_0_m2_VVV); // x386: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m2_]
        vreal x387 = stencil(AtDD11, stencil_idx_0_0_2_VVV); // x387: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__2_]
        vreal x388 = stencil(AtDD11, stencil_idx_0_0_m1_VVV); // x388: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m1_]
        vreal x389 = stencil(AtDD11, stencil_idx_0_0_1_VVV); // x389: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__1_]
        vreal x378 = stencil(AtDD11, stencil_idx_m2_0_0_VVV); // x378: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m2__0__0_]
        vreal x379 = stencil(AtDD11, stencil_idx_2_0_0_VVV); // x379: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__2__0__0_]
        vreal x380 = stencil(AtDD11, stencil_idx_m1_0_0_VVV); // x380: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m1__0__0_]
        vreal x381 = stencil(AtDD11, stencil_idx_1_0_0_VVV); // x381: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__1__0__0_]
        vreal x377 = ((-5.0 / 16.0) * stencil(AtDD11, stencil_idx_0_0_0_VVV)); // x377: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__0_]
        store(At_rhsDD11, stencil_idx_0_0_0_VVV, (access(At_rhsDD11, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x377 + ((-3.0 / 32.0) * ((x378 + x379))) + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_m3_0_0_VVV) + stencil(AtDD11, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x380 + x381))))) + (DYI * (x377 + ((-3.0 / 32.0) * ((x382 + x383))) + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_0_m3_0_VVV) + stencil(AtDD11, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x384 + x385))))) + (DZI * (x377 + ((-3.0 / 32.0) * ((x386 + x387))) + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_0_0_m3_VVV) + stencil(AtDD11, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x388 + x389)))))))));
        vreal x395 = stencil(AtDD12, stencil_idx_0_m2_0_VVV); // x395: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m2__0_]
        vreal x396 = stencil(AtDD12, stencil_idx_0_2_0_VVV); // x396: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__2__0_]
        vreal x397 = stencil(AtDD12, stencil_idx_0_m1_0_VVV); // x397: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m1__0_]
        vreal x398 = stencil(AtDD12, stencil_idx_0_1_0_VVV); // x398: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__1__0_]
        vreal x399 = stencil(AtDD12, stencil_idx_0_0_m2_VVV); // x399: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m2_]
        vreal x400 = stencil(AtDD12, stencil_idx_0_0_2_VVV); // x400: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__2_]
        vreal x401 = stencil(AtDD12, stencil_idx_0_0_m1_VVV); // x401: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m1_]
        vreal x402 = stencil(AtDD12, stencil_idx_0_0_1_VVV); // x402: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__1_]
        vreal x391 = stencil(AtDD12, stencil_idx_m2_0_0_VVV); // x391: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m2__0__0_]
        vreal x392 = stencil(AtDD12, stencil_idx_2_0_0_VVV); // x392: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__2__0__0_]
        vreal x393 = stencil(AtDD12, stencil_idx_m1_0_0_VVV); // x393: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m1__0__0_]
        vreal x394 = stencil(AtDD12, stencil_idx_1_0_0_VVV); // x394: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__1__0__0_]
        vreal x390 = ((-5.0 / 16.0) * stencil(AtDD12, stencil_idx_0_0_0_VVV)); // x390: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__0_]
        store(At_rhsDD12, stencil_idx_0_0_0_VVV, (access(At_rhsDD12, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x390 + ((-3.0 / 32.0) * ((x391 + x392))) + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_m3_0_0_VVV) + stencil(AtDD12, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x393 + x394))))) + (DYI * (x390 + ((-3.0 / 32.0) * ((x395 + x396))) + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_0_m3_0_VVV) + stencil(AtDD12, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x397 + x398))))) + (DZI * (x390 + ((-3.0 / 32.0) * ((x399 + x400))) + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_0_0_m3_VVV) + stencil(AtDD12, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x401 + x402)))))))));
        vreal x408 = stencil(AtDD22, stencil_idx_0_m2_0_VVV); // x408: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m2__0_]
        vreal x409 = stencil(AtDD22, stencil_idx_0_2_0_VVV); // x409: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__2__0_]
        vreal x410 = stencil(AtDD22, stencil_idx_0_m1_0_VVV); // x410: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m1__0_]
        vreal x411 = stencil(AtDD22, stencil_idx_0_1_0_VVV); // x411: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__1__0_]
        vreal x412 = stencil(AtDD22, stencil_idx_0_0_m2_VVV); // x412: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m2_]
        vreal x413 = stencil(AtDD22, stencil_idx_0_0_2_VVV); // x413: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__2_]
        vreal x414 = stencil(AtDD22, stencil_idx_0_0_m1_VVV); // x414: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m1_]
        vreal x415 = stencil(AtDD22, stencil_idx_0_0_1_VVV); // x415: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__1_]
        vreal x404 = stencil(AtDD22, stencil_idx_m2_0_0_VVV); // x404: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m2__0__0_]
        vreal x405 = stencil(AtDD22, stencil_idx_2_0_0_VVV); // x405: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__2__0__0_]
        vreal x406 = stencil(AtDD22, stencil_idx_m1_0_0_VVV); // x406: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m1__0__0_]
        vreal x407 = stencil(AtDD22, stencil_idx_1_0_0_VVV); // x407: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__1__0__0_]
        vreal x403 = ((-5.0 / 16.0) * stencil(AtDD22, stencil_idx_0_0_0_VVV)); // x403: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__0_]
        store(At_rhsDD22, stencil_idx_0_0_0_VVV, (access(At_rhsDD22, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x403 + ((-3.0 / 32.0) * ((x404 + x405))) + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_m3_0_0_VVV) + stencil(AtDD22, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x406 + x407))))) + (DYI * (x403 + ((-3.0 / 32.0) * ((x408 + x409))) + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_0_m3_0_VVV) + stencil(AtDD22, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x410 + x411))))) + (DZI * (x403 + ((-3.0 / 32.0) * ((x412 + x413))) + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_0_0_m3_VVV) + stencil(AtDD22, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x414 + x415)))))))));    
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
        vreal x421 = stencil(ConfConnectU0, stencil_idx_0_m2_0_VVV); // x421: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__m2__0_]
        vreal x422 = stencil(ConfConnectU0, stencil_idx_0_2_0_VVV); // x422: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__2__0_]
        vreal x423 = stencil(ConfConnectU0, stencil_idx_0_m1_0_VVV); // x423: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__m1__0_]
        vreal x424 = stencil(ConfConnectU0, stencil_idx_0_1_0_VVV); // x424: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__1__0_]
        vreal x425 = stencil(ConfConnectU0, stencil_idx_0_0_m2_VVV); // x425: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__m2_]
        vreal x426 = stencil(ConfConnectU0, stencil_idx_0_0_2_VVV); // x426: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__2_]
        vreal x427 = stencil(ConfConnectU0, stencil_idx_0_0_m1_VVV); // x427: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__m1_]
        vreal x428 = stencil(ConfConnectU0, stencil_idx_0_0_1_VVV); // x428: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__1_]
        vreal x417 = stencil(ConfConnectU0, stencil_idx_m2_0_0_VVV); // x417: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__m2__0__0_]
        vreal x418 = stencil(ConfConnectU0, stencil_idx_2_0_0_VVV); // x418: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__2__0__0_]
        vreal x419 = stencil(ConfConnectU0, stencil_idx_m1_0_0_VVV); // x419: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__m1__0__0_]
        vreal x420 = stencil(ConfConnectU0, stencil_idx_1_0_0_VVV); // x420: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__1__0__0_]
        vreal x416 = ((-5.0 / 16.0) * stencil(ConfConnectU0, stencil_idx_0_0_0_VVV)); // x416: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU0__0__0__0_]
        store(ConfConnect_rhsU0, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU0, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x416 + ((-3.0 / 32.0) * ((x417 + x418))) + ((1.0 / 64.0) * ((stencil(ConfConnectU0, stencil_idx_m3_0_0_VVV) + stencil(ConfConnectU0, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x419 + x420))))) + (DYI * (x416 + ((-3.0 / 32.0) * ((x421 + x422))) + ((1.0 / 64.0) * ((stencil(ConfConnectU0, stencil_idx_0_m3_0_VVV) + stencil(ConfConnectU0, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x423 + x424))))) + (DZI * (x416 + ((-3.0 / 32.0) * ((x425 + x426))) + ((1.0 / 64.0) * ((stencil(ConfConnectU0, stencil_idx_0_0_m3_VVV) + stencil(ConfConnectU0, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x427 + x428)))))))));
        vreal x434 = stencil(ConfConnectU1, stencil_idx_0_m2_0_VVV); // x434: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__m2__0_]
        vreal x435 = stencil(ConfConnectU1, stencil_idx_0_2_0_VVV); // x435: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__2__0_]
        vreal x436 = stencil(ConfConnectU1, stencil_idx_0_m1_0_VVV); // x436: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__m1__0_]
        vreal x437 = stencil(ConfConnectU1, stencil_idx_0_1_0_VVV); // x437: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__1__0_]
        vreal x438 = stencil(ConfConnectU1, stencil_idx_0_0_m2_VVV); // x438: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__m2_]
        vreal x439 = stencil(ConfConnectU1, stencil_idx_0_0_2_VVV); // x439: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__2_]
        vreal x440 = stencil(ConfConnectU1, stencil_idx_0_0_m1_VVV); // x440: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__m1_]
        vreal x441 = stencil(ConfConnectU1, stencil_idx_0_0_1_VVV); // x441: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__1_]
        vreal x430 = stencil(ConfConnectU1, stencil_idx_m2_0_0_VVV); // x430: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__m2__0__0_]
        vreal x431 = stencil(ConfConnectU1, stencil_idx_2_0_0_VVV); // x431: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__2__0__0_]
        vreal x432 = stencil(ConfConnectU1, stencil_idx_m1_0_0_VVV); // x432: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__m1__0__0_]
        vreal x433 = stencil(ConfConnectU1, stencil_idx_1_0_0_VVV); // x433: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__1__0__0_]
        vreal x429 = ((-5.0 / 16.0) * stencil(ConfConnectU1, stencil_idx_0_0_0_VVV)); // x429: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU1__0__0__0_]
        store(ConfConnect_rhsU1, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU1, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x429 + ((-3.0 / 32.0) * ((x430 + x431))) + ((1.0 / 64.0) * ((stencil(ConfConnectU1, stencil_idx_m3_0_0_VVV) + stencil(ConfConnectU1, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x432 + x433))))) + (DYI * (x429 + ((-3.0 / 32.0) * ((x434 + x435))) + ((1.0 / 64.0) * ((stencil(ConfConnectU1, stencil_idx_0_m3_0_VVV) + stencil(ConfConnectU1, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x436 + x437))))) + (DZI * (x429 + ((-3.0 / 32.0) * ((x438 + x439))) + ((1.0 / 64.0) * ((stencil(ConfConnectU1, stencil_idx_0_0_m3_VVV) + stencil(ConfConnectU1, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x440 + x441)))))))));
        vreal x447 = stencil(ConfConnectU2, stencil_idx_0_m2_0_VVV); // x447: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__m2__0_]
        vreal x448 = stencil(ConfConnectU2, stencil_idx_0_2_0_VVV); // x448: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__2__0_]
        vreal x449 = stencil(ConfConnectU2, stencil_idx_0_m1_0_VVV); // x449: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__m1__0_]
        vreal x450 = stencil(ConfConnectU2, stencil_idx_0_1_0_VVV); // x450: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__1__0_]
        vreal x451 = stencil(ConfConnectU2, stencil_idx_0_0_m2_VVV); // x451: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__m2_]
        vreal x452 = stencil(ConfConnectU2, stencil_idx_0_0_2_VVV); // x452: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__2_]
        vreal x453 = stencil(ConfConnectU2, stencil_idx_0_0_m1_VVV); // x453: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__m1_]
        vreal x454 = stencil(ConfConnectU2, stencil_idx_0_0_1_VVV); // x454: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__1_]
        vreal x443 = stencil(ConfConnectU2, stencil_idx_m2_0_0_VVV); // x443: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__m2__0__0_]
        vreal x444 = stencil(ConfConnectU2, stencil_idx_2_0_0_VVV); // x444: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__2__0__0_]
        vreal x445 = stencil(ConfConnectU2, stencil_idx_m1_0_0_VVV); // x445: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__m1__0__0_]
        vreal x446 = stencil(ConfConnectU2, stencil_idx_1_0_0_VVV); // x446: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__1__0__0_]
        vreal x442 = ((-5.0 / 16.0) * stencil(ConfConnectU2, stencil_idx_0_0_0_VVV)); // x442: Dependency! Liveness = 1; [__dummy_stencil__ConfConnectU2__0__0__0_]
        store(ConfConnect_rhsU2, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU2, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x442 + ((-3.0 / 32.0) * ((x443 + x444))) + ((1.0 / 64.0) * ((stencil(ConfConnectU2, stencil_idx_m3_0_0_VVV) + stencil(ConfConnectU2, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x445 + x446))))) + (DYI * (x442 + ((-3.0 / 32.0) * ((x447 + x448))) + ((1.0 / 64.0) * ((stencil(ConfConnectU2, stencil_idx_0_m3_0_VVV) + stencil(ConfConnectU2, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x449 + x450))))) + (DZI * (x442 + ((-3.0 / 32.0) * ((x451 + x452))) + ((1.0 / 64.0) * ((stencil(ConfConnectU2, stencil_idx_0_0_m3_VVV) + stencil(ConfConnectU2, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x453 + x454)))))))));    
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
        vreal x456 = stencil(evo_lapse, stencil_idx_m2_0_0_VVV); // x456: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m2__0__0_]
        vreal x457 = stencil(evo_lapse, stencil_idx_2_0_0_VVV); // x457: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__2__0__0_]
        vreal x458 = ((x456 + x457)); // x458: Dependency! Liveness = 2; [x456, x457]
        vreal x459 = stencil(evo_lapse, stencil_idx_m1_0_0_VVV); // x459: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m1__0__0_]
        vreal x460 = stencil(evo_lapse, stencil_idx_1_0_0_VVV); // x460: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__1__0__0_]
        vreal x461 = ((x459 + x460)); // x461: Dependency! Liveness = 2; [x459, x460]
        vreal x462 = stencil(evo_lapse, stencil_idx_0_m2_0_VVV); // x462: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m2__0_]
        vreal x463 = stencil(evo_lapse, stencil_idx_0_2_0_VVV); // x463: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__2__0_]
        vreal x464 = ((x462 + x463)); // x464: Dependency! Liveness = 2; [x462, x463]
        vreal x465 = stencil(evo_lapse, stencil_idx_0_m1_0_VVV); // x465: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m1__0_]
        vreal x466 = stencil(evo_lapse, stencil_idx_0_1_0_VVV); // x466: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__1__0_]
        vreal x467 = ((x465 + x466)); // x467: Dependency! Liveness = 2; [x465, x466]
        vreal x468 = stencil(evo_lapse, stencil_idx_0_0_m2_VVV); // x468: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m2_]
        vreal x469 = stencil(evo_lapse, stencil_idx_0_0_2_VVV); // x469: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__2_]
        vreal x470 = ((x468 + x469)); // x470: Dependency! Liveness = 2; [x468, x469]
        vreal x471 = stencil(evo_lapse, stencil_idx_0_0_m1_VVV); // x471: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m1_]
        vreal x472 = stencil(evo_lapse, stencil_idx_0_0_1_VVV); // x472: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__1_]
        vreal x473 = ((x471 + x472)); // x473: Dependency! Liveness = 2; [x471, x472]
        vreal x455 = ((-5.0 / 16.0) * stencil(evo_lapse, stencil_idx_0_0_0_VVV)); // x455: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__0_]
        store(evo_lapse_rhs, stencil_idx_0_0_0_VVV, (access(evo_lapse_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x455 + ((-3.0 / 32.0) * x458) + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_m3_0_0_VVV) + stencil(evo_lapse, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x461))) + (DYI * (x455 + ((-3.0 / 32.0) * x464) + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_0_m3_0_VVV) + stencil(evo_lapse, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x467))) + (DZI * (x455 + ((-3.0 / 32.0) * x470) + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_0_0_m3_VVV) + stencil(evo_lapse, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x473)))))));    
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
        vreal x475 = stencil(evo_shiftU0, stencil_idx_m2_0_0_VVV); // x475: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m2__0__0_]
        vreal x476 = stencil(evo_shiftU0, stencil_idx_2_0_0_VVV); // x476: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__2__0__0_]
        vreal x477 = ((x475 + x476)); // x477: Dependency! Liveness = 2; [x475, x476]
        vreal x478 = stencil(evo_shiftU0, stencil_idx_m1_0_0_VVV); // x478: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m1__0__0_]
        vreal x479 = stencil(evo_shiftU0, stencil_idx_1_0_0_VVV); // x479: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__1__0__0_]
        vreal x480 = ((x478 + x479)); // x480: Dependency! Liveness = 2; [x478, x479]
        vreal x481 = stencil(evo_shiftU0, stencil_idx_0_m2_0_VVV); // x481: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m2__0_]
        vreal x482 = stencil(evo_shiftU0, stencil_idx_0_2_0_VVV); // x482: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__2__0_]
        vreal x483 = ((x481 + x482)); // x483: Dependency! Liveness = 2; [x481, x482]
        vreal x484 = stencil(evo_shiftU0, stencil_idx_0_m1_0_VVV); // x484: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m1__0_]
        vreal x485 = stencil(evo_shiftU0, stencil_idx_0_1_0_VVV); // x485: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__1__0_]
        vreal x486 = ((x484 + x485)); // x486: Dependency! Liveness = 2; [x484, x485]
        vreal x487 = stencil(evo_shiftU0, stencil_idx_0_0_m2_VVV); // x487: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m2_]
        vreal x488 = stencil(evo_shiftU0, stencil_idx_0_0_2_VVV); // x488: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__2_]
        vreal x489 = ((x487 + x488)); // x489: Dependency! Liveness = 2; [x487, x488]
        vreal x490 = stencil(evo_shiftU0, stencil_idx_0_0_m1_VVV); // x490: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m1_]
        vreal x491 = stencil(evo_shiftU0, stencil_idx_0_0_1_VVV); // x491: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__1_]
        vreal x492 = ((x490 + x491)); // x492: Dependency! Liveness = 2; [x490, x491]
        vreal x474 = ((-5.0 / 16.0) * stencil(evo_shiftU0, stencil_idx_0_0_0_VVV)); // x474: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__0_]
        store(evo_shift_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU0, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x474 + ((-3.0 / 32.0) * x477) + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU0, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x480))) + (DYI * (x474 + ((-3.0 / 32.0) * x483) + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU0, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x486))) + (DZI * (x474 + ((-3.0 / 32.0) * x489) + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU0, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x492)))))));
        vreal x494 = stencil(evo_shiftU1, stencil_idx_m2_0_0_VVV); // x494: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m2__0__0_]
        vreal x495 = stencil(evo_shiftU1, stencil_idx_2_0_0_VVV); // x495: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__2__0__0_]
        vreal x496 = ((x494 + x495)); // x496: Dependency! Liveness = 2; [x494, x495]
        vreal x497 = stencil(evo_shiftU1, stencil_idx_m1_0_0_VVV); // x497: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m1__0__0_]
        vreal x498 = stencil(evo_shiftU1, stencil_idx_1_0_0_VVV); // x498: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__1__0__0_]
        vreal x499 = ((x497 + x498)); // x499: Dependency! Liveness = 2; [x497, x498]
        vreal x500 = stencil(evo_shiftU1, stencil_idx_0_m2_0_VVV); // x500: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m2__0_]
        vreal x501 = stencil(evo_shiftU1, stencil_idx_0_2_0_VVV); // x501: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__2__0_]
        vreal x502 = ((x500 + x501)); // x502: Dependency! Liveness = 2; [x500, x501]
        vreal x503 = stencil(evo_shiftU1, stencil_idx_0_m1_0_VVV); // x503: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m1__0_]
        vreal x504 = stencil(evo_shiftU1, stencil_idx_0_1_0_VVV); // x504: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__1__0_]
        vreal x505 = ((x503 + x504)); // x505: Dependency! Liveness = 2; [x503, x504]
        vreal x506 = stencil(evo_shiftU1, stencil_idx_0_0_m2_VVV); // x506: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m2_]
        vreal x507 = stencil(evo_shiftU1, stencil_idx_0_0_2_VVV); // x507: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__2_]
        vreal x508 = ((x506 + x507)); // x508: Dependency! Liveness = 2; [x506, x507]
        vreal x509 = stencil(evo_shiftU1, stencil_idx_0_0_m1_VVV); // x509: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m1_]
        vreal x510 = stencil(evo_shiftU1, stencil_idx_0_0_1_VVV); // x510: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__1_]
        vreal x511 = ((x509 + x510)); // x511: Dependency! Liveness = 2; [x509, x510]
        vreal x493 = ((-5.0 / 16.0) * stencil(evo_shiftU1, stencil_idx_0_0_0_VVV)); // x493: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__0_]
        store(evo_shift_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU1, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x493 + ((-3.0 / 32.0) * x496) + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU1, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x499))) + (DYI * (x493 + ((-3.0 / 32.0) * x502) + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU1, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x505))) + (DZI * (x493 + ((-3.0 / 32.0) * x508) + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU1, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x511)))))));
        vreal x513 = stencil(evo_shiftU2, stencil_idx_m2_0_0_VVV); // x513: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m2__0__0_]
        vreal x514 = stencil(evo_shiftU2, stencil_idx_2_0_0_VVV); // x514: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__2__0__0_]
        vreal x515 = ((x513 + x514)); // x515: Dependency! Liveness = 2; [x513, x514]
        vreal x516 = stencil(evo_shiftU2, stencil_idx_m1_0_0_VVV); // x516: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m1__0__0_]
        vreal x517 = stencil(evo_shiftU2, stencil_idx_1_0_0_VVV); // x517: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__1__0__0_]
        vreal x518 = ((x516 + x517)); // x518: Dependency! Liveness = 2; [x516, x517]
        vreal x519 = stencil(evo_shiftU2, stencil_idx_0_m2_0_VVV); // x519: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m2__0_]
        vreal x520 = stencil(evo_shiftU2, stencil_idx_0_2_0_VVV); // x520: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__2__0_]
        vreal x521 = ((x519 + x520)); // x521: Dependency! Liveness = 2; [x519, x520]
        vreal x522 = stencil(evo_shiftU2, stencil_idx_0_m1_0_VVV); // x522: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m1__0_]
        vreal x523 = stencil(evo_shiftU2, stencil_idx_0_1_0_VVV); // x523: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__1__0_]
        vreal x524 = ((x522 + x523)); // x524: Dependency! Liveness = 2; [x522, x523]
        vreal x525 = stencil(evo_shiftU2, stencil_idx_0_0_m2_VVV); // x525: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m2_]
        vreal x526 = stencil(evo_shiftU2, stencil_idx_0_0_2_VVV); // x526: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__2_]
        vreal x527 = ((x525 + x526)); // x527: Dependency! Liveness = 2; [x525, x526]
        vreal x528 = stencil(evo_shiftU2, stencil_idx_0_0_m1_VVV); // x528: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m1_]
        vreal x529 = stencil(evo_shiftU2, stencil_idx_0_0_1_VVV); // x529: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__1_]
        vreal x530 = ((x528 + x529)); // x530: Dependency! Liveness = 2; [x528, x529]
        vreal x512 = ((-5.0 / 16.0) * stencil(evo_shiftU2, stencil_idx_0_0_0_VVV)); // x512: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__0_]
        store(evo_shift_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU2, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x512 + ((-3.0 / 32.0) * x515) + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU2, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x518))) + (DYI * (x512 + ((-3.0 / 32.0) * x521) + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU2, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x524))) + (DZI * (x512 + ((-3.0 / 32.0) * x527) + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU2, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x530)))))));    
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
        vreal x536 = stencil(shift_BU0, stencil_idx_0_m2_0_VVV); // x536: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__m2__0_]
        vreal x537 = stencil(shift_BU0, stencil_idx_0_2_0_VVV); // x537: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__2__0_]
        vreal x538 = stencil(shift_BU0, stencil_idx_0_m1_0_VVV); // x538: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__m1__0_]
        vreal x539 = stencil(shift_BU0, stencil_idx_0_1_0_VVV); // x539: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__1__0_]
        vreal x540 = stencil(shift_BU0, stencil_idx_0_0_m2_VVV); // x540: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__m2_]
        vreal x541 = stencil(shift_BU0, stencil_idx_0_0_2_VVV); // x541: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__2_]
        vreal x542 = stencil(shift_BU0, stencil_idx_0_0_m1_VVV); // x542: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__m1_]
        vreal x543 = stencil(shift_BU0, stencil_idx_0_0_1_VVV); // x543: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__1_]
        vreal x532 = stencil(shift_BU0, stencil_idx_m2_0_0_VVV); // x532: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__m2__0__0_]
        vreal x533 = stencil(shift_BU0, stencil_idx_2_0_0_VVV); // x533: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__2__0__0_]
        vreal x534 = stencil(shift_BU0, stencil_idx_m1_0_0_VVV); // x534: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__m1__0__0_]
        vreal x535 = stencil(shift_BU0, stencil_idx_1_0_0_VVV); // x535: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__1__0__0_]
        vreal x531 = ((-5.0 / 16.0) * stencil(shift_BU0, stencil_idx_0_0_0_VVV)); // x531: Dependency! Liveness = 1; [__dummy_stencil__shift_BU0__0__0__0_]
        store(shift_B_rhsU0, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU0, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x531 + ((-3.0 / 32.0) * ((x532 + x533))) + ((1.0 / 64.0) * ((stencil(shift_BU0, stencil_idx_m3_0_0_VVV) + stencil(shift_BU0, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x534 + x535))))) + (DYI * (x531 + ((-3.0 / 32.0) * ((x536 + x537))) + ((1.0 / 64.0) * ((stencil(shift_BU0, stencil_idx_0_m3_0_VVV) + stencil(shift_BU0, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x538 + x539))))) + (DZI * (x531 + ((-3.0 / 32.0) * ((x540 + x541))) + ((1.0 / 64.0) * ((stencil(shift_BU0, stencil_idx_0_0_m3_VVV) + stencil(shift_BU0, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x542 + x543)))))))));
        vreal x549 = stencil(shift_BU1, stencil_idx_0_m2_0_VVV); // x549: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__m2__0_]
        vreal x550 = stencil(shift_BU1, stencil_idx_0_2_0_VVV); // x550: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__2__0_]
        vreal x551 = stencil(shift_BU1, stencil_idx_0_m1_0_VVV); // x551: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__m1__0_]
        vreal x552 = stencil(shift_BU1, stencil_idx_0_1_0_VVV); // x552: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__1__0_]
        vreal x553 = stencil(shift_BU1, stencil_idx_0_0_m2_VVV); // x553: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__m2_]
        vreal x554 = stencil(shift_BU1, stencil_idx_0_0_2_VVV); // x554: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__2_]
        vreal x555 = stencil(shift_BU1, stencil_idx_0_0_m1_VVV); // x555: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__m1_]
        vreal x556 = stencil(shift_BU1, stencil_idx_0_0_1_VVV); // x556: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__1_]
        vreal x545 = stencil(shift_BU1, stencil_idx_m2_0_0_VVV); // x545: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__m2__0__0_]
        vreal x546 = stencil(shift_BU1, stencil_idx_2_0_0_VVV); // x546: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__2__0__0_]
        vreal x547 = stencil(shift_BU1, stencil_idx_m1_0_0_VVV); // x547: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__m1__0__0_]
        vreal x548 = stencil(shift_BU1, stencil_idx_1_0_0_VVV); // x548: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__1__0__0_]
        vreal x544 = ((-5.0 / 16.0) * stencil(shift_BU1, stencil_idx_0_0_0_VVV)); // x544: Dependency! Liveness = 1; [__dummy_stencil__shift_BU1__0__0__0_]
        store(shift_B_rhsU1, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU1, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x544 + ((-3.0 / 32.0) * ((x545 + x546))) + ((1.0 / 64.0) * ((stencil(shift_BU1, stencil_idx_m3_0_0_VVV) + stencil(shift_BU1, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x547 + x548))))) + (DYI * (x544 + ((-3.0 / 32.0) * ((x549 + x550))) + ((1.0 / 64.0) * ((stencil(shift_BU1, stencil_idx_0_m3_0_VVV) + stencil(shift_BU1, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x551 + x552))))) + (DZI * (x544 + ((-3.0 / 32.0) * ((x553 + x554))) + ((1.0 / 64.0) * ((stencil(shift_BU1, stencil_idx_0_0_m3_VVV) + stencil(shift_BU1, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x555 + x556)))))))));
        vreal x562 = stencil(shift_BU2, stencil_idx_0_m2_0_VVV); // x562: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__m2__0_]
        vreal x563 = stencil(shift_BU2, stencil_idx_0_2_0_VVV); // x563: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__2__0_]
        vreal x564 = stencil(shift_BU2, stencil_idx_0_m1_0_VVV); // x564: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__m1__0_]
        vreal x565 = stencil(shift_BU2, stencil_idx_0_1_0_VVV); // x565: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__1__0_]
        vreal x566 = stencil(shift_BU2, stencil_idx_0_0_m2_VVV); // x566: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__m2_]
        vreal x567 = stencil(shift_BU2, stencil_idx_0_0_2_VVV); // x567: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__2_]
        vreal x568 = stencil(shift_BU2, stencil_idx_0_0_m1_VVV); // x568: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__m1_]
        vreal x569 = stencil(shift_BU2, stencil_idx_0_0_1_VVV); // x569: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__1_]
        vreal x558 = stencil(shift_BU2, stencil_idx_m2_0_0_VVV); // x558: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__m2__0__0_]
        vreal x559 = stencil(shift_BU2, stencil_idx_2_0_0_VVV); // x559: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__2__0__0_]
        vreal x560 = stencil(shift_BU2, stencil_idx_m1_0_0_VVV); // x560: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__m1__0__0_]
        vreal x561 = stencil(shift_BU2, stencil_idx_1_0_0_VVV); // x561: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__1__0__0_]
        vreal x557 = ((-5.0 / 16.0) * stencil(shift_BU2, stencil_idx_0_0_0_VVV)); // x557: Dependency! Liveness = 1; [__dummy_stencil__shift_BU2__0__0__0_]
        store(shift_B_rhsU2, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU2, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x557 + ((-3.0 / 32.0) * ((x558 + x559))) + ((1.0 / 64.0) * ((stencil(shift_BU2, stencil_idx_m3_0_0_VVV) + stencil(shift_BU2, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x560 + x561))))) + (DYI * (x557 + ((-3.0 / 32.0) * ((x562 + x563))) + ((1.0 / 64.0) * ((stencil(shift_BU2, stencil_idx_0_m3_0_VVV) + stencil(shift_BU2, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x564 + x565))))) + (DZI * (x557 + ((-3.0 / 32.0) * ((x566 + x567))) + ((1.0 / 64.0) * ((stencil(shift_BU2, stencil_idx_0_0_m3_VVV) + stencil(shift_BU2, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x568 + x569)))))))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}