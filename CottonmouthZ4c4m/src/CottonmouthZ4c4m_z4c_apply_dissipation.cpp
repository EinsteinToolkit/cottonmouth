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
void z4c_apply_dissipation(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_z4c_apply_dissipation;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("z4c_apply_dissipation");
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
    #define Theta_layout VVV_layout
    #define Theta_rhs_layout VVV_layout
    #define chi_layout VVV_layout
    #define chi_rhs_layout VVV_layout
    #define evo_GammatU0_layout VVV_layout
    #define evo_GammatU1_layout VVV_layout
    #define evo_GammatU2_layout VVV_layout
    #define evo_Gammat_rhsU0_layout VVV_layout
    #define evo_Gammat_rhsU1_layout VVV_layout
    #define evo_Gammat_rhsU2_layout VVV_layout
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
    #define trK_layout VVV_layout
    #define trK_rhs_layout VVV_layout
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    CCTK_ASSERT((cctk_nghostzones[0] >= 3));
    CCTK_ASSERT((cctk_nghostzones[1] >= 3));
    CCTK_ASSERT((cctk_nghostzones[2] >= 3));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // z4c_apply_dissipation loop 0
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
        vreal x232 = stencil(Theta, stencil_idx_0_m2_0_VVV); // x232: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__m2__0_]
        vreal x233 = stencil(Theta, stencil_idx_0_2_0_VVV); // x233: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__2__0_]
        vreal x234 = stencil(Theta, stencil_idx_0_m1_0_VVV); // x234: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__m1__0_]
        vreal x235 = stencil(Theta, stencil_idx_0_1_0_VVV); // x235: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__1__0_]
        vreal x236 = stencil(Theta, stencil_idx_0_0_m2_VVV); // x236: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__m2_]
        vreal x237 = stencil(Theta, stencil_idx_0_0_2_VVV); // x237: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__2_]
        vreal x238 = stencil(Theta, stencil_idx_0_0_m1_VVV); // x238: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__m1_]
        vreal x239 = stencil(Theta, stencil_idx_0_0_1_VVV); // x239: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__1_]
        vreal x228 = stencil(Theta, stencil_idx_m2_0_0_VVV); // x228: Dependency! Liveness = 1; [__dummy_stencil__Theta__m2__0__0_]
        vreal x229 = stencil(Theta, stencil_idx_2_0_0_VVV); // x229: Dependency! Liveness = 1; [__dummy_stencil__Theta__2__0__0_]
        vreal x230 = stencil(Theta, stencil_idx_m1_0_0_VVV); // x230: Dependency! Liveness = 1; [__dummy_stencil__Theta__m1__0__0_]
        vreal x231 = stencil(Theta, stencil_idx_1_0_0_VVV); // x231: Dependency! Liveness = 1; [__dummy_stencil__Theta__1__0__0_]
        vreal x227 = ((-5.0 / 16.0) * stencil(Theta, stencil_idx_0_0_0_VVV)); // x227: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__0_]
        store(Theta_rhs, stencil_idx_0_0_0_VVV, (access(Theta_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x227 + ((-3.0 / 32.0) * ((x228 + x229))) + ((1.0 / 64.0) * ((stencil(Theta, stencil_idx_m3_0_0_VVV) + stencil(Theta, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x230 + x231))))) + (DYI * (x227 + ((-3.0 / 32.0) * ((x232 + x233))) + ((1.0 / 64.0) * ((stencil(Theta, stencil_idx_0_m3_0_VVV) + stencil(Theta, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x234 + x235))))) + (DZI * (x227 + ((-3.0 / 32.0) * ((x236 + x237))) + ((1.0 / 64.0) * ((stencil(Theta, stencil_idx_0_0_m3_VVV) + stencil(Theta, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x238 + x239)))))))));    
    });
    // z4c_apply_dissipation loop 1
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
        vreal x241 = stencil(chi, stencil_idx_m2_0_0_VVV); // x241: Dependency! Liveness = 1; [__dummy_stencil__chi__m2__0__0_]
        vreal x242 = stencil(chi, stencil_idx_2_0_0_VVV); // x242: Dependency! Liveness = 1; [__dummy_stencil__chi__2__0__0_]
        vreal x243 = ((x241 + x242)); // x243: Dependency! Liveness = 2; [x241, x242]
        x241 = stencil(chi, stencil_idx_m1_0_0_VVV); // x244: Dependency! Liveness = 1; [__dummy_stencil__chi__m1__0__0_]
        x242 = stencil(chi, stencil_idx_1_0_0_VVV); // x245: Dependency! Liveness = 1; [__dummy_stencil__chi__1__0__0_]
        vreal x246 = ((x241 + x242)); // x246: Dependency! Liveness = 2; [x244, x245]
        vreal x247 = stencil(chi, stencil_idx_0_m2_0_VVV); // x247: Dependency! Liveness = 1; [__dummy_stencil__chi__0__m2__0_]
        vreal x248 = stencil(chi, stencil_idx_0_2_0_VVV); // x248: Dependency! Liveness = 1; [__dummy_stencil__chi__0__2__0_]
        vreal x249 = ((x247 + x248)); // x249: Dependency! Liveness = 2; [x247, x248]
        x247 = stencil(chi, stencil_idx_0_m1_0_VVV); // x250: Dependency! Liveness = 1; [__dummy_stencil__chi__0__m1__0_]
        x248 = stencil(chi, stencil_idx_0_1_0_VVV); // x251: Dependency! Liveness = 1; [__dummy_stencil__chi__0__1__0_]
        vreal x252 = ((x247 + x248)); // x252: Dependency! Liveness = 2; [x250, x251]
        vreal x253 = stencil(chi, stencil_idx_0_0_m2_VVV); // x253: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__m2_]
        vreal x254 = stencil(chi, stencil_idx_0_0_2_VVV); // x254: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__2_]
        vreal x255 = ((x253 + x254)); // x255: Dependency! Liveness = 2; [x253, x254]
        x253 = stencil(chi, stencil_idx_0_0_m1_VVV); // x256: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__m1_]
        x254 = stencil(chi, stencil_idx_0_0_1_VVV); // x257: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__1_]
        vreal x258 = ((x253 + x254)); // x258: Dependency! Liveness = 2; [x256, x257]
        vreal x240 = ((-5.0 / 16.0) * stencil(chi, stencil_idx_0_0_0_VVV)); // x240: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__0_]
        store(chi_rhs, stencil_idx_0_0_0_VVV, (access(chi_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x240 + ((-3.0 / 32.0) * x243) + ((1.0 / 64.0) * ((stencil(chi, stencil_idx_m3_0_0_VVV) + stencil(chi, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x246))) + (DYI * (x240 + ((-3.0 / 32.0) * x249) + ((1.0 / 64.0) * ((stencil(chi, stencil_idx_0_m3_0_VVV) + stencil(chi, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x252))) + (DZI * (x240 + ((-3.0 / 32.0) * x255) + ((1.0 / 64.0) * ((stencil(chi, stencil_idx_0_0_m3_VVV) + stencil(chi, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x258)))))));    
    });
    // z4c_apply_dissipation loop 2
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
        vreal x264 = stencil(trK, stencil_idx_0_m2_0_VVV); // x264: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m2__0_]
        vreal x265 = stencil(trK, stencil_idx_0_2_0_VVV); // x265: Dependency! Liveness = 1; [__dummy_stencil__trK__0__2__0_]
        vreal x266 = stencil(trK, stencil_idx_0_m1_0_VVV); // x266: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m1__0_]
        vreal x267 = stencil(trK, stencil_idx_0_1_0_VVV); // x267: Dependency! Liveness = 1; [__dummy_stencil__trK__0__1__0_]
        vreal x268 = stencil(trK, stencil_idx_0_0_m2_VVV); // x268: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m2_]
        vreal x269 = stencil(trK, stencil_idx_0_0_2_VVV); // x269: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__2_]
        vreal x270 = stencil(trK, stencil_idx_0_0_m1_VVV); // x270: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m1_]
        vreal x271 = stencil(trK, stencil_idx_0_0_1_VVV); // x271: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__1_]
        vreal x260 = stencil(trK, stencil_idx_m2_0_0_VVV); // x260: Dependency! Liveness = 1; [__dummy_stencil__trK__m2__0__0_]
        vreal x261 = stencil(trK, stencil_idx_2_0_0_VVV); // x261: Dependency! Liveness = 1; [__dummy_stencil__trK__2__0__0_]
        vreal x262 = stencil(trK, stencil_idx_m1_0_0_VVV); // x262: Dependency! Liveness = 1; [__dummy_stencil__trK__m1__0__0_]
        vreal x263 = stencil(trK, stencil_idx_1_0_0_VVV); // x263: Dependency! Liveness = 1; [__dummy_stencil__trK__1__0__0_]
        vreal x259 = ((-5.0 / 16.0) * stencil(trK, stencil_idx_0_0_0_VVV)); // x259: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__0_]
        store(trK_rhs, stencil_idx_0_0_0_VVV, (access(trK_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x259 + ((-3.0 / 32.0) * ((x260 + x261))) + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_m3_0_0_VVV) + stencil(trK, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x262 + x263))))) + (DYI * (x259 + ((-3.0 / 32.0) * ((x264 + x265))) + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_0_m3_0_VVV) + stencil(trK, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x266 + x267))))) + (DZI * (x259 + ((-3.0 / 32.0) * ((x268 + x269))) + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_0_0_m3_VVV) + stencil(trK, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x270 + x271)))))))));    
    });
    // z4c_apply_dissipation loop 3
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
        vreal x277 = stencil(evo_GammatU0, stencil_idx_0_m2_0_VVV); // x277: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__m2__0_]
        vreal x278 = stencil(evo_GammatU0, stencil_idx_0_2_0_VVV); // x278: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__2__0_]
        vreal x279 = stencil(evo_GammatU0, stencil_idx_0_m1_0_VVV); // x279: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__m1__0_]
        vreal x280 = stencil(evo_GammatU0, stencil_idx_0_1_0_VVV); // x280: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__1__0_]
        vreal x281 = stencil(evo_GammatU0, stencil_idx_0_0_m2_VVV); // x281: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__m2_]
        vreal x282 = stencil(evo_GammatU0, stencil_idx_0_0_2_VVV); // x282: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__2_]
        vreal x283 = stencil(evo_GammatU0, stencil_idx_0_0_m1_VVV); // x283: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__m1_]
        vreal x284 = stencil(evo_GammatU0, stencil_idx_0_0_1_VVV); // x284: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__1_]
        vreal x273 = stencil(evo_GammatU0, stencil_idx_m2_0_0_VVV); // x273: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__m2__0__0_]
        vreal x274 = stencil(evo_GammatU0, stencil_idx_2_0_0_VVV); // x274: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__2__0__0_]
        vreal x275 = stencil(evo_GammatU0, stencil_idx_m1_0_0_VVV); // x275: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__m1__0__0_]
        vreal x276 = stencil(evo_GammatU0, stencil_idx_1_0_0_VVV); // x276: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__1__0__0_]
        vreal x272 = ((-5.0 / 16.0) * stencil(evo_GammatU0, stencil_idx_0_0_0_VVV)); // x272: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__0_]
        store(evo_Gammat_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU0, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x272 + ((-3.0 / 32.0) * ((x273 + x274))) + ((1.0 / 64.0) * ((stencil(evo_GammatU0, stencil_idx_m3_0_0_VVV) + stencil(evo_GammatU0, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x275 + x276))))) + (DYI * (x272 + ((-3.0 / 32.0) * ((x277 + x278))) + ((1.0 / 64.0) * ((stencil(evo_GammatU0, stencil_idx_0_m3_0_VVV) + stencil(evo_GammatU0, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x279 + x280))))) + (DZI * (x272 + ((-3.0 / 32.0) * ((x281 + x282))) + ((1.0 / 64.0) * ((stencil(evo_GammatU0, stencil_idx_0_0_m3_VVV) + stencil(evo_GammatU0, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x283 + x284)))))))));
        x272 = stencil(evo_GammatU1, stencil_idx_0_m2_0_VVV); // x290: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__m2__0_]
        x273 = stencil(evo_GammatU1, stencil_idx_0_2_0_VVV); // x291: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__2__0_]
        x274 = stencil(evo_GammatU1, stencil_idx_0_m1_0_VVV); // x292: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__m1__0_]
        x275 = stencil(evo_GammatU1, stencil_idx_0_1_0_VVV); // x293: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__1__0_]
        x276 = stencil(evo_GammatU1, stencil_idx_0_0_m2_VVV); // x294: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__m2_]
        x277 = stencil(evo_GammatU1, stencil_idx_0_0_2_VVV); // x295: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__2_]
        x278 = stencil(evo_GammatU1, stencil_idx_0_0_m1_VVV); // x296: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__m1_]
        x279 = stencil(evo_GammatU1, stencil_idx_0_0_1_VVV); // x297: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__1_]
        x280 = stencil(evo_GammatU1, stencil_idx_m2_0_0_VVV); // x286: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__m2__0__0_]
        x281 = stencil(evo_GammatU1, stencil_idx_2_0_0_VVV); // x287: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__2__0__0_]
        x282 = stencil(evo_GammatU1, stencil_idx_m1_0_0_VVV); // x288: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__m1__0__0_]
        x283 = stencil(evo_GammatU1, stencil_idx_1_0_0_VVV); // x289: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__1__0__0_]
        x284 = ((-5.0 / 16.0) * stencil(evo_GammatU1, stencil_idx_0_0_0_VVV)); // x285: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__0_]
        store(evo_Gammat_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU1, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x284 + ((-3.0 / 32.0) * ((x280 + x281))) + ((1.0 / 64.0) * ((stencil(evo_GammatU1, stencil_idx_m3_0_0_VVV) + stencil(evo_GammatU1, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x282 + x283))))) + (DYI * (x284 + ((-3.0 / 32.0) * ((x272 + x273))) + ((1.0 / 64.0) * ((stencil(evo_GammatU1, stencil_idx_0_m3_0_VVV) + stencil(evo_GammatU1, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x274 + x275))))) + (DZI * (x284 + ((-3.0 / 32.0) * ((x276 + x277))) + ((1.0 / 64.0) * ((stencil(evo_GammatU1, stencil_idx_0_0_m3_VVV) + stencil(evo_GammatU1, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x278 + x279)))))))));
        vreal x303 = stencil(evo_GammatU2, stencil_idx_0_m2_0_VVV); // x303: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__m2__0_]
        vreal x304 = stencil(evo_GammatU2, stencil_idx_0_2_0_VVV); // x304: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__2__0_]
        vreal x305 = stencil(evo_GammatU2, stencil_idx_0_m1_0_VVV); // x305: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__m1__0_]
        vreal x306 = stencil(evo_GammatU2, stencil_idx_0_1_0_VVV); // x306: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__1__0_]
        vreal x307 = stencil(evo_GammatU2, stencil_idx_0_0_m2_VVV); // x307: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__m2_]
        vreal x308 = stencil(evo_GammatU2, stencil_idx_0_0_2_VVV); // x308: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__2_]
        vreal x309 = stencil(evo_GammatU2, stencil_idx_0_0_m1_VVV); // x309: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__m1_]
        vreal x310 = stencil(evo_GammatU2, stencil_idx_0_0_1_VVV); // x310: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__1_]
        vreal x299 = stencil(evo_GammatU2, stencil_idx_m2_0_0_VVV); // x299: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__m2__0__0_]
        vreal x300 = stencil(evo_GammatU2, stencil_idx_2_0_0_VVV); // x300: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__2__0__0_]
        vreal x301 = stencil(evo_GammatU2, stencil_idx_m1_0_0_VVV); // x301: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__m1__0__0_]
        vreal x302 = stencil(evo_GammatU2, stencil_idx_1_0_0_VVV); // x302: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__1__0__0_]
        vreal x298 = ((-5.0 / 16.0) * stencil(evo_GammatU2, stencil_idx_0_0_0_VVV)); // x298: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__0_]
        store(evo_Gammat_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU2, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x298 + ((-3.0 / 32.0) * ((x299 + x300))) + ((1.0 / 64.0) * ((stencil(evo_GammatU2, stencil_idx_m3_0_0_VVV) + stencil(evo_GammatU2, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x301 + x302))))) + (DYI * (x298 + ((-3.0 / 32.0) * ((x303 + x304))) + ((1.0 / 64.0) * ((stencil(evo_GammatU2, stencil_idx_0_m3_0_VVV) + stencil(evo_GammatU2, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x305 + x306))))) + (DZI * (x298 + ((-3.0 / 32.0) * ((x307 + x308))) + ((1.0 / 64.0) * ((stencil(evo_GammatU2, stencil_idx_0_0_m3_VVV) + stencil(evo_GammatU2, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x309 + x310)))))))));    
    });
    // z4c_apply_dissipation loop 4
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
        vreal x170 = stencil(gtDD00, stencil_idx_m2_0_0_VVV); // x170: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m2__0__0_]
        vreal x171 = stencil(gtDD00, stencil_idx_2_0_0_VVV); // x171: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__2__0__0_]
        vreal x312 = ((x170 + x171)); // x312: Dependency! Liveness = 2; [x170, x171]
        x170 = stencil(gtDD00, stencil_idx_m1_0_0_VVV); // x172: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m1__0__0_]
        x171 = stencil(gtDD00, stencil_idx_1_0_0_VVV); // x173: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__1__0__0_]
        vreal x313 = ((x170 + x171)); // x313: Dependency! Liveness = 2; [x172, x173]
        vreal x109 = stencil(gtDD00, stencil_idx_0_m2_0_VVV); // x109: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m2__0_]
        vreal x110 = stencil(gtDD00, stencil_idx_0_2_0_VVV); // x110: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__2__0_]
        vreal x314 = ((x109 + x110)); // x314: Dependency! Liveness = 2; [x109, x110]
        x109 = stencil(gtDD00, stencil_idx_0_m1_0_VVV); // x111: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m1__0_]
        x110 = stencil(gtDD00, stencil_idx_0_1_0_VVV); // x112: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__1__0_]
        vreal x315 = ((x109 + x110)); // x315: Dependency! Liveness = 2; [x111, x112]
        vreal x94 = stencil(gtDD00, stencil_idx_0_0_m2_VVV); // x94: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m2_]
        vreal x95 = stencil(gtDD00, stencil_idx_0_0_2_VVV); // x95: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__2_]
        vreal x316 = ((x94 + x95)); // x316: Dependency! Liveness = 2; [x94, x95]
        x94 = stencil(gtDD00, stencil_idx_0_0_m1_VVV); // x96: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m1_]
        x95 = stencil(gtDD00, stencil_idx_0_0_1_VVV); // x97: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__1_]
        vreal x317 = ((x94 + x95)); // x317: Dependency! Liveness = 2; [x96, x97]
        vreal x311 = ((-5.0 / 16.0) * stencil(gtDD00, stencil_idx_0_0_0_VVV)); // x311: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__0_]
        store(gt_rhsDD00, stencil_idx_0_0_0_VVV, (access(gt_rhsDD00, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x311 + ((-3.0 / 32.0) * x312) + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_m3_0_0_VVV) + stencil(gtDD00, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x313))) + (DYI * (x311 + ((-3.0 / 32.0) * x314) + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_0_m3_0_VVV) + stencil(gtDD00, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x315))) + (DZI * (x311 + ((-3.0 / 32.0) * x316) + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_0_0_m3_VVV) + stencil(gtDD00, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x317)))))));
        x311 = stencil(gtDD01, stencil_idx_m2_0_0_VVV); // x175: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m2__0__0_]
        x312 = stencil(gtDD01, stencil_idx_2_0_0_VVV); // x176: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__2__0__0_]
        x313 = ((x311 + x312)); // x319: Dependency! Liveness = 2; [x175, x176]
        x314 = stencil(gtDD01, stencil_idx_m1_0_0_VVV); // x177: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m1__0__0_]
        x315 = stencil(gtDD01, stencil_idx_1_0_0_VVV); // x178: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__1__0__0_]
        x316 = ((x314 + x315)); // x320: Dependency! Liveness = 2; [x177, x178]
        x317 = stencil(gtDD01, stencil_idx_0_m2_0_VVV); // x128: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m2__0_]
        vreal x129 = stencil(gtDD01, stencil_idx_0_2_0_VVV); // x129: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__2__0_]
        vreal x321 = ((x129 + x317)); // x321: Dependency! Liveness = 2; [x128, x129]
        x129 = stencil(gtDD01, stencil_idx_0_m1_0_VVV); // x130: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m1__0_]
        vreal x131 = stencil(gtDD01, stencil_idx_0_1_0_VVV); // x131: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__1__0_]
        vreal x322 = ((x129 + x131)); // x322: Dependency! Liveness = 2; [x130, x131]
        x131 = stencil(gtDD01, stencil_idx_0_0_m2_VVV); // x75: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m2_]
        vreal x76 = stencil(gtDD01, stencil_idx_0_0_2_VVV); // x76: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__2_]
        vreal x323 = ((x131 + x76)); // x323: Dependency! Liveness = 2; [x75, x76]
        x76 = stencil(gtDD01, stencil_idx_0_0_m1_VVV); // x77: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m1_]
        vreal x78 = stencil(gtDD01, stencil_idx_0_0_1_VVV); // x78: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__1_]
        vreal x324 = ((x76 + x78)); // x324: Dependency! Liveness = 2; [x77, x78]
        x78 = ((-5.0 / 16.0) * stencil(gtDD01, stencil_idx_0_0_0_VVV)); // x318: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__0_]
        store(gt_rhsDD01, stencil_idx_0_0_0_VVV, (access(gt_rhsDD01, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x78 + ((-3.0 / 32.0) * x313) + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_m3_0_0_VVV) + stencil(gtDD01, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x316))) + (DYI * (x78 + ((-3.0 / 32.0) * x321) + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_0_m3_0_VVV) + stencil(gtDD01, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x322))) + (DZI * (x78 + ((-3.0 / 32.0) * x323) + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_0_0_m3_VVV) + stencil(gtDD01, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x324)))))));
        x321 = stencil(gtDD02, stencil_idx_m2_0_0_VVV); // x184: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m2__0__0_]
        x322 = stencil(gtDD02, stencil_idx_2_0_0_VVV); // x185: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__2__0__0_]
        x323 = ((x321 + x322)); // x326: Dependency! Liveness = 2; [x184, x185]
        x324 = stencil(gtDD02, stencil_idx_m1_0_0_VVV); // x186: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m1__0__0_]
        vreal x187 = stencil(gtDD02, stencil_idx_1_0_0_VVV); // x187: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__1__0__0_]
        vreal x327 = ((x187 + x324)); // x327: Dependency! Liveness = 2; [x186, x187]
        x187 = stencil(gtDD02, stencil_idx_0_m2_0_VVV); // x70: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m2__0_]
        vreal x71 = stencil(gtDD02, stencil_idx_0_2_0_VVV); // x71: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__2__0_]
        vreal x328 = ((x187 + x71)); // x328: Dependency! Liveness = 2; [x70, x71]
        x71 = stencil(gtDD02, stencil_idx_0_m1_0_VVV); // x72: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m1__0_]
        vreal x73 = stencil(gtDD02, stencil_idx_0_1_0_VVV); // x73: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__1__0_]
        vreal x329 = ((x71 + x73)); // x329: Dependency! Liveness = 2; [x72, x73]
        x73 = stencil(gtDD02, stencil_idx_0_0_m2_VVV); // x161: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m2_]
        vreal x162 = stencil(gtDD02, stencil_idx_0_0_2_VVV); // x162: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__2_]
        vreal x330 = ((x162 + x73)); // x330: Dependency! Liveness = 2; [x161, x162]
        x162 = stencil(gtDD02, stencil_idx_0_0_m1_VVV); // x163: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m1_]
        vreal x164 = stencil(gtDD02, stencil_idx_0_0_1_VVV); // x164: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__1_]
        vreal x331 = ((x162 + x164)); // x331: Dependency! Liveness = 2; [x163, x164]
        x164 = ((-5.0 / 16.0) * stencil(gtDD02, stencil_idx_0_0_0_VVV)); // x325: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__0_]
        store(gt_rhsDD02, stencil_idx_0_0_0_VVV, (access(gt_rhsDD02, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x164 + ((-3.0 / 32.0) * x323) + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_m3_0_0_VVV) + stencil(gtDD02, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x327))) + (DYI * (x164 + ((-3.0 / 32.0) * x328) + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_0_m3_0_VVV) + stencil(gtDD02, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x329))) + (DZI * (x164 + ((-3.0 / 32.0) * x330) + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_0_0_m3_VVV) + stencil(gtDD02, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x331)))))));
        x327 = stencil(gtDD11, stencil_idx_m2_0_0_VVV); // x103: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m2__0__0_]
        x328 = stencil(gtDD11, stencil_idx_2_0_0_VVV); // x104: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__2__0__0_]
        x329 = ((x327 + x328)); // x333: Dependency! Liveness = 2; [x103, x104]
        x330 = stencil(gtDD11, stencil_idx_m1_0_0_VVV); // x105: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m1__0__0_]
        x331 = stencil(gtDD11, stencil_idx_1_0_0_VVV); // x106: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__1__0__0_]
        vreal x334 = ((x330 + x331)); // x334: Dependency! Liveness = 2; [x105, x106]
        vreal x122 = stencil(gtDD11, stencil_idx_0_m2_0_VVV); // x122: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m2__0_]
        vreal x123 = stencil(gtDD11, stencil_idx_0_2_0_VVV); // x123: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__2__0_]
        vreal x335 = ((x122 + x123)); // x335: Dependency! Liveness = 2; [x122, x123]
        x122 = stencil(gtDD11, stencil_idx_0_m1_0_VVV); // x124: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m1__0_]
        x123 = stencil(gtDD11, stencil_idx_0_1_0_VVV); // x125: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__1__0_]
        vreal x336 = ((x122 + x123)); // x336: Dependency! Liveness = 2; [x124, x125]
        vreal x60 = stencil(gtDD11, stencil_idx_0_0_m2_VVV); // x60: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m2_]
        vreal x61 = stencil(gtDD11, stencil_idx_0_0_2_VVV); // x61: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__2_]
        vreal x337 = ((x60 + x61)); // x337: Dependency! Liveness = 2; [x60, x61]
        x60 = stencil(gtDD11, stencil_idx_0_0_m1_VVV); // x62: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m1_]
        x61 = stencil(gtDD11, stencil_idx_0_0_1_VVV); // x63: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__1_]
        vreal x338 = ((x60 + x61)); // x338: Dependency! Liveness = 2; [x62, x63]
        vreal x332 = ((-5.0 / 16.0) * stencil(gtDD11, stencil_idx_0_0_0_VVV)); // x332: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__0_]
        store(gt_rhsDD11, stencil_idx_0_0_0_VVV, (access(gt_rhsDD11, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x332 + ((-3.0 / 32.0) * x329) + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_m3_0_0_VVV) + stencil(gtDD11, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x334))) + (DYI * (x332 + ((-3.0 / 32.0) * x335) + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_0_m3_0_VVV) + stencil(gtDD11, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x336))) + (DZI * (x332 + ((-3.0 / 32.0) * x337) + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_0_0_m3_VVV) + stencil(gtDD11, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x338)))))));
        x332 = stencil(gtDD12, stencil_idx_m2_0_0_VVV); // x81: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m2__0__0_]
        x334 = stencil(gtDD12, stencil_idx_2_0_0_VVV); // x82: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__2__0__0_]
        x335 = ((x332 + x334)); // x340: Dependency! Liveness = 2; [x81, x82]
        x336 = stencil(gtDD12, stencil_idx_m1_0_0_VVV); // x83: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m1__0__0_]
        x337 = stencil(gtDD12, stencil_idx_1_0_0_VVV); // x84: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__1__0__0_]
        x338 = ((x336 + x337)); // x341: Dependency! Liveness = 2; [x83, x84]
        vreal x135 = stencil(gtDD12, stencil_idx_0_m2_0_VVV); // x135: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m2__0_]
        vreal x136 = stencil(gtDD12, stencil_idx_0_2_0_VVV); // x136: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__2__0_]
        vreal x342 = ((x135 + x136)); // x342: Dependency! Liveness = 2; [x135, x136]
        x135 = stencil(gtDD12, stencil_idx_0_m1_0_VVV); // x137: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m1__0_]
        x136 = stencil(gtDD12, stencil_idx_0_1_0_VVV); // x138: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__1__0_]
        vreal x343 = ((x135 + x136)); // x343: Dependency! Liveness = 2; [x137, x138]
        vreal x153 = stencil(gtDD12, stencil_idx_0_0_m2_VVV); // x153: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m2_]
        vreal x154 = stencil(gtDD12, stencil_idx_0_0_2_VVV); // x154: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__2_]
        vreal x344 = ((x153 + x154)); // x344: Dependency! Liveness = 2; [x153, x154]
        x153 = stencil(gtDD12, stencil_idx_0_0_m1_VVV); // x155: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m1_]
        x154 = stencil(gtDD12, stencil_idx_0_0_1_VVV); // x156: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__1_]
        vreal x345 = ((x153 + x154)); // x345: Dependency! Liveness = 2; [x155, x156]
        vreal x339 = ((-5.0 / 16.0) * stencil(gtDD12, stencil_idx_0_0_0_VVV)); // x339: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__0_]
        store(gt_rhsDD12, stencil_idx_0_0_0_VVV, (access(gt_rhsDD12, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x339 + ((-3.0 / 32.0) * x335) + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_m3_0_0_VVV) + stencil(gtDD12, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x338))) + (DYI * (x339 + ((-3.0 / 32.0) * x342) + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_0_m3_0_VVV) + stencil(gtDD12, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x343))) + (DZI * (x339 + ((-3.0 / 32.0) * x344) + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_0_0_m3_VVV) + stencil(gtDD12, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x345)))))));
        x339 = stencil(gtDD22, stencil_idx_m2_0_0_VVV); // x88: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m2__0__0_]
        x342 = stencil(gtDD22, stencil_idx_2_0_0_VVV); // x89: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__2__0__0_]
        x343 = ((x339 + x342)); // x347: Dependency! Liveness = 2; [x88, x89]
        x344 = stencil(gtDD22, stencil_idx_m1_0_0_VVV); // x90: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m1__0__0_]
        x345 = stencil(gtDD22, stencil_idx_1_0_0_VVV); // x91: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__1__0__0_]
        vreal x348 = ((x344 + x345)); // x348: Dependency! Liveness = 2; [x90, x91]
        vreal x49 = stencil(gtDD22, stencil_idx_0_m2_0_VVV); // x49: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m2__0_]
        vreal x50 = stencil(gtDD22, stencil_idx_0_2_0_VVV); // x50: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__2__0_]
        vreal x349 = ((x49 + x50)); // x349: Dependency! Liveness = 2; [x49, x50]
        x49 = stencil(gtDD22, stencil_idx_0_m1_0_VVV); // x51: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m1__0_]
        x50 = stencil(gtDD22, stencil_idx_0_1_0_VVV); // x52: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__1__0_]
        vreal x350 = ((x49 + x50)); // x350: Dependency! Liveness = 2; [x51, x52]
        vreal x147 = stencil(gtDD22, stencil_idx_0_0_m2_VVV); // x147: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m2_]
        vreal x148 = stencil(gtDD22, stencil_idx_0_0_2_VVV); // x148: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__2_]
        vreal x351 = ((x147 + x148)); // x351: Dependency! Liveness = 2; [x147, x148]
        x147 = stencil(gtDD22, stencil_idx_0_0_m1_VVV); // x149: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m1_]
        x148 = stencil(gtDD22, stencil_idx_0_0_1_VVV); // x150: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__1_]
        vreal x352 = ((x147 + x148)); // x352: Dependency! Liveness = 2; [x149, x150]
        vreal x346 = ((-5.0 / 16.0) * stencil(gtDD22, stencil_idx_0_0_0_VVV)); // x346: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__0_]
        store(gt_rhsDD22, stencil_idx_0_0_0_VVV, (access(gt_rhsDD22, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x346 + ((-3.0 / 32.0) * x343) + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_m3_0_0_VVV) + stencil(gtDD22, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x348))) + (DYI * (x346 + ((-3.0 / 32.0) * x349) + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_0_m3_0_VVV) + stencil(gtDD22, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x350))) + (DZI * (x346 + ((-3.0 / 32.0) * x351) + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_0_0_m3_VVV) + stencil(gtDD22, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x352)))))));    
    });
    // z4c_apply_dissipation loop 5
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
        vreal x358 = stencil(AtDD00, stencil_idx_0_m2_0_VVV); // x358: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m2__0_]
        vreal x359 = stencil(AtDD00, stencil_idx_0_2_0_VVV); // x359: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__2__0_]
        vreal x360 = stencil(AtDD00, stencil_idx_0_m1_0_VVV); // x360: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m1__0_]
        vreal x361 = stencil(AtDD00, stencil_idx_0_1_0_VVV); // x361: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__1__0_]
        vreal x362 = stencil(AtDD00, stencil_idx_0_0_m2_VVV); // x362: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m2_]
        vreal x363 = stencil(AtDD00, stencil_idx_0_0_2_VVV); // x363: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__2_]
        vreal x364 = stencil(AtDD00, stencil_idx_0_0_m1_VVV); // x364: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m1_]
        vreal x365 = stencil(AtDD00, stencil_idx_0_0_1_VVV); // x365: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__1_]
        vreal x354 = stencil(AtDD00, stencil_idx_m2_0_0_VVV); // x354: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m2__0__0_]
        vreal x355 = stencil(AtDD00, stencil_idx_2_0_0_VVV); // x355: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__2__0__0_]
        vreal x356 = stencil(AtDD00, stencil_idx_m1_0_0_VVV); // x356: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m1__0__0_]
        vreal x357 = stencil(AtDD00, stencil_idx_1_0_0_VVV); // x357: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__1__0__0_]
        vreal x353 = ((-5.0 / 16.0) * stencil(AtDD00, stencil_idx_0_0_0_VVV)); // x353: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__0_]
        store(At_rhsDD00, stencil_idx_0_0_0_VVV, (access(At_rhsDD00, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x353 + ((-3.0 / 32.0) * ((x354 + x355))) + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_m3_0_0_VVV) + stencil(AtDD00, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x356 + x357))))) + (DYI * (x353 + ((-3.0 / 32.0) * ((x358 + x359))) + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_0_m3_0_VVV) + stencil(AtDD00, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x360 + x361))))) + (DZI * (x353 + ((-3.0 / 32.0) * ((x362 + x363))) + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_0_0_m3_VVV) + stencil(AtDD00, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x364 + x365)))))))));
        x353 = stencil(AtDD01, stencil_idx_0_m2_0_VVV); // x371: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m2__0_]
        x354 = stencil(AtDD01, stencil_idx_0_2_0_VVV); // x372: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__2__0_]
        x355 = stencil(AtDD01, stencil_idx_0_m1_0_VVV); // x373: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m1__0_]
        x356 = stencil(AtDD01, stencil_idx_0_1_0_VVV); // x374: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__1__0_]
        x357 = stencil(AtDD01, stencil_idx_0_0_m2_VVV); // x375: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m2_]
        x358 = stencil(AtDD01, stencil_idx_0_0_2_VVV); // x376: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__2_]
        x359 = stencil(AtDD01, stencil_idx_0_0_m1_VVV); // x377: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m1_]
        x360 = stencil(AtDD01, stencil_idx_0_0_1_VVV); // x378: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__1_]
        x361 = stencil(AtDD01, stencil_idx_m2_0_0_VVV); // x367: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m2__0__0_]
        x362 = stencil(AtDD01, stencil_idx_2_0_0_VVV); // x368: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__2__0__0_]
        x363 = stencil(AtDD01, stencil_idx_m1_0_0_VVV); // x369: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m1__0__0_]
        x364 = stencil(AtDD01, stencil_idx_1_0_0_VVV); // x370: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__1__0__0_]
        x365 = ((-5.0 / 16.0) * stencil(AtDD01, stencil_idx_0_0_0_VVV)); // x366: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__0_]
        store(At_rhsDD01, stencil_idx_0_0_0_VVV, (access(At_rhsDD01, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x365 + ((-3.0 / 32.0) * ((x361 + x362))) + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_m3_0_0_VVV) + stencil(AtDD01, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x363 + x364))))) + (DYI * (x365 + ((-3.0 / 32.0) * ((x353 + x354))) + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_0_m3_0_VVV) + stencil(AtDD01, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x355 + x356))))) + (DZI * (x365 + ((-3.0 / 32.0) * ((x357 + x358))) + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_0_0_m3_VVV) + stencil(AtDD01, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x359 + x360)))))))));
        vreal x384 = stencil(AtDD02, stencil_idx_0_m2_0_VVV); // x384: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m2__0_]
        vreal x385 = stencil(AtDD02, stencil_idx_0_2_0_VVV); // x385: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__2__0_]
        vreal x386 = stencil(AtDD02, stencil_idx_0_m1_0_VVV); // x386: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m1__0_]
        vreal x387 = stencil(AtDD02, stencil_idx_0_1_0_VVV); // x387: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__1__0_]
        vreal x388 = stencil(AtDD02, stencil_idx_0_0_m2_VVV); // x388: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m2_]
        vreal x389 = stencil(AtDD02, stencil_idx_0_0_2_VVV); // x389: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__2_]
        vreal x390 = stencil(AtDD02, stencil_idx_0_0_m1_VVV); // x390: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m1_]
        vreal x391 = stencil(AtDD02, stencil_idx_0_0_1_VVV); // x391: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__1_]
        vreal x380 = stencil(AtDD02, stencil_idx_m2_0_0_VVV); // x380: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m2__0__0_]
        vreal x381 = stencil(AtDD02, stencil_idx_2_0_0_VVV); // x381: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__2__0__0_]
        vreal x382 = stencil(AtDD02, stencil_idx_m1_0_0_VVV); // x382: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m1__0__0_]
        vreal x383 = stencil(AtDD02, stencil_idx_1_0_0_VVV); // x383: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__1__0__0_]
        vreal x379 = ((-5.0 / 16.0) * stencil(AtDD02, stencil_idx_0_0_0_VVV)); // x379: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__0_]
        store(At_rhsDD02, stencil_idx_0_0_0_VVV, (access(At_rhsDD02, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x379 + ((-3.0 / 32.0) * ((x380 + x381))) + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_m3_0_0_VVV) + stencil(AtDD02, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x382 + x383))))) + (DYI * (x379 + ((-3.0 / 32.0) * ((x384 + x385))) + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_0_m3_0_VVV) + stencil(AtDD02, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x386 + x387))))) + (DZI * (x379 + ((-3.0 / 32.0) * ((x388 + x389))) + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_0_0_m3_VVV) + stencil(AtDD02, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x390 + x391)))))))));
        x379 = stencil(AtDD11, stencil_idx_0_m2_0_VVV); // x397: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m2__0_]
        x380 = stencil(AtDD11, stencil_idx_0_2_0_VVV); // x398: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__2__0_]
        x381 = stencil(AtDD11, stencil_idx_0_m1_0_VVV); // x399: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m1__0_]
        x382 = stencil(AtDD11, stencil_idx_0_1_0_VVV); // x400: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__1__0_]
        x383 = stencil(AtDD11, stencil_idx_0_0_m2_VVV); // x401: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m2_]
        x384 = stencil(AtDD11, stencil_idx_0_0_2_VVV); // x402: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__2_]
        x385 = stencil(AtDD11, stencil_idx_0_0_m1_VVV); // x403: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m1_]
        x386 = stencil(AtDD11, stencil_idx_0_0_1_VVV); // x404: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__1_]
        x387 = stencil(AtDD11, stencil_idx_m2_0_0_VVV); // x393: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m2__0__0_]
        x388 = stencil(AtDD11, stencil_idx_2_0_0_VVV); // x394: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__2__0__0_]
        x389 = stencil(AtDD11, stencil_idx_m1_0_0_VVV); // x395: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m1__0__0_]
        x390 = stencil(AtDD11, stencil_idx_1_0_0_VVV); // x396: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__1__0__0_]
        x391 = ((-5.0 / 16.0) * stencil(AtDD11, stencil_idx_0_0_0_VVV)); // x392: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__0_]
        store(At_rhsDD11, stencil_idx_0_0_0_VVV, (access(At_rhsDD11, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x391 + ((-3.0 / 32.0) * ((x387 + x388))) + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_m3_0_0_VVV) + stencil(AtDD11, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x389 + x390))))) + (DYI * (x391 + ((-3.0 / 32.0) * ((x379 + x380))) + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_0_m3_0_VVV) + stencil(AtDD11, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x381 + x382))))) + (DZI * (x391 + ((-3.0 / 32.0) * ((x383 + x384))) + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_0_0_m3_VVV) + stencil(AtDD11, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x385 + x386)))))))));
        vreal x410 = stencil(AtDD12, stencil_idx_0_m2_0_VVV); // x410: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m2__0_]
        vreal x411 = stencil(AtDD12, stencil_idx_0_2_0_VVV); // x411: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__2__0_]
        vreal x412 = stencil(AtDD12, stencil_idx_0_m1_0_VVV); // x412: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m1__0_]
        vreal x413 = stencil(AtDD12, stencil_idx_0_1_0_VVV); // x413: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__1__0_]
        vreal x414 = stencil(AtDD12, stencil_idx_0_0_m2_VVV); // x414: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m2_]
        vreal x415 = stencil(AtDD12, stencil_idx_0_0_2_VVV); // x415: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__2_]
        vreal x416 = stencil(AtDD12, stencil_idx_0_0_m1_VVV); // x416: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m1_]
        vreal x417 = stencil(AtDD12, stencil_idx_0_0_1_VVV); // x417: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__1_]
        vreal x406 = stencil(AtDD12, stencil_idx_m2_0_0_VVV); // x406: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m2__0__0_]
        vreal x407 = stencil(AtDD12, stencil_idx_2_0_0_VVV); // x407: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__2__0__0_]
        vreal x408 = stencil(AtDD12, stencil_idx_m1_0_0_VVV); // x408: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m1__0__0_]
        vreal x409 = stencil(AtDD12, stencil_idx_1_0_0_VVV); // x409: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__1__0__0_]
        vreal x405 = ((-5.0 / 16.0) * stencil(AtDD12, stencil_idx_0_0_0_VVV)); // x405: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__0_]
        store(At_rhsDD12, stencil_idx_0_0_0_VVV, (access(At_rhsDD12, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x405 + ((-3.0 / 32.0) * ((x406 + x407))) + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_m3_0_0_VVV) + stencil(AtDD12, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x408 + x409))))) + (DYI * (x405 + ((-3.0 / 32.0) * ((x410 + x411))) + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_0_m3_0_VVV) + stencil(AtDD12, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x412 + x413))))) + (DZI * (x405 + ((-3.0 / 32.0) * ((x414 + x415))) + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_0_0_m3_VVV) + stencil(AtDD12, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x416 + x417)))))))));
        x405 = stencil(AtDD22, stencil_idx_0_m2_0_VVV); // x423: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m2__0_]
        x406 = stencil(AtDD22, stencil_idx_0_2_0_VVV); // x424: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__2__0_]
        x407 = stencil(AtDD22, stencil_idx_0_m1_0_VVV); // x425: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m1__0_]
        x408 = stencil(AtDD22, stencil_idx_0_1_0_VVV); // x426: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__1__0_]
        x409 = stencil(AtDD22, stencil_idx_0_0_m2_VVV); // x427: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m2_]
        x410 = stencil(AtDD22, stencil_idx_0_0_2_VVV); // x428: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__2_]
        x411 = stencil(AtDD22, stencil_idx_0_0_m1_VVV); // x429: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m1_]
        x412 = stencil(AtDD22, stencil_idx_0_0_1_VVV); // x430: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__1_]
        x413 = stencil(AtDD22, stencil_idx_m2_0_0_VVV); // x419: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m2__0__0_]
        x414 = stencil(AtDD22, stencil_idx_2_0_0_VVV); // x420: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__2__0__0_]
        x415 = stencil(AtDD22, stencil_idx_m1_0_0_VVV); // x421: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m1__0__0_]
        x416 = stencil(AtDD22, stencil_idx_1_0_0_VVV); // x422: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__1__0__0_]
        x417 = ((-5.0 / 16.0) * stencil(AtDD22, stencil_idx_0_0_0_VVV)); // x418: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__0_]
        store(At_rhsDD22, stencil_idx_0_0_0_VVV, (access(At_rhsDD22, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x417 + ((-3.0 / 32.0) * ((x413 + x414))) + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_m3_0_0_VVV) + stencil(AtDD22, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x415 + x416))))) + (DYI * (x417 + ((-3.0 / 32.0) * ((x405 + x406))) + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_0_m3_0_VVV) + stencil(AtDD22, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x407 + x408))))) + (DZI * (x417 + ((-3.0 / 32.0) * ((x409 + x410))) + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_0_0_m3_VVV) + stencil(AtDD22, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x411 + x412)))))))));    
    });
    // z4c_apply_dissipation loop 6
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
        vreal x432 = stencil(evo_lapse, stencil_idx_m2_0_0_VVV); // x432: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m2__0__0_]
        vreal x433 = stencil(evo_lapse, stencil_idx_2_0_0_VVV); // x433: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__2__0__0_]
        vreal x434 = ((x432 + x433)); // x434: Dependency! Liveness = 2; [x432, x433]
        x432 = stencil(evo_lapse, stencil_idx_m1_0_0_VVV); // x435: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m1__0__0_]
        x433 = stencil(evo_lapse, stencil_idx_1_0_0_VVV); // x436: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__1__0__0_]
        vreal x437 = ((x432 + x433)); // x437: Dependency! Liveness = 2; [x435, x436]
        vreal x438 = stencil(evo_lapse, stencil_idx_0_m2_0_VVV); // x438: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m2__0_]
        vreal x439 = stencil(evo_lapse, stencil_idx_0_2_0_VVV); // x439: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__2__0_]
        vreal x440 = ((x438 + x439)); // x440: Dependency! Liveness = 2; [x438, x439]
        x438 = stencil(evo_lapse, stencil_idx_0_m1_0_VVV); // x441: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m1__0_]
        x439 = stencil(evo_lapse, stencil_idx_0_1_0_VVV); // x442: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__1__0_]
        vreal x443 = ((x438 + x439)); // x443: Dependency! Liveness = 2; [x441, x442]
        vreal x444 = stencil(evo_lapse, stencil_idx_0_0_m2_VVV); // x444: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m2_]
        vreal x445 = stencil(evo_lapse, stencil_idx_0_0_2_VVV); // x445: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__2_]
        vreal x446 = ((x444 + x445)); // x446: Dependency! Liveness = 2; [x444, x445]
        x444 = stencil(evo_lapse, stencil_idx_0_0_m1_VVV); // x447: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m1_]
        x445 = stencil(evo_lapse, stencil_idx_0_0_1_VVV); // x448: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__1_]
        vreal x449 = ((x444 + x445)); // x449: Dependency! Liveness = 2; [x447, x448]
        vreal x431 = ((-5.0 / 16.0) * stencil(evo_lapse, stencil_idx_0_0_0_VVV)); // x431: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__0_]
        store(evo_lapse_rhs, stencil_idx_0_0_0_VVV, (access(evo_lapse_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x431 + ((-3.0 / 32.0) * x434) + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_m3_0_0_VVV) + stencil(evo_lapse, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x437))) + (DYI * (x431 + ((-3.0 / 32.0) * x440) + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_0_m3_0_VVV) + stencil(evo_lapse, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x443))) + (DZI * (x431 + ((-3.0 / 32.0) * x446) + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_0_0_m3_VVV) + stencil(evo_lapse, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x449)))))));    
    });
    // z4c_apply_dissipation loop 7
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
        vreal x451 = stencil(evo_shiftU0, stencil_idx_m2_0_0_VVV); // x451: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m2__0__0_]
        vreal x452 = stencil(evo_shiftU0, stencil_idx_2_0_0_VVV); // x452: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__2__0__0_]
        vreal x453 = ((x451 + x452)); // x453: Dependency! Liveness = 2; [x451, x452]
        x451 = stencil(evo_shiftU0, stencil_idx_m1_0_0_VVV); // x454: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m1__0__0_]
        x452 = stencil(evo_shiftU0, stencil_idx_1_0_0_VVV); // x455: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__1__0__0_]
        vreal x456 = ((x451 + x452)); // x456: Dependency! Liveness = 2; [x454, x455]
        vreal x457 = stencil(evo_shiftU0, stencil_idx_0_m2_0_VVV); // x457: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m2__0_]
        vreal x458 = stencil(evo_shiftU0, stencil_idx_0_2_0_VVV); // x458: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__2__0_]
        vreal x459 = ((x457 + x458)); // x459: Dependency! Liveness = 2; [x457, x458]
        x457 = stencil(evo_shiftU0, stencil_idx_0_m1_0_VVV); // x460: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m1__0_]
        x458 = stencil(evo_shiftU0, stencil_idx_0_1_0_VVV); // x461: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__1__0_]
        vreal x462 = ((x457 + x458)); // x462: Dependency! Liveness = 2; [x460, x461]
        vreal x463 = stencil(evo_shiftU0, stencil_idx_0_0_m2_VVV); // x463: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m2_]
        vreal x464 = stencil(evo_shiftU0, stencil_idx_0_0_2_VVV); // x464: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__2_]
        vreal x465 = ((x463 + x464)); // x465: Dependency! Liveness = 2; [x463, x464]
        x463 = stencil(evo_shiftU0, stencil_idx_0_0_m1_VVV); // x466: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m1_]
        x464 = stencil(evo_shiftU0, stencil_idx_0_0_1_VVV); // x467: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__1_]
        vreal x468 = ((x463 + x464)); // x468: Dependency! Liveness = 2; [x466, x467]
        vreal x450 = ((-5.0 / 16.0) * stencil(evo_shiftU0, stencil_idx_0_0_0_VVV)); // x450: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__0_]
        store(evo_shift_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU0, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x450 + ((-3.0 / 32.0) * x453) + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU0, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x456))) + (DYI * (x450 + ((-3.0 / 32.0) * x459) + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU0, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x462))) + (DZI * (x450 + ((-3.0 / 32.0) * x465) + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU0, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x468)))))));
        x450 = stencil(evo_shiftU1, stencil_idx_m2_0_0_VVV); // x470: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m2__0__0_]
        x453 = stencil(evo_shiftU1, stencil_idx_2_0_0_VVV); // x471: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__2__0__0_]
        x456 = ((x450 + x453)); // x472: Dependency! Liveness = 2; [x470, x471]
        x459 = stencil(evo_shiftU1, stencil_idx_m1_0_0_VVV); // x473: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m1__0__0_]
        x462 = stencil(evo_shiftU1, stencil_idx_1_0_0_VVV); // x474: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__1__0__0_]
        x465 = ((x459 + x462)); // x475: Dependency! Liveness = 2; [x473, x474]
        x468 = stencil(evo_shiftU1, stencil_idx_0_m2_0_VVV); // x476: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m2__0_]
        vreal x477 = stencil(evo_shiftU1, stencil_idx_0_2_0_VVV); // x477: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__2__0_]
        vreal x478 = ((x468 + x477)); // x478: Dependency! Liveness = 2; [x476, x477]
        x477 = stencil(evo_shiftU1, stencil_idx_0_m1_0_VVV); // x479: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m1__0_]
        vreal x480 = stencil(evo_shiftU1, stencil_idx_0_1_0_VVV); // x480: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__1__0_]
        vreal x481 = ((x477 + x480)); // x481: Dependency! Liveness = 2; [x479, x480]
        x480 = stencil(evo_shiftU1, stencil_idx_0_0_m2_VVV); // x482: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m2_]
        vreal x483 = stencil(evo_shiftU1, stencil_idx_0_0_2_VVV); // x483: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__2_]
        vreal x484 = ((x480 + x483)); // x484: Dependency! Liveness = 2; [x482, x483]
        x483 = stencil(evo_shiftU1, stencil_idx_0_0_m1_VVV); // x485: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m1_]
        vreal x486 = stencil(evo_shiftU1, stencil_idx_0_0_1_VVV); // x486: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__1_]
        vreal x487 = ((x483 + x486)); // x487: Dependency! Liveness = 2; [x485, x486]
        x486 = ((-5.0 / 16.0) * stencil(evo_shiftU1, stencil_idx_0_0_0_VVV)); // x469: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__0_]
        store(evo_shift_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU1, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x486 + ((-3.0 / 32.0) * x456) + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU1, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x465))) + (DYI * (x486 + ((-3.0 / 32.0) * x478) + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU1, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x481))) + (DZI * (x486 + ((-3.0 / 32.0) * x484) + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU1, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x487)))))));
        x478 = stencil(evo_shiftU2, stencil_idx_m2_0_0_VVV); // x489: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m2__0__0_]
        x481 = stencil(evo_shiftU2, stencil_idx_2_0_0_VVV); // x490: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__2__0__0_]
        x484 = ((x478 + x481)); // x491: Dependency! Liveness = 2; [x489, x490]
        x487 = stencil(evo_shiftU2, stencil_idx_m1_0_0_VVV); // x492: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m1__0__0_]
        vreal x493 = stencil(evo_shiftU2, stencil_idx_1_0_0_VVV); // x493: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__1__0__0_]
        vreal x494 = ((x487 + x493)); // x494: Dependency! Liveness = 2; [x492, x493]
        x493 = stencil(evo_shiftU2, stencil_idx_0_m2_0_VVV); // x495: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m2__0_]
        vreal x496 = stencil(evo_shiftU2, stencil_idx_0_2_0_VVV); // x496: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__2__0_]
        vreal x497 = ((x493 + x496)); // x497: Dependency! Liveness = 2; [x495, x496]
        x496 = stencil(evo_shiftU2, stencil_idx_0_m1_0_VVV); // x498: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m1__0_]
        vreal x499 = stencil(evo_shiftU2, stencil_idx_0_1_0_VVV); // x499: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__1__0_]
        vreal x500 = ((x496 + x499)); // x500: Dependency! Liveness = 2; [x498, x499]
        x499 = stencil(evo_shiftU2, stencil_idx_0_0_m2_VVV); // x501: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m2_]
        vreal x502 = stencil(evo_shiftU2, stencil_idx_0_0_2_VVV); // x502: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__2_]
        vreal x503 = ((x499 + x502)); // x503: Dependency! Liveness = 2; [x501, x502]
        x502 = stencil(evo_shiftU2, stencil_idx_0_0_m1_VVV); // x504: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m1_]
        vreal x505 = stencil(evo_shiftU2, stencil_idx_0_0_1_VVV); // x505: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__1_]
        vreal x506 = ((x502 + x505)); // x506: Dependency! Liveness = 2; [x504, x505]
        x505 = ((-5.0 / 16.0) * stencil(evo_shiftU2, stencil_idx_0_0_0_VVV)); // x488: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__0_]
        store(evo_shift_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU2, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x505 + ((-3.0 / 32.0) * x484) + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU2, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x494))) + (DYI * (x505 + ((-3.0 / 32.0) * x497) + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU2, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x500))) + (DZI * (x505 + ((-3.0 / 32.0) * x503) + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU2, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x506)))))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}