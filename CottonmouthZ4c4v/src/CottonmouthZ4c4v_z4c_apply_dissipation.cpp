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
        vreal x246 = stencil(Theta, stencil_idx_0_m2_0_VVV); // x246: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__m2__0_]
        vreal x247 = stencil(Theta, stencil_idx_0_2_0_VVV); // x247: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__2__0_]
        vreal x248 = stencil(Theta, stencil_idx_0_m1_0_VVV); // x248: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__m1__0_]
        vreal x249 = stencil(Theta, stencil_idx_0_1_0_VVV); // x249: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__1__0_]
        vreal x250 = stencil(Theta, stencil_idx_0_0_m2_VVV); // x250: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__m2_]
        vreal x251 = stencil(Theta, stencil_idx_0_0_2_VVV); // x251: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__2_]
        vreal x252 = stencil(Theta, stencil_idx_0_0_m1_VVV); // x252: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__m1_]
        vreal x253 = stencil(Theta, stencil_idx_0_0_1_VVV); // x253: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__1_]
        vreal x242 = stencil(Theta, stencil_idx_m2_0_0_VVV); // x242: Dependency! Liveness = 1; [__dummy_stencil__Theta__m2__0__0_]
        vreal x243 = stencil(Theta, stencil_idx_2_0_0_VVV); // x243: Dependency! Liveness = 1; [__dummy_stencil__Theta__2__0__0_]
        vreal x244 = stencil(Theta, stencil_idx_m1_0_0_VVV); // x244: Dependency! Liveness = 1; [__dummy_stencil__Theta__m1__0__0_]
        vreal x245 = stencil(Theta, stencil_idx_1_0_0_VVV); // x245: Dependency! Liveness = 1; [__dummy_stencil__Theta__1__0__0_]
        vreal x241 = (-20 * stencil(Theta, stencil_idx_0_0_0_VVV)); // x241: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__0_]
        vreal x254 = ((1.0 / 64.0) * dissipation_epsilon); // x254: Dependency! Liveness = 1; [dissipation_epsilon]
        store(Theta_rhs, stencil_idx_0_0_0_VVV, (access(Theta_rhs, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x241 + (-6 * ((x242 + x243))) + (15 * ((x244 + x245))) + ((stencil(Theta, stencil_idx_m3_0_0_VVV) + stencil(Theta, stencil_idx_3_0_0_VVV))))) + (DYI * (x241 + (-6 * ((x246 + x247))) + (15 * ((x248 + x249))) + ((stencil(Theta, stencil_idx_0_m3_0_VVV) + stencil(Theta, stencil_idx_0_3_0_VVV))))) + (DZI * (x241 + (-6 * ((x250 + x251))) + (15 * ((x252 + x253))) + ((stencil(Theta, stencil_idx_0_0_m3_VVV) + stencil(Theta, stencil_idx_0_0_3_VVV)))))))));    
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
        vreal x256 = stencil(chi, stencil_idx_m2_0_0_VVV); // x256: Dependency! Liveness = 1; [__dummy_stencil__chi__m2__0__0_]
        vreal x257 = stencil(chi, stencil_idx_2_0_0_VVV); // x257: Dependency! Liveness = 1; [__dummy_stencil__chi__2__0__0_]
        vreal x258 = ((x256 + x257)); // x258: Dependency! Liveness = 2; [x256, x257]
        vreal x259 = stencil(chi, stencil_idx_m1_0_0_VVV); // x259: Dependency! Liveness = 1; [__dummy_stencil__chi__m1__0__0_]
        vreal x260 = stencil(chi, stencil_idx_1_0_0_VVV); // x260: Dependency! Liveness = 1; [__dummy_stencil__chi__1__0__0_]
        vreal x261 = ((x259 + x260)); // x261: Dependency! Liveness = 2; [x259, x260]
        vreal x262 = stencil(chi, stencil_idx_0_m2_0_VVV); // x262: Dependency! Liveness = 1; [__dummy_stencil__chi__0__m2__0_]
        vreal x263 = stencil(chi, stencil_idx_0_2_0_VVV); // x263: Dependency! Liveness = 1; [__dummy_stencil__chi__0__2__0_]
        vreal x264 = ((x262 + x263)); // x264: Dependency! Liveness = 2; [x262, x263]
        vreal x265 = stencil(chi, stencil_idx_0_m1_0_VVV); // x265: Dependency! Liveness = 1; [__dummy_stencil__chi__0__m1__0_]
        vreal x266 = stencil(chi, stencil_idx_0_1_0_VVV); // x266: Dependency! Liveness = 1; [__dummy_stencil__chi__0__1__0_]
        vreal x267 = ((x265 + x266)); // x267: Dependency! Liveness = 2; [x265, x266]
        vreal x268 = stencil(chi, stencil_idx_0_0_m2_VVV); // x268: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__m2_]
        vreal x269 = stencil(chi, stencil_idx_0_0_2_VVV); // x269: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__2_]
        vreal x270 = ((x268 + x269)); // x270: Dependency! Liveness = 2; [x268, x269]
        vreal x271 = stencil(chi, stencil_idx_0_0_m1_VVV); // x271: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__m1_]
        vreal x272 = stencil(chi, stencil_idx_0_0_1_VVV); // x272: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__1_]
        vreal x273 = ((x271 + x272)); // x273: Dependency! Liveness = 2; [x271, x272]
        vreal x255 = (-20 * stencil(chi, stencil_idx_0_0_0_VVV)); // x255: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__0_]
        vreal x254 = ((1.0 / 64.0) * dissipation_epsilon); // x254: Dependency! Liveness = 1; [dissipation_epsilon]
        store(chi_rhs, stencil_idx_0_0_0_VVV, (access(chi_rhs, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x255 + (-6 * x258) + (15 * x261) + ((stencil(chi, stencil_idx_m3_0_0_VVV) + stencil(chi, stencil_idx_3_0_0_VVV))))) + (DYI * (x255 + (-6 * x264) + (15 * x267) + ((stencil(chi, stencil_idx_0_m3_0_VVV) + stencil(chi, stencil_idx_0_3_0_VVV))))) + (DZI * (x255 + (-6 * x270) + (15 * x273) + ((stencil(chi, stencil_idx_0_0_m3_VVV) + stencil(chi, stencil_idx_0_0_3_VVV)))))))));    
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
        vreal x279 = stencil(trK, stencil_idx_0_m2_0_VVV); // x279: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m2__0_]
        vreal x280 = stencil(trK, stencil_idx_0_2_0_VVV); // x280: Dependency! Liveness = 1; [__dummy_stencil__trK__0__2__0_]
        vreal x281 = stencil(trK, stencil_idx_0_m1_0_VVV); // x281: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m1__0_]
        vreal x282 = stencil(trK, stencil_idx_0_1_0_VVV); // x282: Dependency! Liveness = 1; [__dummy_stencil__trK__0__1__0_]
        vreal x283 = stencil(trK, stencil_idx_0_0_m2_VVV); // x283: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m2_]
        vreal x284 = stencil(trK, stencil_idx_0_0_2_VVV); // x284: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__2_]
        vreal x285 = stencil(trK, stencil_idx_0_0_m1_VVV); // x285: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m1_]
        vreal x286 = stencil(trK, stencil_idx_0_0_1_VVV); // x286: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__1_]
        vreal x275 = stencil(trK, stencil_idx_m2_0_0_VVV); // x275: Dependency! Liveness = 1; [__dummy_stencil__trK__m2__0__0_]
        vreal x276 = stencil(trK, stencil_idx_2_0_0_VVV); // x276: Dependency! Liveness = 1; [__dummy_stencil__trK__2__0__0_]
        vreal x277 = stencil(trK, stencil_idx_m1_0_0_VVV); // x277: Dependency! Liveness = 1; [__dummy_stencil__trK__m1__0__0_]
        vreal x278 = stencil(trK, stencil_idx_1_0_0_VVV); // x278: Dependency! Liveness = 1; [__dummy_stencil__trK__1__0__0_]
        vreal x274 = (-20 * stencil(trK, stencil_idx_0_0_0_VVV)); // x274: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__0_]
        vreal x254 = ((1.0 / 64.0) * dissipation_epsilon); // x254: Dependency! Liveness = 1; [dissipation_epsilon]
        store(trK_rhs, stencil_idx_0_0_0_VVV, (access(trK_rhs, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x274 + (-6 * ((x275 + x276))) + (15 * ((x277 + x278))) + ((stencil(trK, stencil_idx_m3_0_0_VVV) + stencil(trK, stencil_idx_3_0_0_VVV))))) + (DYI * (x274 + (-6 * ((x279 + x280))) + (15 * ((x281 + x282))) + ((stencil(trK, stencil_idx_0_m3_0_VVV) + stencil(trK, stencil_idx_0_3_0_VVV))))) + (DZI * (x274 + (-6 * ((x283 + x284))) + (15 * ((x285 + x286))) + ((stencil(trK, stencil_idx_0_0_m3_VVV) + stencil(trK, stencil_idx_0_0_3_VVV)))))))));    
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
        vreal x292 = stencil(evo_GammatU0, stencil_idx_0_m2_0_VVV); // x292: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__m2__0_]
        vreal x293 = stencil(evo_GammatU0, stencil_idx_0_2_0_VVV); // x293: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__2__0_]
        vreal x294 = stencil(evo_GammatU0, stencil_idx_0_m1_0_VVV); // x294: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__m1__0_]
        vreal x295 = stencil(evo_GammatU0, stencil_idx_0_1_0_VVV); // x295: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__1__0_]
        vreal x296 = stencil(evo_GammatU0, stencil_idx_0_0_m2_VVV); // x296: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__m2_]
        vreal x297 = stencil(evo_GammatU0, stencil_idx_0_0_2_VVV); // x297: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__2_]
        vreal x298 = stencil(evo_GammatU0, stencil_idx_0_0_m1_VVV); // x298: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__m1_]
        vreal x299 = stencil(evo_GammatU0, stencil_idx_0_0_1_VVV); // x299: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__1_]
        vreal x288 = stencil(evo_GammatU0, stencil_idx_m2_0_0_VVV); // x288: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__m2__0__0_]
        vreal x289 = stencil(evo_GammatU0, stencil_idx_2_0_0_VVV); // x289: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__2__0__0_]
        vreal x290 = stencil(evo_GammatU0, stencil_idx_m1_0_0_VVV); // x290: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__m1__0__0_]
        vreal x291 = stencil(evo_GammatU0, stencil_idx_1_0_0_VVV); // x291: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__1__0__0_]
        vreal x287 = (-20 * stencil(evo_GammatU0, stencil_idx_0_0_0_VVV)); // x287: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__0_]
        vreal x254 = ((1.0 / 64.0) * dissipation_epsilon); // x254: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_Gammat_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU0, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x287 + (-6 * ((x288 + x289))) + (15 * ((x290 + x291))) + ((stencil(evo_GammatU0, stencil_idx_m3_0_0_VVV) + stencil(evo_GammatU0, stencil_idx_3_0_0_VVV))))) + (DYI * (x287 + (-6 * ((x292 + x293))) + (15 * ((x294 + x295))) + ((stencil(evo_GammatU0, stencil_idx_0_m3_0_VVV) + stencil(evo_GammatU0, stencil_idx_0_3_0_VVV))))) + (DZI * (x287 + (-6 * ((x296 + x297))) + (15 * ((x298 + x299))) + ((stencil(evo_GammatU0, stencil_idx_0_0_m3_VVV) + stencil(evo_GammatU0, stencil_idx_0_0_3_VVV)))))))));
        vreal x305 = stencil(evo_GammatU1, stencil_idx_0_m2_0_VVV); // x305: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__m2__0_]
        vreal x306 = stencil(evo_GammatU1, stencil_idx_0_2_0_VVV); // x306: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__2__0_]
        vreal x307 = stencil(evo_GammatU1, stencil_idx_0_m1_0_VVV); // x307: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__m1__0_]
        vreal x308 = stencil(evo_GammatU1, stencil_idx_0_1_0_VVV); // x308: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__1__0_]
        vreal x309 = stencil(evo_GammatU1, stencil_idx_0_0_m2_VVV); // x309: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__m2_]
        vreal x310 = stencil(evo_GammatU1, stencil_idx_0_0_2_VVV); // x310: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__2_]
        vreal x311 = stencil(evo_GammatU1, stencil_idx_0_0_m1_VVV); // x311: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__m1_]
        vreal x312 = stencil(evo_GammatU1, stencil_idx_0_0_1_VVV); // x312: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__1_]
        vreal x301 = stencil(evo_GammatU1, stencil_idx_m2_0_0_VVV); // x301: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__m2__0__0_]
        vreal x302 = stencil(evo_GammatU1, stencil_idx_2_0_0_VVV); // x302: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__2__0__0_]
        vreal x303 = stencil(evo_GammatU1, stencil_idx_m1_0_0_VVV); // x303: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__m1__0__0_]
        vreal x304 = stencil(evo_GammatU1, stencil_idx_1_0_0_VVV); // x304: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__1__0__0_]
        vreal x300 = (-20 * stencil(evo_GammatU1, stencil_idx_0_0_0_VVV)); // x300: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__0_]
        store(evo_Gammat_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU1, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x300 + (-6 * ((x301 + x302))) + (15 * ((x303 + x304))) + ((stencil(evo_GammatU1, stencil_idx_m3_0_0_VVV) + stencil(evo_GammatU1, stencil_idx_3_0_0_VVV))))) + (DYI * (x300 + (-6 * ((x305 + x306))) + (15 * ((x307 + x308))) + ((stencil(evo_GammatU1, stencil_idx_0_m3_0_VVV) + stencil(evo_GammatU1, stencil_idx_0_3_0_VVV))))) + (DZI * (x300 + (-6 * ((x309 + x310))) + (15 * ((x311 + x312))) + ((stencil(evo_GammatU1, stencil_idx_0_0_m3_VVV) + stencil(evo_GammatU1, stencil_idx_0_0_3_VVV)))))))));
        vreal x318 = stencil(evo_GammatU2, stencil_idx_0_m2_0_VVV); // x318: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__m2__0_]
        vreal x319 = stencil(evo_GammatU2, stencil_idx_0_2_0_VVV); // x319: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__2__0_]
        vreal x320 = stencil(evo_GammatU2, stencil_idx_0_m1_0_VVV); // x320: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__m1__0_]
        vreal x321 = stencil(evo_GammatU2, stencil_idx_0_1_0_VVV); // x321: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__1__0_]
        vreal x322 = stencil(evo_GammatU2, stencil_idx_0_0_m2_VVV); // x322: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__m2_]
        vreal x323 = stencil(evo_GammatU2, stencil_idx_0_0_2_VVV); // x323: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__2_]
        vreal x324 = stencil(evo_GammatU2, stencil_idx_0_0_m1_VVV); // x324: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__m1_]
        vreal x325 = stencil(evo_GammatU2, stencil_idx_0_0_1_VVV); // x325: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__1_]
        vreal x314 = stencil(evo_GammatU2, stencil_idx_m2_0_0_VVV); // x314: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__m2__0__0_]
        vreal x315 = stencil(evo_GammatU2, stencil_idx_2_0_0_VVV); // x315: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__2__0__0_]
        vreal x316 = stencil(evo_GammatU2, stencil_idx_m1_0_0_VVV); // x316: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__m1__0__0_]
        vreal x317 = stencil(evo_GammatU2, stencil_idx_1_0_0_VVV); // x317: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__1__0__0_]
        vreal x313 = (-20 * stencil(evo_GammatU2, stencil_idx_0_0_0_VVV)); // x313: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__0_]
        store(evo_Gammat_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU2, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x313 + (-6 * ((x314 + x315))) + (15 * ((x316 + x317))) + ((stencil(evo_GammatU2, stencil_idx_m3_0_0_VVV) + stencil(evo_GammatU2, stencil_idx_3_0_0_VVV))))) + (DYI * (x313 + (-6 * ((x318 + x319))) + (15 * ((x320 + x321))) + ((stencil(evo_GammatU2, stencil_idx_0_m3_0_VVV) + stencil(evo_GammatU2, stencil_idx_0_3_0_VVV))))) + (DZI * (x313 + (-6 * ((x322 + x323))) + (15 * ((x324 + x325))) + ((stencil(evo_GammatU2, stencil_idx_0_0_m3_VVV) + stencil(evo_GammatU2, stencil_idx_0_0_3_VVV)))))))));    
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
        vreal x169 = stencil(gtDD00, stencil_idx_m2_0_0_VVV); // x169: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m2__0__0_]
        vreal x170 = stencil(gtDD00, stencil_idx_2_0_0_VVV); // x170: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__2__0__0_]
        vreal x327 = ((x169 + x170)); // x327: Dependency! Liveness = 2; [x169, x170]
        vreal x171 = stencil(gtDD00, stencil_idx_m1_0_0_VVV); // x171: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m1__0__0_]
        vreal x172 = stencil(gtDD00, stencil_idx_1_0_0_VVV); // x172: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__1__0__0_]
        vreal x328 = ((x171 + x172)); // x328: Dependency! Liveness = 2; [x171, x172]
        vreal x102 = stencil(gtDD00, stencil_idx_0_m2_0_VVV); // x102: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m2__0_]
        vreal x103 = stencil(gtDD00, stencil_idx_0_2_0_VVV); // x103: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__2__0_]
        vreal x329 = ((x102 + x103)); // x329: Dependency! Liveness = 2; [x102, x103]
        vreal x104 = stencil(gtDD00, stencil_idx_0_m1_0_VVV); // x104: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m1__0_]
        vreal x105 = stencil(gtDD00, stencil_idx_0_1_0_VVV); // x105: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__1__0_]
        vreal x330 = ((x104 + x105)); // x330: Dependency! Liveness = 2; [x104, x105]
        vreal x90 = stencil(gtDD00, stencil_idx_0_0_m2_VVV); // x90: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m2_]
        vreal x91 = stencil(gtDD00, stencil_idx_0_0_2_VVV); // x91: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__2_]
        vreal x331 = ((x90 + x91)); // x331: Dependency! Liveness = 2; [x90, x91]
        vreal x92 = stencil(gtDD00, stencil_idx_0_0_m1_VVV); // x92: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m1_]
        vreal x93 = stencil(gtDD00, stencil_idx_0_0_1_VVV); // x93: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__1_]
        vreal x332 = ((x92 + x93)); // x332: Dependency! Liveness = 2; [x92, x93]
        vreal x326 = (-20 * stencil(gtDD00, stencil_idx_0_0_0_VVV)); // x326: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__0_]
        vreal x254 = ((1.0 / 64.0) * dissipation_epsilon); // x254: Dependency! Liveness = 1; [dissipation_epsilon]
        store(gt_rhsDD00, stencil_idx_0_0_0_VVV, (access(gt_rhsDD00, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x326 + (-6 * x327) + (15 * x328) + ((stencil(gtDD00, stencil_idx_m3_0_0_VVV) + stencil(gtDD00, stencil_idx_3_0_0_VVV))))) + (DYI * (x326 + (-6 * x329) + (15 * x330) + ((stencil(gtDD00, stencil_idx_0_m3_0_VVV) + stencil(gtDD00, stencil_idx_0_3_0_VVV))))) + (DZI * (x326 + (-6 * x331) + (15 * x332) + ((stencil(gtDD00, stencil_idx_0_0_m3_VVV) + stencil(gtDD00, stencil_idx_0_0_3_VVV)))))))));
        vreal x183 = stencil(gtDD01, stencil_idx_m2_0_0_VVV); // x183: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m2__0__0_]
        vreal x184 = stencil(gtDD01, stencil_idx_2_0_0_VVV); // x184: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__2__0__0_]
        vreal x334 = ((x183 + x184)); // x334: Dependency! Liveness = 2; [x183, x184]
        vreal x185 = stencil(gtDD01, stencil_idx_m1_0_0_VVV); // x185: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m1__0__0_]
        vreal x186 = stencil(gtDD01, stencil_idx_1_0_0_VVV); // x186: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__1__0__0_]
        vreal x335 = ((x185 + x186)); // x335: Dependency! Liveness = 2; [x185, x186]
        vreal x154 = stencil(gtDD01, stencil_idx_0_m2_0_VVV); // x154: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m2__0_]
        vreal x155 = stencil(gtDD01, stencil_idx_0_2_0_VVV); // x155: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__2__0_]
        vreal x336 = ((x154 + x155)); // x336: Dependency! Liveness = 2; [x154, x155]
        vreal x156 = stencil(gtDD01, stencil_idx_0_m1_0_VVV); // x156: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m1__0_]
        vreal x157 = stencil(gtDD01, stencil_idx_0_1_0_VVV); // x157: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__1__0_]
        vreal x337 = ((x156 + x157)); // x337: Dependency! Liveness = 2; [x156, x157]
        vreal x65 = stencil(gtDD01, stencil_idx_0_0_m2_VVV); // x65: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m2_]
        vreal x66 = stencil(gtDD01, stencil_idx_0_0_2_VVV); // x66: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__2_]
        vreal x338 = ((x65 + x66)); // x338: Dependency! Liveness = 2; [x65, x66]
        vreal x67 = stencil(gtDD01, stencil_idx_0_0_m1_VVV); // x67: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m1_]
        vreal x68 = stencil(gtDD01, stencil_idx_0_0_1_VVV); // x68: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__1_]
        vreal x339 = ((x67 + x68)); // x339: Dependency! Liveness = 2; [x67, x68]
        vreal x333 = (-20 * stencil(gtDD01, stencil_idx_0_0_0_VVV)); // x333: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__0_]
        store(gt_rhsDD01, stencil_idx_0_0_0_VVV, (access(gt_rhsDD01, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x333 + (-6 * x334) + (15 * x335) + ((stencil(gtDD01, stencil_idx_m3_0_0_VVV) + stencil(gtDD01, stencil_idx_3_0_0_VVV))))) + (DYI * (x333 + (-6 * x336) + (15 * x337) + ((stencil(gtDD01, stencil_idx_0_m3_0_VVV) + stencil(gtDD01, stencil_idx_0_3_0_VVV))))) + (DZI * (x333 + (-6 * x338) + (15 * x339) + ((stencil(gtDD01, stencil_idx_0_0_m3_VVV) + stencil(gtDD01, stencil_idx_0_0_3_VVV)))))))));
        vreal x109 = stencil(gtDD11, stencil_idx_m2_0_0_VVV); // x109: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m2__0__0_]
        vreal x110 = stencil(gtDD11, stencil_idx_2_0_0_VVV); // x110: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__2__0__0_]
        vreal x341 = ((x109 + x110)); // x341: Dependency! Liveness = 2; [x109, x110]
        vreal x111 = stencil(gtDD11, stencil_idx_m1_0_0_VVV); // x111: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m1__0__0_]
        vreal x112 = stencil(gtDD11, stencil_idx_1_0_0_VVV); // x112: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__1__0__0_]
        vreal x342 = ((x111 + x112)); // x342: Dependency! Liveness = 2; [x111, x112]
        vreal x147 = stencil(gtDD11, stencil_idx_0_m2_0_VVV); // x147: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m2__0_]
        vreal x148 = stencil(gtDD11, stencil_idx_0_2_0_VVV); // x148: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__2__0_]
        vreal x343 = ((x147 + x148)); // x343: Dependency! Liveness = 2; [x147, x148]
        vreal x149 = stencil(gtDD11, stencil_idx_0_m1_0_VVV); // x149: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m1__0_]
        vreal x150 = stencil(gtDD11, stencil_idx_0_1_0_VVV); // x150: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__1__0_]
        vreal x344 = ((x149 + x150)); // x344: Dependency! Liveness = 2; [x149, x150]
        vreal x50 = stencil(gtDD11, stencil_idx_0_0_m2_VVV); // x50: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m2_]
        vreal x51 = stencil(gtDD11, stencil_idx_0_0_2_VVV); // x51: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__2_]
        vreal x345 = ((x50 + x51)); // x345: Dependency! Liveness = 2; [x50, x51]
        vreal x52 = stencil(gtDD11, stencil_idx_0_0_m1_VVV); // x52: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m1_]
        vreal x53 = stencil(gtDD11, stencil_idx_0_0_1_VVV); // x53: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__1_]
        vreal x346 = ((x52 + x53)); // x346: Dependency! Liveness = 2; [x52, x53]
        vreal x340 = (-20 * stencil(gtDD11, stencil_idx_0_0_0_VVV)); // x340: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__0_]
        store(gt_rhsDD11, stencil_idx_0_0_0_VVV, (access(gt_rhsDD11, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x340 + (-6 * x341) + (15 * x342) + ((stencil(gtDD11, stencil_idx_m3_0_0_VVV) + stencil(gtDD11, stencil_idx_3_0_0_VVV))))) + (DYI * (x340 + (-6 * x343) + (15 * x344) + ((stencil(gtDD11, stencil_idx_0_m3_0_VVV) + stencil(gtDD11, stencil_idx_0_3_0_VVV))))) + (DZI * (x340 + (-6 * x345) + (15 * x346) + ((stencil(gtDD11, stencil_idx_0_0_m3_VVV) + stencil(gtDD11, stencil_idx_0_0_3_VVV)))))))));
        vreal x175 = stencil(gtDD02, stencil_idx_m2_0_0_VVV); // x175: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m2__0__0_]
        vreal x176 = stencil(gtDD02, stencil_idx_2_0_0_VVV); // x176: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__2__0__0_]
        vreal x348 = ((x175 + x176)); // x348: Dependency! Liveness = 2; [x175, x176]
        vreal x177 = stencil(gtDD02, stencil_idx_m1_0_0_VVV); // x177: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m1__0__0_]
        vreal x178 = stencil(gtDD02, stencil_idx_1_0_0_VVV); // x178: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__1__0__0_]
        vreal x349 = ((x177 + x178)); // x349: Dependency! Liveness = 2; [x177, x178]
        vreal x59 = stencil(gtDD02, stencil_idx_0_m2_0_VVV); // x59: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m2__0_]
        vreal x60 = stencil(gtDD02, stencil_idx_0_2_0_VVV); // x60: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__2__0_]
        vreal x350 = ((x59 + x60)); // x350: Dependency! Liveness = 2; [x59, x60]
        vreal x61 = stencil(gtDD02, stencil_idx_0_m1_0_VVV); // x61: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m1__0_]
        vreal x62 = stencil(gtDD02, stencil_idx_0_1_0_VVV); // x62: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__1__0_]
        vreal x351 = ((x61 + x62)); // x351: Dependency! Liveness = 2; [x61, x62]
        vreal x137 = stencil(gtDD02, stencil_idx_0_0_m2_VVV); // x137: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m2_]
        vreal x138 = stencil(gtDD02, stencil_idx_0_0_2_VVV); // x138: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__2_]
        vreal x352 = ((x137 + x138)); // x352: Dependency! Liveness = 2; [x137, x138]
        vreal x139 = stencil(gtDD02, stencil_idx_0_0_m1_VVV); // x139: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m1_]
        vreal x140 = stencil(gtDD02, stencil_idx_0_0_1_VVV); // x140: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__1_]
        vreal x353 = ((x139 + x140)); // x353: Dependency! Liveness = 2; [x139, x140]
        vreal x347 = (-20 * stencil(gtDD02, stencil_idx_0_0_0_VVV)); // x347: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__0_]
        store(gt_rhsDD02, stencil_idx_0_0_0_VVV, (access(gt_rhsDD02, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x347 + (-6 * x348) + (15 * x349) + ((stencil(gtDD02, stencil_idx_m3_0_0_VVV) + stencil(gtDD02, stencil_idx_3_0_0_VVV))))) + (DYI * (x347 + (-6 * x350) + (15 * x351) + ((stencil(gtDD02, stencil_idx_0_m3_0_VVV) + stencil(gtDD02, stencil_idx_0_3_0_VVV))))) + (DZI * (x347 + (-6 * x352) + (15 * x353) + ((stencil(gtDD02, stencil_idx_0_0_m3_VVV) + stencil(gtDD02, stencil_idx_0_0_3_VVV)))))))));
        vreal x71 = stencil(gtDD12, stencil_idx_m2_0_0_VVV); // x71: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m2__0__0_]
        vreal x72 = stencil(gtDD12, stencil_idx_2_0_0_VVV); // x72: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__2__0__0_]
        vreal x355 = ((x71 + x72)); // x355: Dependency! Liveness = 2; [x71, x72]
        vreal x73 = stencil(gtDD12, stencil_idx_m1_0_0_VVV); // x73: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m1__0__0_]
        vreal x74 = stencil(gtDD12, stencil_idx_1_0_0_VVV); // x74: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__1__0__0_]
        vreal x356 = ((x73 + x74)); // x356: Dependency! Liveness = 2; [x73, x74]
        vreal x161 = stencil(gtDD12, stencil_idx_0_m2_0_VVV); // x161: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m2__0_]
        vreal x162 = stencil(gtDD12, stencil_idx_0_2_0_VVV); // x162: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__2__0_]
        vreal x357 = ((x161 + x162)); // x357: Dependency! Liveness = 2; [x161, x162]
        vreal x163 = stencil(gtDD12, stencil_idx_0_m1_0_VVV); // x163: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m1__0_]
        vreal x164 = stencil(gtDD12, stencil_idx_0_1_0_VVV); // x164: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__1__0_]
        vreal x358 = ((x163 + x164)); // x358: Dependency! Liveness = 2; [x163, x164]
        vreal x130 = stencil(gtDD12, stencil_idx_0_0_m2_VVV); // x130: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m2_]
        vreal x131 = stencil(gtDD12, stencil_idx_0_0_2_VVV); // x131: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__2_]
        vreal x359 = ((x130 + x131)); // x359: Dependency! Liveness = 2; [x130, x131]
        vreal x132 = stencil(gtDD12, stencil_idx_0_0_m1_VVV); // x132: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m1_]
        vreal x133 = stencil(gtDD12, stencil_idx_0_0_1_VVV); // x133: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__1_]
        vreal x360 = ((x132 + x133)); // x360: Dependency! Liveness = 2; [x132, x133]
        vreal x354 = (-20 * stencil(gtDD12, stencil_idx_0_0_0_VVV)); // x354: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__0_]
        store(gt_rhsDD12, stencil_idx_0_0_0_VVV, (access(gt_rhsDD12, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x354 + (-6 * x355) + (15 * x356) + ((stencil(gtDD12, stencil_idx_m3_0_0_VVV) + stencil(gtDD12, stencil_idx_3_0_0_VVV))))) + (DYI * (x354 + (-6 * x357) + (15 * x358) + ((stencil(gtDD12, stencil_idx_0_m3_0_VVV) + stencil(gtDD12, stencil_idx_0_3_0_VVV))))) + (DZI * (x354 + (-6 * x359) + (15 * x360) + ((stencil(gtDD12, stencil_idx_0_0_m3_VVV) + stencil(gtDD12, stencil_idx_0_0_3_VVV)))))))));
        vreal x83 = stencil(gtDD22, stencil_idx_m2_0_0_VVV); // x83: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m2__0__0_]
        vreal x84 = stencil(gtDD22, stencil_idx_2_0_0_VVV); // x84: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__2__0__0_]
        vreal x362 = ((x83 + x84)); // x362: Dependency! Liveness = 2; [x83, x84]
        vreal x85 = stencil(gtDD22, stencil_idx_m1_0_0_VVV); // x85: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m1__0__0_]
        vreal x86 = stencil(gtDD22, stencil_idx_1_0_0_VVV); // x86: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__1__0__0_]
        vreal x363 = ((x85 + x86)); // x363: Dependency! Liveness = 2; [x85, x86]
        vreal x41 = stencil(gtDD22, stencil_idx_0_m2_0_VVV); // x41: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m2__0_]
        vreal x42 = stencil(gtDD22, stencil_idx_0_2_0_VVV); // x42: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__2__0_]
        vreal x364 = ((x41 + x42)); // x364: Dependency! Liveness = 2; [x41, x42]
        vreal x43 = stencil(gtDD22, stencil_idx_0_m1_0_VVV); // x43: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m1__0_]
        vreal x44 = stencil(gtDD22, stencil_idx_0_1_0_VVV); // x44: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__1__0_]
        vreal x365 = ((x43 + x44)); // x365: Dependency! Liveness = 2; [x43, x44]
        vreal x123 = stencil(gtDD22, stencil_idx_0_0_m2_VVV); // x123: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m2_]
        vreal x124 = stencil(gtDD22, stencil_idx_0_0_2_VVV); // x124: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__2_]
        vreal x366 = ((x123 + x124)); // x366: Dependency! Liveness = 2; [x123, x124]
        vreal x125 = stencil(gtDD22, stencil_idx_0_0_m1_VVV); // x125: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m1_]
        vreal x126 = stencil(gtDD22, stencil_idx_0_0_1_VVV); // x126: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__1_]
        vreal x367 = ((x125 + x126)); // x367: Dependency! Liveness = 2; [x125, x126]
        vreal x361 = (-20 * stencil(gtDD22, stencil_idx_0_0_0_VVV)); // x361: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__0_]
        store(gt_rhsDD22, stencil_idx_0_0_0_VVV, (access(gt_rhsDD22, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x361 + (-6 * x362) + (15 * x363) + ((stencil(gtDD22, stencil_idx_m3_0_0_VVV) + stencil(gtDD22, stencil_idx_3_0_0_VVV))))) + (DYI * (x361 + (-6 * x364) + (15 * x365) + ((stencil(gtDD22, stencil_idx_0_m3_0_VVV) + stencil(gtDD22, stencil_idx_0_3_0_VVV))))) + (DZI * (x361 + (-6 * x366) + (15 * x367) + ((stencil(gtDD22, stencil_idx_0_0_m3_VVV) + stencil(gtDD22, stencil_idx_0_0_3_VVV)))))))));    
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
        vreal x373 = stencil(AtDD00, stencil_idx_0_m2_0_VVV); // x373: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m2__0_]
        vreal x374 = stencil(AtDD00, stencil_idx_0_2_0_VVV); // x374: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__2__0_]
        vreal x375 = stencil(AtDD00, stencil_idx_0_m1_0_VVV); // x375: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m1__0_]
        vreal x376 = stencil(AtDD00, stencil_idx_0_1_0_VVV); // x376: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__1__0_]
        vreal x377 = stencil(AtDD00, stencil_idx_0_0_m2_VVV); // x377: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m2_]
        vreal x378 = stencil(AtDD00, stencil_idx_0_0_2_VVV); // x378: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__2_]
        vreal x379 = stencil(AtDD00, stencil_idx_0_0_m1_VVV); // x379: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m1_]
        vreal x380 = stencil(AtDD00, stencil_idx_0_0_1_VVV); // x380: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__1_]
        vreal x369 = stencil(AtDD00, stencil_idx_m2_0_0_VVV); // x369: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m2__0__0_]
        vreal x370 = stencil(AtDD00, stencil_idx_2_0_0_VVV); // x370: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__2__0__0_]
        vreal x371 = stencil(AtDD00, stencil_idx_m1_0_0_VVV); // x371: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m1__0__0_]
        vreal x372 = stencil(AtDD00, stencil_idx_1_0_0_VVV); // x372: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__1__0__0_]
        vreal x368 = (-20 * stencil(AtDD00, stencil_idx_0_0_0_VVV)); // x368: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__0_]
        vreal x254 = ((1.0 / 64.0) * dissipation_epsilon); // x254: Dependency! Liveness = 1; [dissipation_epsilon]
        store(At_rhsDD00, stencil_idx_0_0_0_VVV, (access(At_rhsDD00, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x368 + (-6 * ((x369 + x370))) + (15 * ((x371 + x372))) + ((stencil(AtDD00, stencil_idx_m3_0_0_VVV) + stencil(AtDD00, stencil_idx_3_0_0_VVV))))) + (DYI * (x368 + (-6 * ((x373 + x374))) + (15 * ((x375 + x376))) + ((stencil(AtDD00, stencil_idx_0_m3_0_VVV) + stencil(AtDD00, stencil_idx_0_3_0_VVV))))) + (DZI * (x368 + (-6 * ((x377 + x378))) + (15 * ((x379 + x380))) + ((stencil(AtDD00, stencil_idx_0_0_m3_VVV) + stencil(AtDD00, stencil_idx_0_0_3_VVV)))))))));
        vreal x386 = stencil(AtDD01, stencil_idx_0_m2_0_VVV); // x386: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m2__0_]
        vreal x387 = stencil(AtDD01, stencil_idx_0_2_0_VVV); // x387: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__2__0_]
        vreal x388 = stencil(AtDD01, stencil_idx_0_m1_0_VVV); // x388: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m1__0_]
        vreal x389 = stencil(AtDD01, stencil_idx_0_1_0_VVV); // x389: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__1__0_]
        vreal x390 = stencil(AtDD01, stencil_idx_0_0_m2_VVV); // x390: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m2_]
        vreal x391 = stencil(AtDD01, stencil_idx_0_0_2_VVV); // x391: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__2_]
        vreal x392 = stencil(AtDD01, stencil_idx_0_0_m1_VVV); // x392: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m1_]
        vreal x393 = stencil(AtDD01, stencil_idx_0_0_1_VVV); // x393: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__1_]
        vreal x382 = stencil(AtDD01, stencil_idx_m2_0_0_VVV); // x382: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m2__0__0_]
        vreal x383 = stencil(AtDD01, stencil_idx_2_0_0_VVV); // x383: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__2__0__0_]
        vreal x384 = stencil(AtDD01, stencil_idx_m1_0_0_VVV); // x384: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m1__0__0_]
        vreal x385 = stencil(AtDD01, stencil_idx_1_0_0_VVV); // x385: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__1__0__0_]
        vreal x381 = (-20 * stencil(AtDD01, stencil_idx_0_0_0_VVV)); // x381: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__0_]
        store(At_rhsDD01, stencil_idx_0_0_0_VVV, (access(At_rhsDD01, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x381 + (-6 * ((x382 + x383))) + (15 * ((x384 + x385))) + ((stencil(AtDD01, stencil_idx_m3_0_0_VVV) + stencil(AtDD01, stencil_idx_3_0_0_VVV))))) + (DYI * (x381 + (-6 * ((x386 + x387))) + (15 * ((x388 + x389))) + ((stencil(AtDD01, stencil_idx_0_m3_0_VVV) + stencil(AtDD01, stencil_idx_0_3_0_VVV))))) + (DZI * (x381 + (-6 * ((x390 + x391))) + (15 * ((x392 + x393))) + ((stencil(AtDD01, stencil_idx_0_0_m3_VVV) + stencil(AtDD01, stencil_idx_0_0_3_VVV)))))))));
        vreal x399 = stencil(AtDD11, stencil_idx_0_m2_0_VVV); // x399: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m2__0_]
        vreal x400 = stencil(AtDD11, stencil_idx_0_2_0_VVV); // x400: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__2__0_]
        vreal x401 = stencil(AtDD11, stencil_idx_0_m1_0_VVV); // x401: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m1__0_]
        vreal x402 = stencil(AtDD11, stencil_idx_0_1_0_VVV); // x402: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__1__0_]
        vreal x403 = stencil(AtDD11, stencil_idx_0_0_m2_VVV); // x403: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m2_]
        vreal x404 = stencil(AtDD11, stencil_idx_0_0_2_VVV); // x404: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__2_]
        vreal x405 = stencil(AtDD11, stencil_idx_0_0_m1_VVV); // x405: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m1_]
        vreal x406 = stencil(AtDD11, stencil_idx_0_0_1_VVV); // x406: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__1_]
        vreal x395 = stencil(AtDD11, stencil_idx_m2_0_0_VVV); // x395: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m2__0__0_]
        vreal x396 = stencil(AtDD11, stencil_idx_2_0_0_VVV); // x396: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__2__0__0_]
        vreal x397 = stencil(AtDD11, stencil_idx_m1_0_0_VVV); // x397: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m1__0__0_]
        vreal x398 = stencil(AtDD11, stencil_idx_1_0_0_VVV); // x398: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__1__0__0_]
        vreal x394 = (-20 * stencil(AtDD11, stencil_idx_0_0_0_VVV)); // x394: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__0_]
        store(At_rhsDD11, stencil_idx_0_0_0_VVV, (access(At_rhsDD11, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x394 + (-6 * ((x395 + x396))) + (15 * ((x397 + x398))) + ((stencil(AtDD11, stencil_idx_m3_0_0_VVV) + stencil(AtDD11, stencil_idx_3_0_0_VVV))))) + (DYI * (x394 + (-6 * ((x399 + x400))) + (15 * ((x401 + x402))) + ((stencil(AtDD11, stencil_idx_0_m3_0_VVV) + stencil(AtDD11, stencil_idx_0_3_0_VVV))))) + (DZI * (x394 + (-6 * ((x403 + x404))) + (15 * ((x405 + x406))) + ((stencil(AtDD11, stencil_idx_0_0_m3_VVV) + stencil(AtDD11, stencil_idx_0_0_3_VVV)))))))));
        vreal x412 = stencil(AtDD02, stencil_idx_0_m2_0_VVV); // x412: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m2__0_]
        vreal x413 = stencil(AtDD02, stencil_idx_0_2_0_VVV); // x413: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__2__0_]
        vreal x414 = stencil(AtDD02, stencil_idx_0_m1_0_VVV); // x414: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m1__0_]
        vreal x415 = stencil(AtDD02, stencil_idx_0_1_0_VVV); // x415: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__1__0_]
        vreal x416 = stencil(AtDD02, stencil_idx_0_0_m2_VVV); // x416: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m2_]
        vreal x417 = stencil(AtDD02, stencil_idx_0_0_2_VVV); // x417: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__2_]
        vreal x418 = stencil(AtDD02, stencil_idx_0_0_m1_VVV); // x418: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m1_]
        vreal x419 = stencil(AtDD02, stencil_idx_0_0_1_VVV); // x419: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__1_]
        vreal x408 = stencil(AtDD02, stencil_idx_m2_0_0_VVV); // x408: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m2__0__0_]
        vreal x409 = stencil(AtDD02, stencil_idx_2_0_0_VVV); // x409: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__2__0__0_]
        vreal x410 = stencil(AtDD02, stencil_idx_m1_0_0_VVV); // x410: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m1__0__0_]
        vreal x411 = stencil(AtDD02, stencil_idx_1_0_0_VVV); // x411: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__1__0__0_]
        vreal x407 = (-20 * stencil(AtDD02, stencil_idx_0_0_0_VVV)); // x407: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__0_]
        store(At_rhsDD02, stencil_idx_0_0_0_VVV, (access(At_rhsDD02, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x407 + (-6 * ((x408 + x409))) + (15 * ((x410 + x411))) + ((stencil(AtDD02, stencil_idx_m3_0_0_VVV) + stencil(AtDD02, stencil_idx_3_0_0_VVV))))) + (DYI * (x407 + (-6 * ((x412 + x413))) + (15 * ((x414 + x415))) + ((stencil(AtDD02, stencil_idx_0_m3_0_VVV) + stencil(AtDD02, stencil_idx_0_3_0_VVV))))) + (DZI * (x407 + (-6 * ((x416 + x417))) + (15 * ((x418 + x419))) + ((stencil(AtDD02, stencil_idx_0_0_m3_VVV) + stencil(AtDD02, stencil_idx_0_0_3_VVV)))))))));
        vreal x425 = stencil(AtDD12, stencil_idx_0_m2_0_VVV); // x425: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m2__0_]
        vreal x426 = stencil(AtDD12, stencil_idx_0_2_0_VVV); // x426: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__2__0_]
        vreal x427 = stencil(AtDD12, stencil_idx_0_m1_0_VVV); // x427: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m1__0_]
        vreal x428 = stencil(AtDD12, stencil_idx_0_1_0_VVV); // x428: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__1__0_]
        vreal x429 = stencil(AtDD12, stencil_idx_0_0_m2_VVV); // x429: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m2_]
        vreal x430 = stencil(AtDD12, stencil_idx_0_0_2_VVV); // x430: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__2_]
        vreal x431 = stencil(AtDD12, stencil_idx_0_0_m1_VVV); // x431: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m1_]
        vreal x432 = stencil(AtDD12, stencil_idx_0_0_1_VVV); // x432: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__1_]
        vreal x421 = stencil(AtDD12, stencil_idx_m2_0_0_VVV); // x421: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m2__0__0_]
        vreal x422 = stencil(AtDD12, stencil_idx_2_0_0_VVV); // x422: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__2__0__0_]
        vreal x423 = stencil(AtDD12, stencil_idx_m1_0_0_VVV); // x423: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m1__0__0_]
        vreal x424 = stencil(AtDD12, stencil_idx_1_0_0_VVV); // x424: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__1__0__0_]
        vreal x420 = (-20 * stencil(AtDD12, stencil_idx_0_0_0_VVV)); // x420: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__0_]
        store(At_rhsDD12, stencil_idx_0_0_0_VVV, (access(At_rhsDD12, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x420 + (-6 * ((x421 + x422))) + (15 * ((x423 + x424))) + ((stencil(AtDD12, stencil_idx_m3_0_0_VVV) + stencil(AtDD12, stencil_idx_3_0_0_VVV))))) + (DYI * (x420 + (-6 * ((x425 + x426))) + (15 * ((x427 + x428))) + ((stencil(AtDD12, stencil_idx_0_m3_0_VVV) + stencil(AtDD12, stencil_idx_0_3_0_VVV))))) + (DZI * (x420 + (-6 * ((x429 + x430))) + (15 * ((x431 + x432))) + ((stencil(AtDD12, stencil_idx_0_0_m3_VVV) + stencil(AtDD12, stencil_idx_0_0_3_VVV)))))))));
        vreal x438 = stencil(AtDD22, stencil_idx_0_m2_0_VVV); // x438: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m2__0_]
        vreal x439 = stencil(AtDD22, stencil_idx_0_2_0_VVV); // x439: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__2__0_]
        vreal x440 = stencil(AtDD22, stencil_idx_0_m1_0_VVV); // x440: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m1__0_]
        vreal x441 = stencil(AtDD22, stencil_idx_0_1_0_VVV); // x441: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__1__0_]
        vreal x442 = stencil(AtDD22, stencil_idx_0_0_m2_VVV); // x442: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m2_]
        vreal x443 = stencil(AtDD22, stencil_idx_0_0_2_VVV); // x443: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__2_]
        vreal x444 = stencil(AtDD22, stencil_idx_0_0_m1_VVV); // x444: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m1_]
        vreal x445 = stencil(AtDD22, stencil_idx_0_0_1_VVV); // x445: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__1_]
        vreal x434 = stencil(AtDD22, stencil_idx_m2_0_0_VVV); // x434: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m2__0__0_]
        vreal x435 = stencil(AtDD22, stencil_idx_2_0_0_VVV); // x435: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__2__0__0_]
        vreal x436 = stencil(AtDD22, stencil_idx_m1_0_0_VVV); // x436: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m1__0__0_]
        vreal x437 = stencil(AtDD22, stencil_idx_1_0_0_VVV); // x437: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__1__0__0_]
        vreal x433 = (-20 * stencil(AtDD22, stencil_idx_0_0_0_VVV)); // x433: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__0_]
        store(At_rhsDD22, stencil_idx_0_0_0_VVV, (access(At_rhsDD22, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x433 + (-6 * ((x434 + x435))) + (15 * ((x436 + x437))) + ((stencil(AtDD22, stencil_idx_m3_0_0_VVV) + stencil(AtDD22, stencil_idx_3_0_0_VVV))))) + (DYI * (x433 + (-6 * ((x438 + x439))) + (15 * ((x440 + x441))) + ((stencil(AtDD22, stencil_idx_0_m3_0_VVV) + stencil(AtDD22, stencil_idx_0_3_0_VVV))))) + (DZI * (x433 + (-6 * ((x442 + x443))) + (15 * ((x444 + x445))) + ((stencil(AtDD22, stencil_idx_0_0_m3_VVV) + stencil(AtDD22, stencil_idx_0_0_3_VVV)))))))));    
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
        vreal x447 = stencil(evo_lapse, stencil_idx_m2_0_0_VVV); // x447: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m2__0__0_]
        vreal x448 = stencil(evo_lapse, stencil_idx_2_0_0_VVV); // x448: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__2__0__0_]
        vreal x449 = ((x447 + x448)); // x449: Dependency! Liveness = 2; [x447, x448]
        vreal x450 = stencil(evo_lapse, stencil_idx_m1_0_0_VVV); // x450: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m1__0__0_]
        vreal x451 = stencil(evo_lapse, stencil_idx_1_0_0_VVV); // x451: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__1__0__0_]
        vreal x452 = ((x450 + x451)); // x452: Dependency! Liveness = 2; [x450, x451]
        vreal x453 = stencil(evo_lapse, stencil_idx_0_m2_0_VVV); // x453: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m2__0_]
        vreal x454 = stencil(evo_lapse, stencil_idx_0_2_0_VVV); // x454: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__2__0_]
        vreal x455 = ((x453 + x454)); // x455: Dependency! Liveness = 2; [x453, x454]
        vreal x456 = stencil(evo_lapse, stencil_idx_0_m1_0_VVV); // x456: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m1__0_]
        vreal x457 = stencil(evo_lapse, stencil_idx_0_1_0_VVV); // x457: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__1__0_]
        vreal x458 = ((x456 + x457)); // x458: Dependency! Liveness = 2; [x456, x457]
        vreal x459 = stencil(evo_lapse, stencil_idx_0_0_m2_VVV); // x459: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m2_]
        vreal x460 = stencil(evo_lapse, stencil_idx_0_0_2_VVV); // x460: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__2_]
        vreal x461 = ((x459 + x460)); // x461: Dependency! Liveness = 2; [x459, x460]
        vreal x462 = stencil(evo_lapse, stencil_idx_0_0_m1_VVV); // x462: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m1_]
        vreal x463 = stencil(evo_lapse, stencil_idx_0_0_1_VVV); // x463: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__1_]
        vreal x464 = ((x462 + x463)); // x464: Dependency! Liveness = 2; [x462, x463]
        vreal x446 = (-20 * stencil(evo_lapse, stencil_idx_0_0_0_VVV)); // x446: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__0_]
        vreal x254 = ((1.0 / 64.0) * dissipation_epsilon); // x254: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_lapse_rhs, stencil_idx_0_0_0_VVV, (access(evo_lapse_rhs, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x446 + (-6 * x449) + (15 * x452) + ((stencil(evo_lapse, stencil_idx_m3_0_0_VVV) + stencil(evo_lapse, stencil_idx_3_0_0_VVV))))) + (DYI * (x446 + (-6 * x455) + (15 * x458) + ((stencil(evo_lapse, stencil_idx_0_m3_0_VVV) + stencil(evo_lapse, stencil_idx_0_3_0_VVV))))) + (DZI * (x446 + (-6 * x461) + (15 * x464) + ((stencil(evo_lapse, stencil_idx_0_0_m3_VVV) + stencil(evo_lapse, stencil_idx_0_0_3_VVV)))))))));    
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
        vreal x466 = stencil(evo_shiftU0, stencil_idx_m2_0_0_VVV); // x466: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m2__0__0_]
        vreal x467 = stencil(evo_shiftU0, stencil_idx_2_0_0_VVV); // x467: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__2__0__0_]
        vreal x468 = ((x466 + x467)); // x468: Dependency! Liveness = 2; [x466, x467]
        vreal x469 = stencil(evo_shiftU0, stencil_idx_m1_0_0_VVV); // x469: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m1__0__0_]
        vreal x470 = stencil(evo_shiftU0, stencil_idx_1_0_0_VVV); // x470: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__1__0__0_]
        vreal x471 = ((x469 + x470)); // x471: Dependency! Liveness = 2; [x469, x470]
        vreal x472 = stencil(evo_shiftU0, stencil_idx_0_m2_0_VVV); // x472: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m2__0_]
        vreal x473 = stencil(evo_shiftU0, stencil_idx_0_2_0_VVV); // x473: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__2__0_]
        vreal x474 = ((x472 + x473)); // x474: Dependency! Liveness = 2; [x472, x473]
        vreal x475 = stencil(evo_shiftU0, stencil_idx_0_m1_0_VVV); // x475: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m1__0_]
        vreal x476 = stencil(evo_shiftU0, stencil_idx_0_1_0_VVV); // x476: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__1__0_]
        vreal x477 = ((x475 + x476)); // x477: Dependency! Liveness = 2; [x475, x476]
        vreal x478 = stencil(evo_shiftU0, stencil_idx_0_0_m2_VVV); // x478: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m2_]
        vreal x479 = stencil(evo_shiftU0, stencil_idx_0_0_2_VVV); // x479: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__2_]
        vreal x480 = ((x478 + x479)); // x480: Dependency! Liveness = 2; [x478, x479]
        vreal x481 = stencil(evo_shiftU0, stencil_idx_0_0_m1_VVV); // x481: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m1_]
        vreal x482 = stencil(evo_shiftU0, stencil_idx_0_0_1_VVV); // x482: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__1_]
        vreal x483 = ((x481 + x482)); // x483: Dependency! Liveness = 2; [x481, x482]
        vreal x465 = (-20 * stencil(evo_shiftU0, stencil_idx_0_0_0_VVV)); // x465: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__0_]
        vreal x254 = ((1.0 / 64.0) * dissipation_epsilon); // x254: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_shift_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU0, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x465 + (-6 * x468) + (15 * x471) + ((stencil(evo_shiftU0, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU0, stencil_idx_3_0_0_VVV))))) + (DYI * (x465 + (-6 * x474) + (15 * x477) + ((stencil(evo_shiftU0, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU0, stencil_idx_0_3_0_VVV))))) + (DZI * (x465 + (-6 * x480) + (15 * x483) + ((stencil(evo_shiftU0, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU0, stencil_idx_0_0_3_VVV)))))))));
        vreal x485 = stencil(evo_shiftU1, stencil_idx_m2_0_0_VVV); // x485: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m2__0__0_]
        vreal x486 = stencil(evo_shiftU1, stencil_idx_2_0_0_VVV); // x486: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__2__0__0_]
        vreal x487 = ((x485 + x486)); // x487: Dependency! Liveness = 2; [x485, x486]
        vreal x488 = stencil(evo_shiftU1, stencil_idx_m1_0_0_VVV); // x488: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m1__0__0_]
        vreal x489 = stencil(evo_shiftU1, stencil_idx_1_0_0_VVV); // x489: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__1__0__0_]
        vreal x490 = ((x488 + x489)); // x490: Dependency! Liveness = 2; [x488, x489]
        vreal x491 = stencil(evo_shiftU1, stencil_idx_0_m2_0_VVV); // x491: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m2__0_]
        vreal x492 = stencil(evo_shiftU1, stencil_idx_0_2_0_VVV); // x492: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__2__0_]
        vreal x493 = ((x491 + x492)); // x493: Dependency! Liveness = 2; [x491, x492]
        vreal x494 = stencil(evo_shiftU1, stencil_idx_0_m1_0_VVV); // x494: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m1__0_]
        vreal x495 = stencil(evo_shiftU1, stencil_idx_0_1_0_VVV); // x495: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__1__0_]
        vreal x496 = ((x494 + x495)); // x496: Dependency! Liveness = 2; [x494, x495]
        vreal x497 = stencil(evo_shiftU1, stencil_idx_0_0_m2_VVV); // x497: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m2_]
        vreal x498 = stencil(evo_shiftU1, stencil_idx_0_0_2_VVV); // x498: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__2_]
        vreal x499 = ((x497 + x498)); // x499: Dependency! Liveness = 2; [x497, x498]
        vreal x500 = stencil(evo_shiftU1, stencil_idx_0_0_m1_VVV); // x500: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m1_]
        vreal x501 = stencil(evo_shiftU1, stencil_idx_0_0_1_VVV); // x501: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__1_]
        vreal x502 = ((x500 + x501)); // x502: Dependency! Liveness = 2; [x500, x501]
        vreal x484 = (-20 * stencil(evo_shiftU1, stencil_idx_0_0_0_VVV)); // x484: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__0_]
        store(evo_shift_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU1, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x484 + (-6 * x487) + (15 * x490) + ((stencil(evo_shiftU1, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU1, stencil_idx_3_0_0_VVV))))) + (DYI * (x484 + (-6 * x493) + (15 * x496) + ((stencil(evo_shiftU1, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU1, stencil_idx_0_3_0_VVV))))) + (DZI * (x484 + (-6 * x499) + (15 * x502) + ((stencil(evo_shiftU1, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU1, stencil_idx_0_0_3_VVV)))))))));
        vreal x504 = stencil(evo_shiftU2, stencil_idx_m2_0_0_VVV); // x504: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m2__0__0_]
        vreal x505 = stencil(evo_shiftU2, stencil_idx_2_0_0_VVV); // x505: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__2__0__0_]
        vreal x506 = ((x504 + x505)); // x506: Dependency! Liveness = 2; [x504, x505]
        vreal x507 = stencil(evo_shiftU2, stencil_idx_m1_0_0_VVV); // x507: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m1__0__0_]
        vreal x508 = stencil(evo_shiftU2, stencil_idx_1_0_0_VVV); // x508: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__1__0__0_]
        vreal x509 = ((x507 + x508)); // x509: Dependency! Liveness = 2; [x507, x508]
        vreal x510 = stencil(evo_shiftU2, stencil_idx_0_m2_0_VVV); // x510: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m2__0_]
        vreal x511 = stencil(evo_shiftU2, stencil_idx_0_2_0_VVV); // x511: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__2__0_]
        vreal x512 = ((x510 + x511)); // x512: Dependency! Liveness = 2; [x510, x511]
        vreal x513 = stencil(evo_shiftU2, stencil_idx_0_m1_0_VVV); // x513: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m1__0_]
        vreal x514 = stencil(evo_shiftU2, stencil_idx_0_1_0_VVV); // x514: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__1__0_]
        vreal x515 = ((x513 + x514)); // x515: Dependency! Liveness = 2; [x513, x514]
        vreal x516 = stencil(evo_shiftU2, stencil_idx_0_0_m2_VVV); // x516: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m2_]
        vreal x517 = stencil(evo_shiftU2, stencil_idx_0_0_2_VVV); // x517: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__2_]
        vreal x518 = ((x516 + x517)); // x518: Dependency! Liveness = 2; [x516, x517]
        vreal x519 = stencil(evo_shiftU2, stencil_idx_0_0_m1_VVV); // x519: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m1_]
        vreal x520 = stencil(evo_shiftU2, stencil_idx_0_0_1_VVV); // x520: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__1_]
        vreal x521 = ((x519 + x520)); // x521: Dependency! Liveness = 2; [x519, x520]
        vreal x503 = (-20 * stencil(evo_shiftU2, stencil_idx_0_0_0_VVV)); // x503: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__0_]
        store(evo_shift_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU2, stencil_idx_0_0_0_VVV) + (x254 * ((DXI * (x503 + (-6 * x506) + (15 * x509) + ((stencil(evo_shiftU2, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU2, stencil_idx_3_0_0_VVV))))) + (DYI * (x503 + (-6 * x512) + (15 * x515) + ((stencil(evo_shiftU2, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU2, stencil_idx_0_3_0_VVV))))) + (DZI * (x503 + (-6 * x518) + (15 * x521) + ((stencil(evo_shiftU2, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU2, stencil_idx_0_0_3_VVV)))))))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}