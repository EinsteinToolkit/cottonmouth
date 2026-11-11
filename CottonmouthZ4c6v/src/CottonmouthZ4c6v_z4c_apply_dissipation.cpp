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
    CCTK_ASSERT((cctk_nghostzones[0] >= 4));
    CCTK_ASSERT((cctk_nghostzones[1] >= 4));
    CCTK_ASSERT((cctk_nghostzones[2] >= 4));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // z4c_apply_dissipation loop 0
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
        vreal x284 = stencil(Theta, stencil_idx_0_m3_0_VVV); // x284: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__m3__0_]
        vreal x285 = stencil(Theta, stencil_idx_0_3_0_VVV); // x285: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__3__0_]
        vreal x286 = stencil(Theta, stencil_idx_0_m2_0_VVV); // x286: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__m2__0_]
        vreal x287 = stencil(Theta, stencil_idx_0_2_0_VVV); // x287: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__2__0_]
        vreal x288 = stencil(Theta, stencil_idx_0_m1_0_VVV); // x288: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__m1__0_]
        vreal x289 = stencil(Theta, stencil_idx_0_1_0_VVV); // x289: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__1__0_]
        vreal x290 = stencil(Theta, stencil_idx_0_0_m3_VVV); // x290: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__m3_]
        vreal x291 = stencil(Theta, stencil_idx_0_0_3_VVV); // x291: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__3_]
        vreal x292 = stencil(Theta, stencil_idx_0_0_m2_VVV); // x292: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__m2_]
        vreal x293 = stencil(Theta, stencil_idx_0_0_2_VVV); // x293: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__2_]
        vreal x294 = stencil(Theta, stencil_idx_0_0_m1_VVV); // x294: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__m1_]
        vreal x295 = stencil(Theta, stencil_idx_0_0_1_VVV); // x295: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__1_]
        vreal x278 = stencil(Theta, stencil_idx_m3_0_0_VVV); // x278: Dependency! Liveness = 1; [__dummy_stencil__Theta__m3__0__0_]
        vreal x279 = stencil(Theta, stencil_idx_3_0_0_VVV); // x279: Dependency! Liveness = 1; [__dummy_stencil__Theta__3__0__0_]
        vreal x280 = stencil(Theta, stencil_idx_m2_0_0_VVV); // x280: Dependency! Liveness = 1; [__dummy_stencil__Theta__m2__0__0_]
        vreal x281 = stencil(Theta, stencil_idx_2_0_0_VVV); // x281: Dependency! Liveness = 1; [__dummy_stencil__Theta__2__0__0_]
        vreal x282 = stencil(Theta, stencil_idx_m1_0_0_VVV); // x282: Dependency! Liveness = 1; [__dummy_stencil__Theta__m1__0__0_]
        vreal x283 = stencil(Theta, stencil_idx_1_0_0_VVV); // x283: Dependency! Liveness = 1; [__dummy_stencil__Theta__1__0__0_]
        vreal x277 = (70 * stencil(Theta, stencil_idx_0_0_0_VVV)); // x277: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__0_]
        vreal x296 = ((1.0 / 256.0) * dissipation_epsilon); // x296: Dependency! Liveness = 1; [dissipation_epsilon]
        store(Theta_rhs, stencil_idx_0_0_0_VVV, (access(Theta_rhs, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x277 + (-56 * ((x282 + x283))) + (-8 * ((x278 + x279))) + (28 * ((x280 + x281))) + ((stencil(Theta, stencil_idx_m4_0_0_VVV) + stencil(Theta, stencil_idx_4_0_0_VVV))))) + (DYI * (x277 + (-56 * ((x288 + x289))) + (-8 * ((x284 + x285))) + (28 * ((x286 + x287))) + ((stencil(Theta, stencil_idx_0_m4_0_VVV) + stencil(Theta, stencil_idx_0_4_0_VVV))))) + (DZI * (x277 + (-56 * ((x294 + x295))) + (-8 * ((x290 + x291))) + (28 * ((x292 + x293))) + ((stencil(Theta, stencil_idx_0_0_m4_VVV) + stencil(Theta, stencil_idx_0_0_4_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 1
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
        vreal x298 = stencil(chi, stencil_idx_m3_0_0_VVV); // x298: Dependency! Liveness = 1; [__dummy_stencil__chi__m3__0__0_]
        vreal x299 = stencil(chi, stencil_idx_3_0_0_VVV); // x299: Dependency! Liveness = 1; [__dummy_stencil__chi__3__0__0_]
        vreal x300 = ((x298 + x299)); // x300: Dependency! Liveness = 2; [x298, x299]
        vreal x301 = stencil(chi, stencil_idx_m2_0_0_VVV); // x301: Dependency! Liveness = 1; [__dummy_stencil__chi__m2__0__0_]
        vreal x302 = stencil(chi, stencil_idx_2_0_0_VVV); // x302: Dependency! Liveness = 1; [__dummy_stencil__chi__2__0__0_]
        vreal x303 = ((x301 + x302)); // x303: Dependency! Liveness = 2; [x301, x302]
        vreal x304 = stencil(chi, stencil_idx_m1_0_0_VVV); // x304: Dependency! Liveness = 1; [__dummy_stencil__chi__m1__0__0_]
        vreal x305 = stencil(chi, stencil_idx_1_0_0_VVV); // x305: Dependency! Liveness = 1; [__dummy_stencil__chi__1__0__0_]
        vreal x306 = ((x304 + x305)); // x306: Dependency! Liveness = 2; [x304, x305]
        vreal x307 = stencil(chi, stencil_idx_0_m3_0_VVV); // x307: Dependency! Liveness = 1; [__dummy_stencil__chi__0__m3__0_]
        vreal x308 = stencil(chi, stencil_idx_0_3_0_VVV); // x308: Dependency! Liveness = 1; [__dummy_stencil__chi__0__3__0_]
        vreal x309 = ((x307 + x308)); // x309: Dependency! Liveness = 2; [x307, x308]
        vreal x310 = stencil(chi, stencil_idx_0_m2_0_VVV); // x310: Dependency! Liveness = 1; [__dummy_stencil__chi__0__m2__0_]
        vreal x311 = stencil(chi, stencil_idx_0_2_0_VVV); // x311: Dependency! Liveness = 1; [__dummy_stencil__chi__0__2__0_]
        vreal x312 = ((x310 + x311)); // x312: Dependency! Liveness = 2; [x310, x311]
        vreal x313 = stencil(chi, stencil_idx_0_m1_0_VVV); // x313: Dependency! Liveness = 1; [__dummy_stencil__chi__0__m1__0_]
        vreal x314 = stencil(chi, stencil_idx_0_1_0_VVV); // x314: Dependency! Liveness = 1; [__dummy_stencil__chi__0__1__0_]
        vreal x315 = ((x313 + x314)); // x315: Dependency! Liveness = 2; [x313, x314]
        vreal x316 = stencil(chi, stencil_idx_0_0_m3_VVV); // x316: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__m3_]
        vreal x317 = stencil(chi, stencil_idx_0_0_3_VVV); // x317: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__3_]
        vreal x318 = ((x316 + x317)); // x318: Dependency! Liveness = 2; [x316, x317]
        vreal x319 = stencil(chi, stencil_idx_0_0_m2_VVV); // x319: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__m2_]
        vreal x320 = stencil(chi, stencil_idx_0_0_2_VVV); // x320: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__2_]
        vreal x321 = ((x319 + x320)); // x321: Dependency! Liveness = 2; [x319, x320]
        vreal x322 = stencil(chi, stencil_idx_0_0_m1_VVV); // x322: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__m1_]
        vreal x323 = stencil(chi, stencil_idx_0_0_1_VVV); // x323: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__1_]
        vreal x324 = ((x322 + x323)); // x324: Dependency! Liveness = 2; [x322, x323]
        vreal x297 = (70 * stencil(chi, stencil_idx_0_0_0_VVV)); // x297: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__0_]
        vreal x296 = ((1.0 / 256.0) * dissipation_epsilon); // x296: Dependency! Liveness = 1; [dissipation_epsilon]
        store(chi_rhs, stencil_idx_0_0_0_VVV, (access(chi_rhs, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x297 + (-56 * x306) + (-8 * x300) + (28 * x303) + ((stencil(chi, stencil_idx_m4_0_0_VVV) + stencil(chi, stencil_idx_4_0_0_VVV))))) + (DYI * (x297 + (-56 * x315) + (-8 * x309) + (28 * x312) + ((stencil(chi, stencil_idx_0_m4_0_VVV) + stencil(chi, stencil_idx_0_4_0_VVV))))) + (DZI * (x297 + (-56 * x324) + (-8 * x318) + (28 * x321) + ((stencil(chi, stencil_idx_0_0_m4_VVV) + stencil(chi, stencil_idx_0_0_4_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 2
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
        vreal x332 = stencil(trK, stencil_idx_0_m3_0_VVV); // x332: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m3__0_]
        vreal x333 = stencil(trK, stencil_idx_0_3_0_VVV); // x333: Dependency! Liveness = 1; [__dummy_stencil__trK__0__3__0_]
        vreal x334 = stencil(trK, stencil_idx_0_m2_0_VVV); // x334: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m2__0_]
        vreal x335 = stencil(trK, stencil_idx_0_2_0_VVV); // x335: Dependency! Liveness = 1; [__dummy_stencil__trK__0__2__0_]
        vreal x336 = stencil(trK, stencil_idx_0_m1_0_VVV); // x336: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m1__0_]
        vreal x337 = stencil(trK, stencil_idx_0_1_0_VVV); // x337: Dependency! Liveness = 1; [__dummy_stencil__trK__0__1__0_]
        vreal x338 = stencil(trK, stencil_idx_0_0_m3_VVV); // x338: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m3_]
        vreal x339 = stencil(trK, stencil_idx_0_0_3_VVV); // x339: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__3_]
        vreal x340 = stencil(trK, stencil_idx_0_0_m2_VVV); // x340: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m2_]
        vreal x341 = stencil(trK, stencil_idx_0_0_2_VVV); // x341: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__2_]
        vreal x342 = stencil(trK, stencil_idx_0_0_m1_VVV); // x342: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m1_]
        vreal x343 = stencil(trK, stencil_idx_0_0_1_VVV); // x343: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__1_]
        vreal x326 = stencil(trK, stencil_idx_m3_0_0_VVV); // x326: Dependency! Liveness = 1; [__dummy_stencil__trK__m3__0__0_]
        vreal x327 = stencil(trK, stencil_idx_3_0_0_VVV); // x327: Dependency! Liveness = 1; [__dummy_stencil__trK__3__0__0_]
        vreal x328 = stencil(trK, stencil_idx_m2_0_0_VVV); // x328: Dependency! Liveness = 1; [__dummy_stencil__trK__m2__0__0_]
        vreal x329 = stencil(trK, stencil_idx_2_0_0_VVV); // x329: Dependency! Liveness = 1; [__dummy_stencil__trK__2__0__0_]
        vreal x330 = stencil(trK, stencil_idx_m1_0_0_VVV); // x330: Dependency! Liveness = 1; [__dummy_stencil__trK__m1__0__0_]
        vreal x331 = stencil(trK, stencil_idx_1_0_0_VVV); // x331: Dependency! Liveness = 1; [__dummy_stencil__trK__1__0__0_]
        vreal x325 = (70 * stencil(trK, stencil_idx_0_0_0_VVV)); // x325: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__0_]
        vreal x296 = ((1.0 / 256.0) * dissipation_epsilon); // x296: Dependency! Liveness = 1; [dissipation_epsilon]
        store(trK_rhs, stencil_idx_0_0_0_VVV, (access(trK_rhs, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x325 + (-56 * ((x330 + x331))) + (-8 * ((x326 + x327))) + (28 * ((x328 + x329))) + ((stencil(trK, stencil_idx_m4_0_0_VVV) + stencil(trK, stencil_idx_4_0_0_VVV))))) + (DYI * (x325 + (-56 * ((x336 + x337))) + (-8 * ((x332 + x333))) + (28 * ((x334 + x335))) + ((stencil(trK, stencil_idx_0_m4_0_VVV) + stencil(trK, stencil_idx_0_4_0_VVV))))) + (DZI * (x325 + (-56 * ((x342 + x343))) + (-8 * ((x338 + x339))) + (28 * ((x340 + x341))) + ((stencil(trK, stencil_idx_0_0_m4_VVV) + stencil(trK, stencil_idx_0_0_4_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 3
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
        vreal x351 = stencil(evo_GammatU0, stencil_idx_0_m3_0_VVV); // x351: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__m3__0_]
        vreal x352 = stencil(evo_GammatU0, stencil_idx_0_3_0_VVV); // x352: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__3__0_]
        vreal x353 = stencil(evo_GammatU0, stencil_idx_0_m2_0_VVV); // x353: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__m2__0_]
        vreal x354 = stencil(evo_GammatU0, stencil_idx_0_2_0_VVV); // x354: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__2__0_]
        vreal x355 = stencil(evo_GammatU0, stencil_idx_0_m1_0_VVV); // x355: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__m1__0_]
        vreal x356 = stencil(evo_GammatU0, stencil_idx_0_1_0_VVV); // x356: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__1__0_]
        vreal x357 = stencil(evo_GammatU0, stencil_idx_0_0_m3_VVV); // x357: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__m3_]
        vreal x358 = stencil(evo_GammatU0, stencil_idx_0_0_3_VVV); // x358: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__3_]
        vreal x359 = stencil(evo_GammatU0, stencil_idx_0_0_m2_VVV); // x359: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__m2_]
        vreal x360 = stencil(evo_GammatU0, stencil_idx_0_0_2_VVV); // x360: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__2_]
        vreal x361 = stencil(evo_GammatU0, stencil_idx_0_0_m1_VVV); // x361: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__m1_]
        vreal x362 = stencil(evo_GammatU0, stencil_idx_0_0_1_VVV); // x362: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__1_]
        vreal x345 = stencil(evo_GammatU0, stencil_idx_m3_0_0_VVV); // x345: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__m3__0__0_]
        vreal x346 = stencil(evo_GammatU0, stencil_idx_3_0_0_VVV); // x346: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__3__0__0_]
        vreal x347 = stencil(evo_GammatU0, stencil_idx_m2_0_0_VVV); // x347: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__m2__0__0_]
        vreal x348 = stencil(evo_GammatU0, stencil_idx_2_0_0_VVV); // x348: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__2__0__0_]
        vreal x349 = stencil(evo_GammatU0, stencil_idx_m1_0_0_VVV); // x349: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__m1__0__0_]
        vreal x350 = stencil(evo_GammatU0, stencil_idx_1_0_0_VVV); // x350: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__1__0__0_]
        vreal x344 = (70 * stencil(evo_GammatU0, stencil_idx_0_0_0_VVV)); // x344: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__0_]
        vreal x296 = ((1.0 / 256.0) * dissipation_epsilon); // x296: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_Gammat_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU0, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x344 + (-56 * ((x349 + x350))) + (-8 * ((x345 + x346))) + (28 * ((x347 + x348))) + ((stencil(evo_GammatU0, stencil_idx_m4_0_0_VVV) + stencil(evo_GammatU0, stencil_idx_4_0_0_VVV))))) + (DYI * (x344 + (-56 * ((x355 + x356))) + (-8 * ((x351 + x352))) + (28 * ((x353 + x354))) + ((stencil(evo_GammatU0, stencil_idx_0_m4_0_VVV) + stencil(evo_GammatU0, stencil_idx_0_4_0_VVV))))) + (DZI * (x344 + (-56 * ((x361 + x362))) + (-8 * ((x357 + x358))) + (28 * ((x359 + x360))) + ((stencil(evo_GammatU0, stencil_idx_0_0_m4_VVV) + stencil(evo_GammatU0, stencil_idx_0_0_4_VVV)))))))));
        vreal x370 = stencil(evo_GammatU1, stencil_idx_0_m3_0_VVV); // x370: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__m3__0_]
        vreal x371 = stencil(evo_GammatU1, stencil_idx_0_3_0_VVV); // x371: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__3__0_]
        vreal x372 = stencil(evo_GammatU1, stencil_idx_0_m2_0_VVV); // x372: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__m2__0_]
        vreal x373 = stencil(evo_GammatU1, stencil_idx_0_2_0_VVV); // x373: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__2__0_]
        vreal x374 = stencil(evo_GammatU1, stencil_idx_0_m1_0_VVV); // x374: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__m1__0_]
        vreal x375 = stencil(evo_GammatU1, stencil_idx_0_1_0_VVV); // x375: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__1__0_]
        vreal x376 = stencil(evo_GammatU1, stencil_idx_0_0_m3_VVV); // x376: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__m3_]
        vreal x377 = stencil(evo_GammatU1, stencil_idx_0_0_3_VVV); // x377: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__3_]
        vreal x378 = stencil(evo_GammatU1, stencil_idx_0_0_m2_VVV); // x378: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__m2_]
        vreal x379 = stencil(evo_GammatU1, stencil_idx_0_0_2_VVV); // x379: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__2_]
        vreal x380 = stencil(evo_GammatU1, stencil_idx_0_0_m1_VVV); // x380: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__m1_]
        vreal x381 = stencil(evo_GammatU1, stencil_idx_0_0_1_VVV); // x381: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__1_]
        vreal x364 = stencil(evo_GammatU1, stencil_idx_m3_0_0_VVV); // x364: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__m3__0__0_]
        vreal x365 = stencil(evo_GammatU1, stencil_idx_3_0_0_VVV); // x365: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__3__0__0_]
        vreal x366 = stencil(evo_GammatU1, stencil_idx_m2_0_0_VVV); // x366: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__m2__0__0_]
        vreal x367 = stencil(evo_GammatU1, stencil_idx_2_0_0_VVV); // x367: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__2__0__0_]
        vreal x368 = stencil(evo_GammatU1, stencil_idx_m1_0_0_VVV); // x368: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__m1__0__0_]
        vreal x369 = stencil(evo_GammatU1, stencil_idx_1_0_0_VVV); // x369: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__1__0__0_]
        vreal x363 = (70 * stencil(evo_GammatU1, stencil_idx_0_0_0_VVV)); // x363: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__0_]
        store(evo_Gammat_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU1, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x363 + (-56 * ((x368 + x369))) + (-8 * ((x364 + x365))) + (28 * ((x366 + x367))) + ((stencil(evo_GammatU1, stencil_idx_m4_0_0_VVV) + stencil(evo_GammatU1, stencil_idx_4_0_0_VVV))))) + (DYI * (x363 + (-56 * ((x374 + x375))) + (-8 * ((x370 + x371))) + (28 * ((x372 + x373))) + ((stencil(evo_GammatU1, stencil_idx_0_m4_0_VVV) + stencil(evo_GammatU1, stencil_idx_0_4_0_VVV))))) + (DZI * (x363 + (-56 * ((x380 + x381))) + (-8 * ((x376 + x377))) + (28 * ((x378 + x379))) + ((stencil(evo_GammatU1, stencil_idx_0_0_m4_VVV) + stencil(evo_GammatU1, stencil_idx_0_0_4_VVV)))))))));
        vreal x389 = stencil(evo_GammatU2, stencil_idx_0_m3_0_VVV); // x389: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__m3__0_]
        vreal x390 = stencil(evo_GammatU2, stencil_idx_0_3_0_VVV); // x390: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__3__0_]
        vreal x391 = stencil(evo_GammatU2, stencil_idx_0_m2_0_VVV); // x391: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__m2__0_]
        vreal x392 = stencil(evo_GammatU2, stencil_idx_0_2_0_VVV); // x392: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__2__0_]
        vreal x393 = stencil(evo_GammatU2, stencil_idx_0_m1_0_VVV); // x393: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__m1__0_]
        vreal x394 = stencil(evo_GammatU2, stencil_idx_0_1_0_VVV); // x394: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__1__0_]
        vreal x395 = stencil(evo_GammatU2, stencil_idx_0_0_m3_VVV); // x395: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__m3_]
        vreal x396 = stencil(evo_GammatU2, stencil_idx_0_0_3_VVV); // x396: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__3_]
        vreal x397 = stencil(evo_GammatU2, stencil_idx_0_0_m2_VVV); // x397: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__m2_]
        vreal x398 = stencil(evo_GammatU2, stencil_idx_0_0_2_VVV); // x398: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__2_]
        vreal x399 = stencil(evo_GammatU2, stencil_idx_0_0_m1_VVV); // x399: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__m1_]
        vreal x400 = stencil(evo_GammatU2, stencil_idx_0_0_1_VVV); // x400: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__1_]
        vreal x383 = stencil(evo_GammatU2, stencil_idx_m3_0_0_VVV); // x383: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__m3__0__0_]
        vreal x384 = stencil(evo_GammatU2, stencil_idx_3_0_0_VVV); // x384: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__3__0__0_]
        vreal x385 = stencil(evo_GammatU2, stencil_idx_m2_0_0_VVV); // x385: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__m2__0__0_]
        vreal x386 = stencil(evo_GammatU2, stencil_idx_2_0_0_VVV); // x386: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__2__0__0_]
        vreal x387 = stencil(evo_GammatU2, stencil_idx_m1_0_0_VVV); // x387: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__m1__0__0_]
        vreal x388 = stencil(evo_GammatU2, stencil_idx_1_0_0_VVV); // x388: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__1__0__0_]
        vreal x382 = (70 * stencil(evo_GammatU2, stencil_idx_0_0_0_VVV)); // x382: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__0_]
        store(evo_Gammat_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU2, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x382 + (-56 * ((x387 + x388))) + (-8 * ((x383 + x384))) + (28 * ((x385 + x386))) + ((stencil(evo_GammatU2, stencil_idx_m4_0_0_VVV) + stencil(evo_GammatU2, stencil_idx_4_0_0_VVV))))) + (DYI * (x382 + (-56 * ((x393 + x394))) + (-8 * ((x389 + x390))) + (28 * ((x391 + x392))) + ((stencil(evo_GammatU2, stencil_idx_0_m4_0_VVV) + stencil(evo_GammatU2, stencil_idx_0_4_0_VVV))))) + (DZI * (x382 + (-56 * ((x399 + x400))) + (-8 * ((x395 + x396))) + (28 * ((x397 + x398))) + ((stencil(evo_GammatU2, stencil_idx_0_0_m4_VVV) + stencil(evo_GammatU2, stencil_idx_0_0_4_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 4
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
        vreal x199 = stencil(gtDD00, stencil_idx_3_0_0_VVV); // x199: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__3__0__0_]
        vreal x200 = stencil(gtDD00, stencil_idx_m3_0_0_VVV); // x200: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m3__0__0_]
        vreal x402 = ((x199 + x200)); // x402: Dependency! Liveness = 2; [x199, x200]
        vreal x201 = stencil(gtDD00, stencil_idx_m2_0_0_VVV); // x201: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m2__0__0_]
        vreal x202 = stencil(gtDD00, stencil_idx_2_0_0_VVV); // x202: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__2__0__0_]
        vreal x403 = ((x201 + x202)); // x403: Dependency! Liveness = 2; [x201, x202]
        vreal x203 = stencil(gtDD00, stencil_idx_m1_0_0_VVV); // x203: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m1__0__0_]
        vreal x204 = stencil(gtDD00, stencil_idx_1_0_0_VVV); // x204: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__1__0__0_]
        vreal x404 = ((x203 + x204)); // x404: Dependency! Liveness = 2; [x203, x204]
        vreal x116 = stencil(gtDD00, stencil_idx_0_3_0_VVV); // x116: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__3__0_]
        vreal x117 = stencil(gtDD00, stencil_idx_0_m3_0_VVV); // x117: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m3__0_]
        vreal x405 = ((x116 + x117)); // x405: Dependency! Liveness = 2; [x116, x117]
        vreal x118 = stencil(gtDD00, stencil_idx_0_m2_0_VVV); // x118: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m2__0_]
        vreal x119 = stencil(gtDD00, stencil_idx_0_2_0_VVV); // x119: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__2__0_]
        vreal x406 = ((x118 + x119)); // x406: Dependency! Liveness = 2; [x118, x119]
        vreal x120 = stencil(gtDD00, stencil_idx_0_m1_0_VVV); // x120: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m1__0_]
        vreal x121 = stencil(gtDD00, stencil_idx_0_1_0_VVV); // x121: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__1__0_]
        vreal x407 = ((x120 + x121)); // x407: Dependency! Liveness = 2; [x120, x121]
        vreal x102 = stencil(gtDD00, stencil_idx_0_0_3_VVV); // x102: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__3_]
        vreal x103 = stencil(gtDD00, stencil_idx_0_0_m3_VVV); // x103: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m3_]
        vreal x408 = ((x102 + x103)); // x408: Dependency! Liveness = 2; [x102, x103]
        vreal x104 = stencil(gtDD00, stencil_idx_0_0_m2_VVV); // x104: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m2_]
        vreal x105 = stencil(gtDD00, stencil_idx_0_0_2_VVV); // x105: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__2_]
        vreal x409 = ((x104 + x105)); // x409: Dependency! Liveness = 2; [x104, x105]
        vreal x106 = stencil(gtDD00, stencil_idx_0_0_m1_VVV); // x106: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m1_]
        vreal x107 = stencil(gtDD00, stencil_idx_0_0_1_VVV); // x107: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__1_]
        vreal x410 = ((x106 + x107)); // x410: Dependency! Liveness = 2; [x106, x107]
        vreal x401 = (70 * stencil(gtDD00, stencil_idx_0_0_0_VVV)); // x401: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__0_]
        vreal x296 = ((1.0 / 256.0) * dissipation_epsilon); // x296: Dependency! Liveness = 1; [dissipation_epsilon]
        store(gt_rhsDD00, stencil_idx_0_0_0_VVV, (access(gt_rhsDD00, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x401 + (-56 * x404) + (-8 * x402) + (28 * x403) + ((stencil(gtDD00, stencil_idx_m4_0_0_VVV) + stencil(gtDD00, stencil_idx_4_0_0_VVV))))) + (DYI * (x401 + (-56 * x407) + (-8 * x405) + (28 * x406) + ((stencil(gtDD00, stencil_idx_0_m4_0_VVV) + stencil(gtDD00, stencil_idx_0_4_0_VVV))))) + (DZI * (x401 + (-56 * x410) + (-8 * x408) + (28 * x409) + ((stencil(gtDD00, stencil_idx_0_0_m4_VVV) + stencil(gtDD00, stencil_idx_0_0_4_VVV)))))))));
        vreal x217 = stencil(gtDD01, stencil_idx_3_0_0_VVV); // x217: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__3__0__0_]
        vreal x218 = stencil(gtDD01, stencil_idx_m3_0_0_VVV); // x218: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m3__0__0_]
        vreal x412 = ((x217 + x218)); // x412: Dependency! Liveness = 2; [x217, x218]
        vreal x219 = stencil(gtDD01, stencil_idx_m2_0_0_VVV); // x219: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m2__0__0_]
        vreal x220 = stencil(gtDD01, stencil_idx_2_0_0_VVV); // x220: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__2__0__0_]
        vreal x413 = ((x219 + x220)); // x413: Dependency! Liveness = 2; [x219, x220]
        vreal x221 = stencil(gtDD01, stencil_idx_m1_0_0_VVV); // x221: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m1__0__0_]
        vreal x222 = stencil(gtDD01, stencil_idx_1_0_0_VVV); // x222: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__1__0__0_]
        vreal x414 = ((x221 + x222)); // x414: Dependency! Liveness = 2; [x221, x222]
        vreal x180 = stencil(gtDD01, stencil_idx_0_3_0_VVV); // x180: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__3__0_]
        vreal x181 = stencil(gtDD01, stencil_idx_0_m3_0_VVV); // x181: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m3__0_]
        vreal x415 = ((x180 + x181)); // x415: Dependency! Liveness = 2; [x180, x181]
        vreal x182 = stencil(gtDD01, stencil_idx_0_m2_0_VVV); // x182: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m2__0_]
        vreal x183 = stencil(gtDD01, stencil_idx_0_2_0_VVV); // x183: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__2__0_]
        vreal x416 = ((x182 + x183)); // x416: Dependency! Liveness = 2; [x182, x183]
        vreal x184 = stencil(gtDD01, stencil_idx_0_m1_0_VVV); // x184: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m1__0_]
        vreal x185 = stencil(gtDD01, stencil_idx_0_1_0_VVV); // x185: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__1__0_]
        vreal x417 = ((x184 + x185)); // x417: Dependency! Liveness = 2; [x184, x185]
        vreal x71 = stencil(gtDD01, stencil_idx_0_0_3_VVV); // x71: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__3_]
        vreal x72 = stencil(gtDD01, stencil_idx_0_0_m3_VVV); // x72: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m3_]
        vreal x418 = ((x71 + x72)); // x418: Dependency! Liveness = 2; [x71, x72]
        vreal x73 = stencil(gtDD01, stencil_idx_0_0_m2_VVV); // x73: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m2_]
        vreal x74 = stencil(gtDD01, stencil_idx_0_0_2_VVV); // x74: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__2_]
        vreal x419 = ((x73 + x74)); // x419: Dependency! Liveness = 2; [x73, x74]
        vreal x75 = stencil(gtDD01, stencil_idx_0_0_m1_VVV); // x75: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m1_]
        vreal x76 = stencil(gtDD01, stencil_idx_0_0_1_VVV); // x76: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__1_]
        vreal x420 = ((x75 + x76)); // x420: Dependency! Liveness = 2; [x75, x76]
        vreal x411 = (70 * stencil(gtDD01, stencil_idx_0_0_0_VVV)); // x411: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__0_]
        store(gt_rhsDD01, stencil_idx_0_0_0_VVV, (access(gt_rhsDD01, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x411 + (-56 * x414) + (-8 * x412) + (28 * x413) + ((stencil(gtDD01, stencil_idx_m4_0_0_VVV) + stencil(gtDD01, stencil_idx_4_0_0_VVV))))) + (DYI * (x411 + (-56 * x417) + (-8 * x415) + (28 * x416) + ((stencil(gtDD01, stencil_idx_0_m4_0_VVV) + stencil(gtDD01, stencil_idx_0_4_0_VVV))))) + (DZI * (x411 + (-56 * x420) + (-8 * x418) + (28 * x419) + ((stencil(gtDD01, stencil_idx_0_0_m4_VVV) + stencil(gtDD01, stencil_idx_0_0_4_VVV)))))))));
        vreal x125 = stencil(gtDD11, stencil_idx_3_0_0_VVV); // x125: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__3__0__0_]
        vreal x126 = stencil(gtDD11, stencil_idx_m3_0_0_VVV); // x126: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m3__0__0_]
        vreal x422 = ((x125 + x126)); // x422: Dependency! Liveness = 2; [x125, x126]
        vreal x127 = stencil(gtDD11, stencil_idx_m2_0_0_VVV); // x127: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m2__0__0_]
        vreal x128 = stencil(gtDD11, stencil_idx_2_0_0_VVV); // x128: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__2__0__0_]
        vreal x423 = ((x127 + x128)); // x423: Dependency! Liveness = 2; [x127, x128]
        vreal x129 = stencil(gtDD11, stencil_idx_m1_0_0_VVV); // x129: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m1__0__0_]
        vreal x130 = stencil(gtDD11, stencil_idx_1_0_0_VVV); // x130: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__1__0__0_]
        vreal x424 = ((x129 + x130)); // x424: Dependency! Liveness = 2; [x129, x130]
        vreal x171 = stencil(gtDD11, stencil_idx_0_3_0_VVV); // x171: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__3__0_]
        vreal x172 = stencil(gtDD11, stencil_idx_0_m3_0_VVV); // x172: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m3__0_]
        vreal x425 = ((x171 + x172)); // x425: Dependency! Liveness = 2; [x171, x172]
        vreal x173 = stencil(gtDD11, stencil_idx_0_m2_0_VVV); // x173: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m2__0_]
        vreal x174 = stencil(gtDD11, stencil_idx_0_2_0_VVV); // x174: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__2__0_]
        vreal x426 = ((x173 + x174)); // x426: Dependency! Liveness = 2; [x173, x174]
        vreal x175 = stencil(gtDD11, stencil_idx_0_m1_0_VVV); // x175: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m1__0_]
        vreal x176 = stencil(gtDD11, stencil_idx_0_1_0_VVV); // x176: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__1__0_]
        vreal x427 = ((x175 + x176)); // x427: Dependency! Liveness = 2; [x175, x176]
        vreal x52 = stencil(gtDD11, stencil_idx_0_0_3_VVV); // x52: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__3_]
        vreal x53 = stencil(gtDD11, stencil_idx_0_0_m3_VVV); // x53: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m3_]
        vreal x428 = ((x52 + x53)); // x428: Dependency! Liveness = 2; [x52, x53]
        vreal x54 = stencil(gtDD11, stencil_idx_0_0_m2_VVV); // x54: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m2_]
        vreal x55 = stencil(gtDD11, stencil_idx_0_0_2_VVV); // x55: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__2_]
        vreal x429 = ((x54 + x55)); // x429: Dependency! Liveness = 2; [x54, x55]
        vreal x56 = stencil(gtDD11, stencil_idx_0_0_m1_VVV); // x56: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m1_]
        vreal x57 = stencil(gtDD11, stencil_idx_0_0_1_VVV); // x57: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__1_]
        vreal x430 = ((x56 + x57)); // x430: Dependency! Liveness = 2; [x56, x57]
        vreal x421 = (70 * stencil(gtDD11, stencil_idx_0_0_0_VVV)); // x421: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__0_]
        store(gt_rhsDD11, stencil_idx_0_0_0_VVV, (access(gt_rhsDD11, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x421 + (-56 * x424) + (-8 * x422) + (28 * x423) + ((stencil(gtDD11, stencil_idx_m4_0_0_VVV) + stencil(gtDD11, stencil_idx_4_0_0_VVV))))) + (DYI * (x421 + (-56 * x427) + (-8 * x425) + (28 * x426) + ((stencil(gtDD11, stencil_idx_0_m4_0_VVV) + stencil(gtDD11, stencil_idx_0_4_0_VVV))))) + (DZI * (x421 + (-56 * x430) + (-8 * x428) + (28 * x429) + ((stencil(gtDD11, stencil_idx_0_0_m4_VVV) + stencil(gtDD11, stencil_idx_0_0_4_VVV)))))))));
        vreal x207 = stencil(gtDD02, stencil_idx_3_0_0_VVV); // x207: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__3__0__0_]
        vreal x208 = stencil(gtDD02, stencil_idx_m3_0_0_VVV); // x208: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m3__0__0_]
        vreal x432 = ((x207 + x208)); // x432: Dependency! Liveness = 2; [x207, x208]
        vreal x209 = stencil(gtDD02, stencil_idx_m2_0_0_VVV); // x209: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m2__0__0_]
        vreal x210 = stencil(gtDD02, stencil_idx_2_0_0_VVV); // x210: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__2__0__0_]
        vreal x433 = ((x209 + x210)); // x433: Dependency! Liveness = 2; [x209, x210]
        vreal x211 = stencil(gtDD02, stencil_idx_m1_0_0_VVV); // x211: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m1__0__0_]
        vreal x212 = stencil(gtDD02, stencil_idx_1_0_0_VVV); // x212: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__1__0__0_]
        vreal x434 = ((x211 + x212)); // x434: Dependency! Liveness = 2; [x211, x212]
        vreal x63 = stencil(gtDD02, stencil_idx_0_3_0_VVV); // x63: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__3__0_]
        vreal x64 = stencil(gtDD02, stencil_idx_0_m3_0_VVV); // x64: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m3__0_]
        vreal x435 = ((x63 + x64)); // x435: Dependency! Liveness = 2; [x63, x64]
        vreal x65 = stencil(gtDD02, stencil_idx_0_m2_0_VVV); // x65: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m2__0_]
        vreal x66 = stencil(gtDD02, stencil_idx_0_2_0_VVV); // x66: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__2__0_]
        vreal x436 = ((x65 + x66)); // x436: Dependency! Liveness = 2; [x65, x66]
        vreal x67 = stencil(gtDD02, stencil_idx_0_m1_0_VVV); // x67: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m1__0_]
        vreal x68 = stencil(gtDD02, stencil_idx_0_1_0_VVV); // x68: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__1__0_]
        vreal x437 = ((x67 + x68)); // x437: Dependency! Liveness = 2; [x67, x68]
        vreal x159 = stencil(gtDD02, stencil_idx_0_0_3_VVV); // x159: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__3_]
        vreal x160 = stencil(gtDD02, stencil_idx_0_0_m3_VVV); // x160: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m3_]
        vreal x438 = ((x159 + x160)); // x438: Dependency! Liveness = 2; [x159, x160]
        vreal x161 = stencil(gtDD02, stencil_idx_0_0_m2_VVV); // x161: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m2_]
        vreal x162 = stencil(gtDD02, stencil_idx_0_0_2_VVV); // x162: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__2_]
        vreal x439 = ((x161 + x162)); // x439: Dependency! Liveness = 2; [x161, x162]
        vreal x163 = stencil(gtDD02, stencil_idx_0_0_m1_VVV); // x163: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m1_]
        vreal x164 = stencil(gtDD02, stencil_idx_0_0_1_VVV); // x164: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__1_]
        vreal x440 = ((x163 + x164)); // x440: Dependency! Liveness = 2; [x163, x164]
        vreal x431 = (70 * stencil(gtDD02, stencil_idx_0_0_0_VVV)); // x431: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__0_]
        store(gt_rhsDD02, stencil_idx_0_0_0_VVV, (access(gt_rhsDD02, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x431 + (-56 * x434) + (-8 * x432) + (28 * x433) + ((stencil(gtDD02, stencil_idx_m4_0_0_VVV) + stencil(gtDD02, stencil_idx_4_0_0_VVV))))) + (DYI * (x431 + (-56 * x437) + (-8 * x435) + (28 * x436) + ((stencil(gtDD02, stencil_idx_0_m4_0_VVV) + stencil(gtDD02, stencil_idx_0_4_0_VVV))))) + (DZI * (x431 + (-56 * x440) + (-8 * x438) + (28 * x439) + ((stencil(gtDD02, stencil_idx_0_0_m4_VVV) + stencil(gtDD02, stencil_idx_0_0_4_VVV)))))))));
        vreal x79 = stencil(gtDD12, stencil_idx_3_0_0_VVV); // x79: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__3__0__0_]
        vreal x80 = stencil(gtDD12, stencil_idx_m3_0_0_VVV); // x80: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m3__0__0_]
        vreal x442 = ((x79 + x80)); // x442: Dependency! Liveness = 2; [x79, x80]
        vreal x81 = stencil(gtDD12, stencil_idx_m2_0_0_VVV); // x81: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m2__0__0_]
        vreal x82 = stencil(gtDD12, stencil_idx_2_0_0_VVV); // x82: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__2__0__0_]
        vreal x443 = ((x81 + x82)); // x443: Dependency! Liveness = 2; [x81, x82]
        vreal x83 = stencil(gtDD12, stencil_idx_m1_0_0_VVV); // x83: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m1__0__0_]
        vreal x84 = stencil(gtDD12, stencil_idx_1_0_0_VVV); // x84: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__1__0__0_]
        vreal x444 = ((x83 + x84)); // x444: Dependency! Liveness = 2; [x83, x84]
        vreal x189 = stencil(gtDD12, stencil_idx_0_3_0_VVV); // x189: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__3__0_]
        vreal x190 = stencil(gtDD12, stencil_idx_0_m3_0_VVV); // x190: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m3__0_]
        vreal x445 = ((x189 + x190)); // x445: Dependency! Liveness = 2; [x189, x190]
        vreal x191 = stencil(gtDD12, stencil_idx_0_m2_0_VVV); // x191: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m2__0_]
        vreal x192 = stencil(gtDD12, stencil_idx_0_2_0_VVV); // x192: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__2__0_]
        vreal x446 = ((x191 + x192)); // x446: Dependency! Liveness = 2; [x191, x192]
        vreal x193 = stencil(gtDD12, stencil_idx_0_m1_0_VVV); // x193: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m1__0_]
        vreal x194 = stencil(gtDD12, stencil_idx_0_1_0_VVV); // x194: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__1__0_]
        vreal x447 = ((x193 + x194)); // x447: Dependency! Liveness = 2; [x193, x194]
        vreal x150 = stencil(gtDD12, stencil_idx_0_0_3_VVV); // x150: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__3_]
        vreal x151 = stencil(gtDD12, stencil_idx_0_0_m3_VVV); // x151: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m3_]
        vreal x448 = ((x150 + x151)); // x448: Dependency! Liveness = 2; [x150, x151]
        vreal x152 = stencil(gtDD12, stencil_idx_0_0_m2_VVV); // x152: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m2_]
        vreal x153 = stencil(gtDD12, stencil_idx_0_0_2_VVV); // x153: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__2_]
        vreal x449 = ((x152 + x153)); // x449: Dependency! Liveness = 2; [x152, x153]
        vreal x154 = stencil(gtDD12, stencil_idx_0_0_m1_VVV); // x154: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m1_]
        vreal x155 = stencil(gtDD12, stencil_idx_0_0_1_VVV); // x155: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__1_]
        vreal x450 = ((x154 + x155)); // x450: Dependency! Liveness = 2; [x154, x155]
        vreal x441 = (70 * stencil(gtDD12, stencil_idx_0_0_0_VVV)); // x441: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__0_]
        store(gt_rhsDD12, stencil_idx_0_0_0_VVV, (access(gt_rhsDD12, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x441 + (-56 * x444) + (-8 * x442) + (28 * x443) + ((stencil(gtDD12, stencil_idx_m4_0_0_VVV) + stencil(gtDD12, stencil_idx_4_0_0_VVV))))) + (DYI * (x441 + (-56 * x447) + (-8 * x445) + (28 * x446) + ((stencil(gtDD12, stencil_idx_0_m4_0_VVV) + stencil(gtDD12, stencil_idx_0_4_0_VVV))))) + (DZI * (x441 + (-56 * x450) + (-8 * x448) + (28 * x449) + ((stencil(gtDD12, stencil_idx_0_0_m4_VVV) + stencil(gtDD12, stencil_idx_0_0_4_VVV)))))))));
        vreal x93 = stencil(gtDD22, stencil_idx_3_0_0_VVV); // x93: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__3__0__0_]
        vreal x94 = stencil(gtDD22, stencil_idx_m3_0_0_VVV); // x94: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m3__0__0_]
        vreal x452 = ((x93 + x94)); // x452: Dependency! Liveness = 2; [x93, x94]
        vreal x95 = stencil(gtDD22, stencil_idx_m2_0_0_VVV); // x95: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m2__0__0_]
        vreal x96 = stencil(gtDD22, stencil_idx_2_0_0_VVV); // x96: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__2__0__0_]
        vreal x453 = ((x95 + x96)); // x453: Dependency! Liveness = 2; [x95, x96]
        vreal x97 = stencil(gtDD22, stencil_idx_m1_0_0_VVV); // x97: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m1__0__0_]
        vreal x98 = stencil(gtDD22, stencil_idx_1_0_0_VVV); // x98: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__1__0__0_]
        vreal x454 = ((x97 + x98)); // x454: Dependency! Liveness = 2; [x97, x98]
        vreal x41 = stencil(gtDD22, stencil_idx_0_3_0_VVV); // x41: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__3__0_]
        vreal x42 = stencil(gtDD22, stencil_idx_0_m3_0_VVV); // x42: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m3__0_]
        vreal x455 = ((x41 + x42)); // x455: Dependency! Liveness = 2; [x41, x42]
        vreal x43 = stencil(gtDD22, stencil_idx_0_m2_0_VVV); // x43: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m2__0_]
        vreal x44 = stencil(gtDD22, stencil_idx_0_2_0_VVV); // x44: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__2__0_]
        vreal x456 = ((x43 + x44)); // x456: Dependency! Liveness = 2; [x43, x44]
        vreal x45 = stencil(gtDD22, stencil_idx_0_m1_0_VVV); // x45: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m1__0_]
        vreal x46 = stencil(gtDD22, stencil_idx_0_1_0_VVV); // x46: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__1__0_]
        vreal x457 = ((x45 + x46)); // x457: Dependency! Liveness = 2; [x45, x46]
        vreal x141 = stencil(gtDD22, stencil_idx_0_0_3_VVV); // x141: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__3_]
        vreal x142 = stencil(gtDD22, stencil_idx_0_0_m3_VVV); // x142: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m3_]
        vreal x458 = ((x141 + x142)); // x458: Dependency! Liveness = 2; [x141, x142]
        vreal x143 = stencil(gtDD22, stencil_idx_0_0_m2_VVV); // x143: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m2_]
        vreal x144 = stencil(gtDD22, stencil_idx_0_0_2_VVV); // x144: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__2_]
        vreal x459 = ((x143 + x144)); // x459: Dependency! Liveness = 2; [x143, x144]
        vreal x145 = stencil(gtDD22, stencil_idx_0_0_m1_VVV); // x145: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m1_]
        vreal x146 = stencil(gtDD22, stencil_idx_0_0_1_VVV); // x146: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__1_]
        vreal x460 = ((x145 + x146)); // x460: Dependency! Liveness = 2; [x145, x146]
        vreal x451 = (70 * stencil(gtDD22, stencil_idx_0_0_0_VVV)); // x451: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__0_]
        store(gt_rhsDD22, stencil_idx_0_0_0_VVV, (access(gt_rhsDD22, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x451 + (-56 * x454) + (-8 * x452) + (28 * x453) + ((stencil(gtDD22, stencil_idx_m4_0_0_VVV) + stencil(gtDD22, stencil_idx_4_0_0_VVV))))) + (DYI * (x451 + (-56 * x457) + (-8 * x455) + (28 * x456) + ((stencil(gtDD22, stencil_idx_0_m4_0_VVV) + stencil(gtDD22, stencil_idx_0_4_0_VVV))))) + (DZI * (x451 + (-56 * x460) + (-8 * x458) + (28 * x459) + ((stencil(gtDD22, stencil_idx_0_0_m4_VVV) + stencil(gtDD22, stencil_idx_0_0_4_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 5
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
        vreal x468 = stencil(AtDD00, stencil_idx_0_m3_0_VVV); // x468: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m3__0_]
        vreal x469 = stencil(AtDD00, stencil_idx_0_3_0_VVV); // x469: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__3__0_]
        vreal x470 = stencil(AtDD00, stencil_idx_0_m2_0_VVV); // x470: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m2__0_]
        vreal x471 = stencil(AtDD00, stencil_idx_0_2_0_VVV); // x471: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__2__0_]
        vreal x472 = stencil(AtDD00, stencil_idx_0_m1_0_VVV); // x472: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m1__0_]
        vreal x473 = stencil(AtDD00, stencil_idx_0_1_0_VVV); // x473: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__1__0_]
        vreal x474 = stencil(AtDD00, stencil_idx_0_0_m3_VVV); // x474: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m3_]
        vreal x475 = stencil(AtDD00, stencil_idx_0_0_3_VVV); // x475: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__3_]
        vreal x476 = stencil(AtDD00, stencil_idx_0_0_m2_VVV); // x476: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m2_]
        vreal x477 = stencil(AtDD00, stencil_idx_0_0_2_VVV); // x477: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__2_]
        vreal x478 = stencil(AtDD00, stencil_idx_0_0_m1_VVV); // x478: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m1_]
        vreal x479 = stencil(AtDD00, stencil_idx_0_0_1_VVV); // x479: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__1_]
        vreal x462 = stencil(AtDD00, stencil_idx_m3_0_0_VVV); // x462: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m3__0__0_]
        vreal x463 = stencil(AtDD00, stencil_idx_3_0_0_VVV); // x463: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__3__0__0_]
        vreal x464 = stencil(AtDD00, stencil_idx_m2_0_0_VVV); // x464: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m2__0__0_]
        vreal x465 = stencil(AtDD00, stencil_idx_2_0_0_VVV); // x465: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__2__0__0_]
        vreal x466 = stencil(AtDD00, stencil_idx_m1_0_0_VVV); // x466: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m1__0__0_]
        vreal x467 = stencil(AtDD00, stencil_idx_1_0_0_VVV); // x467: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__1__0__0_]
        vreal x461 = (70 * stencil(AtDD00, stencil_idx_0_0_0_VVV)); // x461: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__0_]
        vreal x296 = ((1.0 / 256.0) * dissipation_epsilon); // x296: Dependency! Liveness = 1; [dissipation_epsilon]
        store(At_rhsDD00, stencil_idx_0_0_0_VVV, (access(At_rhsDD00, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x461 + (-56 * ((x466 + x467))) + (-8 * ((x462 + x463))) + (28 * ((x464 + x465))) + ((stencil(AtDD00, stencil_idx_m4_0_0_VVV) + stencil(AtDD00, stencil_idx_4_0_0_VVV))))) + (DYI * (x461 + (-56 * ((x472 + x473))) + (-8 * ((x468 + x469))) + (28 * ((x470 + x471))) + ((stencil(AtDD00, stencil_idx_0_m4_0_VVV) + stencil(AtDD00, stencil_idx_0_4_0_VVV))))) + (DZI * (x461 + (-56 * ((x478 + x479))) + (-8 * ((x474 + x475))) + (28 * ((x476 + x477))) + ((stencil(AtDD00, stencil_idx_0_0_m4_VVV) + stencil(AtDD00, stencil_idx_0_0_4_VVV)))))))));
        vreal x487 = stencil(AtDD01, stencil_idx_0_m3_0_VVV); // x487: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m3__0_]
        vreal x488 = stencil(AtDD01, stencil_idx_0_3_0_VVV); // x488: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__3__0_]
        vreal x489 = stencil(AtDD01, stencil_idx_0_m2_0_VVV); // x489: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m2__0_]
        vreal x490 = stencil(AtDD01, stencil_idx_0_2_0_VVV); // x490: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__2__0_]
        vreal x491 = stencil(AtDD01, stencil_idx_0_m1_0_VVV); // x491: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m1__0_]
        vreal x492 = stencil(AtDD01, stencil_idx_0_1_0_VVV); // x492: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__1__0_]
        vreal x493 = stencil(AtDD01, stencil_idx_0_0_m3_VVV); // x493: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m3_]
        vreal x494 = stencil(AtDD01, stencil_idx_0_0_3_VVV); // x494: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__3_]
        vreal x495 = stencil(AtDD01, stencil_idx_0_0_m2_VVV); // x495: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m2_]
        vreal x496 = stencil(AtDD01, stencil_idx_0_0_2_VVV); // x496: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__2_]
        vreal x497 = stencil(AtDD01, stencil_idx_0_0_m1_VVV); // x497: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m1_]
        vreal x498 = stencil(AtDD01, stencil_idx_0_0_1_VVV); // x498: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__1_]
        vreal x481 = stencil(AtDD01, stencil_idx_m3_0_0_VVV); // x481: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m3__0__0_]
        vreal x482 = stencil(AtDD01, stencil_idx_3_0_0_VVV); // x482: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__3__0__0_]
        vreal x483 = stencil(AtDD01, stencil_idx_m2_0_0_VVV); // x483: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m2__0__0_]
        vreal x484 = stencil(AtDD01, stencil_idx_2_0_0_VVV); // x484: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__2__0__0_]
        vreal x485 = stencil(AtDD01, stencil_idx_m1_0_0_VVV); // x485: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m1__0__0_]
        vreal x486 = stencil(AtDD01, stencil_idx_1_0_0_VVV); // x486: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__1__0__0_]
        vreal x480 = (70 * stencil(AtDD01, stencil_idx_0_0_0_VVV)); // x480: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__0_]
        store(At_rhsDD01, stencil_idx_0_0_0_VVV, (access(At_rhsDD01, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x480 + (-56 * ((x485 + x486))) + (-8 * ((x481 + x482))) + (28 * ((x483 + x484))) + ((stencil(AtDD01, stencil_idx_m4_0_0_VVV) + stencil(AtDD01, stencil_idx_4_0_0_VVV))))) + (DYI * (x480 + (-56 * ((x491 + x492))) + (-8 * ((x487 + x488))) + (28 * ((x489 + x490))) + ((stencil(AtDD01, stencil_idx_0_m4_0_VVV) + stencil(AtDD01, stencil_idx_0_4_0_VVV))))) + (DZI * (x480 + (-56 * ((x497 + x498))) + (-8 * ((x493 + x494))) + (28 * ((x495 + x496))) + ((stencil(AtDD01, stencil_idx_0_0_m4_VVV) + stencil(AtDD01, stencil_idx_0_0_4_VVV)))))))));
        vreal x506 = stencil(AtDD11, stencil_idx_0_m3_0_VVV); // x506: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m3__0_]
        vreal x507 = stencil(AtDD11, stencil_idx_0_3_0_VVV); // x507: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__3__0_]
        vreal x508 = stencil(AtDD11, stencil_idx_0_m2_0_VVV); // x508: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m2__0_]
        vreal x509 = stencil(AtDD11, stencil_idx_0_2_0_VVV); // x509: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__2__0_]
        vreal x510 = stencil(AtDD11, stencil_idx_0_m1_0_VVV); // x510: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m1__0_]
        vreal x511 = stencil(AtDD11, stencil_idx_0_1_0_VVV); // x511: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__1__0_]
        vreal x512 = stencil(AtDD11, stencil_idx_0_0_m3_VVV); // x512: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m3_]
        vreal x513 = stencil(AtDD11, stencil_idx_0_0_3_VVV); // x513: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__3_]
        vreal x514 = stencil(AtDD11, stencil_idx_0_0_m2_VVV); // x514: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m2_]
        vreal x515 = stencil(AtDD11, stencil_idx_0_0_2_VVV); // x515: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__2_]
        vreal x516 = stencil(AtDD11, stencil_idx_0_0_m1_VVV); // x516: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m1_]
        vreal x517 = stencil(AtDD11, stencil_idx_0_0_1_VVV); // x517: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__1_]
        vreal x500 = stencil(AtDD11, stencil_idx_m3_0_0_VVV); // x500: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m3__0__0_]
        vreal x501 = stencil(AtDD11, stencil_idx_3_0_0_VVV); // x501: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__3__0__0_]
        vreal x502 = stencil(AtDD11, stencil_idx_m2_0_0_VVV); // x502: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m2__0__0_]
        vreal x503 = stencil(AtDD11, stencil_idx_2_0_0_VVV); // x503: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__2__0__0_]
        vreal x504 = stencil(AtDD11, stencil_idx_m1_0_0_VVV); // x504: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m1__0__0_]
        vreal x505 = stencil(AtDD11, stencil_idx_1_0_0_VVV); // x505: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__1__0__0_]
        vreal x499 = (70 * stencil(AtDD11, stencil_idx_0_0_0_VVV)); // x499: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__0_]
        store(At_rhsDD11, stencil_idx_0_0_0_VVV, (access(At_rhsDD11, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x499 + (-56 * ((x504 + x505))) + (-8 * ((x500 + x501))) + (28 * ((x502 + x503))) + ((stencil(AtDD11, stencil_idx_m4_0_0_VVV) + stencil(AtDD11, stencil_idx_4_0_0_VVV))))) + (DYI * (x499 + (-56 * ((x510 + x511))) + (-8 * ((x506 + x507))) + (28 * ((x508 + x509))) + ((stencil(AtDD11, stencil_idx_0_m4_0_VVV) + stencil(AtDD11, stencil_idx_0_4_0_VVV))))) + (DZI * (x499 + (-56 * ((x516 + x517))) + (-8 * ((x512 + x513))) + (28 * ((x514 + x515))) + ((stencil(AtDD11, stencil_idx_0_0_m4_VVV) + stencil(AtDD11, stencil_idx_0_0_4_VVV)))))))));
        vreal x525 = stencil(AtDD02, stencil_idx_0_m3_0_VVV); // x525: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m3__0_]
        vreal x526 = stencil(AtDD02, stencil_idx_0_3_0_VVV); // x526: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__3__0_]
        vreal x527 = stencil(AtDD02, stencil_idx_0_m2_0_VVV); // x527: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m2__0_]
        vreal x528 = stencil(AtDD02, stencil_idx_0_2_0_VVV); // x528: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__2__0_]
        vreal x529 = stencil(AtDD02, stencil_idx_0_m1_0_VVV); // x529: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m1__0_]
        vreal x530 = stencil(AtDD02, stencil_idx_0_1_0_VVV); // x530: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__1__0_]
        vreal x531 = stencil(AtDD02, stencil_idx_0_0_m3_VVV); // x531: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m3_]
        vreal x532 = stencil(AtDD02, stencil_idx_0_0_3_VVV); // x532: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__3_]
        vreal x533 = stencil(AtDD02, stencil_idx_0_0_m2_VVV); // x533: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m2_]
        vreal x534 = stencil(AtDD02, stencil_idx_0_0_2_VVV); // x534: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__2_]
        vreal x535 = stencil(AtDD02, stencil_idx_0_0_m1_VVV); // x535: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m1_]
        vreal x536 = stencil(AtDD02, stencil_idx_0_0_1_VVV); // x536: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__1_]
        vreal x519 = stencil(AtDD02, stencil_idx_m3_0_0_VVV); // x519: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m3__0__0_]
        vreal x520 = stencil(AtDD02, stencil_idx_3_0_0_VVV); // x520: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__3__0__0_]
        vreal x521 = stencil(AtDD02, stencil_idx_m2_0_0_VVV); // x521: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m2__0__0_]
        vreal x522 = stencil(AtDD02, stencil_idx_2_0_0_VVV); // x522: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__2__0__0_]
        vreal x523 = stencil(AtDD02, stencil_idx_m1_0_0_VVV); // x523: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m1__0__0_]
        vreal x524 = stencil(AtDD02, stencil_idx_1_0_0_VVV); // x524: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__1__0__0_]
        vreal x518 = (70 * stencil(AtDD02, stencil_idx_0_0_0_VVV)); // x518: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__0_]
        store(At_rhsDD02, stencil_idx_0_0_0_VVV, (access(At_rhsDD02, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x518 + (-56 * ((x523 + x524))) + (-8 * ((x519 + x520))) + (28 * ((x521 + x522))) + ((stencil(AtDD02, stencil_idx_m4_0_0_VVV) + stencil(AtDD02, stencil_idx_4_0_0_VVV))))) + (DYI * (x518 + (-56 * ((x529 + x530))) + (-8 * ((x525 + x526))) + (28 * ((x527 + x528))) + ((stencil(AtDD02, stencil_idx_0_m4_0_VVV) + stencil(AtDD02, stencil_idx_0_4_0_VVV))))) + (DZI * (x518 + (-56 * ((x535 + x536))) + (-8 * ((x531 + x532))) + (28 * ((x533 + x534))) + ((stencil(AtDD02, stencil_idx_0_0_m4_VVV) + stencil(AtDD02, stencil_idx_0_0_4_VVV)))))))));
        vreal x544 = stencil(AtDD12, stencil_idx_0_m3_0_VVV); // x544: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m3__0_]
        vreal x545 = stencil(AtDD12, stencil_idx_0_3_0_VVV); // x545: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__3__0_]
        vreal x546 = stencil(AtDD12, stencil_idx_0_m2_0_VVV); // x546: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m2__0_]
        vreal x547 = stencil(AtDD12, stencil_idx_0_2_0_VVV); // x547: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__2__0_]
        vreal x548 = stencil(AtDD12, stencil_idx_0_m1_0_VVV); // x548: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m1__0_]
        vreal x549 = stencil(AtDD12, stencil_idx_0_1_0_VVV); // x549: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__1__0_]
        vreal x550 = stencil(AtDD12, stencil_idx_0_0_m3_VVV); // x550: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m3_]
        vreal x551 = stencil(AtDD12, stencil_idx_0_0_3_VVV); // x551: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__3_]
        vreal x552 = stencil(AtDD12, stencil_idx_0_0_m2_VVV); // x552: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m2_]
        vreal x553 = stencil(AtDD12, stencil_idx_0_0_2_VVV); // x553: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__2_]
        vreal x554 = stencil(AtDD12, stencil_idx_0_0_m1_VVV); // x554: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m1_]
        vreal x555 = stencil(AtDD12, stencil_idx_0_0_1_VVV); // x555: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__1_]
        vreal x538 = stencil(AtDD12, stencil_idx_m3_0_0_VVV); // x538: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m3__0__0_]
        vreal x539 = stencil(AtDD12, stencil_idx_3_0_0_VVV); // x539: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__3__0__0_]
        vreal x540 = stencil(AtDD12, stencil_idx_m2_0_0_VVV); // x540: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m2__0__0_]
        vreal x541 = stencil(AtDD12, stencil_idx_2_0_0_VVV); // x541: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__2__0__0_]
        vreal x542 = stencil(AtDD12, stencil_idx_m1_0_0_VVV); // x542: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m1__0__0_]
        vreal x543 = stencil(AtDD12, stencil_idx_1_0_0_VVV); // x543: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__1__0__0_]
        vreal x537 = (70 * stencil(AtDD12, stencil_idx_0_0_0_VVV)); // x537: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__0_]
        store(At_rhsDD12, stencil_idx_0_0_0_VVV, (access(At_rhsDD12, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x537 + (-56 * ((x542 + x543))) + (-8 * ((x538 + x539))) + (28 * ((x540 + x541))) + ((stencil(AtDD12, stencil_idx_m4_0_0_VVV) + stencil(AtDD12, stencil_idx_4_0_0_VVV))))) + (DYI * (x537 + (-56 * ((x548 + x549))) + (-8 * ((x544 + x545))) + (28 * ((x546 + x547))) + ((stencil(AtDD12, stencil_idx_0_m4_0_VVV) + stencil(AtDD12, stencil_idx_0_4_0_VVV))))) + (DZI * (x537 + (-56 * ((x554 + x555))) + (-8 * ((x550 + x551))) + (28 * ((x552 + x553))) + ((stencil(AtDD12, stencil_idx_0_0_m4_VVV) + stencil(AtDD12, stencil_idx_0_0_4_VVV)))))))));
        vreal x563 = stencil(AtDD22, stencil_idx_0_m3_0_VVV); // x563: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m3__0_]
        vreal x564 = stencil(AtDD22, stencil_idx_0_3_0_VVV); // x564: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__3__0_]
        vreal x565 = stencil(AtDD22, stencil_idx_0_m2_0_VVV); // x565: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m2__0_]
        vreal x566 = stencil(AtDD22, stencil_idx_0_2_0_VVV); // x566: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__2__0_]
        vreal x567 = stencil(AtDD22, stencil_idx_0_m1_0_VVV); // x567: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m1__0_]
        vreal x568 = stencil(AtDD22, stencil_idx_0_1_0_VVV); // x568: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__1__0_]
        vreal x569 = stencil(AtDD22, stencil_idx_0_0_m3_VVV); // x569: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m3_]
        vreal x570 = stencil(AtDD22, stencil_idx_0_0_3_VVV); // x570: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__3_]
        vreal x571 = stencil(AtDD22, stencil_idx_0_0_m2_VVV); // x571: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m2_]
        vreal x572 = stencil(AtDD22, stencil_idx_0_0_2_VVV); // x572: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__2_]
        vreal x573 = stencil(AtDD22, stencil_idx_0_0_m1_VVV); // x573: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m1_]
        vreal x574 = stencil(AtDD22, stencil_idx_0_0_1_VVV); // x574: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__1_]
        vreal x557 = stencil(AtDD22, stencil_idx_m3_0_0_VVV); // x557: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m3__0__0_]
        vreal x558 = stencil(AtDD22, stencil_idx_3_0_0_VVV); // x558: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__3__0__0_]
        vreal x559 = stencil(AtDD22, stencil_idx_m2_0_0_VVV); // x559: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m2__0__0_]
        vreal x560 = stencil(AtDD22, stencil_idx_2_0_0_VVV); // x560: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__2__0__0_]
        vreal x561 = stencil(AtDD22, stencil_idx_m1_0_0_VVV); // x561: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m1__0__0_]
        vreal x562 = stencil(AtDD22, stencil_idx_1_0_0_VVV); // x562: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__1__0__0_]
        vreal x556 = (70 * stencil(AtDD22, stencil_idx_0_0_0_VVV)); // x556: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__0_]
        store(At_rhsDD22, stencil_idx_0_0_0_VVV, (access(At_rhsDD22, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x556 + (-56 * ((x561 + x562))) + (-8 * ((x557 + x558))) + (28 * ((x559 + x560))) + ((stencil(AtDD22, stencil_idx_m4_0_0_VVV) + stencil(AtDD22, stencil_idx_4_0_0_VVV))))) + (DYI * (x556 + (-56 * ((x567 + x568))) + (-8 * ((x563 + x564))) + (28 * ((x565 + x566))) + ((stencil(AtDD22, stencil_idx_0_m4_0_VVV) + stencil(AtDD22, stencil_idx_0_4_0_VVV))))) + (DZI * (x556 + (-56 * ((x573 + x574))) + (-8 * ((x569 + x570))) + (28 * ((x571 + x572))) + ((stencil(AtDD22, stencil_idx_0_0_m4_VVV) + stencil(AtDD22, stencil_idx_0_0_4_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 6
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
        vreal x576 = stencil(evo_lapse, stencil_idx_m3_0_0_VVV); // x576: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m3__0__0_]
        vreal x577 = stencil(evo_lapse, stencil_idx_3_0_0_VVV); // x577: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__3__0__0_]
        vreal x578 = ((x576 + x577)); // x578: Dependency! Liveness = 2; [x576, x577]
        vreal x579 = stencil(evo_lapse, stencil_idx_m2_0_0_VVV); // x579: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m2__0__0_]
        vreal x580 = stencil(evo_lapse, stencil_idx_2_0_0_VVV); // x580: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__2__0__0_]
        vreal x581 = ((x579 + x580)); // x581: Dependency! Liveness = 2; [x579, x580]
        vreal x582 = stencil(evo_lapse, stencil_idx_m1_0_0_VVV); // x582: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m1__0__0_]
        vreal x583 = stencil(evo_lapse, stencil_idx_1_0_0_VVV); // x583: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__1__0__0_]
        vreal x584 = ((x582 + x583)); // x584: Dependency! Liveness = 2; [x582, x583]
        vreal x585 = stencil(evo_lapse, stencil_idx_0_m3_0_VVV); // x585: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m3__0_]
        vreal x586 = stencil(evo_lapse, stencil_idx_0_3_0_VVV); // x586: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__3__0_]
        vreal x587 = ((x585 + x586)); // x587: Dependency! Liveness = 2; [x585, x586]
        vreal x588 = stencil(evo_lapse, stencil_idx_0_m2_0_VVV); // x588: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m2__0_]
        vreal x589 = stencil(evo_lapse, stencil_idx_0_2_0_VVV); // x589: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__2__0_]
        vreal x590 = ((x588 + x589)); // x590: Dependency! Liveness = 2; [x588, x589]
        vreal x591 = stencil(evo_lapse, stencil_idx_0_m1_0_VVV); // x591: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m1__0_]
        vreal x592 = stencil(evo_lapse, stencil_idx_0_1_0_VVV); // x592: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__1__0_]
        vreal x593 = ((x591 + x592)); // x593: Dependency! Liveness = 2; [x591, x592]
        vreal x594 = stencil(evo_lapse, stencil_idx_0_0_m3_VVV); // x594: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m3_]
        vreal x595 = stencil(evo_lapse, stencil_idx_0_0_3_VVV); // x595: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__3_]
        vreal x596 = ((x594 + x595)); // x596: Dependency! Liveness = 2; [x594, x595]
        vreal x597 = stencil(evo_lapse, stencil_idx_0_0_m2_VVV); // x597: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m2_]
        vreal x598 = stencil(evo_lapse, stencil_idx_0_0_2_VVV); // x598: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__2_]
        vreal x599 = ((x597 + x598)); // x599: Dependency! Liveness = 2; [x597, x598]
        vreal x600 = stencil(evo_lapse, stencil_idx_0_0_m1_VVV); // x600: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m1_]
        vreal x601 = stencil(evo_lapse, stencil_idx_0_0_1_VVV); // x601: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__1_]
        vreal x602 = ((x600 + x601)); // x602: Dependency! Liveness = 2; [x600, x601]
        vreal x575 = (70 * stencil(evo_lapse, stencil_idx_0_0_0_VVV)); // x575: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__0_]
        vreal x296 = ((1.0 / 256.0) * dissipation_epsilon); // x296: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_lapse_rhs, stencil_idx_0_0_0_VVV, (access(evo_lapse_rhs, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x575 + (-56 * x584) + (-8 * x578) + (28 * x581) + ((stencil(evo_lapse, stencil_idx_m4_0_0_VVV) + stencil(evo_lapse, stencil_idx_4_0_0_VVV))))) + (DYI * (x575 + (-56 * x593) + (-8 * x587) + (28 * x590) + ((stencil(evo_lapse, stencil_idx_0_m4_0_VVV) + stencil(evo_lapse, stencil_idx_0_4_0_VVV))))) + (DZI * (x575 + (-56 * x602) + (-8 * x596) + (28 * x599) + ((stencil(evo_lapse, stencil_idx_0_0_m4_VVV) + stencil(evo_lapse, stencil_idx_0_0_4_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 7
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
        vreal x604 = stencil(evo_shiftU0, stencil_idx_m3_0_0_VVV); // x604: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m3__0__0_]
        vreal x605 = stencil(evo_shiftU0, stencil_idx_3_0_0_VVV); // x605: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__3__0__0_]
        vreal x606 = ((x604 + x605)); // x606: Dependency! Liveness = 2; [x604, x605]
        vreal x607 = stencil(evo_shiftU0, stencil_idx_m2_0_0_VVV); // x607: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m2__0__0_]
        vreal x608 = stencil(evo_shiftU0, stencil_idx_2_0_0_VVV); // x608: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__2__0__0_]
        vreal x609 = ((x607 + x608)); // x609: Dependency! Liveness = 2; [x607, x608]
        vreal x610 = stencil(evo_shiftU0, stencil_idx_m1_0_0_VVV); // x610: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m1__0__0_]
        vreal x611 = stencil(evo_shiftU0, stencil_idx_1_0_0_VVV); // x611: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__1__0__0_]
        vreal x612 = ((x610 + x611)); // x612: Dependency! Liveness = 2; [x610, x611]
        vreal x613 = stencil(evo_shiftU0, stencil_idx_0_m3_0_VVV); // x613: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m3__0_]
        vreal x614 = stencil(evo_shiftU0, stencil_idx_0_3_0_VVV); // x614: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__3__0_]
        vreal x615 = ((x613 + x614)); // x615: Dependency! Liveness = 2; [x613, x614]
        vreal x616 = stencil(evo_shiftU0, stencil_idx_0_m2_0_VVV); // x616: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m2__0_]
        vreal x617 = stencil(evo_shiftU0, stencil_idx_0_2_0_VVV); // x617: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__2__0_]
        vreal x618 = ((x616 + x617)); // x618: Dependency! Liveness = 2; [x616, x617]
        vreal x619 = stencil(evo_shiftU0, stencil_idx_0_m1_0_VVV); // x619: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m1__0_]
        vreal x620 = stencil(evo_shiftU0, stencil_idx_0_1_0_VVV); // x620: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__1__0_]
        vreal x621 = ((x619 + x620)); // x621: Dependency! Liveness = 2; [x619, x620]
        vreal x622 = stencil(evo_shiftU0, stencil_idx_0_0_m3_VVV); // x622: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m3_]
        vreal x623 = stencil(evo_shiftU0, stencil_idx_0_0_3_VVV); // x623: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__3_]
        vreal x624 = ((x622 + x623)); // x624: Dependency! Liveness = 2; [x622, x623]
        vreal x625 = stencil(evo_shiftU0, stencil_idx_0_0_m2_VVV); // x625: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m2_]
        vreal x626 = stencil(evo_shiftU0, stencil_idx_0_0_2_VVV); // x626: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__2_]
        vreal x627 = ((x625 + x626)); // x627: Dependency! Liveness = 2; [x625, x626]
        vreal x628 = stencil(evo_shiftU0, stencil_idx_0_0_m1_VVV); // x628: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m1_]
        vreal x629 = stencil(evo_shiftU0, stencil_idx_0_0_1_VVV); // x629: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__1_]
        vreal x630 = ((x628 + x629)); // x630: Dependency! Liveness = 2; [x628, x629]
        vreal x603 = (70 * stencil(evo_shiftU0, stencil_idx_0_0_0_VVV)); // x603: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__0_]
        vreal x296 = ((1.0 / 256.0) * dissipation_epsilon); // x296: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_shift_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU0, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x603 + (-56 * x612) + (-8 * x606) + (28 * x609) + ((stencil(evo_shiftU0, stencil_idx_m4_0_0_VVV) + stencil(evo_shiftU0, stencil_idx_4_0_0_VVV))))) + (DYI * (x603 + (-56 * x621) + (-8 * x615) + (28 * x618) + ((stencil(evo_shiftU0, stencil_idx_0_m4_0_VVV) + stencil(evo_shiftU0, stencil_idx_0_4_0_VVV))))) + (DZI * (x603 + (-56 * x630) + (-8 * x624) + (28 * x627) + ((stencil(evo_shiftU0, stencil_idx_0_0_m4_VVV) + stencil(evo_shiftU0, stencil_idx_0_0_4_VVV)))))))));
        vreal x632 = stencil(evo_shiftU1, stencil_idx_m3_0_0_VVV); // x632: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m3__0__0_]
        vreal x633 = stencil(evo_shiftU1, stencil_idx_3_0_0_VVV); // x633: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__3__0__0_]
        vreal x634 = ((x632 + x633)); // x634: Dependency! Liveness = 2; [x632, x633]
        vreal x635 = stencil(evo_shiftU1, stencil_idx_m2_0_0_VVV); // x635: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m2__0__0_]
        vreal x636 = stencil(evo_shiftU1, stencil_idx_2_0_0_VVV); // x636: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__2__0__0_]
        vreal x637 = ((x635 + x636)); // x637: Dependency! Liveness = 2; [x635, x636]
        vreal x638 = stencil(evo_shiftU1, stencil_idx_m1_0_0_VVV); // x638: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m1__0__0_]
        vreal x639 = stencil(evo_shiftU1, stencil_idx_1_0_0_VVV); // x639: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__1__0__0_]
        vreal x640 = ((x638 + x639)); // x640: Dependency! Liveness = 2; [x638, x639]
        vreal x641 = stencil(evo_shiftU1, stencil_idx_0_m3_0_VVV); // x641: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m3__0_]
        vreal x642 = stencil(evo_shiftU1, stencil_idx_0_3_0_VVV); // x642: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__3__0_]
        vreal x643 = ((x641 + x642)); // x643: Dependency! Liveness = 2; [x641, x642]
        vreal x644 = stencil(evo_shiftU1, stencil_idx_0_m2_0_VVV); // x644: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m2__0_]
        vreal x645 = stencil(evo_shiftU1, stencil_idx_0_2_0_VVV); // x645: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__2__0_]
        vreal x646 = ((x644 + x645)); // x646: Dependency! Liveness = 2; [x644, x645]
        vreal x647 = stencil(evo_shiftU1, stencil_idx_0_m1_0_VVV); // x647: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m1__0_]
        vreal x648 = stencil(evo_shiftU1, stencil_idx_0_1_0_VVV); // x648: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__1__0_]
        vreal x649 = ((x647 + x648)); // x649: Dependency! Liveness = 2; [x647, x648]
        vreal x650 = stencil(evo_shiftU1, stencil_idx_0_0_m3_VVV); // x650: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m3_]
        vreal x651 = stencil(evo_shiftU1, stencil_idx_0_0_3_VVV); // x651: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__3_]
        vreal x652 = ((x650 + x651)); // x652: Dependency! Liveness = 2; [x650, x651]
        vreal x653 = stencil(evo_shiftU1, stencil_idx_0_0_m2_VVV); // x653: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m2_]
        vreal x654 = stencil(evo_shiftU1, stencil_idx_0_0_2_VVV); // x654: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__2_]
        vreal x655 = ((x653 + x654)); // x655: Dependency! Liveness = 2; [x653, x654]
        vreal x656 = stencil(evo_shiftU1, stencil_idx_0_0_m1_VVV); // x656: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m1_]
        vreal x657 = stencil(evo_shiftU1, stencil_idx_0_0_1_VVV); // x657: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__1_]
        vreal x658 = ((x656 + x657)); // x658: Dependency! Liveness = 2; [x656, x657]
        vreal x631 = (70 * stencil(evo_shiftU1, stencil_idx_0_0_0_VVV)); // x631: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__0_]
        store(evo_shift_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU1, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x631 + (-56 * x640) + (-8 * x634) + (28 * x637) + ((stencil(evo_shiftU1, stencil_idx_m4_0_0_VVV) + stencil(evo_shiftU1, stencil_idx_4_0_0_VVV))))) + (DYI * (x631 + (-56 * x649) + (-8 * x643) + (28 * x646) + ((stencil(evo_shiftU1, stencil_idx_0_m4_0_VVV) + stencil(evo_shiftU1, stencil_idx_0_4_0_VVV))))) + (DZI * (x631 + (-56 * x658) + (-8 * x652) + (28 * x655) + ((stencil(evo_shiftU1, stencil_idx_0_0_m4_VVV) + stencil(evo_shiftU1, stencil_idx_0_0_4_VVV)))))))));
        vreal x660 = stencil(evo_shiftU2, stencil_idx_m3_0_0_VVV); // x660: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m3__0__0_]
        vreal x661 = stencil(evo_shiftU2, stencil_idx_3_0_0_VVV); // x661: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__3__0__0_]
        vreal x662 = ((x660 + x661)); // x662: Dependency! Liveness = 2; [x660, x661]
        vreal x663 = stencil(evo_shiftU2, stencil_idx_m2_0_0_VVV); // x663: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m2__0__0_]
        vreal x664 = stencil(evo_shiftU2, stencil_idx_2_0_0_VVV); // x664: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__2__0__0_]
        vreal x665 = ((x663 + x664)); // x665: Dependency! Liveness = 2; [x663, x664]
        vreal x666 = stencil(evo_shiftU2, stencil_idx_m1_0_0_VVV); // x666: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m1__0__0_]
        vreal x667 = stencil(evo_shiftU2, stencil_idx_1_0_0_VVV); // x667: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__1__0__0_]
        vreal x668 = ((x666 + x667)); // x668: Dependency! Liveness = 2; [x666, x667]
        vreal x669 = stencil(evo_shiftU2, stencil_idx_0_m3_0_VVV); // x669: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m3__0_]
        vreal x670 = stencil(evo_shiftU2, stencil_idx_0_3_0_VVV); // x670: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__3__0_]
        vreal x671 = ((x669 + x670)); // x671: Dependency! Liveness = 2; [x669, x670]
        vreal x672 = stencil(evo_shiftU2, stencil_idx_0_m2_0_VVV); // x672: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m2__0_]
        vreal x673 = stencil(evo_shiftU2, stencil_idx_0_2_0_VVV); // x673: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__2__0_]
        vreal x674 = ((x672 + x673)); // x674: Dependency! Liveness = 2; [x672, x673]
        vreal x675 = stencil(evo_shiftU2, stencil_idx_0_m1_0_VVV); // x675: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m1__0_]
        vreal x676 = stencil(evo_shiftU2, stencil_idx_0_1_0_VVV); // x676: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__1__0_]
        vreal x677 = ((x675 + x676)); // x677: Dependency! Liveness = 2; [x675, x676]
        vreal x678 = stencil(evo_shiftU2, stencil_idx_0_0_m3_VVV); // x678: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m3_]
        vreal x679 = stencil(evo_shiftU2, stencil_idx_0_0_3_VVV); // x679: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__3_]
        vreal x680 = ((x678 + x679)); // x680: Dependency! Liveness = 2; [x678, x679]
        vreal x681 = stencil(evo_shiftU2, stencil_idx_0_0_m2_VVV); // x681: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m2_]
        vreal x682 = stencil(evo_shiftU2, stencil_idx_0_0_2_VVV); // x682: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__2_]
        vreal x683 = ((x681 + x682)); // x683: Dependency! Liveness = 2; [x681, x682]
        vreal x684 = stencil(evo_shiftU2, stencil_idx_0_0_m1_VVV); // x684: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m1_]
        vreal x685 = stencil(evo_shiftU2, stencil_idx_0_0_1_VVV); // x685: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__1_]
        vreal x686 = ((x684 + x685)); // x686: Dependency! Liveness = 2; [x684, x685]
        vreal x659 = (70 * stencil(evo_shiftU2, stencil_idx_0_0_0_VVV)); // x659: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__0_]
        store(evo_shift_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU2, stencil_idx_0_0_0_VVV) + (x296 * ((DXI * (x659 + (-56 * x668) + (-8 * x662) + (28 * x665) + ((stencil(evo_shiftU2, stencil_idx_m4_0_0_VVV) + stencil(evo_shiftU2, stencil_idx_4_0_0_VVV))))) + (DYI * (x659 + (-56 * x677) + (-8 * x671) + (28 * x674) + ((stencil(evo_shiftU2, stencil_idx_0_m4_0_VVV) + stencil(evo_shiftU2, stencil_idx_0_4_0_VVV))))) + (DZI * (x659 + (-56 * x686) + (-8 * x680) + (28 * x683) + ((stencil(evo_shiftU2, stencil_idx_0_0_m4_VVV) + stencil(evo_shiftU2, stencil_idx_0_0_4_VVV)))))))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}