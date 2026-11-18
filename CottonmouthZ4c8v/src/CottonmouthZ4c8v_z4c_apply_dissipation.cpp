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
    CCTK_ASSERT((cctk_nghostzones[0] >= 5));
    CCTK_ASSERT((cctk_nghostzones[1] >= 5));
    CCTK_ASSERT((cctk_nghostzones[2] >= 5));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // z4c_apply_dissipation loop 0
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
        vreal x322 = stencil(Theta, stencil_idx_0_m4_0_VVV); // x322: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__m4__0_]
        vreal x323 = stencil(Theta, stencil_idx_0_4_0_VVV); // x323: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__4__0_]
        vreal x324 = stencil(Theta, stencil_idx_0_m3_0_VVV); // x324: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__m3__0_]
        vreal x325 = stencil(Theta, stencil_idx_0_3_0_VVV); // x325: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__3__0_]
        vreal x326 = stencil(Theta, stencil_idx_0_m2_0_VVV); // x326: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__m2__0_]
        vreal x327 = stencil(Theta, stencil_idx_0_2_0_VVV); // x327: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__2__0_]
        vreal x328 = stencil(Theta, stencil_idx_0_m1_0_VVV); // x328: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__m1__0_]
        vreal x329 = stencil(Theta, stencil_idx_0_1_0_VVV); // x329: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__1__0_]
        vreal x330 = stencil(Theta, stencil_idx_0_0_m4_VVV); // x330: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__m4_]
        vreal x331 = stencil(Theta, stencil_idx_0_0_4_VVV); // x331: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__4_]
        vreal x332 = stencil(Theta, stencil_idx_0_0_m3_VVV); // x332: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__m3_]
        vreal x333 = stencil(Theta, stencil_idx_0_0_3_VVV); // x333: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__3_]
        vreal x334 = stencil(Theta, stencil_idx_0_0_m2_VVV); // x334: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__m2_]
        vreal x335 = stencil(Theta, stencil_idx_0_0_2_VVV); // x335: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__2_]
        vreal x336 = stencil(Theta, stencil_idx_0_0_m1_VVV); // x336: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__m1_]
        vreal x337 = stencil(Theta, stencil_idx_0_0_1_VVV); // x337: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__1_]
        vreal x314 = stencil(Theta, stencil_idx_m4_0_0_VVV); // x314: Dependency! Liveness = 1; [__dummy_stencil__Theta__m4__0__0_]
        vreal x315 = stencil(Theta, stencil_idx_4_0_0_VVV); // x315: Dependency! Liveness = 1; [__dummy_stencil__Theta__4__0__0_]
        vreal x316 = stencil(Theta, stencil_idx_m3_0_0_VVV); // x316: Dependency! Liveness = 1; [__dummy_stencil__Theta__m3__0__0_]
        vreal x317 = stencil(Theta, stencil_idx_3_0_0_VVV); // x317: Dependency! Liveness = 1; [__dummy_stencil__Theta__3__0__0_]
        vreal x318 = stencil(Theta, stencil_idx_m2_0_0_VVV); // x318: Dependency! Liveness = 1; [__dummy_stencil__Theta__m2__0__0_]
        vreal x319 = stencil(Theta, stencil_idx_2_0_0_VVV); // x319: Dependency! Liveness = 1; [__dummy_stencil__Theta__2__0__0_]
        vreal x320 = stencil(Theta, stencil_idx_m1_0_0_VVV); // x320: Dependency! Liveness = 1; [__dummy_stencil__Theta__m1__0__0_]
        vreal x321 = stencil(Theta, stencil_idx_1_0_0_VVV); // x321: Dependency! Liveness = 1; [__dummy_stencil__Theta__1__0__0_]
        vreal x313 = (-252 * stencil(Theta, stencil_idx_0_0_0_VVV)); // x313: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__0_]
        vreal x338 = ((1.0 / 1024.0) * dissipation_epsilon); // x338: Dependency! Liveness = 1; [dissipation_epsilon]
        store(Theta_rhs, stencil_idx_0_0_0_VVV, (access(Theta_rhs, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x313 + (-120 * ((x318 + x319))) + (-10 * ((x314 + x315))) + (45 * ((x316 + x317))) + (210 * ((x320 + x321))) + ((stencil(Theta, stencil_idx_m5_0_0_VVV) + stencil(Theta, stencil_idx_5_0_0_VVV))))) + (DYI * (x313 + (-120 * ((x326 + x327))) + (-10 * ((x322 + x323))) + (45 * ((x324 + x325))) + (210 * ((x328 + x329))) + ((stencil(Theta, stencil_idx_0_m5_0_VVV) + stencil(Theta, stencil_idx_0_5_0_VVV))))) + (DZI * (x313 + (-120 * ((x334 + x335))) + (-10 * ((x330 + x331))) + (45 * ((x332 + x333))) + (210 * ((x336 + x337))) + ((stencil(Theta, stencil_idx_0_0_m5_VVV) + stencil(Theta, stencil_idx_0_0_5_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 1
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
        vreal x340 = stencil(chi, stencil_idx_m4_0_0_VVV); // x340: Dependency! Liveness = 1; [__dummy_stencil__chi__m4__0__0_]
        vreal x341 = stencil(chi, stencil_idx_4_0_0_VVV); // x341: Dependency! Liveness = 1; [__dummy_stencil__chi__4__0__0_]
        vreal x342 = ((x340 + x341)); // x342: Dependency! Liveness = 2; [x340, x341]
        vreal x343 = stencil(chi, stencil_idx_m3_0_0_VVV); // x343: Dependency! Liveness = 1; [__dummy_stencil__chi__m3__0__0_]
        vreal x344 = stencil(chi, stencil_idx_3_0_0_VVV); // x344: Dependency! Liveness = 1; [__dummy_stencil__chi__3__0__0_]
        vreal x345 = ((x343 + x344)); // x345: Dependency! Liveness = 2; [x343, x344]
        vreal x346 = stencil(chi, stencil_idx_m2_0_0_VVV); // x346: Dependency! Liveness = 1; [__dummy_stencil__chi__m2__0__0_]
        vreal x347 = stencil(chi, stencil_idx_2_0_0_VVV); // x347: Dependency! Liveness = 1; [__dummy_stencil__chi__2__0__0_]
        vreal x348 = ((x346 + x347)); // x348: Dependency! Liveness = 2; [x346, x347]
        vreal x349 = stencil(chi, stencil_idx_m1_0_0_VVV); // x349: Dependency! Liveness = 1; [__dummy_stencil__chi__m1__0__0_]
        vreal x350 = stencil(chi, stencil_idx_1_0_0_VVV); // x350: Dependency! Liveness = 1; [__dummy_stencil__chi__1__0__0_]
        vreal x351 = ((x349 + x350)); // x351: Dependency! Liveness = 2; [x349, x350]
        vreal x352 = stencil(chi, stencil_idx_0_m4_0_VVV); // x352: Dependency! Liveness = 1; [__dummy_stencil__chi__0__m4__0_]
        vreal x353 = stencil(chi, stencil_idx_0_4_0_VVV); // x353: Dependency! Liveness = 1; [__dummy_stencil__chi__0__4__0_]
        vreal x354 = ((x352 + x353)); // x354: Dependency! Liveness = 2; [x352, x353]
        vreal x355 = stencil(chi, stencil_idx_0_m3_0_VVV); // x355: Dependency! Liveness = 1; [__dummy_stencil__chi__0__m3__0_]
        vreal x356 = stencil(chi, stencil_idx_0_3_0_VVV); // x356: Dependency! Liveness = 1; [__dummy_stencil__chi__0__3__0_]
        vreal x357 = ((x355 + x356)); // x357: Dependency! Liveness = 2; [x355, x356]
        vreal x358 = stencil(chi, stencil_idx_0_m2_0_VVV); // x358: Dependency! Liveness = 1; [__dummy_stencil__chi__0__m2__0_]
        vreal x359 = stencil(chi, stencil_idx_0_2_0_VVV); // x359: Dependency! Liveness = 1; [__dummy_stencil__chi__0__2__0_]
        vreal x360 = ((x358 + x359)); // x360: Dependency! Liveness = 2; [x358, x359]
        vreal x361 = stencil(chi, stencil_idx_0_m1_0_VVV); // x361: Dependency! Liveness = 1; [__dummy_stencil__chi__0__m1__0_]
        vreal x362 = stencil(chi, stencil_idx_0_1_0_VVV); // x362: Dependency! Liveness = 1; [__dummy_stencil__chi__0__1__0_]
        vreal x363 = ((x361 + x362)); // x363: Dependency! Liveness = 2; [x361, x362]
        vreal x364 = stencil(chi, stencil_idx_0_0_m4_VVV); // x364: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__m4_]
        vreal x365 = stencil(chi, stencil_idx_0_0_4_VVV); // x365: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__4_]
        vreal x366 = ((x364 + x365)); // x366: Dependency! Liveness = 2; [x364, x365]
        vreal x367 = stencil(chi, stencil_idx_0_0_m3_VVV); // x367: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__m3_]
        vreal x368 = stencil(chi, stencil_idx_0_0_3_VVV); // x368: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__3_]
        vreal x369 = ((x367 + x368)); // x369: Dependency! Liveness = 2; [x367, x368]
        vreal x370 = stencil(chi, stencil_idx_0_0_m2_VVV); // x370: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__m2_]
        vreal x371 = stencil(chi, stencil_idx_0_0_2_VVV); // x371: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__2_]
        vreal x372 = ((x370 + x371)); // x372: Dependency! Liveness = 2; [x370, x371]
        vreal x373 = stencil(chi, stencil_idx_0_0_m1_VVV); // x373: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__m1_]
        vreal x374 = stencil(chi, stencil_idx_0_0_1_VVV); // x374: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__1_]
        vreal x375 = ((x373 + x374)); // x375: Dependency! Liveness = 2; [x373, x374]
        vreal x339 = (-252 * stencil(chi, stencil_idx_0_0_0_VVV)); // x339: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__0_]
        vreal x338 = ((1.0 / 1024.0) * dissipation_epsilon); // x338: Dependency! Liveness = 1; [dissipation_epsilon]
        store(chi_rhs, stencil_idx_0_0_0_VVV, (access(chi_rhs, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x339 + (-120 * x348) + (-10 * x342) + (45 * x345) + (210 * x351) + ((stencil(chi, stencil_idx_m5_0_0_VVV) + stencil(chi, stencil_idx_5_0_0_VVV))))) + (DYI * (x339 + (-120 * x360) + (-10 * x354) + (45 * x357) + (210 * x363) + ((stencil(chi, stencil_idx_0_m5_0_VVV) + stencil(chi, stencil_idx_0_5_0_VVV))))) + (DZI * (x339 + (-120 * x372) + (-10 * x366) + (45 * x369) + (210 * x375) + ((stencil(chi, stencil_idx_0_0_m5_VVV) + stencil(chi, stencil_idx_0_0_5_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 2
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
        vreal x385 = stencil(trK, stencil_idx_0_m4_0_VVV); // x385: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m4__0_]
        vreal x386 = stencil(trK, stencil_idx_0_4_0_VVV); // x386: Dependency! Liveness = 1; [__dummy_stencil__trK__0__4__0_]
        vreal x387 = stencil(trK, stencil_idx_0_m3_0_VVV); // x387: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m3__0_]
        vreal x388 = stencil(trK, stencil_idx_0_3_0_VVV); // x388: Dependency! Liveness = 1; [__dummy_stencil__trK__0__3__0_]
        vreal x389 = stencil(trK, stencil_idx_0_m2_0_VVV); // x389: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m2__0_]
        vreal x390 = stencil(trK, stencil_idx_0_2_0_VVV); // x390: Dependency! Liveness = 1; [__dummy_stencil__trK__0__2__0_]
        vreal x391 = stencil(trK, stencil_idx_0_m1_0_VVV); // x391: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m1__0_]
        vreal x392 = stencil(trK, stencil_idx_0_1_0_VVV); // x392: Dependency! Liveness = 1; [__dummy_stencil__trK__0__1__0_]
        vreal x393 = stencil(trK, stencil_idx_0_0_m4_VVV); // x393: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m4_]
        vreal x394 = stencil(trK, stencil_idx_0_0_4_VVV); // x394: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__4_]
        vreal x395 = stencil(trK, stencil_idx_0_0_m3_VVV); // x395: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m3_]
        vreal x396 = stencil(trK, stencil_idx_0_0_3_VVV); // x396: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__3_]
        vreal x397 = stencil(trK, stencil_idx_0_0_m2_VVV); // x397: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m2_]
        vreal x398 = stencil(trK, stencil_idx_0_0_2_VVV); // x398: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__2_]
        vreal x399 = stencil(trK, stencil_idx_0_0_m1_VVV); // x399: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m1_]
        vreal x400 = stencil(trK, stencil_idx_0_0_1_VVV); // x400: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__1_]
        vreal x377 = stencil(trK, stencil_idx_m4_0_0_VVV); // x377: Dependency! Liveness = 1; [__dummy_stencil__trK__m4__0__0_]
        vreal x378 = stencil(trK, stencil_idx_4_0_0_VVV); // x378: Dependency! Liveness = 1; [__dummy_stencil__trK__4__0__0_]
        vreal x379 = stencil(trK, stencil_idx_m3_0_0_VVV); // x379: Dependency! Liveness = 1; [__dummy_stencil__trK__m3__0__0_]
        vreal x380 = stencil(trK, stencil_idx_3_0_0_VVV); // x380: Dependency! Liveness = 1; [__dummy_stencil__trK__3__0__0_]
        vreal x381 = stencil(trK, stencil_idx_m2_0_0_VVV); // x381: Dependency! Liveness = 1; [__dummy_stencil__trK__m2__0__0_]
        vreal x382 = stencil(trK, stencil_idx_2_0_0_VVV); // x382: Dependency! Liveness = 1; [__dummy_stencil__trK__2__0__0_]
        vreal x383 = stencil(trK, stencil_idx_m1_0_0_VVV); // x383: Dependency! Liveness = 1; [__dummy_stencil__trK__m1__0__0_]
        vreal x384 = stencil(trK, stencil_idx_1_0_0_VVV); // x384: Dependency! Liveness = 1; [__dummy_stencil__trK__1__0__0_]
        vreal x376 = (-252 * stencil(trK, stencil_idx_0_0_0_VVV)); // x376: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__0_]
        vreal x338 = ((1.0 / 1024.0) * dissipation_epsilon); // x338: Dependency! Liveness = 1; [dissipation_epsilon]
        store(trK_rhs, stencil_idx_0_0_0_VVV, (access(trK_rhs, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x376 + (-120 * ((x381 + x382))) + (-10 * ((x377 + x378))) + (45 * ((x379 + x380))) + (210 * ((x383 + x384))) + ((stencil(trK, stencil_idx_m5_0_0_VVV) + stencil(trK, stencil_idx_5_0_0_VVV))))) + (DYI * (x376 + (-120 * ((x389 + x390))) + (-10 * ((x385 + x386))) + (45 * ((x387 + x388))) + (210 * ((x391 + x392))) + ((stencil(trK, stencil_idx_0_m5_0_VVV) + stencil(trK, stencil_idx_0_5_0_VVV))))) + (DZI * (x376 + (-120 * ((x397 + x398))) + (-10 * ((x393 + x394))) + (45 * ((x395 + x396))) + (210 * ((x399 + x400))) + ((stencil(trK, stencil_idx_0_0_m5_VVV) + stencil(trK, stencil_idx_0_0_5_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 3
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
        vreal x410 = stencil(evo_GammatU0, stencil_idx_0_m4_0_VVV); // x410: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__m4__0_]
        vreal x411 = stencil(evo_GammatU0, stencil_idx_0_4_0_VVV); // x411: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__4__0_]
        vreal x412 = stencil(evo_GammatU0, stencil_idx_0_m3_0_VVV); // x412: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__m3__0_]
        vreal x413 = stencil(evo_GammatU0, stencil_idx_0_3_0_VVV); // x413: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__3__0_]
        vreal x414 = stencil(evo_GammatU0, stencil_idx_0_m2_0_VVV); // x414: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__m2__0_]
        vreal x415 = stencil(evo_GammatU0, stencil_idx_0_2_0_VVV); // x415: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__2__0_]
        vreal x416 = stencil(evo_GammatU0, stencil_idx_0_m1_0_VVV); // x416: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__m1__0_]
        vreal x417 = stencil(evo_GammatU0, stencil_idx_0_1_0_VVV); // x417: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__1__0_]
        vreal x418 = stencil(evo_GammatU0, stencil_idx_0_0_m4_VVV); // x418: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__m4_]
        vreal x419 = stencil(evo_GammatU0, stencil_idx_0_0_4_VVV); // x419: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__4_]
        vreal x420 = stencil(evo_GammatU0, stencil_idx_0_0_m3_VVV); // x420: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__m3_]
        vreal x421 = stencil(evo_GammatU0, stencil_idx_0_0_3_VVV); // x421: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__3_]
        vreal x422 = stencil(evo_GammatU0, stencil_idx_0_0_m2_VVV); // x422: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__m2_]
        vreal x423 = stencil(evo_GammatU0, stencil_idx_0_0_2_VVV); // x423: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__2_]
        vreal x424 = stencil(evo_GammatU0, stencil_idx_0_0_m1_VVV); // x424: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__m1_]
        vreal x425 = stencil(evo_GammatU0, stencil_idx_0_0_1_VVV); // x425: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__1_]
        vreal x402 = stencil(evo_GammatU0, stencil_idx_m4_0_0_VVV); // x402: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__m4__0__0_]
        vreal x403 = stencil(evo_GammatU0, stencil_idx_4_0_0_VVV); // x403: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__4__0__0_]
        vreal x404 = stencil(evo_GammatU0, stencil_idx_m3_0_0_VVV); // x404: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__m3__0__0_]
        vreal x405 = stencil(evo_GammatU0, stencil_idx_3_0_0_VVV); // x405: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__3__0__0_]
        vreal x406 = stencil(evo_GammatU0, stencil_idx_m2_0_0_VVV); // x406: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__m2__0__0_]
        vreal x407 = stencil(evo_GammatU0, stencil_idx_2_0_0_VVV); // x407: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__2__0__0_]
        vreal x408 = stencil(evo_GammatU0, stencil_idx_m1_0_0_VVV); // x408: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__m1__0__0_]
        vreal x409 = stencil(evo_GammatU0, stencil_idx_1_0_0_VVV); // x409: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__1__0__0_]
        vreal x401 = (-252 * stencil(evo_GammatU0, stencil_idx_0_0_0_VVV)); // x401: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__0_]
        vreal x338 = ((1.0 / 1024.0) * dissipation_epsilon); // x338: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_Gammat_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU0, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x401 + (-120 * ((x406 + x407))) + (-10 * ((x402 + x403))) + (45 * ((x404 + x405))) + (210 * ((x408 + x409))) + ((stencil(evo_GammatU0, stencil_idx_m5_0_0_VVV) + stencil(evo_GammatU0, stencil_idx_5_0_0_VVV))))) + (DYI * (x401 + (-120 * ((x414 + x415))) + (-10 * ((x410 + x411))) + (45 * ((x412 + x413))) + (210 * ((x416 + x417))) + ((stencil(evo_GammatU0, stencil_idx_0_m5_0_VVV) + stencil(evo_GammatU0, stencil_idx_0_5_0_VVV))))) + (DZI * (x401 + (-120 * ((x422 + x423))) + (-10 * ((x418 + x419))) + (45 * ((x420 + x421))) + (210 * ((x424 + x425))) + ((stencil(evo_GammatU0, stencil_idx_0_0_m5_VVV) + stencil(evo_GammatU0, stencil_idx_0_0_5_VVV)))))))));
        vreal x435 = stencil(evo_GammatU1, stencil_idx_0_m4_0_VVV); // x435: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__m4__0_]
        vreal x436 = stencil(evo_GammatU1, stencil_idx_0_4_0_VVV); // x436: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__4__0_]
        vreal x437 = stencil(evo_GammatU1, stencil_idx_0_m3_0_VVV); // x437: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__m3__0_]
        vreal x438 = stencil(evo_GammatU1, stencil_idx_0_3_0_VVV); // x438: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__3__0_]
        vreal x439 = stencil(evo_GammatU1, stencil_idx_0_m2_0_VVV); // x439: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__m2__0_]
        vreal x440 = stencil(evo_GammatU1, stencil_idx_0_2_0_VVV); // x440: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__2__0_]
        vreal x441 = stencil(evo_GammatU1, stencil_idx_0_m1_0_VVV); // x441: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__m1__0_]
        vreal x442 = stencil(evo_GammatU1, stencil_idx_0_1_0_VVV); // x442: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__1__0_]
        vreal x443 = stencil(evo_GammatU1, stencil_idx_0_0_m4_VVV); // x443: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__m4_]
        vreal x444 = stencil(evo_GammatU1, stencil_idx_0_0_4_VVV); // x444: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__4_]
        vreal x445 = stencil(evo_GammatU1, stencil_idx_0_0_m3_VVV); // x445: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__m3_]
        vreal x446 = stencil(evo_GammatU1, stencil_idx_0_0_3_VVV); // x446: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__3_]
        vreal x447 = stencil(evo_GammatU1, stencil_idx_0_0_m2_VVV); // x447: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__m2_]
        vreal x448 = stencil(evo_GammatU1, stencil_idx_0_0_2_VVV); // x448: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__2_]
        vreal x449 = stencil(evo_GammatU1, stencil_idx_0_0_m1_VVV); // x449: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__m1_]
        vreal x450 = stencil(evo_GammatU1, stencil_idx_0_0_1_VVV); // x450: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__1_]
        vreal x427 = stencil(evo_GammatU1, stencil_idx_m4_0_0_VVV); // x427: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__m4__0__0_]
        vreal x428 = stencil(evo_GammatU1, stencil_idx_4_0_0_VVV); // x428: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__4__0__0_]
        vreal x429 = stencil(evo_GammatU1, stencil_idx_m3_0_0_VVV); // x429: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__m3__0__0_]
        vreal x430 = stencil(evo_GammatU1, stencil_idx_3_0_0_VVV); // x430: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__3__0__0_]
        vreal x431 = stencil(evo_GammatU1, stencil_idx_m2_0_0_VVV); // x431: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__m2__0__0_]
        vreal x432 = stencil(evo_GammatU1, stencil_idx_2_0_0_VVV); // x432: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__2__0__0_]
        vreal x433 = stencil(evo_GammatU1, stencil_idx_m1_0_0_VVV); // x433: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__m1__0__0_]
        vreal x434 = stencil(evo_GammatU1, stencil_idx_1_0_0_VVV); // x434: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__1__0__0_]
        vreal x426 = (-252 * stencil(evo_GammatU1, stencil_idx_0_0_0_VVV)); // x426: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__0_]
        store(evo_Gammat_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU1, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x426 + (-120 * ((x431 + x432))) + (-10 * ((x427 + x428))) + (45 * ((x429 + x430))) + (210 * ((x433 + x434))) + ((stencil(evo_GammatU1, stencil_idx_m5_0_0_VVV) + stencil(evo_GammatU1, stencil_idx_5_0_0_VVV))))) + (DYI * (x426 + (-120 * ((x439 + x440))) + (-10 * ((x435 + x436))) + (45 * ((x437 + x438))) + (210 * ((x441 + x442))) + ((stencil(evo_GammatU1, stencil_idx_0_m5_0_VVV) + stencil(evo_GammatU1, stencil_idx_0_5_0_VVV))))) + (DZI * (x426 + (-120 * ((x447 + x448))) + (-10 * ((x443 + x444))) + (45 * ((x445 + x446))) + (210 * ((x449 + x450))) + ((stencil(evo_GammatU1, stencil_idx_0_0_m5_VVV) + stencil(evo_GammatU1, stencil_idx_0_0_5_VVV)))))))));
        vreal x460 = stencil(evo_GammatU2, stencil_idx_0_m4_0_VVV); // x460: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__m4__0_]
        vreal x461 = stencil(evo_GammatU2, stencil_idx_0_4_0_VVV); // x461: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__4__0_]
        vreal x462 = stencil(evo_GammatU2, stencil_idx_0_m3_0_VVV); // x462: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__m3__0_]
        vreal x463 = stencil(evo_GammatU2, stencil_idx_0_3_0_VVV); // x463: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__3__0_]
        vreal x464 = stencil(evo_GammatU2, stencil_idx_0_m2_0_VVV); // x464: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__m2__0_]
        vreal x465 = stencil(evo_GammatU2, stencil_idx_0_2_0_VVV); // x465: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__2__0_]
        vreal x466 = stencil(evo_GammatU2, stencil_idx_0_m1_0_VVV); // x466: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__m1__0_]
        vreal x467 = stencil(evo_GammatU2, stencil_idx_0_1_0_VVV); // x467: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__1__0_]
        vreal x468 = stencil(evo_GammatU2, stencil_idx_0_0_m4_VVV); // x468: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__m4_]
        vreal x469 = stencil(evo_GammatU2, stencil_idx_0_0_4_VVV); // x469: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__4_]
        vreal x470 = stencil(evo_GammatU2, stencil_idx_0_0_m3_VVV); // x470: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__m3_]
        vreal x471 = stencil(evo_GammatU2, stencil_idx_0_0_3_VVV); // x471: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__3_]
        vreal x472 = stencil(evo_GammatU2, stencil_idx_0_0_m2_VVV); // x472: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__m2_]
        vreal x473 = stencil(evo_GammatU2, stencil_idx_0_0_2_VVV); // x473: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__2_]
        vreal x474 = stencil(evo_GammatU2, stencil_idx_0_0_m1_VVV); // x474: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__m1_]
        vreal x475 = stencil(evo_GammatU2, stencil_idx_0_0_1_VVV); // x475: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__1_]
        vreal x452 = stencil(evo_GammatU2, stencil_idx_m4_0_0_VVV); // x452: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__m4__0__0_]
        vreal x453 = stencil(evo_GammatU2, stencil_idx_4_0_0_VVV); // x453: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__4__0__0_]
        vreal x454 = stencil(evo_GammatU2, stencil_idx_m3_0_0_VVV); // x454: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__m3__0__0_]
        vreal x455 = stencil(evo_GammatU2, stencil_idx_3_0_0_VVV); // x455: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__3__0__0_]
        vreal x456 = stencil(evo_GammatU2, stencil_idx_m2_0_0_VVV); // x456: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__m2__0__0_]
        vreal x457 = stencil(evo_GammatU2, stencil_idx_2_0_0_VVV); // x457: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__2__0__0_]
        vreal x458 = stencil(evo_GammatU2, stencil_idx_m1_0_0_VVV); // x458: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__m1__0__0_]
        vreal x459 = stencil(evo_GammatU2, stencil_idx_1_0_0_VVV); // x459: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__1__0__0_]
        vreal x451 = (-252 * stencil(evo_GammatU2, stencil_idx_0_0_0_VVV)); // x451: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__0_]
        store(evo_Gammat_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU2, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x451 + (-120 * ((x456 + x457))) + (-10 * ((x452 + x453))) + (45 * ((x454 + x455))) + (210 * ((x458 + x459))) + ((stencil(evo_GammatU2, stencil_idx_m5_0_0_VVV) + stencil(evo_GammatU2, stencil_idx_5_0_0_VVV))))) + (DYI * (x451 + (-120 * ((x464 + x465))) + (-10 * ((x460 + x461))) + (45 * ((x462 + x463))) + (210 * ((x466 + x467))) + ((stencil(evo_GammatU2, stencil_idx_0_m5_0_VVV) + stencil(evo_GammatU2, stencil_idx_0_5_0_VVV))))) + (DZI * (x451 + (-120 * ((x472 + x473))) + (-10 * ((x468 + x469))) + (45 * ((x470 + x471))) + (210 * ((x474 + x475))) + ((stencil(evo_GammatU2, stencil_idx_0_0_m5_VVV) + stencil(evo_GammatU2, stencil_idx_0_0_5_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 4
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
        vreal x229 = stencil(gtDD00, stencil_idx_m4_0_0_VVV); // x229: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m4__0__0_]
        vreal x234 = stencil(gtDD00, stencil_idx_4_0_0_VVV); // x234: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__4__0__0_]
        vreal x477 = ((x229 + x234)); // x477: Dependency! Liveness = 2; [x229, x234]
        vreal x230 = stencil(gtDD00, stencil_idx_m3_0_0_VVV); // x230: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m3__0__0_]
        vreal x233 = stencil(gtDD00, stencil_idx_3_0_0_VVV); // x233: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__3__0__0_]
        vreal x478 = ((x230 + x233)); // x478: Dependency! Liveness = 2; [x230, x233]
        vreal x231 = stencil(gtDD00, stencil_idx_m2_0_0_VVV); // x231: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m2__0__0_]
        vreal x232 = stencil(gtDD00, stencil_idx_2_0_0_VVV); // x232: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__2__0__0_]
        vreal x479 = ((x231 + x232)); // x479: Dependency! Liveness = 2; [x231, x232]
        vreal x235 = stencil(gtDD00, stencil_idx_m1_0_0_VVV); // x235: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m1__0__0_]
        vreal x236 = stencil(gtDD00, stencil_idx_1_0_0_VVV); // x236: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__1__0__0_]
        vreal x480 = ((x235 + x236)); // x480: Dependency! Liveness = 2; [x235, x236]
        vreal x130 = stencil(gtDD00, stencil_idx_0_m4_0_VVV); // x130: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m4__0_]
        vreal x135 = stencil(gtDD00, stencil_idx_0_4_0_VVV); // x135: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__4__0_]
        vreal x481 = ((x130 + x135)); // x481: Dependency! Liveness = 2; [x130, x135]
        vreal x131 = stencil(gtDD00, stencil_idx_0_m3_0_VVV); // x131: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m3__0_]
        vreal x134 = stencil(gtDD00, stencil_idx_0_3_0_VVV); // x134: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__3__0_]
        vreal x482 = ((x131 + x134)); // x482: Dependency! Liveness = 2; [x131, x134]
        vreal x132 = stencil(gtDD00, stencil_idx_0_m2_0_VVV); // x132: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m2__0_]
        vreal x133 = stencil(gtDD00, stencil_idx_0_2_0_VVV); // x133: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__2__0_]
        vreal x483 = ((x132 + x133)); // x483: Dependency! Liveness = 2; [x132, x133]
        vreal x136 = stencil(gtDD00, stencil_idx_0_m1_0_VVV); // x136: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m1__0_]
        vreal x137 = stencil(gtDD00, stencil_idx_0_1_0_VVV); // x137: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__1__0_]
        vreal x484 = ((x136 + x137)); // x484: Dependency! Liveness = 2; [x136, x137]
        vreal x114 = stencil(gtDD00, stencil_idx_0_0_m4_VVV); // x114: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m4_]
        vreal x119 = stencil(gtDD00, stencil_idx_0_0_4_VVV); // x119: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__4_]
        vreal x485 = ((x114 + x119)); // x485: Dependency! Liveness = 2; [x114, x119]
        vreal x115 = stencil(gtDD00, stencil_idx_0_0_m3_VVV); // x115: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m3_]
        vreal x118 = stencil(gtDD00, stencil_idx_0_0_3_VVV); // x118: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__3_]
        vreal x486 = ((x115 + x118)); // x486: Dependency! Liveness = 2; [x115, x118]
        vreal x116 = stencil(gtDD00, stencil_idx_0_0_m2_VVV); // x116: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m2_]
        vreal x117 = stencil(gtDD00, stencil_idx_0_0_2_VVV); // x117: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__2_]
        vreal x487 = ((x116 + x117)); // x487: Dependency! Liveness = 2; [x116, x117]
        vreal x120 = stencil(gtDD00, stencil_idx_0_0_m1_VVV); // x120: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m1_]
        vreal x121 = stencil(gtDD00, stencil_idx_0_0_1_VVV); // x121: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__1_]
        vreal x488 = ((x120 + x121)); // x488: Dependency! Liveness = 2; [x120, x121]
        vreal x476 = (-252 * stencil(gtDD00, stencil_idx_0_0_0_VVV)); // x476: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__0_]
        vreal x338 = ((1.0 / 1024.0) * dissipation_epsilon); // x338: Dependency! Liveness = 1; [dissipation_epsilon]
        store(gt_rhsDD00, stencil_idx_0_0_0_VVV, (access(gt_rhsDD00, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x476 + (-120 * x479) + (-10 * x477) + (45 * x478) + (210 * x480) + ((stencil(gtDD00, stencil_idx_m5_0_0_VVV) + stencil(gtDD00, stencil_idx_5_0_0_VVV))))) + (DYI * (x476 + (-120 * x483) + (-10 * x481) + (45 * x482) + (210 * x484) + ((stencil(gtDD00, stencil_idx_0_m5_0_VVV) + stencil(gtDD00, stencil_idx_0_5_0_VVV))))) + (DZI * (x476 + (-120 * x487) + (-10 * x485) + (45 * x486) + (210 * x488) + ((stencil(gtDD00, stencil_idx_0_0_m5_VVV) + stencil(gtDD00, stencil_idx_0_0_5_VVV)))))))));
        vreal x251 = stencil(gtDD01, stencil_idx_m4_0_0_VVV); // x251: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m4__0__0_]
        vreal x256 = stencil(gtDD01, stencil_idx_4_0_0_VVV); // x256: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__4__0__0_]
        vreal x490 = ((x251 + x256)); // x490: Dependency! Liveness = 2; [x251, x256]
        vreal x252 = stencil(gtDD01, stencil_idx_m3_0_0_VVV); // x252: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m3__0__0_]
        vreal x255 = stencil(gtDD01, stencil_idx_3_0_0_VVV); // x255: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__3__0__0_]
        vreal x491 = ((x252 + x255)); // x491: Dependency! Liveness = 2; [x252, x255]
        vreal x253 = stencil(gtDD01, stencil_idx_m2_0_0_VVV); // x253: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m2__0__0_]
        vreal x254 = stencil(gtDD01, stencil_idx_2_0_0_VVV); // x254: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__2__0__0_]
        vreal x492 = ((x253 + x254)); // x492: Dependency! Liveness = 2; [x253, x254]
        vreal x257 = stencil(gtDD01, stencil_idx_m1_0_0_VVV); // x257: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m1__0__0_]
        vreal x258 = stencil(gtDD01, stencil_idx_1_0_0_VVV); // x258: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__1__0__0_]
        vreal x493 = ((x257 + x258)); // x493: Dependency! Liveness = 2; [x257, x258]
        vreal x206 = stencil(gtDD01, stencil_idx_0_m4_0_VVV); // x206: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m4__0_]
        vreal x211 = stencil(gtDD01, stencil_idx_0_4_0_VVV); // x211: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__4__0_]
        vreal x494 = ((x206 + x211)); // x494: Dependency! Liveness = 2; [x206, x211]
        vreal x207 = stencil(gtDD01, stencil_idx_0_m3_0_VVV); // x207: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m3__0_]
        vreal x210 = stencil(gtDD01, stencil_idx_0_3_0_VVV); // x210: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__3__0_]
        vreal x495 = ((x207 + x210)); // x495: Dependency! Liveness = 2; [x207, x210]
        vreal x208 = stencil(gtDD01, stencil_idx_0_m2_0_VVV); // x208: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m2__0_]
        vreal x209 = stencil(gtDD01, stencil_idx_0_2_0_VVV); // x209: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__2__0_]
        vreal x496 = ((x208 + x209)); // x496: Dependency! Liveness = 2; [x208, x209]
        vreal x212 = stencil(gtDD01, stencil_idx_0_m1_0_VVV); // x212: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m1__0_]
        vreal x213 = stencil(gtDD01, stencil_idx_0_1_0_VVV); // x213: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__1__0_]
        vreal x497 = ((x212 + x213)); // x497: Dependency! Liveness = 2; [x212, x213]
        vreal x77 = stencil(gtDD01, stencil_idx_0_0_m4_VVV); // x77: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m4_]
        vreal x82 = stencil(gtDD01, stencil_idx_0_0_4_VVV); // x82: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__4_]
        vreal x498 = ((x77 + x82)); // x498: Dependency! Liveness = 2; [x77, x82]
        vreal x78 = stencil(gtDD01, stencil_idx_0_0_m3_VVV); // x78: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m3_]
        vreal x81 = stencil(gtDD01, stencil_idx_0_0_3_VVV); // x81: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__3_]
        vreal x499 = ((x78 + x81)); // x499: Dependency! Liveness = 2; [x78, x81]
        vreal x79 = stencil(gtDD01, stencil_idx_0_0_m2_VVV); // x79: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m2_]
        vreal x80 = stencil(gtDD01, stencil_idx_0_0_2_VVV); // x80: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__2_]
        vreal x500 = ((x79 + x80)); // x500: Dependency! Liveness = 2; [x79, x80]
        vreal x83 = stencil(gtDD01, stencil_idx_0_0_m1_VVV); // x83: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m1_]
        vreal x84 = stencil(gtDD01, stencil_idx_0_0_1_VVV); // x84: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__1_]
        vreal x501 = ((x83 + x84)); // x501: Dependency! Liveness = 2; [x83, x84]
        vreal x489 = (-252 * stencil(gtDD01, stencil_idx_0_0_0_VVV)); // x489: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__0_]
        store(gt_rhsDD01, stencil_idx_0_0_0_VVV, (access(gt_rhsDD01, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x489 + (-120 * x492) + (-10 * x490) + (45 * x491) + (210 * x493) + ((stencil(gtDD01, stencil_idx_m5_0_0_VVV) + stencil(gtDD01, stencil_idx_5_0_0_VVV))))) + (DYI * (x489 + (-120 * x496) + (-10 * x494) + (45 * x495) + (210 * x497) + ((stencil(gtDD01, stencil_idx_0_m5_0_VVV) + stencil(gtDD01, stencil_idx_0_5_0_VVV))))) + (DZI * (x489 + (-120 * x500) + (-10 * x498) + (45 * x499) + (210 * x501) + ((stencil(gtDD01, stencil_idx_0_0_m5_VVV) + stencil(gtDD01, stencil_idx_0_0_5_VVV)))))))));
        vreal x141 = stencil(gtDD11, stencil_idx_m4_0_0_VVV); // x141: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m4__0__0_]
        vreal x146 = stencil(gtDD11, stencil_idx_4_0_0_VVV); // x146: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__4__0__0_]
        vreal x503 = ((x141 + x146)); // x503: Dependency! Liveness = 2; [x141, x146]
        vreal x142 = stencil(gtDD11, stencil_idx_m3_0_0_VVV); // x142: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m3__0__0_]
        vreal x145 = stencil(gtDD11, stencil_idx_3_0_0_VVV); // x145: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__3__0__0_]
        vreal x504 = ((x142 + x145)); // x504: Dependency! Liveness = 2; [x142, x145]
        vreal x143 = stencil(gtDD11, stencil_idx_m2_0_0_VVV); // x143: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m2__0__0_]
        vreal x144 = stencil(gtDD11, stencil_idx_2_0_0_VVV); // x144: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__2__0__0_]
        vreal x505 = ((x143 + x144)); // x505: Dependency! Liveness = 2; [x143, x144]
        vreal x147 = stencil(gtDD11, stencil_idx_m1_0_0_VVV); // x147: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m1__0__0_]
        vreal x148 = stencil(gtDD11, stencil_idx_1_0_0_VVV); // x148: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__1__0__0_]
        vreal x506 = ((x147 + x148)); // x506: Dependency! Liveness = 2; [x147, x148]
        vreal x195 = stencil(gtDD11, stencil_idx_0_m4_0_VVV); // x195: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m4__0_]
        vreal x200 = stencil(gtDD11, stencil_idx_0_4_0_VVV); // x200: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__4__0_]
        vreal x507 = ((x195 + x200)); // x507: Dependency! Liveness = 2; [x195, x200]
        vreal x196 = stencil(gtDD11, stencil_idx_0_m3_0_VVV); // x196: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m3__0_]
        vreal x199 = stencil(gtDD11, stencil_idx_0_3_0_VVV); // x199: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__3__0_]
        vreal x508 = ((x196 + x199)); // x508: Dependency! Liveness = 2; [x196, x199]
        vreal x197 = stencil(gtDD11, stencil_idx_0_m2_0_VVV); // x197: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m2__0_]
        vreal x198 = stencil(gtDD11, stencil_idx_0_2_0_VVV); // x198: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__2__0_]
        vreal x509 = ((x197 + x198)); // x509: Dependency! Liveness = 2; [x197, x198]
        vreal x201 = stencil(gtDD11, stencil_idx_0_m1_0_VVV); // x201: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m1__0_]
        vreal x202 = stencil(gtDD11, stencil_idx_0_1_0_VVV); // x202: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__1__0_]
        vreal x510 = ((x201 + x202)); // x510: Dependency! Liveness = 2; [x201, x202]
        vreal x54 = stencil(gtDD11, stencil_idx_0_0_m4_VVV); // x54: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m4_]
        vreal x59 = stencil(gtDD11, stencil_idx_0_0_4_VVV); // x59: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__4_]
        vreal x511 = ((x54 + x59)); // x511: Dependency! Liveness = 2; [x54, x59]
        vreal x55 = stencil(gtDD11, stencil_idx_0_0_m3_VVV); // x55: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m3_]
        vreal x58 = stencil(gtDD11, stencil_idx_0_0_3_VVV); // x58: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__3_]
        vreal x512 = ((x55 + x58)); // x512: Dependency! Liveness = 2; [x55, x58]
        vreal x56 = stencil(gtDD11, stencil_idx_0_0_m2_VVV); // x56: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m2_]
        vreal x57 = stencil(gtDD11, stencil_idx_0_0_2_VVV); // x57: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__2_]
        vreal x513 = ((x56 + x57)); // x513: Dependency! Liveness = 2; [x56, x57]
        vreal x60 = stencil(gtDD11, stencil_idx_0_0_m1_VVV); // x60: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m1_]
        vreal x61 = stencil(gtDD11, stencil_idx_0_0_1_VVV); // x61: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__1_]
        vreal x514 = ((x60 + x61)); // x514: Dependency! Liveness = 2; [x60, x61]
        vreal x502 = (-252 * stencil(gtDD11, stencil_idx_0_0_0_VVV)); // x502: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__0_]
        store(gt_rhsDD11, stencil_idx_0_0_0_VVV, (access(gt_rhsDD11, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x502 + (-120 * x505) + (-10 * x503) + (45 * x504) + (210 * x506) + ((stencil(gtDD11, stencil_idx_m5_0_0_VVV) + stencil(gtDD11, stencil_idx_5_0_0_VVV))))) + (DYI * (x502 + (-120 * x509) + (-10 * x507) + (45 * x508) + (210 * x510) + ((stencil(gtDD11, stencil_idx_0_m5_0_VVV) + stencil(gtDD11, stencil_idx_0_5_0_VVV))))) + (DZI * (x502 + (-120 * x513) + (-10 * x511) + (45 * x512) + (210 * x514) + ((stencil(gtDD11, stencil_idx_0_0_m5_VVV) + stencil(gtDD11, stencil_idx_0_0_5_VVV)))))))));
        vreal x239 = stencil(gtDD02, stencil_idx_m4_0_0_VVV); // x239: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m4__0__0_]
        vreal x244 = stencil(gtDD02, stencil_idx_4_0_0_VVV); // x244: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__4__0__0_]
        vreal x516 = ((x239 + x244)); // x516: Dependency! Liveness = 2; [x239, x244]
        vreal x240 = stencil(gtDD02, stencil_idx_m3_0_0_VVV); // x240: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m3__0__0_]
        vreal x243 = stencil(gtDD02, stencil_idx_3_0_0_VVV); // x243: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__3__0__0_]
        vreal x517 = ((x240 + x243)); // x517: Dependency! Liveness = 2; [x240, x243]
        vreal x241 = stencil(gtDD02, stencil_idx_m2_0_0_VVV); // x241: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m2__0__0_]
        vreal x242 = stencil(gtDD02, stencil_idx_2_0_0_VVV); // x242: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__2__0__0_]
        vreal x518 = ((x241 + x242)); // x518: Dependency! Liveness = 2; [x241, x242]
        vreal x245 = stencil(gtDD02, stencil_idx_m1_0_0_VVV); // x245: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m1__0__0_]
        vreal x246 = stencil(gtDD02, stencil_idx_1_0_0_VVV); // x246: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__1__0__0_]
        vreal x519 = ((x245 + x246)); // x519: Dependency! Liveness = 2; [x245, x246]
        vreal x67 = stencil(gtDD02, stencil_idx_0_m4_0_VVV); // x67: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m4__0_]
        vreal x72 = stencil(gtDD02, stencil_idx_0_4_0_VVV); // x72: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__4__0_]
        vreal x520 = ((x67 + x72)); // x520: Dependency! Liveness = 2; [x67, x72]
        vreal x68 = stencil(gtDD02, stencil_idx_0_m3_0_VVV); // x68: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m3__0_]
        vreal x71 = stencil(gtDD02, stencil_idx_0_3_0_VVV); // x71: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__3__0_]
        vreal x521 = ((x68 + x71)); // x521: Dependency! Liveness = 2; [x68, x71]
        vreal x69 = stencil(gtDD02, stencil_idx_0_m2_0_VVV); // x69: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m2__0_]
        vreal x70 = stencil(gtDD02, stencil_idx_0_2_0_VVV); // x70: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__2__0_]
        vreal x522 = ((x69 + x70)); // x522: Dependency! Liveness = 2; [x69, x70]
        vreal x73 = stencil(gtDD02, stencil_idx_0_m1_0_VVV); // x73: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m1__0_]
        vreal x74 = stencil(gtDD02, stencil_idx_0_1_0_VVV); // x74: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__1__0_]
        vreal x523 = ((x73 + x74)); // x523: Dependency! Liveness = 2; [x73, x74]
        vreal x181 = stencil(gtDD02, stencil_idx_0_0_m4_VVV); // x181: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m4_]
        vreal x186 = stencil(gtDD02, stencil_idx_0_0_4_VVV); // x186: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__4_]
        vreal x524 = ((x181 + x186)); // x524: Dependency! Liveness = 2; [x181, x186]
        vreal x182 = stencil(gtDD02, stencil_idx_0_0_m3_VVV); // x182: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m3_]
        vreal x185 = stencil(gtDD02, stencil_idx_0_0_3_VVV); // x185: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__3_]
        vreal x525 = ((x182 + x185)); // x525: Dependency! Liveness = 2; [x182, x185]
        vreal x183 = stencil(gtDD02, stencil_idx_0_0_m2_VVV); // x183: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m2_]
        vreal x184 = stencil(gtDD02, stencil_idx_0_0_2_VVV); // x184: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__2_]
        vreal x526 = ((x183 + x184)); // x526: Dependency! Liveness = 2; [x183, x184]
        vreal x187 = stencil(gtDD02, stencil_idx_0_0_m1_VVV); // x187: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m1_]
        vreal x188 = stencil(gtDD02, stencil_idx_0_0_1_VVV); // x188: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__1_]
        vreal x527 = ((x187 + x188)); // x527: Dependency! Liveness = 2; [x187, x188]
        vreal x515 = (-252 * stencil(gtDD02, stencil_idx_0_0_0_VVV)); // x515: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__0_]
        store(gt_rhsDD02, stencil_idx_0_0_0_VVV, (access(gt_rhsDD02, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x515 + (-120 * x518) + (-10 * x516) + (45 * x517) + (210 * x519) + ((stencil(gtDD02, stencil_idx_m5_0_0_VVV) + stencil(gtDD02, stencil_idx_5_0_0_VVV))))) + (DYI * (x515 + (-120 * x522) + (-10 * x520) + (45 * x521) + (210 * x523) + ((stencil(gtDD02, stencil_idx_0_m5_0_VVV) + stencil(gtDD02, stencil_idx_0_5_0_VVV))))) + (DZI * (x515 + (-120 * x526) + (-10 * x524) + (45 * x525) + (210 * x527) + ((stencil(gtDD02, stencil_idx_0_0_m5_VVV) + stencil(gtDD02, stencil_idx_0_0_5_VVV)))))))));
        vreal x87 = stencil(gtDD12, stencil_idx_m4_0_0_VVV); // x87: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m4__0__0_]
        vreal x92 = stencil(gtDD12, stencil_idx_4_0_0_VVV); // x92: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__4__0__0_]
        vreal x529 = ((x87 + x92)); // x529: Dependency! Liveness = 2; [x87, x92]
        vreal x88 = stencil(gtDD12, stencil_idx_m3_0_0_VVV); // x88: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m3__0__0_]
        vreal x91 = stencil(gtDD12, stencil_idx_3_0_0_VVV); // x91: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__3__0__0_]
        vreal x530 = ((x88 + x91)); // x530: Dependency! Liveness = 2; [x88, x91]
        vreal x89 = stencil(gtDD12, stencil_idx_m2_0_0_VVV); // x89: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m2__0__0_]
        vreal x90 = stencil(gtDD12, stencil_idx_2_0_0_VVV); // x90: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__2__0__0_]
        vreal x531 = ((x89 + x90)); // x531: Dependency! Liveness = 2; [x89, x90]
        vreal x93 = stencil(gtDD12, stencil_idx_m1_0_0_VVV); // x93: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m1__0__0_]
        vreal x94 = stencil(gtDD12, stencil_idx_1_0_0_VVV); // x94: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__1__0__0_]
        vreal x532 = ((x93 + x94)); // x532: Dependency! Liveness = 2; [x93, x94]
        vreal x217 = stencil(gtDD12, stencil_idx_0_m4_0_VVV); // x217: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m4__0_]
        vreal x222 = stencil(gtDD12, stencil_idx_0_4_0_VVV); // x222: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__4__0_]
        vreal x533 = ((x217 + x222)); // x533: Dependency! Liveness = 2; [x217, x222]
        vreal x218 = stencil(gtDD12, stencil_idx_0_m3_0_VVV); // x218: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m3__0_]
        vreal x221 = stencil(gtDD12, stencil_idx_0_3_0_VVV); // x221: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__3__0_]
        vreal x534 = ((x218 + x221)); // x534: Dependency! Liveness = 2; [x218, x221]
        vreal x219 = stencil(gtDD12, stencil_idx_0_m2_0_VVV); // x219: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m2__0_]
        vreal x220 = stencil(gtDD12, stencil_idx_0_2_0_VVV); // x220: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__2__0_]
        vreal x535 = ((x219 + x220)); // x535: Dependency! Liveness = 2; [x219, x220]
        vreal x223 = stencil(gtDD12, stencil_idx_0_m1_0_VVV); // x223: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m1__0_]
        vreal x224 = stencil(gtDD12, stencil_idx_0_1_0_VVV); // x224: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__1__0_]
        vreal x536 = ((x223 + x224)); // x536: Dependency! Liveness = 2; [x223, x224]
        vreal x170 = stencil(gtDD12, stencil_idx_0_0_m4_VVV); // x170: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m4_]
        vreal x175 = stencil(gtDD12, stencil_idx_0_0_4_VVV); // x175: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__4_]
        vreal x537 = ((x170 + x175)); // x537: Dependency! Liveness = 2; [x170, x175]
        vreal x171 = stencil(gtDD12, stencil_idx_0_0_m3_VVV); // x171: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m3_]
        vreal x174 = stencil(gtDD12, stencil_idx_0_0_3_VVV); // x174: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__3_]
        vreal x538 = ((x171 + x174)); // x538: Dependency! Liveness = 2; [x171, x174]
        vreal x172 = stencil(gtDD12, stencil_idx_0_0_m2_VVV); // x172: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m2_]
        vreal x173 = stencil(gtDD12, stencil_idx_0_0_2_VVV); // x173: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__2_]
        vreal x539 = ((x172 + x173)); // x539: Dependency! Liveness = 2; [x172, x173]
        vreal x176 = stencil(gtDD12, stencil_idx_0_0_m1_VVV); // x176: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m1_]
        vreal x177 = stencil(gtDD12, stencil_idx_0_0_1_VVV); // x177: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__1_]
        vreal x540 = ((x176 + x177)); // x540: Dependency! Liveness = 2; [x176, x177]
        vreal x528 = (-252 * stencil(gtDD12, stencil_idx_0_0_0_VVV)); // x528: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__0_]
        store(gt_rhsDD12, stencil_idx_0_0_0_VVV, (access(gt_rhsDD12, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x528 + (-120 * x531) + (-10 * x529) + (45 * x530) + (210 * x532) + ((stencil(gtDD12, stencil_idx_m5_0_0_VVV) + stencil(gtDD12, stencil_idx_5_0_0_VVV))))) + (DYI * (x528 + (-120 * x535) + (-10 * x533) + (45 * x534) + (210 * x536) + ((stencil(gtDD12, stencil_idx_0_m5_0_VVV) + stencil(gtDD12, stencil_idx_0_5_0_VVV))))) + (DZI * (x528 + (-120 * x539) + (-10 * x537) + (45 * x538) + (210 * x540) + ((stencil(gtDD12, stencil_idx_0_0_m5_VVV) + stencil(gtDD12, stencil_idx_0_0_5_VVV)))))))));
        vreal x103 = stencil(gtDD22, stencil_idx_m4_0_0_VVV); // x103: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m4__0__0_]
        vreal x108 = stencil(gtDD22, stencil_idx_4_0_0_VVV); // x108: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__4__0__0_]
        vreal x542 = ((x103 + x108)); // x542: Dependency! Liveness = 2; [x103, x108]
        vreal x104 = stencil(gtDD22, stencil_idx_m3_0_0_VVV); // x104: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m3__0__0_]
        vreal x107 = stencil(gtDD22, stencil_idx_3_0_0_VVV); // x107: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__3__0__0_]
        vreal x543 = ((x104 + x107)); // x543: Dependency! Liveness = 2; [x104, x107]
        vreal x105 = stencil(gtDD22, stencil_idx_m2_0_0_VVV); // x105: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m2__0__0_]
        vreal x106 = stencil(gtDD22, stencil_idx_2_0_0_VVV); // x106: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__2__0__0_]
        vreal x544 = ((x105 + x106)); // x544: Dependency! Liveness = 2; [x105, x106]
        vreal x109 = stencil(gtDD22, stencil_idx_m1_0_0_VVV); // x109: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m1__0__0_]
        vreal x110 = stencil(gtDD22, stencil_idx_1_0_0_VVV); // x110: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__1__0__0_]
        vreal x545 = ((x109 + x110)); // x545: Dependency! Liveness = 2; [x109, x110]
        vreal x41 = stencil(gtDD22, stencil_idx_0_m4_0_VVV); // x41: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m4__0_]
        vreal x46 = stencil(gtDD22, stencil_idx_0_4_0_VVV); // x46: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__4__0_]
        vreal x546 = ((x41 + x46)); // x546: Dependency! Liveness = 2; [x41, x46]
        vreal x42 = stencil(gtDD22, stencil_idx_0_m3_0_VVV); // x42: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m3__0_]
        vreal x45 = stencil(gtDD22, stencil_idx_0_3_0_VVV); // x45: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__3__0_]
        vreal x547 = ((x42 + x45)); // x547: Dependency! Liveness = 2; [x42, x45]
        vreal x43 = stencil(gtDD22, stencil_idx_0_m2_0_VVV); // x43: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m2__0_]
        vreal x44 = stencil(gtDD22, stencil_idx_0_2_0_VVV); // x44: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__2__0_]
        vreal x548 = ((x43 + x44)); // x548: Dependency! Liveness = 2; [x43, x44]
        vreal x47 = stencil(gtDD22, stencil_idx_0_m1_0_VVV); // x47: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m1__0_]
        vreal x48 = stencil(gtDD22, stencil_idx_0_1_0_VVV); // x48: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__1__0_]
        vreal x549 = ((x47 + x48)); // x549: Dependency! Liveness = 2; [x47, x48]
        vreal x159 = stencil(gtDD22, stencil_idx_0_0_m4_VVV); // x159: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m4_]
        vreal x164 = stencil(gtDD22, stencil_idx_0_0_4_VVV); // x164: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__4_]
        vreal x550 = ((x159 + x164)); // x550: Dependency! Liveness = 2; [x159, x164]
        vreal x160 = stencil(gtDD22, stencil_idx_0_0_m3_VVV); // x160: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m3_]
        vreal x163 = stencil(gtDD22, stencil_idx_0_0_3_VVV); // x163: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__3_]
        vreal x551 = ((x160 + x163)); // x551: Dependency! Liveness = 2; [x160, x163]
        vreal x161 = stencil(gtDD22, stencil_idx_0_0_m2_VVV); // x161: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m2_]
        vreal x162 = stencil(gtDD22, stencil_idx_0_0_2_VVV); // x162: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__2_]
        vreal x552 = ((x161 + x162)); // x552: Dependency! Liveness = 2; [x161, x162]
        vreal x165 = stencil(gtDD22, stencil_idx_0_0_m1_VVV); // x165: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m1_]
        vreal x166 = stencil(gtDD22, stencil_idx_0_0_1_VVV); // x166: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__1_]
        vreal x553 = ((x165 + x166)); // x553: Dependency! Liveness = 2; [x165, x166]
        vreal x541 = (-252 * stencil(gtDD22, stencil_idx_0_0_0_VVV)); // x541: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__0_]
        store(gt_rhsDD22, stencil_idx_0_0_0_VVV, (access(gt_rhsDD22, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x541 + (-120 * x544) + (-10 * x542) + (45 * x543) + (210 * x545) + ((stencil(gtDD22, stencil_idx_m5_0_0_VVV) + stencil(gtDD22, stencil_idx_5_0_0_VVV))))) + (DYI * (x541 + (-120 * x548) + (-10 * x546) + (45 * x547) + (210 * x549) + ((stencil(gtDD22, stencil_idx_0_m5_0_VVV) + stencil(gtDD22, stencil_idx_0_5_0_VVV))))) + (DZI * (x541 + (-120 * x552) + (-10 * x550) + (45 * x551) + (210 * x553) + ((stencil(gtDD22, stencil_idx_0_0_m5_VVV) + stencil(gtDD22, stencil_idx_0_0_5_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 5
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
        vreal x563 = stencil(AtDD00, stencil_idx_0_m4_0_VVV); // x563: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m4__0_]
        vreal x564 = stencil(AtDD00, stencil_idx_0_4_0_VVV); // x564: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__4__0_]
        vreal x565 = stencil(AtDD00, stencil_idx_0_m3_0_VVV); // x565: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m3__0_]
        vreal x566 = stencil(AtDD00, stencil_idx_0_3_0_VVV); // x566: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__3__0_]
        vreal x567 = stencil(AtDD00, stencil_idx_0_m2_0_VVV); // x567: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m2__0_]
        vreal x568 = stencil(AtDD00, stencil_idx_0_2_0_VVV); // x568: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__2__0_]
        vreal x569 = stencil(AtDD00, stencil_idx_0_m1_0_VVV); // x569: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m1__0_]
        vreal x570 = stencil(AtDD00, stencil_idx_0_1_0_VVV); // x570: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__1__0_]
        vreal x571 = stencil(AtDD00, stencil_idx_0_0_m4_VVV); // x571: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m4_]
        vreal x572 = stencil(AtDD00, stencil_idx_0_0_4_VVV); // x572: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__4_]
        vreal x573 = stencil(AtDD00, stencil_idx_0_0_m3_VVV); // x573: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m3_]
        vreal x574 = stencil(AtDD00, stencil_idx_0_0_3_VVV); // x574: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__3_]
        vreal x575 = stencil(AtDD00, stencil_idx_0_0_m2_VVV); // x575: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m2_]
        vreal x576 = stencil(AtDD00, stencil_idx_0_0_2_VVV); // x576: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__2_]
        vreal x577 = stencil(AtDD00, stencil_idx_0_0_m1_VVV); // x577: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m1_]
        vreal x578 = stencil(AtDD00, stencil_idx_0_0_1_VVV); // x578: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__1_]
        vreal x555 = stencil(AtDD00, stencil_idx_m4_0_0_VVV); // x555: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m4__0__0_]
        vreal x556 = stencil(AtDD00, stencil_idx_4_0_0_VVV); // x556: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__4__0__0_]
        vreal x557 = stencil(AtDD00, stencil_idx_m3_0_0_VVV); // x557: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m3__0__0_]
        vreal x558 = stencil(AtDD00, stencil_idx_3_0_0_VVV); // x558: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__3__0__0_]
        vreal x559 = stencil(AtDD00, stencil_idx_m2_0_0_VVV); // x559: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m2__0__0_]
        vreal x560 = stencil(AtDD00, stencil_idx_2_0_0_VVV); // x560: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__2__0__0_]
        vreal x561 = stencil(AtDD00, stencil_idx_m1_0_0_VVV); // x561: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m1__0__0_]
        vreal x562 = stencil(AtDD00, stencil_idx_1_0_0_VVV); // x562: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__1__0__0_]
        vreal x554 = (-252 * stencil(AtDD00, stencil_idx_0_0_0_VVV)); // x554: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__0_]
        vreal x338 = ((1.0 / 1024.0) * dissipation_epsilon); // x338: Dependency! Liveness = 1; [dissipation_epsilon]
        store(At_rhsDD00, stencil_idx_0_0_0_VVV, (access(At_rhsDD00, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x554 + (-120 * ((x559 + x560))) + (-10 * ((x555 + x556))) + (45 * ((x557 + x558))) + (210 * ((x561 + x562))) + ((stencil(AtDD00, stencil_idx_m5_0_0_VVV) + stencil(AtDD00, stencil_idx_5_0_0_VVV))))) + (DYI * (x554 + (-120 * ((x567 + x568))) + (-10 * ((x563 + x564))) + (45 * ((x565 + x566))) + (210 * ((x569 + x570))) + ((stencil(AtDD00, stencil_idx_0_m5_0_VVV) + stencil(AtDD00, stencil_idx_0_5_0_VVV))))) + (DZI * (x554 + (-120 * ((x575 + x576))) + (-10 * ((x571 + x572))) + (45 * ((x573 + x574))) + (210 * ((x577 + x578))) + ((stencil(AtDD00, stencil_idx_0_0_m5_VVV) + stencil(AtDD00, stencil_idx_0_0_5_VVV)))))))));
        vreal x588 = stencil(AtDD01, stencil_idx_0_m4_0_VVV); // x588: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m4__0_]
        vreal x589 = stencil(AtDD01, stencil_idx_0_4_0_VVV); // x589: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__4__0_]
        vreal x590 = stencil(AtDD01, stencil_idx_0_m3_0_VVV); // x590: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m3__0_]
        vreal x591 = stencil(AtDD01, stencil_idx_0_3_0_VVV); // x591: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__3__0_]
        vreal x592 = stencil(AtDD01, stencil_idx_0_m2_0_VVV); // x592: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m2__0_]
        vreal x593 = stencil(AtDD01, stencil_idx_0_2_0_VVV); // x593: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__2__0_]
        vreal x594 = stencil(AtDD01, stencil_idx_0_m1_0_VVV); // x594: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m1__0_]
        vreal x595 = stencil(AtDD01, stencil_idx_0_1_0_VVV); // x595: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__1__0_]
        vreal x596 = stencil(AtDD01, stencil_idx_0_0_m4_VVV); // x596: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m4_]
        vreal x597 = stencil(AtDD01, stencil_idx_0_0_4_VVV); // x597: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__4_]
        vreal x598 = stencil(AtDD01, stencil_idx_0_0_m3_VVV); // x598: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m3_]
        vreal x599 = stencil(AtDD01, stencil_idx_0_0_3_VVV); // x599: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__3_]
        vreal x600 = stencil(AtDD01, stencil_idx_0_0_m2_VVV); // x600: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m2_]
        vreal x601 = stencil(AtDD01, stencil_idx_0_0_2_VVV); // x601: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__2_]
        vreal x602 = stencil(AtDD01, stencil_idx_0_0_m1_VVV); // x602: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m1_]
        vreal x603 = stencil(AtDD01, stencil_idx_0_0_1_VVV); // x603: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__1_]
        vreal x580 = stencil(AtDD01, stencil_idx_m4_0_0_VVV); // x580: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m4__0__0_]
        vreal x581 = stencil(AtDD01, stencil_idx_4_0_0_VVV); // x581: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__4__0__0_]
        vreal x582 = stencil(AtDD01, stencil_idx_m3_0_0_VVV); // x582: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m3__0__0_]
        vreal x583 = stencil(AtDD01, stencil_idx_3_0_0_VVV); // x583: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__3__0__0_]
        vreal x584 = stencil(AtDD01, stencil_idx_m2_0_0_VVV); // x584: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m2__0__0_]
        vreal x585 = stencil(AtDD01, stencil_idx_2_0_0_VVV); // x585: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__2__0__0_]
        vreal x586 = stencil(AtDD01, stencil_idx_m1_0_0_VVV); // x586: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m1__0__0_]
        vreal x587 = stencil(AtDD01, stencil_idx_1_0_0_VVV); // x587: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__1__0__0_]
        vreal x579 = (-252 * stencil(AtDD01, stencil_idx_0_0_0_VVV)); // x579: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__0_]
        store(At_rhsDD01, stencil_idx_0_0_0_VVV, (access(At_rhsDD01, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x579 + (-120 * ((x584 + x585))) + (-10 * ((x580 + x581))) + (45 * ((x582 + x583))) + (210 * ((x586 + x587))) + ((stencil(AtDD01, stencil_idx_m5_0_0_VVV) + stencil(AtDD01, stencil_idx_5_0_0_VVV))))) + (DYI * (x579 + (-120 * ((x592 + x593))) + (-10 * ((x588 + x589))) + (45 * ((x590 + x591))) + (210 * ((x594 + x595))) + ((stencil(AtDD01, stencil_idx_0_m5_0_VVV) + stencil(AtDD01, stencil_idx_0_5_0_VVV))))) + (DZI * (x579 + (-120 * ((x600 + x601))) + (-10 * ((x596 + x597))) + (45 * ((x598 + x599))) + (210 * ((x602 + x603))) + ((stencil(AtDD01, stencil_idx_0_0_m5_VVV) + stencil(AtDD01, stencil_idx_0_0_5_VVV)))))))));
        vreal x613 = stencil(AtDD11, stencil_idx_0_m4_0_VVV); // x613: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m4__0_]
        vreal x614 = stencil(AtDD11, stencil_idx_0_4_0_VVV); // x614: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__4__0_]
        vreal x615 = stencil(AtDD11, stencil_idx_0_m3_0_VVV); // x615: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m3__0_]
        vreal x616 = stencil(AtDD11, stencil_idx_0_3_0_VVV); // x616: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__3__0_]
        vreal x617 = stencil(AtDD11, stencil_idx_0_m2_0_VVV); // x617: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m2__0_]
        vreal x618 = stencil(AtDD11, stencil_idx_0_2_0_VVV); // x618: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__2__0_]
        vreal x619 = stencil(AtDD11, stencil_idx_0_m1_0_VVV); // x619: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m1__0_]
        vreal x620 = stencil(AtDD11, stencil_idx_0_1_0_VVV); // x620: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__1__0_]
        vreal x621 = stencil(AtDD11, stencil_idx_0_0_m4_VVV); // x621: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m4_]
        vreal x622 = stencil(AtDD11, stencil_idx_0_0_4_VVV); // x622: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__4_]
        vreal x623 = stencil(AtDD11, stencil_idx_0_0_m3_VVV); // x623: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m3_]
        vreal x624 = stencil(AtDD11, stencil_idx_0_0_3_VVV); // x624: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__3_]
        vreal x625 = stencil(AtDD11, stencil_idx_0_0_m2_VVV); // x625: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m2_]
        vreal x626 = stencil(AtDD11, stencil_idx_0_0_2_VVV); // x626: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__2_]
        vreal x627 = stencil(AtDD11, stencil_idx_0_0_m1_VVV); // x627: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m1_]
        vreal x628 = stencil(AtDD11, stencil_idx_0_0_1_VVV); // x628: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__1_]
        vreal x605 = stencil(AtDD11, stencil_idx_m4_0_0_VVV); // x605: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m4__0__0_]
        vreal x606 = stencil(AtDD11, stencil_idx_4_0_0_VVV); // x606: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__4__0__0_]
        vreal x607 = stencil(AtDD11, stencil_idx_m3_0_0_VVV); // x607: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m3__0__0_]
        vreal x608 = stencil(AtDD11, stencil_idx_3_0_0_VVV); // x608: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__3__0__0_]
        vreal x609 = stencil(AtDD11, stencil_idx_m2_0_0_VVV); // x609: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m2__0__0_]
        vreal x610 = stencil(AtDD11, stencil_idx_2_0_0_VVV); // x610: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__2__0__0_]
        vreal x611 = stencil(AtDD11, stencil_idx_m1_0_0_VVV); // x611: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m1__0__0_]
        vreal x612 = stencil(AtDD11, stencil_idx_1_0_0_VVV); // x612: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__1__0__0_]
        vreal x604 = (-252 * stencil(AtDD11, stencil_idx_0_0_0_VVV)); // x604: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__0_]
        store(At_rhsDD11, stencil_idx_0_0_0_VVV, (access(At_rhsDD11, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x604 + (-120 * ((x609 + x610))) + (-10 * ((x605 + x606))) + (45 * ((x607 + x608))) + (210 * ((x611 + x612))) + ((stencil(AtDD11, stencil_idx_m5_0_0_VVV) + stencil(AtDD11, stencil_idx_5_0_0_VVV))))) + (DYI * (x604 + (-120 * ((x617 + x618))) + (-10 * ((x613 + x614))) + (45 * ((x615 + x616))) + (210 * ((x619 + x620))) + ((stencil(AtDD11, stencil_idx_0_m5_0_VVV) + stencil(AtDD11, stencil_idx_0_5_0_VVV))))) + (DZI * (x604 + (-120 * ((x625 + x626))) + (-10 * ((x621 + x622))) + (45 * ((x623 + x624))) + (210 * ((x627 + x628))) + ((stencil(AtDD11, stencil_idx_0_0_m5_VVV) + stencil(AtDD11, stencil_idx_0_0_5_VVV)))))))));
        vreal x638 = stencil(AtDD02, stencil_idx_0_m4_0_VVV); // x638: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m4__0_]
        vreal x639 = stencil(AtDD02, stencil_idx_0_4_0_VVV); // x639: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__4__0_]
        vreal x640 = stencil(AtDD02, stencil_idx_0_m3_0_VVV); // x640: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m3__0_]
        vreal x641 = stencil(AtDD02, stencil_idx_0_3_0_VVV); // x641: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__3__0_]
        vreal x642 = stencil(AtDD02, stencil_idx_0_m2_0_VVV); // x642: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m2__0_]
        vreal x643 = stencil(AtDD02, stencil_idx_0_2_0_VVV); // x643: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__2__0_]
        vreal x644 = stencil(AtDD02, stencil_idx_0_m1_0_VVV); // x644: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m1__0_]
        vreal x645 = stencil(AtDD02, stencil_idx_0_1_0_VVV); // x645: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__1__0_]
        vreal x646 = stencil(AtDD02, stencil_idx_0_0_m4_VVV); // x646: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m4_]
        vreal x647 = stencil(AtDD02, stencil_idx_0_0_4_VVV); // x647: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__4_]
        vreal x648 = stencil(AtDD02, stencil_idx_0_0_m3_VVV); // x648: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m3_]
        vreal x649 = stencil(AtDD02, stencil_idx_0_0_3_VVV); // x649: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__3_]
        vreal x650 = stencil(AtDD02, stencil_idx_0_0_m2_VVV); // x650: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m2_]
        vreal x651 = stencil(AtDD02, stencil_idx_0_0_2_VVV); // x651: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__2_]
        vreal x652 = stencil(AtDD02, stencil_idx_0_0_m1_VVV); // x652: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m1_]
        vreal x653 = stencil(AtDD02, stencil_idx_0_0_1_VVV); // x653: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__1_]
        vreal x630 = stencil(AtDD02, stencil_idx_m4_0_0_VVV); // x630: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m4__0__0_]
        vreal x631 = stencil(AtDD02, stencil_idx_4_0_0_VVV); // x631: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__4__0__0_]
        vreal x632 = stencil(AtDD02, stencil_idx_m3_0_0_VVV); // x632: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m3__0__0_]
        vreal x633 = stencil(AtDD02, stencil_idx_3_0_0_VVV); // x633: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__3__0__0_]
        vreal x634 = stencil(AtDD02, stencil_idx_m2_0_0_VVV); // x634: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m2__0__0_]
        vreal x635 = stencil(AtDD02, stencil_idx_2_0_0_VVV); // x635: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__2__0__0_]
        vreal x636 = stencil(AtDD02, stencil_idx_m1_0_0_VVV); // x636: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m1__0__0_]
        vreal x637 = stencil(AtDD02, stencil_idx_1_0_0_VVV); // x637: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__1__0__0_]
        vreal x629 = (-252 * stencil(AtDD02, stencil_idx_0_0_0_VVV)); // x629: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__0_]
        store(At_rhsDD02, stencil_idx_0_0_0_VVV, (access(At_rhsDD02, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x629 + (-120 * ((x634 + x635))) + (-10 * ((x630 + x631))) + (45 * ((x632 + x633))) + (210 * ((x636 + x637))) + ((stencil(AtDD02, stencil_idx_m5_0_0_VVV) + stencil(AtDD02, stencil_idx_5_0_0_VVV))))) + (DYI * (x629 + (-120 * ((x642 + x643))) + (-10 * ((x638 + x639))) + (45 * ((x640 + x641))) + (210 * ((x644 + x645))) + ((stencil(AtDD02, stencil_idx_0_m5_0_VVV) + stencil(AtDD02, stencil_idx_0_5_0_VVV))))) + (DZI * (x629 + (-120 * ((x650 + x651))) + (-10 * ((x646 + x647))) + (45 * ((x648 + x649))) + (210 * ((x652 + x653))) + ((stencil(AtDD02, stencil_idx_0_0_m5_VVV) + stencil(AtDD02, stencil_idx_0_0_5_VVV)))))))));
        vreal x663 = stencil(AtDD12, stencil_idx_0_m4_0_VVV); // x663: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m4__0_]
        vreal x664 = stencil(AtDD12, stencil_idx_0_4_0_VVV); // x664: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__4__0_]
        vreal x665 = stencil(AtDD12, stencil_idx_0_m3_0_VVV); // x665: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m3__0_]
        vreal x666 = stencil(AtDD12, stencil_idx_0_3_0_VVV); // x666: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__3__0_]
        vreal x667 = stencil(AtDD12, stencil_idx_0_m2_0_VVV); // x667: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m2__0_]
        vreal x668 = stencil(AtDD12, stencil_idx_0_2_0_VVV); // x668: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__2__0_]
        vreal x669 = stencil(AtDD12, stencil_idx_0_m1_0_VVV); // x669: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m1__0_]
        vreal x670 = stencil(AtDD12, stencil_idx_0_1_0_VVV); // x670: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__1__0_]
        vreal x671 = stencil(AtDD12, stencil_idx_0_0_m4_VVV); // x671: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m4_]
        vreal x672 = stencil(AtDD12, stencil_idx_0_0_4_VVV); // x672: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__4_]
        vreal x673 = stencil(AtDD12, stencil_idx_0_0_m3_VVV); // x673: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m3_]
        vreal x674 = stencil(AtDD12, stencil_idx_0_0_3_VVV); // x674: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__3_]
        vreal x675 = stencil(AtDD12, stencil_idx_0_0_m2_VVV); // x675: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m2_]
        vreal x676 = stencil(AtDD12, stencil_idx_0_0_2_VVV); // x676: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__2_]
        vreal x677 = stencil(AtDD12, stencil_idx_0_0_m1_VVV); // x677: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m1_]
        vreal x678 = stencil(AtDD12, stencil_idx_0_0_1_VVV); // x678: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__1_]
        vreal x655 = stencil(AtDD12, stencil_idx_m4_0_0_VVV); // x655: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m4__0__0_]
        vreal x656 = stencil(AtDD12, stencil_idx_4_0_0_VVV); // x656: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__4__0__0_]
        vreal x657 = stencil(AtDD12, stencil_idx_m3_0_0_VVV); // x657: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m3__0__0_]
        vreal x658 = stencil(AtDD12, stencil_idx_3_0_0_VVV); // x658: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__3__0__0_]
        vreal x659 = stencil(AtDD12, stencil_idx_m2_0_0_VVV); // x659: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m2__0__0_]
        vreal x660 = stencil(AtDD12, stencil_idx_2_0_0_VVV); // x660: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__2__0__0_]
        vreal x661 = stencil(AtDD12, stencil_idx_m1_0_0_VVV); // x661: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m1__0__0_]
        vreal x662 = stencil(AtDD12, stencil_idx_1_0_0_VVV); // x662: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__1__0__0_]
        vreal x654 = (-252 * stencil(AtDD12, stencil_idx_0_0_0_VVV)); // x654: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__0_]
        store(At_rhsDD12, stencil_idx_0_0_0_VVV, (access(At_rhsDD12, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x654 + (-120 * ((x659 + x660))) + (-10 * ((x655 + x656))) + (45 * ((x657 + x658))) + (210 * ((x661 + x662))) + ((stencil(AtDD12, stencil_idx_m5_0_0_VVV) + stencil(AtDD12, stencil_idx_5_0_0_VVV))))) + (DYI * (x654 + (-120 * ((x667 + x668))) + (-10 * ((x663 + x664))) + (45 * ((x665 + x666))) + (210 * ((x669 + x670))) + ((stencil(AtDD12, stencil_idx_0_m5_0_VVV) + stencil(AtDD12, stencil_idx_0_5_0_VVV))))) + (DZI * (x654 + (-120 * ((x675 + x676))) + (-10 * ((x671 + x672))) + (45 * ((x673 + x674))) + (210 * ((x677 + x678))) + ((stencil(AtDD12, stencil_idx_0_0_m5_VVV) + stencil(AtDD12, stencil_idx_0_0_5_VVV)))))))));
        vreal x688 = stencil(AtDD22, stencil_idx_0_m4_0_VVV); // x688: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m4__0_]
        vreal x689 = stencil(AtDD22, stencil_idx_0_4_0_VVV); // x689: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__4__0_]
        vreal x690 = stencil(AtDD22, stencil_idx_0_m3_0_VVV); // x690: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m3__0_]
        vreal x691 = stencil(AtDD22, stencil_idx_0_3_0_VVV); // x691: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__3__0_]
        vreal x692 = stencil(AtDD22, stencil_idx_0_m2_0_VVV); // x692: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m2__0_]
        vreal x693 = stencil(AtDD22, stencil_idx_0_2_0_VVV); // x693: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__2__0_]
        vreal x694 = stencil(AtDD22, stencil_idx_0_m1_0_VVV); // x694: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m1__0_]
        vreal x695 = stencil(AtDD22, stencil_idx_0_1_0_VVV); // x695: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__1__0_]
        vreal x696 = stencil(AtDD22, stencil_idx_0_0_m4_VVV); // x696: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m4_]
        vreal x697 = stencil(AtDD22, stencil_idx_0_0_4_VVV); // x697: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__4_]
        vreal x698 = stencil(AtDD22, stencil_idx_0_0_m3_VVV); // x698: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m3_]
        vreal x699 = stencil(AtDD22, stencil_idx_0_0_3_VVV); // x699: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__3_]
        vreal x700 = stencil(AtDD22, stencil_idx_0_0_m2_VVV); // x700: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m2_]
        vreal x701 = stencil(AtDD22, stencil_idx_0_0_2_VVV); // x701: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__2_]
        vreal x702 = stencil(AtDD22, stencil_idx_0_0_m1_VVV); // x702: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m1_]
        vreal x703 = stencil(AtDD22, stencil_idx_0_0_1_VVV); // x703: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__1_]
        vreal x680 = stencil(AtDD22, stencil_idx_m4_0_0_VVV); // x680: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m4__0__0_]
        vreal x681 = stencil(AtDD22, stencil_idx_4_0_0_VVV); // x681: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__4__0__0_]
        vreal x682 = stencil(AtDD22, stencil_idx_m3_0_0_VVV); // x682: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m3__0__0_]
        vreal x683 = stencil(AtDD22, stencil_idx_3_0_0_VVV); // x683: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__3__0__0_]
        vreal x684 = stencil(AtDD22, stencil_idx_m2_0_0_VVV); // x684: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m2__0__0_]
        vreal x685 = stencil(AtDD22, stencil_idx_2_0_0_VVV); // x685: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__2__0__0_]
        vreal x686 = stencil(AtDD22, stencil_idx_m1_0_0_VVV); // x686: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m1__0__0_]
        vreal x687 = stencil(AtDD22, stencil_idx_1_0_0_VVV); // x687: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__1__0__0_]
        vreal x679 = (-252 * stencil(AtDD22, stencil_idx_0_0_0_VVV)); // x679: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__0_]
        store(At_rhsDD22, stencil_idx_0_0_0_VVV, (access(At_rhsDD22, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x679 + (-120 * ((x684 + x685))) + (-10 * ((x680 + x681))) + (45 * ((x682 + x683))) + (210 * ((x686 + x687))) + ((stencil(AtDD22, stencil_idx_m5_0_0_VVV) + stencil(AtDD22, stencil_idx_5_0_0_VVV))))) + (DYI * (x679 + (-120 * ((x692 + x693))) + (-10 * ((x688 + x689))) + (45 * ((x690 + x691))) + (210 * ((x694 + x695))) + ((stencil(AtDD22, stencil_idx_0_m5_0_VVV) + stencil(AtDD22, stencil_idx_0_5_0_VVV))))) + (DZI * (x679 + (-120 * ((x700 + x701))) + (-10 * ((x696 + x697))) + (45 * ((x698 + x699))) + (210 * ((x702 + x703))) + ((stencil(AtDD22, stencil_idx_0_0_m5_VVV) + stencil(AtDD22, stencil_idx_0_0_5_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 6
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
        vreal x705 = stencil(evo_lapse, stencil_idx_m4_0_0_VVV); // x705: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m4__0__0_]
        vreal x706 = stencil(evo_lapse, stencil_idx_4_0_0_VVV); // x706: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__4__0__0_]
        vreal x707 = ((x705 + x706)); // x707: Dependency! Liveness = 2; [x705, x706]
        vreal x708 = stencil(evo_lapse, stencil_idx_m3_0_0_VVV); // x708: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m3__0__0_]
        vreal x709 = stencil(evo_lapse, stencil_idx_3_0_0_VVV); // x709: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__3__0__0_]
        vreal x710 = ((x708 + x709)); // x710: Dependency! Liveness = 2; [x708, x709]
        vreal x711 = stencil(evo_lapse, stencil_idx_m2_0_0_VVV); // x711: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m2__0__0_]
        vreal x712 = stencil(evo_lapse, stencil_idx_2_0_0_VVV); // x712: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__2__0__0_]
        vreal x713 = ((x711 + x712)); // x713: Dependency! Liveness = 2; [x711, x712]
        vreal x714 = stencil(evo_lapse, stencil_idx_m1_0_0_VVV); // x714: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m1__0__0_]
        vreal x715 = stencil(evo_lapse, stencil_idx_1_0_0_VVV); // x715: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__1__0__0_]
        vreal x716 = ((x714 + x715)); // x716: Dependency! Liveness = 2; [x714, x715]
        vreal x717 = stencil(evo_lapse, stencil_idx_0_m4_0_VVV); // x717: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m4__0_]
        vreal x718 = stencil(evo_lapse, stencil_idx_0_4_0_VVV); // x718: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__4__0_]
        vreal x719 = ((x717 + x718)); // x719: Dependency! Liveness = 2; [x717, x718]
        vreal x720 = stencil(evo_lapse, stencil_idx_0_m3_0_VVV); // x720: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m3__0_]
        vreal x721 = stencil(evo_lapse, stencil_idx_0_3_0_VVV); // x721: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__3__0_]
        vreal x722 = ((x720 + x721)); // x722: Dependency! Liveness = 2; [x720, x721]
        vreal x723 = stencil(evo_lapse, stencil_idx_0_m2_0_VVV); // x723: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m2__0_]
        vreal x724 = stencil(evo_lapse, stencil_idx_0_2_0_VVV); // x724: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__2__0_]
        vreal x725 = ((x723 + x724)); // x725: Dependency! Liveness = 2; [x723, x724]
        vreal x726 = stencil(evo_lapse, stencil_idx_0_m1_0_VVV); // x726: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m1__0_]
        vreal x727 = stencil(evo_lapse, stencil_idx_0_1_0_VVV); // x727: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__1__0_]
        vreal x728 = ((x726 + x727)); // x728: Dependency! Liveness = 2; [x726, x727]
        vreal x729 = stencil(evo_lapse, stencil_idx_0_0_m4_VVV); // x729: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m4_]
        vreal x730 = stencil(evo_lapse, stencil_idx_0_0_4_VVV); // x730: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__4_]
        vreal x731 = ((x729 + x730)); // x731: Dependency! Liveness = 2; [x729, x730]
        vreal x732 = stencil(evo_lapse, stencil_idx_0_0_m3_VVV); // x732: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m3_]
        vreal x733 = stencil(evo_lapse, stencil_idx_0_0_3_VVV); // x733: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__3_]
        vreal x734 = ((x732 + x733)); // x734: Dependency! Liveness = 2; [x732, x733]
        vreal x735 = stencil(evo_lapse, stencil_idx_0_0_m2_VVV); // x735: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m2_]
        vreal x736 = stencil(evo_lapse, stencil_idx_0_0_2_VVV); // x736: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__2_]
        vreal x737 = ((x735 + x736)); // x737: Dependency! Liveness = 2; [x735, x736]
        vreal x738 = stencil(evo_lapse, stencil_idx_0_0_m1_VVV); // x738: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m1_]
        vreal x739 = stencil(evo_lapse, stencil_idx_0_0_1_VVV); // x739: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__1_]
        vreal x740 = ((x738 + x739)); // x740: Dependency! Liveness = 2; [x738, x739]
        vreal x704 = (-252 * stencil(evo_lapse, stencil_idx_0_0_0_VVV)); // x704: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__0_]
        vreal x338 = ((1.0 / 1024.0) * dissipation_epsilon); // x338: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_lapse_rhs, stencil_idx_0_0_0_VVV, (access(evo_lapse_rhs, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x704 + (-120 * x713) + (-10 * x707) + (45 * x710) + (210 * x716) + ((stencil(evo_lapse, stencil_idx_m5_0_0_VVV) + stencil(evo_lapse, stencil_idx_5_0_0_VVV))))) + (DYI * (x704 + (-120 * x725) + (-10 * x719) + (45 * x722) + (210 * x728) + ((stencil(evo_lapse, stencil_idx_0_m5_0_VVV) + stencil(evo_lapse, stencil_idx_0_5_0_VVV))))) + (DZI * (x704 + (-120 * x737) + (-10 * x731) + (45 * x734) + (210 * x740) + ((stencil(evo_lapse, stencil_idx_0_0_m5_VVV) + stencil(evo_lapse, stencil_idx_0_0_5_VVV)))))))));    
    });
    // z4c_apply_dissipation loop 7
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
        vreal x742 = stencil(evo_shiftU0, stencil_idx_m4_0_0_VVV); // x742: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m4__0__0_]
        vreal x743 = stencil(evo_shiftU0, stencil_idx_4_0_0_VVV); // x743: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__4__0__0_]
        vreal x744 = ((x742 + x743)); // x744: Dependency! Liveness = 2; [x742, x743]
        vreal x745 = stencil(evo_shiftU0, stencil_idx_m3_0_0_VVV); // x745: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m3__0__0_]
        vreal x746 = stencil(evo_shiftU0, stencil_idx_3_0_0_VVV); // x746: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__3__0__0_]
        vreal x747 = ((x745 + x746)); // x747: Dependency! Liveness = 2; [x745, x746]
        vreal x748 = stencil(evo_shiftU0, stencil_idx_m2_0_0_VVV); // x748: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m2__0__0_]
        vreal x749 = stencil(evo_shiftU0, stencil_idx_2_0_0_VVV); // x749: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__2__0__0_]
        vreal x750 = ((x748 + x749)); // x750: Dependency! Liveness = 2; [x748, x749]
        vreal x751 = stencil(evo_shiftU0, stencil_idx_m1_0_0_VVV); // x751: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m1__0__0_]
        vreal x752 = stencil(evo_shiftU0, stencil_idx_1_0_0_VVV); // x752: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__1__0__0_]
        vreal x753 = ((x751 + x752)); // x753: Dependency! Liveness = 2; [x751, x752]
        vreal x754 = stencil(evo_shiftU0, stencil_idx_0_m4_0_VVV); // x754: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m4__0_]
        vreal x755 = stencil(evo_shiftU0, stencil_idx_0_4_0_VVV); // x755: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__4__0_]
        vreal x756 = ((x754 + x755)); // x756: Dependency! Liveness = 2; [x754, x755]
        vreal x757 = stencil(evo_shiftU0, stencil_idx_0_m3_0_VVV); // x757: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m3__0_]
        vreal x758 = stencil(evo_shiftU0, stencil_idx_0_3_0_VVV); // x758: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__3__0_]
        vreal x759 = ((x757 + x758)); // x759: Dependency! Liveness = 2; [x757, x758]
        vreal x760 = stencil(evo_shiftU0, stencil_idx_0_m2_0_VVV); // x760: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m2__0_]
        vreal x761 = stencil(evo_shiftU0, stencil_idx_0_2_0_VVV); // x761: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__2__0_]
        vreal x762 = ((x760 + x761)); // x762: Dependency! Liveness = 2; [x760, x761]
        vreal x763 = stencil(evo_shiftU0, stencil_idx_0_m1_0_VVV); // x763: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m1__0_]
        vreal x764 = stencil(evo_shiftU0, stencil_idx_0_1_0_VVV); // x764: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__1__0_]
        vreal x765 = ((x763 + x764)); // x765: Dependency! Liveness = 2; [x763, x764]
        vreal x766 = stencil(evo_shiftU0, stencil_idx_0_0_m4_VVV); // x766: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m4_]
        vreal x767 = stencil(evo_shiftU0, stencil_idx_0_0_4_VVV); // x767: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__4_]
        vreal x768 = ((x766 + x767)); // x768: Dependency! Liveness = 2; [x766, x767]
        vreal x769 = stencil(evo_shiftU0, stencil_idx_0_0_m3_VVV); // x769: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m3_]
        vreal x770 = stencil(evo_shiftU0, stencil_idx_0_0_3_VVV); // x770: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__3_]
        vreal x771 = ((x769 + x770)); // x771: Dependency! Liveness = 2; [x769, x770]
        vreal x772 = stencil(evo_shiftU0, stencil_idx_0_0_m2_VVV); // x772: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m2_]
        vreal x773 = stencil(evo_shiftU0, stencil_idx_0_0_2_VVV); // x773: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__2_]
        vreal x774 = ((x772 + x773)); // x774: Dependency! Liveness = 2; [x772, x773]
        vreal x775 = stencil(evo_shiftU0, stencil_idx_0_0_m1_VVV); // x775: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m1_]
        vreal x776 = stencil(evo_shiftU0, stencil_idx_0_0_1_VVV); // x776: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__1_]
        vreal x777 = ((x775 + x776)); // x777: Dependency! Liveness = 2; [x775, x776]
        vreal x741 = (-252 * stencil(evo_shiftU0, stencil_idx_0_0_0_VVV)); // x741: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__0_]
        vreal x338 = ((1.0 / 1024.0) * dissipation_epsilon); // x338: Dependency! Liveness = 1; [dissipation_epsilon]
        store(evo_shift_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU0, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x741 + (-120 * x750) + (-10 * x744) + (45 * x747) + (210 * x753) + ((stencil(evo_shiftU0, stencil_idx_m5_0_0_VVV) + stencil(evo_shiftU0, stencil_idx_5_0_0_VVV))))) + (DYI * (x741 + (-120 * x762) + (-10 * x756) + (45 * x759) + (210 * x765) + ((stencil(evo_shiftU0, stencil_idx_0_m5_0_VVV) + stencil(evo_shiftU0, stencil_idx_0_5_0_VVV))))) + (DZI * (x741 + (-120 * x774) + (-10 * x768) + (45 * x771) + (210 * x777) + ((stencil(evo_shiftU0, stencil_idx_0_0_m5_VVV) + stencil(evo_shiftU0, stencil_idx_0_0_5_VVV)))))))));
        vreal x779 = stencil(evo_shiftU1, stencil_idx_m4_0_0_VVV); // x779: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m4__0__0_]
        vreal x780 = stencil(evo_shiftU1, stencil_idx_4_0_0_VVV); // x780: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__4__0__0_]
        vreal x781 = ((x779 + x780)); // x781: Dependency! Liveness = 2; [x779, x780]
        vreal x782 = stencil(evo_shiftU1, stencil_idx_m3_0_0_VVV); // x782: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m3__0__0_]
        vreal x783 = stencil(evo_shiftU1, stencil_idx_3_0_0_VVV); // x783: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__3__0__0_]
        vreal x784 = ((x782 + x783)); // x784: Dependency! Liveness = 2; [x782, x783]
        vreal x785 = stencil(evo_shiftU1, stencil_idx_m2_0_0_VVV); // x785: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m2__0__0_]
        vreal x786 = stencil(evo_shiftU1, stencil_idx_2_0_0_VVV); // x786: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__2__0__0_]
        vreal x787 = ((x785 + x786)); // x787: Dependency! Liveness = 2; [x785, x786]
        vreal x788 = stencil(evo_shiftU1, stencil_idx_m1_0_0_VVV); // x788: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m1__0__0_]
        vreal x789 = stencil(evo_shiftU1, stencil_idx_1_0_0_VVV); // x789: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__1__0__0_]
        vreal x790 = ((x788 + x789)); // x790: Dependency! Liveness = 2; [x788, x789]
        vreal x791 = stencil(evo_shiftU1, stencil_idx_0_m4_0_VVV); // x791: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m4__0_]
        vreal x792 = stencil(evo_shiftU1, stencil_idx_0_4_0_VVV); // x792: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__4__0_]
        vreal x793 = ((x791 + x792)); // x793: Dependency! Liveness = 2; [x791, x792]
        vreal x794 = stencil(evo_shiftU1, stencil_idx_0_m3_0_VVV); // x794: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m3__0_]
        vreal x795 = stencil(evo_shiftU1, stencil_idx_0_3_0_VVV); // x795: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__3__0_]
        vreal x796 = ((x794 + x795)); // x796: Dependency! Liveness = 2; [x794, x795]
        vreal x797 = stencil(evo_shiftU1, stencil_idx_0_m2_0_VVV); // x797: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m2__0_]
        vreal x798 = stencil(evo_shiftU1, stencil_idx_0_2_0_VVV); // x798: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__2__0_]
        vreal x799 = ((x797 + x798)); // x799: Dependency! Liveness = 2; [x797, x798]
        vreal x800 = stencil(evo_shiftU1, stencil_idx_0_m1_0_VVV); // x800: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m1__0_]
        vreal x801 = stencil(evo_shiftU1, stencil_idx_0_1_0_VVV); // x801: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__1__0_]
        vreal x802 = ((x800 + x801)); // x802: Dependency! Liveness = 2; [x800, x801]
        vreal x803 = stencil(evo_shiftU1, stencil_idx_0_0_m4_VVV); // x803: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m4_]
        vreal x804 = stencil(evo_shiftU1, stencil_idx_0_0_4_VVV); // x804: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__4_]
        vreal x805 = ((x803 + x804)); // x805: Dependency! Liveness = 2; [x803, x804]
        vreal x806 = stencil(evo_shiftU1, stencil_idx_0_0_m3_VVV); // x806: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m3_]
        vreal x807 = stencil(evo_shiftU1, stencil_idx_0_0_3_VVV); // x807: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__3_]
        vreal x808 = ((x806 + x807)); // x808: Dependency! Liveness = 2; [x806, x807]
        vreal x809 = stencil(evo_shiftU1, stencil_idx_0_0_m2_VVV); // x809: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m2_]
        vreal x810 = stencil(evo_shiftU1, stencil_idx_0_0_2_VVV); // x810: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__2_]
        vreal x811 = ((x809 + x810)); // x811: Dependency! Liveness = 2; [x809, x810]
        vreal x812 = stencil(evo_shiftU1, stencil_idx_0_0_m1_VVV); // x812: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m1_]
        vreal x813 = stencil(evo_shiftU1, stencil_idx_0_0_1_VVV); // x813: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__1_]
        vreal x814 = ((x812 + x813)); // x814: Dependency! Liveness = 2; [x812, x813]
        vreal x778 = (-252 * stencil(evo_shiftU1, stencil_idx_0_0_0_VVV)); // x778: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__0_]
        store(evo_shift_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU1, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x778 + (-120 * x787) + (-10 * x781) + (45 * x784) + (210 * x790) + ((stencil(evo_shiftU1, stencil_idx_m5_0_0_VVV) + stencil(evo_shiftU1, stencil_idx_5_0_0_VVV))))) + (DYI * (x778 + (-120 * x799) + (-10 * x793) + (45 * x796) + (210 * x802) + ((stencil(evo_shiftU1, stencil_idx_0_m5_0_VVV) + stencil(evo_shiftU1, stencil_idx_0_5_0_VVV))))) + (DZI * (x778 + (-120 * x811) + (-10 * x805) + (45 * x808) + (210 * x814) + ((stencil(evo_shiftU1, stencil_idx_0_0_m5_VVV) + stencil(evo_shiftU1, stencil_idx_0_0_5_VVV)))))))));
        vreal x816 = stencil(evo_shiftU2, stencil_idx_m4_0_0_VVV); // x816: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m4__0__0_]
        vreal x817 = stencil(evo_shiftU2, stencil_idx_4_0_0_VVV); // x817: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__4__0__0_]
        vreal x818 = ((x816 + x817)); // x818: Dependency! Liveness = 2; [x816, x817]
        vreal x819 = stencil(evo_shiftU2, stencil_idx_m3_0_0_VVV); // x819: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m3__0__0_]
        vreal x820 = stencil(evo_shiftU2, stencil_idx_3_0_0_VVV); // x820: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__3__0__0_]
        vreal x821 = ((x819 + x820)); // x821: Dependency! Liveness = 2; [x819, x820]
        vreal x822 = stencil(evo_shiftU2, stencil_idx_m2_0_0_VVV); // x822: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m2__0__0_]
        vreal x823 = stencil(evo_shiftU2, stencil_idx_2_0_0_VVV); // x823: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__2__0__0_]
        vreal x824 = ((x822 + x823)); // x824: Dependency! Liveness = 2; [x822, x823]
        vreal x825 = stencil(evo_shiftU2, stencil_idx_m1_0_0_VVV); // x825: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m1__0__0_]
        vreal x826 = stencil(evo_shiftU2, stencil_idx_1_0_0_VVV); // x826: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__1__0__0_]
        vreal x827 = ((x825 + x826)); // x827: Dependency! Liveness = 2; [x825, x826]
        vreal x828 = stencil(evo_shiftU2, stencil_idx_0_m4_0_VVV); // x828: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m4__0_]
        vreal x829 = stencil(evo_shiftU2, stencil_idx_0_4_0_VVV); // x829: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__4__0_]
        vreal x830 = ((x828 + x829)); // x830: Dependency! Liveness = 2; [x828, x829]
        vreal x831 = stencil(evo_shiftU2, stencil_idx_0_m3_0_VVV); // x831: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m3__0_]
        vreal x832 = stencil(evo_shiftU2, stencil_idx_0_3_0_VVV); // x832: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__3__0_]
        vreal x833 = ((x831 + x832)); // x833: Dependency! Liveness = 2; [x831, x832]
        vreal x834 = stencil(evo_shiftU2, stencil_idx_0_m2_0_VVV); // x834: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m2__0_]
        vreal x835 = stencil(evo_shiftU2, stencil_idx_0_2_0_VVV); // x835: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__2__0_]
        vreal x836 = ((x834 + x835)); // x836: Dependency! Liveness = 2; [x834, x835]
        vreal x837 = stencil(evo_shiftU2, stencil_idx_0_m1_0_VVV); // x837: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m1__0_]
        vreal x838 = stencil(evo_shiftU2, stencil_idx_0_1_0_VVV); // x838: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__1__0_]
        vreal x839 = ((x837 + x838)); // x839: Dependency! Liveness = 2; [x837, x838]
        vreal x840 = stencil(evo_shiftU2, stencil_idx_0_0_m4_VVV); // x840: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m4_]
        vreal x841 = stencil(evo_shiftU2, stencil_idx_0_0_4_VVV); // x841: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__4_]
        vreal x842 = ((x840 + x841)); // x842: Dependency! Liveness = 2; [x840, x841]
        vreal x843 = stencil(evo_shiftU2, stencil_idx_0_0_m3_VVV); // x843: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m3_]
        vreal x844 = stencil(evo_shiftU2, stencil_idx_0_0_3_VVV); // x844: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__3_]
        vreal x845 = ((x843 + x844)); // x845: Dependency! Liveness = 2; [x843, x844]
        vreal x846 = stencil(evo_shiftU2, stencil_idx_0_0_m2_VVV); // x846: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m2_]
        vreal x847 = stencil(evo_shiftU2, stencil_idx_0_0_2_VVV); // x847: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__2_]
        vreal x848 = ((x846 + x847)); // x848: Dependency! Liveness = 2; [x846, x847]
        vreal x849 = stencil(evo_shiftU2, stencil_idx_0_0_m1_VVV); // x849: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m1_]
        vreal x850 = stencil(evo_shiftU2, stencil_idx_0_0_1_VVV); // x850: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__1_]
        vreal x851 = ((x849 + x850)); // x851: Dependency! Liveness = 2; [x849, x850]
        vreal x815 = (-252 * stencil(evo_shiftU2, stencil_idx_0_0_0_VVV)); // x815: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__0_]
        store(evo_shift_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU2, stencil_idx_0_0_0_VVV) + (x338 * ((DXI * (x815 + (-120 * x824) + (-10 * x818) + (45 * x821) + (210 * x827) + ((stencil(evo_shiftU2, stencil_idx_m5_0_0_VVV) + stencil(evo_shiftU2, stencil_idx_5_0_0_VVV))))) + (DYI * (x815 + (-120 * x836) + (-10 * x830) + (45 * x833) + (210 * x839) + ((stencil(evo_shiftU2, stencil_idx_0_m5_0_VVV) + stencil(evo_shiftU2, stencil_idx_0_5_0_VVV))))) + (DZI * (x815 + (-120 * x848) + (-10 * x842) + (45 * x845) + (210 * x851) + ((stencil(evo_shiftU2, stencil_idx_0_0_m5_VVV) + stencil(evo_shiftU2, stencil_idx_0_0_5_VVV)))))))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}