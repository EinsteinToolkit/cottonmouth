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
        vreal x214 = stencil(AtDD00, stencil_idx_0_m2_0_VVV); // x214: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m2__0_]
        vreal x215 = stencil(AtDD00, stencil_idx_0_2_0_VVV); // x215: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__2__0_]
        vreal x216 = stencil(AtDD00, stencil_idx_0_m1_0_VVV); // x216: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__m1__0_]
        vreal x217 = stencil(AtDD00, stencil_idx_0_1_0_VVV); // x217: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__1__0_]
        vreal x218 = stencil(AtDD00, stencil_idx_0_0_m2_VVV); // x218: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m2_]
        vreal x219 = stencil(AtDD00, stencil_idx_0_0_2_VVV); // x219: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__2_]
        vreal x220 = stencil(AtDD00, stencil_idx_0_0_m1_VVV); // x220: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__m1_]
        vreal x221 = stencil(AtDD00, stencil_idx_0_0_1_VVV); // x221: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__1_]
        vreal x210 = stencil(AtDD00, stencil_idx_m2_0_0_VVV); // x210: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m2__0__0_]
        vreal x211 = stencil(AtDD00, stencil_idx_2_0_0_VVV); // x211: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__2__0__0_]
        vreal x212 = stencil(AtDD00, stencil_idx_m1_0_0_VVV); // x212: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__m1__0__0_]
        vreal x213 = stencil(AtDD00, stencil_idx_1_0_0_VVV); // x213: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__1__0__0_]
        vreal x209 = ((-5.0 / 16.0) * stencil(AtDD00, stencil_idx_0_0_0_VVV)); // x209: Dependency! Liveness = 1; [__dummy_stencil__AtDD00__0__0__0_]
        store(At_rhsDD00, stencil_idx_0_0_0_VVV, (access(At_rhsDD00, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x209 + ((-3.0 / 32.0) * ((x210 + x211))) + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_m3_0_0_VVV) + stencil(AtDD00, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x212 + x213))))) + (DYI * (x209 + ((-3.0 / 32.0) * ((x214 + x215))) + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_0_m3_0_VVV) + stencil(AtDD00, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x216 + x217))))) + (DZI * (x209 + ((-3.0 / 32.0) * ((x218 + x219))) + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_0_0_m3_VVV) + stencil(AtDD00, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x220 + x221)))))))));
        x209 = stencil(AtDD01, stencil_idx_0_m2_0_VVV); // x227: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m2__0_]
        x210 = stencil(AtDD01, stencil_idx_0_2_0_VVV); // x228: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__2__0_]
        x211 = stencil(AtDD01, stencil_idx_0_m1_0_VVV); // x229: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__m1__0_]
        x212 = stencil(AtDD01, stencil_idx_0_1_0_VVV); // x230: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__1__0_]
        x213 = stencil(AtDD01, stencil_idx_0_0_m2_VVV); // x231: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m2_]
        x214 = stencil(AtDD01, stencil_idx_0_0_2_VVV); // x232: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__2_]
        x215 = stencil(AtDD01, stencil_idx_0_0_m1_VVV); // x233: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__m1_]
        x216 = stencil(AtDD01, stencil_idx_0_0_1_VVV); // x234: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__1_]
        x217 = stencil(AtDD01, stencil_idx_m2_0_0_VVV); // x223: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m2__0__0_]
        x218 = stencil(AtDD01, stencil_idx_2_0_0_VVV); // x224: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__2__0__0_]
        x219 = stencil(AtDD01, stencil_idx_m1_0_0_VVV); // x225: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__m1__0__0_]
        x220 = stencil(AtDD01, stencil_idx_1_0_0_VVV); // x226: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__1__0__0_]
        x221 = ((-5.0 / 16.0) * stencil(AtDD01, stencil_idx_0_0_0_VVV)); // x222: Dependency! Liveness = 1; [__dummy_stencil__AtDD01__0__0__0_]
        store(At_rhsDD01, stencil_idx_0_0_0_VVV, (access(At_rhsDD01, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x221 + ((-3.0 / 32.0) * ((x217 + x218))) + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_m3_0_0_VVV) + stencil(AtDD01, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x219 + x220))))) + (DYI * (x221 + ((-3.0 / 32.0) * ((x209 + x210))) + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_0_m3_0_VVV) + stencil(AtDD01, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x211 + x212))))) + (DZI * (x221 + ((-3.0 / 32.0) * ((x213 + x214))) + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_0_0_m3_VVV) + stencil(AtDD01, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x215 + x216)))))))));
        vreal x240 = stencil(AtDD02, stencil_idx_0_m2_0_VVV); // x240: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m2__0_]
        vreal x241 = stencil(AtDD02, stencil_idx_0_2_0_VVV); // x241: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__2__0_]
        vreal x242 = stencil(AtDD02, stencil_idx_0_m1_0_VVV); // x242: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__m1__0_]
        vreal x243 = stencil(AtDD02, stencil_idx_0_1_0_VVV); // x243: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__1__0_]
        vreal x244 = stencil(AtDD02, stencil_idx_0_0_m2_VVV); // x244: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m2_]
        vreal x245 = stencil(AtDD02, stencil_idx_0_0_2_VVV); // x245: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__2_]
        vreal x246 = stencil(AtDD02, stencil_idx_0_0_m1_VVV); // x246: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__m1_]
        vreal x247 = stencil(AtDD02, stencil_idx_0_0_1_VVV); // x247: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__1_]
        vreal x236 = stencil(AtDD02, stencil_idx_m2_0_0_VVV); // x236: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m2__0__0_]
        vreal x237 = stencil(AtDD02, stencil_idx_2_0_0_VVV); // x237: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__2__0__0_]
        vreal x238 = stencil(AtDD02, stencil_idx_m1_0_0_VVV); // x238: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__m1__0__0_]
        vreal x239 = stencil(AtDD02, stencil_idx_1_0_0_VVV); // x239: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__1__0__0_]
        vreal x235 = ((-5.0 / 16.0) * stencil(AtDD02, stencil_idx_0_0_0_VVV)); // x235: Dependency! Liveness = 1; [__dummy_stencil__AtDD02__0__0__0_]
        store(At_rhsDD02, stencil_idx_0_0_0_VVV, (access(At_rhsDD02, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x235 + ((-3.0 / 32.0) * ((x236 + x237))) + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_m3_0_0_VVV) + stencil(AtDD02, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x238 + x239))))) + (DYI * (x235 + ((-3.0 / 32.0) * ((x240 + x241))) + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_0_m3_0_VVV) + stencil(AtDD02, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x242 + x243))))) + (DZI * (x235 + ((-3.0 / 32.0) * ((x244 + x245))) + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_0_0_m3_VVV) + stencil(AtDD02, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x246 + x247)))))))));
        x235 = stencil(AtDD11, stencil_idx_0_m2_0_VVV); // x253: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m2__0_]
        x236 = stencil(AtDD11, stencil_idx_0_2_0_VVV); // x254: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__2__0_]
        x237 = stencil(AtDD11, stencil_idx_0_m1_0_VVV); // x255: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__m1__0_]
        x238 = stencil(AtDD11, stencil_idx_0_1_0_VVV); // x256: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__1__0_]
        x239 = stencil(AtDD11, stencil_idx_0_0_m2_VVV); // x257: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m2_]
        x240 = stencil(AtDD11, stencil_idx_0_0_2_VVV); // x258: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__2_]
        x241 = stencil(AtDD11, stencil_idx_0_0_m1_VVV); // x259: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__m1_]
        x242 = stencil(AtDD11, stencil_idx_0_0_1_VVV); // x260: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__1_]
        x243 = stencil(AtDD11, stencil_idx_m2_0_0_VVV); // x249: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m2__0__0_]
        x244 = stencil(AtDD11, stencil_idx_2_0_0_VVV); // x250: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__2__0__0_]
        x245 = stencil(AtDD11, stencil_idx_m1_0_0_VVV); // x251: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__m1__0__0_]
        x246 = stencil(AtDD11, stencil_idx_1_0_0_VVV); // x252: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__1__0__0_]
        x247 = ((-5.0 / 16.0) * stencil(AtDD11, stencil_idx_0_0_0_VVV)); // x248: Dependency! Liveness = 1; [__dummy_stencil__AtDD11__0__0__0_]
        store(At_rhsDD11, stencil_idx_0_0_0_VVV, (access(At_rhsDD11, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x247 + ((-3.0 / 32.0) * ((x243 + x244))) + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_m3_0_0_VVV) + stencil(AtDD11, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x245 + x246))))) + (DYI * (x247 + ((-3.0 / 32.0) * ((x235 + x236))) + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_0_m3_0_VVV) + stencil(AtDD11, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x237 + x238))))) + (DZI * (x247 + ((-3.0 / 32.0) * ((x239 + x240))) + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_0_0_m3_VVV) + stencil(AtDD11, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x241 + x242)))))))));
        vreal x266 = stencil(AtDD12, stencil_idx_0_m2_0_VVV); // x266: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m2__0_]
        vreal x267 = stencil(AtDD12, stencil_idx_0_2_0_VVV); // x267: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__2__0_]
        vreal x268 = stencil(AtDD12, stencil_idx_0_m1_0_VVV); // x268: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__m1__0_]
        vreal x269 = stencil(AtDD12, stencil_idx_0_1_0_VVV); // x269: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__1__0_]
        vreal x270 = stencil(AtDD12, stencil_idx_0_0_m2_VVV); // x270: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m2_]
        vreal x271 = stencil(AtDD12, stencil_idx_0_0_2_VVV); // x271: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__2_]
        vreal x272 = stencil(AtDD12, stencil_idx_0_0_m1_VVV); // x272: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__m1_]
        vreal x273 = stencil(AtDD12, stencil_idx_0_0_1_VVV); // x273: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__1_]
        vreal x262 = stencil(AtDD12, stencil_idx_m2_0_0_VVV); // x262: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m2__0__0_]
        vreal x263 = stencil(AtDD12, stencil_idx_2_0_0_VVV); // x263: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__2__0__0_]
        vreal x264 = stencil(AtDD12, stencil_idx_m1_0_0_VVV); // x264: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__m1__0__0_]
        vreal x265 = stencil(AtDD12, stencil_idx_1_0_0_VVV); // x265: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__1__0__0_]
        vreal x261 = ((-5.0 / 16.0) * stencil(AtDD12, stencil_idx_0_0_0_VVV)); // x261: Dependency! Liveness = 1; [__dummy_stencil__AtDD12__0__0__0_]
        store(At_rhsDD12, stencil_idx_0_0_0_VVV, (access(At_rhsDD12, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x261 + ((-3.0 / 32.0) * ((x262 + x263))) + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_m3_0_0_VVV) + stencil(AtDD12, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x264 + x265))))) + (DYI * (x261 + ((-3.0 / 32.0) * ((x266 + x267))) + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_0_m3_0_VVV) + stencil(AtDD12, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x268 + x269))))) + (DZI * (x261 + ((-3.0 / 32.0) * ((x270 + x271))) + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_0_0_m3_VVV) + stencil(AtDD12, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x272 + x273)))))))));
        x261 = stencil(AtDD22, stencil_idx_0_m2_0_VVV); // x279: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m2__0_]
        x262 = stencil(AtDD22, stencil_idx_0_2_0_VVV); // x280: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__2__0_]
        x263 = stencil(AtDD22, stencil_idx_0_m1_0_VVV); // x281: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__m1__0_]
        x264 = stencil(AtDD22, stencil_idx_0_1_0_VVV); // x282: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__1__0_]
        x265 = stencil(AtDD22, stencil_idx_0_0_m2_VVV); // x283: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m2_]
        x266 = stencil(AtDD22, stencil_idx_0_0_2_VVV); // x284: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__2_]
        x267 = stencil(AtDD22, stencil_idx_0_0_m1_VVV); // x285: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__m1_]
        x268 = stencil(AtDD22, stencil_idx_0_0_1_VVV); // x286: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__1_]
        x269 = stencil(AtDD22, stencil_idx_m2_0_0_VVV); // x275: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m2__0__0_]
        x270 = stencil(AtDD22, stencil_idx_2_0_0_VVV); // x276: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__2__0__0_]
        x271 = stencil(AtDD22, stencil_idx_m1_0_0_VVV); // x277: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__m1__0__0_]
        x272 = stencil(AtDD22, stencil_idx_1_0_0_VVV); // x278: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__1__0__0_]
        x273 = ((-5.0 / 16.0) * stencil(AtDD22, stencil_idx_0_0_0_VVV)); // x274: Dependency! Liveness = 1; [__dummy_stencil__AtDD22__0__0__0_]
        store(At_rhsDD22, stencil_idx_0_0_0_VVV, (access(At_rhsDD22, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x273 + ((-3.0 / 32.0) * ((x269 + x270))) + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_m3_0_0_VVV) + stencil(AtDD22, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x271 + x272))))) + (DYI * (x273 + ((-3.0 / 32.0) * ((x261 + x262))) + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_0_m3_0_VVV) + stencil(AtDD22, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x263 + x264))))) + (DZI * (x273 + ((-3.0 / 32.0) * ((x265 + x266))) + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_0_0_m3_VVV) + stencil(AtDD22, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x267 + x268)))))))));
        vreal x292 = stencil(Theta, stencil_idx_0_m2_0_VVV); // x292: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__m2__0_]
        vreal x293 = stencil(Theta, stencil_idx_0_2_0_VVV); // x293: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__2__0_]
        vreal x294 = stencil(Theta, stencil_idx_0_m1_0_VVV); // x294: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__m1__0_]
        vreal x295 = stencil(Theta, stencil_idx_0_1_0_VVV); // x295: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__1__0_]
        vreal x296 = stencil(Theta, stencil_idx_0_0_m2_VVV); // x296: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__m2_]
        vreal x297 = stencil(Theta, stencil_idx_0_0_2_VVV); // x297: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__2_]
        vreal x298 = stencil(Theta, stencil_idx_0_0_m1_VVV); // x298: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__m1_]
        vreal x299 = stencil(Theta, stencil_idx_0_0_1_VVV); // x299: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__1_]
        vreal x288 = stencil(Theta, stencil_idx_m2_0_0_VVV); // x288: Dependency! Liveness = 1; [__dummy_stencil__Theta__m2__0__0_]
        vreal x289 = stencil(Theta, stencil_idx_2_0_0_VVV); // x289: Dependency! Liveness = 1; [__dummy_stencil__Theta__2__0__0_]
        vreal x290 = stencil(Theta, stencil_idx_m1_0_0_VVV); // x290: Dependency! Liveness = 1; [__dummy_stencil__Theta__m1__0__0_]
        vreal x291 = stencil(Theta, stencil_idx_1_0_0_VVV); // x291: Dependency! Liveness = 1; [__dummy_stencil__Theta__1__0__0_]
        vreal x287 = ((-5.0 / 16.0) * stencil(Theta, stencil_idx_0_0_0_VVV)); // x287: Dependency! Liveness = 1; [__dummy_stencil__Theta__0__0__0_]
        store(Theta_rhs, stencil_idx_0_0_0_VVV, (access(Theta_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x287 + ((-3.0 / 32.0) * ((x288 + x289))) + ((1.0 / 64.0) * ((stencil(Theta, stencil_idx_m3_0_0_VVV) + stencil(Theta, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x290 + x291))))) + (DYI * (x287 + ((-3.0 / 32.0) * ((x292 + x293))) + ((1.0 / 64.0) * ((stencil(Theta, stencil_idx_0_m3_0_VVV) + stencil(Theta, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x294 + x295))))) + (DZI * (x287 + ((-3.0 / 32.0) * ((x296 + x297))) + ((1.0 / 64.0) * ((stencil(Theta, stencil_idx_0_0_m3_VVV) + stencil(Theta, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x298 + x299)))))))));
        x287 = stencil(evo_GammatU0, stencil_idx_0_m2_0_VVV); // x324: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__m2__0_]
        x288 = stencil(evo_GammatU0, stencil_idx_0_2_0_VVV); // x325: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__2__0_]
        x289 = stencil(evo_GammatU0, stencil_idx_0_m1_0_VVV); // x326: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__m1__0_]
        x290 = stencil(evo_GammatU0, stencil_idx_0_1_0_VVV); // x327: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__1__0_]
        x291 = stencil(evo_GammatU0, stencil_idx_0_0_m2_VVV); // x328: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__m2_]
        x292 = stencil(evo_GammatU0, stencil_idx_0_0_2_VVV); // x329: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__2_]
        x293 = stencil(evo_GammatU0, stencil_idx_0_0_m1_VVV); // x330: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__m1_]
        x294 = stencil(evo_GammatU0, stencil_idx_0_0_1_VVV); // x331: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__1_]
        x295 = stencil(evo_GammatU0, stencil_idx_m2_0_0_VVV); // x320: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__m2__0__0_]
        x296 = stencil(evo_GammatU0, stencil_idx_2_0_0_VVV); // x321: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__2__0__0_]
        x297 = stencil(evo_GammatU0, stencil_idx_m1_0_0_VVV); // x322: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__m1__0__0_]
        x298 = stencil(evo_GammatU0, stencil_idx_1_0_0_VVV); // x323: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__1__0__0_]
        x299 = ((-5.0 / 16.0) * stencil(evo_GammatU0, stencil_idx_0_0_0_VVV)); // x319: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU0__0__0__0_]
        store(evo_Gammat_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU0, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x299 + ((-3.0 / 32.0) * ((x295 + x296))) + ((1.0 / 64.0) * ((stencil(evo_GammatU0, stencil_idx_m3_0_0_VVV) + stencil(evo_GammatU0, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x297 + x298))))) + (DYI * (x299 + ((-3.0 / 32.0) * ((x287 + x288))) + ((1.0 / 64.0) * ((stencil(evo_GammatU0, stencil_idx_0_m3_0_VVV) + stencil(evo_GammatU0, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x289 + x290))))) + (DZI * (x299 + ((-3.0 / 32.0) * ((x291 + x292))) + ((1.0 / 64.0) * ((stencil(evo_GammatU0, stencil_idx_0_0_m3_VVV) + stencil(evo_GammatU0, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x293 + x294)))))))));
        vreal x337 = stencil(evo_GammatU1, stencil_idx_0_m2_0_VVV); // x337: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__m2__0_]
        vreal x338 = stencil(evo_GammatU1, stencil_idx_0_2_0_VVV); // x338: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__2__0_]
        vreal x339 = stencil(evo_GammatU1, stencil_idx_0_m1_0_VVV); // x339: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__m1__0_]
        vreal x340 = stencil(evo_GammatU1, stencil_idx_0_1_0_VVV); // x340: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__1__0_]
        vreal x341 = stencil(evo_GammatU1, stencil_idx_0_0_m2_VVV); // x341: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__m2_]
        vreal x342 = stencil(evo_GammatU1, stencil_idx_0_0_2_VVV); // x342: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__2_]
        vreal x343 = stencil(evo_GammatU1, stencil_idx_0_0_m1_VVV); // x343: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__m1_]
        vreal x344 = stencil(evo_GammatU1, stencil_idx_0_0_1_VVV); // x344: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__1_]
        vreal x333 = stencil(evo_GammatU1, stencil_idx_m2_0_0_VVV); // x333: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__m2__0__0_]
        vreal x334 = stencil(evo_GammatU1, stencil_idx_2_0_0_VVV); // x334: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__2__0__0_]
        vreal x335 = stencil(evo_GammatU1, stencil_idx_m1_0_0_VVV); // x335: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__m1__0__0_]
        vreal x336 = stencil(evo_GammatU1, stencil_idx_1_0_0_VVV); // x336: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__1__0__0_]
        vreal x332 = ((-5.0 / 16.0) * stencil(evo_GammatU1, stencil_idx_0_0_0_VVV)); // x332: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU1__0__0__0_]
        store(evo_Gammat_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU1, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x332 + ((-3.0 / 32.0) * ((x333 + x334))) + ((1.0 / 64.0) * ((stencil(evo_GammatU1, stencil_idx_m3_0_0_VVV) + stencil(evo_GammatU1, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x335 + x336))))) + (DYI * (x332 + ((-3.0 / 32.0) * ((x337 + x338))) + ((1.0 / 64.0) * ((stencil(evo_GammatU1, stencil_idx_0_m3_0_VVV) + stencil(evo_GammatU1, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x339 + x340))))) + (DZI * (x332 + ((-3.0 / 32.0) * ((x341 + x342))) + ((1.0 / 64.0) * ((stencil(evo_GammatU1, stencil_idx_0_0_m3_VVV) + stencil(evo_GammatU1, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x343 + x344)))))))));
        x332 = stencil(evo_GammatU2, stencil_idx_0_m2_0_VVV); // x350: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__m2__0_]
        x333 = stencil(evo_GammatU2, stencil_idx_0_2_0_VVV); // x351: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__2__0_]
        x334 = stencil(evo_GammatU2, stencil_idx_0_m1_0_VVV); // x352: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__m1__0_]
        x335 = stencil(evo_GammatU2, stencil_idx_0_1_0_VVV); // x353: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__1__0_]
        x336 = stencil(evo_GammatU2, stencil_idx_0_0_m2_VVV); // x354: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__m2_]
        x337 = stencil(evo_GammatU2, stencil_idx_0_0_2_VVV); // x355: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__2_]
        x338 = stencil(evo_GammatU2, stencil_idx_0_0_m1_VVV); // x356: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__m1_]
        x339 = stencil(evo_GammatU2, stencil_idx_0_0_1_VVV); // x357: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__1_]
        x340 = stencil(evo_GammatU2, stencil_idx_m2_0_0_VVV); // x346: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__m2__0__0_]
        x341 = stencil(evo_GammatU2, stencil_idx_2_0_0_VVV); // x347: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__2__0__0_]
        x342 = stencil(evo_GammatU2, stencil_idx_m1_0_0_VVV); // x348: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__m1__0__0_]
        x343 = stencil(evo_GammatU2, stencil_idx_1_0_0_VVV); // x349: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__1__0__0_]
        x344 = ((-5.0 / 16.0) * stencil(evo_GammatU2, stencil_idx_0_0_0_VVV)); // x345: Dependency! Liveness = 1; [__dummy_stencil__evo_GammatU2__0__0__0_]
        store(evo_Gammat_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_Gammat_rhsU2, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x344 + ((-3.0 / 32.0) * ((x340 + x341))) + ((1.0 / 64.0) * ((stencil(evo_GammatU2, stencil_idx_m3_0_0_VVV) + stencil(evo_GammatU2, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x342 + x343))))) + (DYI * (x344 + ((-3.0 / 32.0) * ((x332 + x333))) + ((1.0 / 64.0) * ((stencil(evo_GammatU2, stencil_idx_0_m3_0_VVV) + stencil(evo_GammatU2, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x334 + x335))))) + (DZI * (x344 + ((-3.0 / 32.0) * ((x336 + x337))) + ((1.0 / 64.0) * ((stencil(evo_GammatU2, stencil_idx_0_0_m3_VVV) + stencil(evo_GammatU2, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x338 + x339)))))))));
        vreal x481 = stencil(trK, stencil_idx_0_m2_0_VVV); // x481: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m2__0_]
        vreal x482 = stencil(trK, stencil_idx_0_2_0_VVV); // x482: Dependency! Liveness = 1; [__dummy_stencil__trK__0__2__0_]
        vreal x483 = stencil(trK, stencil_idx_0_m1_0_VVV); // x483: Dependency! Liveness = 1; [__dummy_stencil__trK__0__m1__0_]
        vreal x484 = stencil(trK, stencil_idx_0_1_0_VVV); // x484: Dependency! Liveness = 1; [__dummy_stencil__trK__0__1__0_]
        vreal x485 = stencil(trK, stencil_idx_0_0_m2_VVV); // x485: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m2_]
        vreal x486 = stencil(trK, stencil_idx_0_0_2_VVV); // x486: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__2_]
        vreal x487 = stencil(trK, stencil_idx_0_0_m1_VVV); // x487: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__m1_]
        vreal x488 = stencil(trK, stencil_idx_0_0_1_VVV); // x488: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__1_]
        vreal x477 = stencil(trK, stencil_idx_m2_0_0_VVV); // x477: Dependency! Liveness = 1; [__dummy_stencil__trK__m2__0__0_]
        vreal x478 = stencil(trK, stencil_idx_2_0_0_VVV); // x478: Dependency! Liveness = 1; [__dummy_stencil__trK__2__0__0_]
        vreal x479 = stencil(trK, stencil_idx_m1_0_0_VVV); // x479: Dependency! Liveness = 1; [__dummy_stencil__trK__m1__0__0_]
        vreal x480 = stencil(trK, stencil_idx_1_0_0_VVV); // x480: Dependency! Liveness = 1; [__dummy_stencil__trK__1__0__0_]
        vreal x476 = ((-5.0 / 16.0) * stencil(trK, stencil_idx_0_0_0_VVV)); // x476: Dependency! Liveness = 1; [__dummy_stencil__trK__0__0__0_]
        store(trK_rhs, stencil_idx_0_0_0_VVV, (access(trK_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x476 + ((-3.0 / 32.0) * ((x477 + x478))) + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_m3_0_0_VVV) + stencil(trK, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * ((x479 + x480))))) + (DYI * (x476 + ((-3.0 / 32.0) * ((x481 + x482))) + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_0_m3_0_VVV) + stencil(trK, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * ((x483 + x484))))) + (DZI * (x476 + ((-3.0 / 32.0) * ((x485 + x486))) + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_0_0_m3_VVV) + stencil(trK, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * ((x487 + x488)))))))));
        x476 = stencil(chi, stencil_idx_m2_0_0_VVV); // x301: Dependency! Liveness = 1; [__dummy_stencil__chi__m2__0__0_]
        x477 = stencil(chi, stencil_idx_2_0_0_VVV); // x302: Dependency! Liveness = 1; [__dummy_stencil__chi__2__0__0_]
        x478 = ((x476 + x477)); // x303: Dependency! Liveness = 2; [x301, x302]
        x479 = stencil(chi, stencil_idx_m1_0_0_VVV); // x304: Dependency! Liveness = 1; [__dummy_stencil__chi__m1__0__0_]
        x480 = stencil(chi, stencil_idx_1_0_0_VVV); // x305: Dependency! Liveness = 1; [__dummy_stencil__chi__1__0__0_]
        x481 = ((x479 + x480)); // x306: Dependency! Liveness = 2; [x304, x305]
        x482 = stencil(chi, stencil_idx_0_m2_0_VVV); // x307: Dependency! Liveness = 1; [__dummy_stencil__chi__0__m2__0_]
        x483 = stencil(chi, stencil_idx_0_2_0_VVV); // x308: Dependency! Liveness = 1; [__dummy_stencil__chi__0__2__0_]
        x484 = ((x482 + x483)); // x309: Dependency! Liveness = 2; [x307, x308]
        x485 = stencil(chi, stencil_idx_0_m1_0_VVV); // x310: Dependency! Liveness = 1; [__dummy_stencil__chi__0__m1__0_]
        x486 = stencil(chi, stencil_idx_0_1_0_VVV); // x311: Dependency! Liveness = 1; [__dummy_stencil__chi__0__1__0_]
        x487 = ((x485 + x486)); // x312: Dependency! Liveness = 2; [x310, x311]
        x488 = stencil(chi, stencil_idx_0_0_m2_VVV); // x313: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__m2_]
        vreal x314 = stencil(chi, stencil_idx_0_0_2_VVV); // x314: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__2_]
        vreal x315 = ((x314 + x488)); // x315: Dependency! Liveness = 2; [x313, x314]
        x314 = stencil(chi, stencil_idx_0_0_m1_VVV); // x316: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__m1_]
        vreal x317 = stencil(chi, stencil_idx_0_0_1_VVV); // x317: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__1_]
        vreal x318 = ((x314 + x317)); // x318: Dependency! Liveness = 2; [x316, x317]
        x317 = ((-5.0 / 16.0) * stencil(chi, stencil_idx_0_0_0_VVV)); // x300: Dependency! Liveness = 1; [__dummy_stencil__chi__0__0__0_]
        store(chi_rhs, stencil_idx_0_0_0_VVV, (access(chi_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x317 + ((-3.0 / 32.0) * x478) + ((1.0 / 64.0) * ((stencil(chi, stencil_idx_m3_0_0_VVV) + stencil(chi, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x481))) + (DYI * (x317 + ((-3.0 / 32.0) * x484) + ((1.0 / 64.0) * ((stencil(chi, stencil_idx_0_m3_0_VVV) + stencil(chi, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x487))) + (DZI * (x317 + ((-3.0 / 32.0) * x315) + ((1.0 / 64.0) * ((stencil(chi, stencil_idx_0_0_m3_VVV) + stencil(chi, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x318)))))));
        x315 = stencil(evo_lapse, stencil_idx_m2_0_0_VVV); // x359: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m2__0__0_]
        x318 = stencil(evo_lapse, stencil_idx_2_0_0_VVV); // x360: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__2__0__0_]
        vreal x361 = ((x315 + x318)); // x361: Dependency! Liveness = 2; [x359, x360]
        vreal x362 = stencil(evo_lapse, stencil_idx_m1_0_0_VVV); // x362: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__m1__0__0_]
        vreal x363 = stencil(evo_lapse, stencil_idx_1_0_0_VVV); // x363: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__1__0__0_]
        vreal x364 = ((x362 + x363)); // x364: Dependency! Liveness = 2; [x362, x363]
        x362 = stencil(evo_lapse, stencil_idx_0_m2_0_VVV); // x365: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m2__0_]
        x363 = stencil(evo_lapse, stencil_idx_0_2_0_VVV); // x366: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__2__0_]
        vreal x367 = ((x362 + x363)); // x367: Dependency! Liveness = 2; [x365, x366]
        vreal x368 = stencil(evo_lapse, stencil_idx_0_m1_0_VVV); // x368: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__m1__0_]
        vreal x369 = stencil(evo_lapse, stencil_idx_0_1_0_VVV); // x369: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__1__0_]
        vreal x370 = ((x368 + x369)); // x370: Dependency! Liveness = 2; [x368, x369]
        x368 = stencil(evo_lapse, stencil_idx_0_0_m2_VVV); // x371: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m2_]
        x369 = stencil(evo_lapse, stencil_idx_0_0_2_VVV); // x372: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__2_]
        vreal x373 = ((x368 + x369)); // x373: Dependency! Liveness = 2; [x371, x372]
        vreal x374 = stencil(evo_lapse, stencil_idx_0_0_m1_VVV); // x374: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__m1_]
        vreal x375 = stencil(evo_lapse, stencil_idx_0_0_1_VVV); // x375: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__1_]
        vreal x376 = ((x374 + x375)); // x376: Dependency! Liveness = 2; [x374, x375]
        x374 = ((-5.0 / 16.0) * stencil(evo_lapse, stencil_idx_0_0_0_VVV)); // x358: Dependency! Liveness = 1; [__dummy_stencil__evo_lapse__0__0__0_]
        store(evo_lapse_rhs, stencil_idx_0_0_0_VVV, (access(evo_lapse_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x374 + ((-3.0 / 32.0) * x361) + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_m3_0_0_VVV) + stencil(evo_lapse, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x364))) + (DYI * (x374 + ((-3.0 / 32.0) * x367) + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_0_m3_0_VVV) + stencil(evo_lapse, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x370))) + (DZI * (x374 + ((-3.0 / 32.0) * x373) + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_0_0_m3_VVV) + stencil(evo_lapse, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x376)))))));
        x361 = stencil(evo_shiftU0, stencil_idx_m2_0_0_VVV); // x378: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m2__0__0_]
        x364 = stencil(evo_shiftU0, stencil_idx_2_0_0_VVV); // x379: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__2__0__0_]
        x367 = ((x361 + x364)); // x380: Dependency! Liveness = 2; [x378, x379]
        x370 = stencil(evo_shiftU0, stencil_idx_m1_0_0_VVV); // x381: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__m1__0__0_]
        x373 = stencil(evo_shiftU0, stencil_idx_1_0_0_VVV); // x382: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__1__0__0_]
        x376 = ((x370 + x373)); // x383: Dependency! Liveness = 2; [x381, x382]
        x375 = stencil(evo_shiftU0, stencil_idx_0_m2_0_VVV); // x384: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m2__0_]
        vreal x385 = stencil(evo_shiftU0, stencil_idx_0_2_0_VVV); // x385: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__2__0_]
        vreal x386 = ((x375 + x385)); // x386: Dependency! Liveness = 2; [x384, x385]
        x385 = stencil(evo_shiftU0, stencil_idx_0_m1_0_VVV); // x387: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__m1__0_]
        vreal x388 = stencil(evo_shiftU0, stencil_idx_0_1_0_VVV); // x388: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__1__0_]
        vreal x389 = ((x385 + x388)); // x389: Dependency! Liveness = 2; [x387, x388]
        x388 = stencil(evo_shiftU0, stencil_idx_0_0_m2_VVV); // x390: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m2_]
        vreal x391 = stencil(evo_shiftU0, stencil_idx_0_0_2_VVV); // x391: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__2_]
        vreal x392 = ((x388 + x391)); // x392: Dependency! Liveness = 2; [x390, x391]
        x391 = stencil(evo_shiftU0, stencil_idx_0_0_m1_VVV); // x393: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__m1_]
        vreal x394 = stencil(evo_shiftU0, stencil_idx_0_0_1_VVV); // x394: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__1_]
        vreal x395 = ((x391 + x394)); // x395: Dependency! Liveness = 2; [x393, x394]
        x394 = ((-5.0 / 16.0) * stencil(evo_shiftU0, stencil_idx_0_0_0_VVV)); // x377: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU0__0__0__0_]
        store(evo_shift_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU0, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x394 + ((-3.0 / 32.0) * x367) + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU0, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x376))) + (DYI * (x394 + ((-3.0 / 32.0) * x386) + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU0, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x389))) + (DZI * (x394 + ((-3.0 / 32.0) * x392) + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU0, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x395)))))));
        x386 = stencil(evo_shiftU1, stencil_idx_m2_0_0_VVV); // x397: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m2__0__0_]
        x389 = stencil(evo_shiftU1, stencil_idx_2_0_0_VVV); // x398: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__2__0__0_]
        x392 = ((x386 + x389)); // x399: Dependency! Liveness = 2; [x397, x398]
        x395 = stencil(evo_shiftU1, stencil_idx_m1_0_0_VVV); // x400: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__m1__0__0_]
        vreal x401 = stencil(evo_shiftU1, stencil_idx_1_0_0_VVV); // x401: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__1__0__0_]
        vreal x402 = ((x395 + x401)); // x402: Dependency! Liveness = 2; [x400, x401]
        x401 = stencil(evo_shiftU1, stencil_idx_0_m2_0_VVV); // x403: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m2__0_]
        vreal x404 = stencil(evo_shiftU1, stencil_idx_0_2_0_VVV); // x404: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__2__0_]
        vreal x405 = ((x401 + x404)); // x405: Dependency! Liveness = 2; [x403, x404]
        x404 = stencil(evo_shiftU1, stencil_idx_0_m1_0_VVV); // x406: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__m1__0_]
        vreal x407 = stencil(evo_shiftU1, stencil_idx_0_1_0_VVV); // x407: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__1__0_]
        vreal x408 = ((x404 + x407)); // x408: Dependency! Liveness = 2; [x406, x407]
        x407 = stencil(evo_shiftU1, stencil_idx_0_0_m2_VVV); // x409: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m2_]
        vreal x410 = stencil(evo_shiftU1, stencil_idx_0_0_2_VVV); // x410: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__2_]
        vreal x411 = ((x407 + x410)); // x411: Dependency! Liveness = 2; [x409, x410]
        x410 = stencil(evo_shiftU1, stencil_idx_0_0_m1_VVV); // x412: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__m1_]
        vreal x413 = stencil(evo_shiftU1, stencil_idx_0_0_1_VVV); // x413: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__1_]
        vreal x414 = ((x410 + x413)); // x414: Dependency! Liveness = 2; [x412, x413]
        x413 = ((-5.0 / 16.0) * stencil(evo_shiftU1, stencil_idx_0_0_0_VVV)); // x396: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU1__0__0__0_]
        store(evo_shift_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU1, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x413 + ((-3.0 / 32.0) * x392) + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU1, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x402))) + (DYI * (x413 + ((-3.0 / 32.0) * x405) + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU1, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x408))) + (DZI * (x413 + ((-3.0 / 32.0) * x411) + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU1, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x414)))))));
        x402 = stencil(evo_shiftU2, stencil_idx_m2_0_0_VVV); // x416: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m2__0__0_]
        x405 = stencil(evo_shiftU2, stencil_idx_2_0_0_VVV); // x417: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__2__0__0_]
        x408 = ((x402 + x405)); // x418: Dependency! Liveness = 2; [x416, x417]
        x411 = stencil(evo_shiftU2, stencil_idx_m1_0_0_VVV); // x419: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__m1__0__0_]
        x414 = stencil(evo_shiftU2, stencil_idx_1_0_0_VVV); // x420: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__1__0__0_]
        vreal x421 = ((x411 + x414)); // x421: Dependency! Liveness = 2; [x419, x420]
        vreal x422 = stencil(evo_shiftU2, stencil_idx_0_m2_0_VVV); // x422: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m2__0_]
        vreal x423 = stencil(evo_shiftU2, stencil_idx_0_2_0_VVV); // x423: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__2__0_]
        vreal x424 = ((x422 + x423)); // x424: Dependency! Liveness = 2; [x422, x423]
        x422 = stencil(evo_shiftU2, stencil_idx_0_m1_0_VVV); // x425: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__m1__0_]
        x423 = stencil(evo_shiftU2, stencil_idx_0_1_0_VVV); // x426: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__1__0_]
        vreal x427 = ((x422 + x423)); // x427: Dependency! Liveness = 2; [x425, x426]
        vreal x428 = stencil(evo_shiftU2, stencil_idx_0_0_m2_VVV); // x428: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m2_]
        vreal x429 = stencil(evo_shiftU2, stencil_idx_0_0_2_VVV); // x429: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__2_]
        vreal x430 = ((x428 + x429)); // x430: Dependency! Liveness = 2; [x428, x429]
        x428 = stencil(evo_shiftU2, stencil_idx_0_0_m1_VVV); // x431: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__m1_]
        x429 = stencil(evo_shiftU2, stencil_idx_0_0_1_VVV); // x432: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__1_]
        vreal x433 = ((x428 + x429)); // x433: Dependency! Liveness = 2; [x431, x432]
        vreal x415 = ((-5.0 / 16.0) * stencil(evo_shiftU2, stencil_idx_0_0_0_VVV)); // x415: Dependency! Liveness = 1; [__dummy_stencil__evo_shiftU2__0__0__0_]
        store(evo_shift_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU2, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x415 + ((-3.0 / 32.0) * x408) + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU2, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x421))) + (DYI * (x415 + ((-3.0 / 32.0) * x424) + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU2, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x427))) + (DZI * (x415 + ((-3.0 / 32.0) * x430) + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU2, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x433)))))));
        x415 = stencil(gtDD00, stencil_idx_m2_0_0_VVV); // x152: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m2__0__0_]
        x421 = stencil(gtDD00, stencil_idx_2_0_0_VVV); // x153: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__2__0__0_]
        x424 = ((x415 + x421)); // x435: Dependency! Liveness = 2; [x152, x153]
        x427 = stencil(gtDD00, stencil_idx_m1_0_0_VVV); // x154: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__m1__0__0_]
        x430 = stencil(gtDD00, stencil_idx_1_0_0_VVV); // x155: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__1__0__0_]
        x433 = ((x427 + x430)); // x436: Dependency! Liveness = 2; [x154, x155]
        vreal x91 = stencil(gtDD00, stencil_idx_0_m2_0_VVV); // x91: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m2__0_]
        vreal x92 = stencil(gtDD00, stencil_idx_0_2_0_VVV); // x92: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__2__0_]
        vreal x437 = ((x91 + x92)); // x437: Dependency! Liveness = 2; [x91, x92]
        x91 = stencil(gtDD00, stencil_idx_0_m1_0_VVV); // x93: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__m1__0_]
        x92 = stencil(gtDD00, stencil_idx_0_1_0_VVV); // x94: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__1__0_]
        vreal x438 = ((x91 + x92)); // x438: Dependency! Liveness = 2; [x93, x94]
        vreal x76 = stencil(gtDD00, stencil_idx_0_0_m2_VVV); // x76: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m2_]
        vreal x77 = stencil(gtDD00, stencil_idx_0_0_2_VVV); // x77: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__2_]
        vreal x439 = ((x76 + x77)); // x439: Dependency! Liveness = 2; [x76, x77]
        x76 = stencil(gtDD00, stencil_idx_0_0_m1_VVV); // x78: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__m1_]
        x77 = stencil(gtDD00, stencil_idx_0_0_1_VVV); // x79: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__1_]
        vreal x440 = ((x76 + x77)); // x440: Dependency! Liveness = 2; [x78, x79]
        vreal x434 = ((-5.0 / 16.0) * stencil(gtDD00, stencil_idx_0_0_0_VVV)); // x434: Dependency! Liveness = 1; [__dummy_stencil__gtDD00__0__0__0_]
        store(gt_rhsDD00, stencil_idx_0_0_0_VVV, (access(gt_rhsDD00, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x434 + ((-3.0 / 32.0) * x424) + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_m3_0_0_VVV) + stencil(gtDD00, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x433))) + (DYI * (x434 + ((-3.0 / 32.0) * x437) + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_0_m3_0_VVV) + stencil(gtDD00, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x438))) + (DZI * (x434 + ((-3.0 / 32.0) * x439) + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_0_0_m3_VVV) + stencil(gtDD00, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x440)))))));
        x434 = stencil(gtDD01, stencil_idx_m2_0_0_VVV); // x157: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m2__0__0_]
        x437 = stencil(gtDD01, stencil_idx_2_0_0_VVV); // x158: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__2__0__0_]
        x438 = ((x434 + x437)); // x442: Dependency! Liveness = 2; [x157, x158]
        x439 = stencil(gtDD01, stencil_idx_m1_0_0_VVV); // x159: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__m1__0__0_]
        x440 = stencil(gtDD01, stencil_idx_1_0_0_VVV); // x160: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__1__0__0_]
        vreal x443 = ((x439 + x440)); // x443: Dependency! Liveness = 2; [x159, x160]
        vreal x118 = stencil(gtDD01, stencil_idx_0_m2_0_VVV); // x118: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m2__0_]
        vreal x119 = stencil(gtDD01, stencil_idx_0_2_0_VVV); // x119: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__2__0_]
        vreal x444 = ((x118 + x119)); // x444: Dependency! Liveness = 2; [x118, x119]
        x118 = stencil(gtDD01, stencil_idx_0_m1_0_VVV); // x120: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__m1__0_]
        x119 = stencil(gtDD01, stencil_idx_0_1_0_VVV); // x121: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__1__0_]
        vreal x445 = ((x118 + x119)); // x445: Dependency! Liveness = 2; [x120, x121]
        vreal x57 = stencil(gtDD01, stencil_idx_0_0_m2_VVV); // x57: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m2_]
        vreal x58 = stencil(gtDD01, stencil_idx_0_0_2_VVV); // x58: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__2_]
        vreal x446 = ((x57 + x58)); // x446: Dependency! Liveness = 2; [x57, x58]
        x57 = stencil(gtDD01, stencil_idx_0_0_m1_VVV); // x59: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__m1_]
        x58 = stencil(gtDD01, stencil_idx_0_0_1_VVV); // x60: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__1_]
        vreal x447 = ((x57 + x58)); // x447: Dependency! Liveness = 2; [x59, x60]
        vreal x441 = ((-5.0 / 16.0) * stencil(gtDD01, stencil_idx_0_0_0_VVV)); // x441: Dependency! Liveness = 1; [__dummy_stencil__gtDD01__0__0__0_]
        store(gt_rhsDD01, stencil_idx_0_0_0_VVV, (access(gt_rhsDD01, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x441 + ((-3.0 / 32.0) * x438) + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_m3_0_0_VVV) + stencil(gtDD01, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x443))) + (DYI * (x441 + ((-3.0 / 32.0) * x444) + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_0_m3_0_VVV) + stencil(gtDD01, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x445))) + (DZI * (x441 + ((-3.0 / 32.0) * x446) + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_0_0_m3_VVV) + stencil(gtDD01, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x447)))))));
        x441 = stencil(gtDD02, stencil_idx_m2_0_0_VVV); // x166: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m2__0__0_]
        x443 = stencil(gtDD02, stencil_idx_2_0_0_VVV); // x167: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__2__0__0_]
        x444 = ((x441 + x443)); // x449: Dependency! Liveness = 2; [x166, x167]
        x445 = stencil(gtDD02, stencil_idx_m1_0_0_VVV); // x168: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__m1__0__0_]
        x446 = stencil(gtDD02, stencil_idx_1_0_0_VVV); // x169: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__1__0__0_]
        x447 = ((x445 + x446)); // x450: Dependency! Liveness = 2; [x168, x169]
        vreal x52 = stencil(gtDD02, stencil_idx_0_m2_0_VVV); // x52: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m2__0_]
        vreal x53 = stencil(gtDD02, stencil_idx_0_2_0_VVV); // x53: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__2__0_]
        vreal x451 = ((x52 + x53)); // x451: Dependency! Liveness = 2; [x52, x53]
        x52 = stencil(gtDD02, stencil_idx_0_m1_0_VVV); // x54: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__m1__0_]
        x53 = stencil(gtDD02, stencil_idx_0_1_0_VVV); // x55: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__1__0_]
        vreal x452 = ((x52 + x53)); // x452: Dependency! Liveness = 2; [x54, x55]
        vreal x143 = stencil(gtDD02, stencil_idx_0_0_m2_VVV); // x143: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m2_]
        vreal x144 = stencil(gtDD02, stencil_idx_0_0_2_VVV); // x144: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__2_]
        vreal x453 = ((x143 + x144)); // x453: Dependency! Liveness = 2; [x143, x144]
        x143 = stencil(gtDD02, stencil_idx_0_0_m1_VVV); // x145: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__m1_]
        x144 = stencil(gtDD02, stencil_idx_0_0_1_VVV); // x146: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__1_]
        vreal x454 = ((x143 + x144)); // x454: Dependency! Liveness = 2; [x145, x146]
        vreal x448 = ((-5.0 / 16.0) * stencil(gtDD02, stencil_idx_0_0_0_VVV)); // x448: Dependency! Liveness = 1; [__dummy_stencil__gtDD02__0__0__0_]
        store(gt_rhsDD02, stencil_idx_0_0_0_VVV, (access(gt_rhsDD02, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x448 + ((-3.0 / 32.0) * x444) + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_m3_0_0_VVV) + stencil(gtDD02, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x447))) + (DYI * (x448 + ((-3.0 / 32.0) * x451) + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_0_m3_0_VVV) + stencil(gtDD02, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x452))) + (DZI * (x448 + ((-3.0 / 32.0) * x453) + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_0_0_m3_VVV) + stencil(gtDD02, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x454)))))));
        x448 = stencil(gtDD11, stencil_idx_m2_0_0_VVV); // x85: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m2__0__0_]
        x451 = stencil(gtDD11, stencil_idx_2_0_0_VVV); // x86: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__2__0__0_]
        x452 = ((x448 + x451)); // x456: Dependency! Liveness = 2; [x85, x86]
        x453 = stencil(gtDD11, stencil_idx_m1_0_0_VVV); // x87: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__m1__0__0_]
        x454 = stencil(gtDD11, stencil_idx_1_0_0_VVV); // x88: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__1__0__0_]
        vreal x457 = ((x453 + x454)); // x457: Dependency! Liveness = 2; [x87, x88]
        vreal x104 = stencil(gtDD11, stencil_idx_0_m2_0_VVV); // x104: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m2__0_]
        vreal x105 = stencil(gtDD11, stencil_idx_0_2_0_VVV); // x105: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__2__0_]
        vreal x458 = ((x104 + x105)); // x458: Dependency! Liveness = 2; [x104, x105]
        x104 = stencil(gtDD11, stencil_idx_0_m1_0_VVV); // x106: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__m1__0_]
        x105 = stencil(gtDD11, stencil_idx_0_1_0_VVV); // x107: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__1__0_]
        vreal x459 = ((x104 + x105)); // x459: Dependency! Liveness = 2; [x106, x107]
        vreal x42 = stencil(gtDD11, stencil_idx_0_0_m2_VVV); // x42: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m2_]
        vreal x43 = stencil(gtDD11, stencil_idx_0_0_2_VVV); // x43: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__2_]
        vreal x460 = ((x42 + x43)); // x460: Dependency! Liveness = 2; [x42, x43]
        x42 = stencil(gtDD11, stencil_idx_0_0_m1_VVV); // x44: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__m1_]
        x43 = stencil(gtDD11, stencil_idx_0_0_1_VVV); // x45: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__1_]
        vreal x461 = ((x42 + x43)); // x461: Dependency! Liveness = 2; [x44, x45]
        vreal x455 = ((-5.0 / 16.0) * stencil(gtDD11, stencil_idx_0_0_0_VVV)); // x455: Dependency! Liveness = 1; [__dummy_stencil__gtDD11__0__0__0_]
        store(gt_rhsDD11, stencil_idx_0_0_0_VVV, (access(gt_rhsDD11, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x455 + ((-3.0 / 32.0) * x452) + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_m3_0_0_VVV) + stencil(gtDD11, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x457))) + (DYI * (x455 + ((-3.0 / 32.0) * x458) + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_0_m3_0_VVV) + stencil(gtDD11, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x459))) + (DZI * (x455 + ((-3.0 / 32.0) * x460) + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_0_0_m3_VVV) + stencil(gtDD11, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x461)))))));
        x455 = stencil(gtDD12, stencil_idx_m2_0_0_VVV); // x63: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m2__0__0_]
        x457 = stencil(gtDD12, stencil_idx_2_0_0_VVV); // x64: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__2__0__0_]
        x458 = ((x455 + x457)); // x463: Dependency! Liveness = 2; [x63, x64]
        x459 = stencil(gtDD12, stencil_idx_m1_0_0_VVV); // x65: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__m1__0__0_]
        x460 = stencil(gtDD12, stencil_idx_1_0_0_VVV); // x66: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__1__0__0_]
        x461 = ((x459 + x460)); // x464: Dependency! Liveness = 2; [x65, x66]
        vreal x109 = stencil(gtDD12, stencil_idx_0_m2_0_VVV); // x109: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m2__0_]
        vreal x110 = stencil(gtDD12, stencil_idx_0_2_0_VVV); // x110: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__2__0_]
        vreal x465 = ((x109 + x110)); // x465: Dependency! Liveness = 2; [x109, x110]
        x109 = stencil(gtDD12, stencil_idx_0_m1_0_VVV); // x111: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__m1__0_]
        x110 = stencil(gtDD12, stencil_idx_0_1_0_VVV); // x112: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__1__0_]
        vreal x466 = ((x109 + x110)); // x466: Dependency! Liveness = 2; [x111, x112]
        vreal x135 = stencil(gtDD12, stencil_idx_0_0_m2_VVV); // x135: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m2_]
        vreal x136 = stencil(gtDD12, stencil_idx_0_0_2_VVV); // x136: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__2_]
        vreal x467 = ((x135 + x136)); // x467: Dependency! Liveness = 2; [x135, x136]
        x135 = stencil(gtDD12, stencil_idx_0_0_m1_VVV); // x137: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__m1_]
        x136 = stencil(gtDD12, stencil_idx_0_0_1_VVV); // x138: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__1_]
        vreal x468 = ((x135 + x136)); // x468: Dependency! Liveness = 2; [x137, x138]
        vreal x462 = ((-5.0 / 16.0) * stencil(gtDD12, stencil_idx_0_0_0_VVV)); // x462: Dependency! Liveness = 1; [__dummy_stencil__gtDD12__0__0__0_]
        store(gt_rhsDD12, stencil_idx_0_0_0_VVV, (access(gt_rhsDD12, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x462 + ((-3.0 / 32.0) * x458) + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_m3_0_0_VVV) + stencil(gtDD12, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x461))) + (DYI * (x462 + ((-3.0 / 32.0) * x465) + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_0_m3_0_VVV) + stencil(gtDD12, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x466))) + (DZI * (x462 + ((-3.0 / 32.0) * x467) + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_0_0_m3_VVV) + stencil(gtDD12, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x468)))))));
        x462 = stencil(gtDD22, stencil_idx_m2_0_0_VVV); // x70: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m2__0__0_]
        x465 = stencil(gtDD22, stencil_idx_2_0_0_VVV); // x71: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__2__0__0_]
        x466 = ((x462 + x465)); // x470: Dependency! Liveness = 2; [x70, x71]
        x467 = stencil(gtDD22, stencil_idx_m1_0_0_VVV); // x72: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__m1__0__0_]
        x468 = stencil(gtDD22, stencil_idx_1_0_0_VVV); // x73: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__1__0__0_]
        vreal x471 = ((x467 + x468)); // x471: Dependency! Liveness = 2; [x72, x73]
        vreal x32 = stencil(gtDD22, stencil_idx_0_m2_0_VVV); // x32: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m2__0_]
        vreal x33 = stencil(gtDD22, stencil_idx_0_2_0_VVV); // x33: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__2__0_]
        vreal x472 = ((x32 + x33)); // x472: Dependency! Liveness = 2; [x32, x33]
        x32 = stencil(gtDD22, stencil_idx_0_m1_0_VVV); // x34: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__m1__0_]
        x33 = stencil(gtDD22, stencil_idx_0_1_0_VVV); // x35: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__1__0_]
        vreal x473 = ((x32 + x33)); // x473: Dependency! Liveness = 2; [x34, x35]
        vreal x129 = stencil(gtDD22, stencil_idx_0_0_m2_VVV); // x129: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m2_]
        vreal x130 = stencil(gtDD22, stencil_idx_0_0_2_VVV); // x130: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__2_]
        vreal x474 = ((x129 + x130)); // x474: Dependency! Liveness = 2; [x129, x130]
        x129 = stencil(gtDD22, stencil_idx_0_0_m1_VVV); // x131: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__m1_]
        x130 = stencil(gtDD22, stencil_idx_0_0_1_VVV); // x132: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__1_]
        vreal x475 = ((x129 + x130)); // x475: Dependency! Liveness = 2; [x131, x132]
        vreal x469 = ((-5.0 / 16.0) * stencil(gtDD22, stencil_idx_0_0_0_VVV)); // x469: Dependency! Liveness = 1; [__dummy_stencil__gtDD22__0__0__0_]
        store(gt_rhsDD22, stencil_idx_0_0_0_VVV, (access(gt_rhsDD22, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x469 + ((-3.0 / 32.0) * x466) + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_m3_0_0_VVV) + stencil(gtDD22, stencil_idx_3_0_0_VVV)))) + ((15.0 / 64.0) * x471))) + (DYI * (x469 + ((-3.0 / 32.0) * x472) + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_0_m3_0_VVV) + stencil(gtDD22, stencil_idx_0_3_0_VVV)))) + ((15.0 / 64.0) * x473))) + (DZI * (x469 + ((-3.0 / 32.0) * x474) + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_0_0_m3_VVV) + stencil(gtDD22, stencil_idx_0_0_3_VVV)))) + ((15.0 / 64.0) * x475)))))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}