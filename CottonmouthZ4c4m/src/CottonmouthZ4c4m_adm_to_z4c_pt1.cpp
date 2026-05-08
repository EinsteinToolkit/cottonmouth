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
void adm_to_z4c_pt1(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_adm_to_z4c_pt1;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("adm_to_z4c_pt1");
    #endif
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define AtDD00_layout VVV_layout
    #define AtDD01_layout VVV_layout
    #define AtDD02_layout VVV_layout
    #define AtDD11_layout VVV_layout
    #define AtDD12_layout VVV_layout
    #define AtDD22_layout VVV_layout
    #define Theta_layout VVV_layout
    #define alp_layout VVV_layout
    #define betax_layout VVV_layout
    #define betay_layout VVV_layout
    #define betaz_layout VVV_layout
    #define chi_layout VVV_layout
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
    #define trK_layout VVV_layout
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // adm_to_z4c_pt1 loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x0 = pow2(access(gyz, stencil_idx_0_0_0_VVV)); // x0: Dependency! Liveness = 4; [gxy, gxz, gyz, x4]
        vreal x1 = pow2(access(gxy, stencil_idx_0_0_0_VVV)); // x1: Dependency! Liveness = 4; [gxy, gxz, gyz, x4]
        vreal x2 = pow2(access(gxz, stencil_idx_0_0_0_VVV)); // x2: Dependency! Liveness = 4; [gxy, gxz, gyz, x4]
        vreal x3 = ((access(gxx, stencil_idx_0_0_0_VVV) * x0) + (access(gyy, stencil_idx_0_0_0_VVV) * x2) + (access(gzz, stencil_idx_0_0_0_VVV) * x1) + (-1 * access(gxx, stencil_idx_0_0_0_VVV) * access(gyy, stencil_idx_0_0_0_VVV) * access(gzz, stencil_idx_0_0_0_VVV)) + (-2 * access(gxy, stencil_idx_0_0_0_VVV) * access(gxz, stencil_idx_0_0_0_VVV) * access(gyz, stencil_idx_0_0_0_VVV))); // x3: Dependency! Liveness = 17; [gxx, gxy, gxz, gyy, gyz, gzz, kxx, kxy, kxz, kyy, kyz, kzz, x0, x1, x2, x3, x4]
        vreal x5 = pown<vreal>(x3, -1); // x5: Dependency! Liveness = 9; [gxx, gxy, gxz, gyy, gyz, gzz, x3, x4, x5]
        vreal x12 = (access(kyy, stencil_idx_0_0_0_VVV) * x5 * (x2 + (-1 * access(gxx, stencil_idx_0_0_0_VVV) * access(gzz, stencil_idx_0_0_0_VVV)))); // x12: Dependency! Liveness = 18; [gxx, gxy, gxz, gyy, gyz, gzz, kxx, kxy, kxz, kyy, kyz, kzz, x0, x1, x2, x3, x4, x5]
        x2 = (access(kzz, stencil_idx_0_0_0_VVV) * x5 * (x1 + (-1 * access(gxx, stencil_idx_0_0_0_VVV) * access(gyy, stencil_idx_0_0_0_VVV)))); // x14: Dependency! Liveness = 16; [gxx, gxy, gxz, gyy, gyz, gzz, kxx, kxy, kxz, kyz, kzz, x0, x1, x3, x4, x5]
        x1 = (access(kxx, stencil_idx_0_0_0_VVV) * x5 * (x0 + (-1 * access(gyy, stencil_idx_0_0_0_VVV) * access(gzz, stencil_idx_0_0_0_VVV)))); // x6: Dependency! Liveness = 14; [gxx, gxy, gxz, gyy, gyz, gzz, kxx, kxy, kxz, kyz, x0, x3, x4, x5]
        x0 = (access(kxz, stencil_idx_0_0_0_VVV) * ((access(gxz, stencil_idx_0_0_0_VVV) * access(gyy, stencil_idx_0_0_0_VVV)) + (-1 * access(gxy, stencil_idx_0_0_0_VVV) * access(gyz, stencil_idx_0_0_0_VVV)))); // x11: Dependency! Liveness = 12; [gxx, gxy, gxz, gyy, gyz, gzz, kxy, kxz, kyz, x3, x4, x5]
        vreal x13 = (access(kyz, stencil_idx_0_0_0_VVV) * ((access(gxx, stencil_idx_0_0_0_VVV) * access(gyz, stencil_idx_0_0_0_VVV)) + (-1 * access(gxy, stencil_idx_0_0_0_VVV) * access(gxz, stencil_idx_0_0_0_VVV)))); // x13: Dependency! Liveness = 11; [gxx, gxy, gxz, gyy, gyz, gzz, kxy, kyz, x3, x4, x5]
        vreal x8 = (access(kxy, stencil_idx_0_0_0_VVV) * ((access(gxy, stencil_idx_0_0_0_VVV) * access(gzz, stencil_idx_0_0_0_VVV)) + (-1 * access(gxz, stencil_idx_0_0_0_VVV) * access(gyz, stencil_idx_0_0_0_VVV)))); // x8: Dependency! Liveness = 10; [gxx, gxy, gxz, gyy, gyz, gzz, kxy, x3, x4, x5]
        vreal x4 = pow(static_cast<vreal>((-(x3))), (-1.0 / 3.0)); // x4: Dependency! Liveness = 9; [gxx, gxy, gxz, gyy, gyz, gzz, x3, x4, x5]
        x3 = ((2.0 / 3.0) * x5); // x9: Dependency! Liveness = 8; [gxx, gxy, gxz, gyy, gyz, gzz, x4, x5]
        vreal x10 = (access(gxx, stencil_idx_0_0_0_VVV) * x3); // x10: Dependency! Liveness = 9; [gxx, gxy, gxz, gyy, gyz, gzz, x4, x5, x9]
        vreal x7 = ((1.0 / 3.0) * access(gxx, stencil_idx_0_0_0_VVV)); // x7: Dependency! Liveness = 5; [gxx, gxy, gxz, gyz, x4]
        store(AtDD00, stencil_idx_0_0_0_VVV, (x4 * (access(kxx, stencil_idx_0_0_0_VVV) + (-1 * x0 * x10) + (-1 * x1 * x7) + (-1 * x10 * x13) + (-1 * x10 * x8) + (-1 * x12 * x7) + (-1 * x2 * x7)))); // AtDD00: Liveness = 11; [AtDD00, kxx, x10, x11, x12, x13, x14, x4, x6, x7, x8]
        x10 = (access(gxy, stencil_idx_0_0_0_VVV) * x3); // x16: Dependency! Liveness = 9; [gxx, gxy, gxz, gyy, gyz, gzz, x4, x5, x9]
        x7 = ((1.0 / 3.0) * access(gxy, stencil_idx_0_0_0_VVV)); // x15: Dependency! Liveness = 4; [gxy, gxz, gyz, x4]
        store(AtDD01, stencil_idx_0_0_0_VVV, (x4 * (access(kxy, stencil_idx_0_0_0_VVV) + (-1 * x0 * x10) + (-1 * x1 * x7) + (-1 * x10 * x13) + (-1 * x10 * x8) + (-1 * x12 * x7) + (-1 * x2 * x7)))); // AtDD01: Liveness = 12; [AtDD01, kxx, kxy, x11, x12, x13, x14, x15, x16, x4, x6, x8]
        vreal x18 = (access(gxz, stencil_idx_0_0_0_VVV) * x3); // x18: Dependency! Liveness = 9; [gxx, gxy, gxz, gyy, gyz, gzz, x4, x5, x9]
        vreal x17 = ((1.0 / 3.0) * access(gxz, stencil_idx_0_0_0_VVV)); // x17: Dependency! Liveness = 3; [gxz, gyz, x4]
        store(AtDD02, stencil_idx_0_0_0_VVV, (x4 * (access(kxz, stencil_idx_0_0_0_VVV) + (-1 * x0 * x18) + (-1 * x1 * x17) + (-1 * x12 * x17) + (-1 * x13 * x18) + (-1 * x17 * x2) + (-1 * x18 * x8)))); // AtDD02: Liveness = 13; [AtDD02, kxx, kxy, kxz, x11, x12, x13, x14, x17, x18, x4, x6, x8]
        x17 = (access(gyy, stencil_idx_0_0_0_VVV) * x3); // x20: Dependency! Liveness = 9; [gxx, gxy, gxz, gyy, gyz, gzz, x4, x5, x9]
        x18 = ((1.0 / 3.0) * access(gyy, stencil_idx_0_0_0_VVV)); // x19: Dependency! Liveness = 7; [gxx, gxy, gxz, gyy, gyz, gzz, x4]
        store(AtDD11, stencil_idx_0_0_0_VVV, (x4 * (access(kyy, stencil_idx_0_0_0_VVV) + (-1 * x0 * x17) + (-1 * x1 * x18) + (-1 * x12 * x18) + (-1 * x13 * x17) + (-1 * x17 * x8) + (-1 * x18 * x2)))); // AtDD11: Liveness = 14; [AtDD11, kxx, kxy, kxz, kyy, x11, x12, x13, x14, x19, x20, x4, x6, x8]
        vreal x22 = (access(gyz, stencil_idx_0_0_0_VVV) * x3); // x22: Dependency! Liveness = 9; [gxx, gxy, gxz, gyy, gyz, gzz, x4, x5, x9]
        vreal x21 = ((1.0 / 3.0) * access(gyz, stencil_idx_0_0_0_VVV)); // x21: Dependency! Liveness = 2; [gyz, x4]
        store(AtDD12, stencil_idx_0_0_0_VVV, (x4 * (access(kyz, stencil_idx_0_0_0_VVV) + (-1 * x0 * x22) + (-1 * x1 * x21) + (-1 * x12 * x21) + (-1 * x13 * x22) + (-1 * x2 * x21) + (-1 * x22 * x8)))); // AtDD12: Liveness = 15; [AtDD12, kxx, kxy, kxz, kyy, kyz, x11, x12, x13, x14, x21, x22, x4, x6, x8]
        x21 = (access(gzz, stencil_idx_0_0_0_VVV) * x3); // x24: Dependency! Liveness = 9; [gxx, gxy, gxz, gyy, gyz, gzz, x4, x5, x9]
        x22 = ((1.0 / 3.0) * access(gzz, stencil_idx_0_0_0_VVV)); // x23: Dependency! Liveness = 6; [gxx, gxy, gxz, gyz, gzz, x4]
        store(AtDD22, stencil_idx_0_0_0_VVV, (x4 * (access(kzz, stencil_idx_0_0_0_VVV) + (-1 * x0 * x21) + (-1 * x1 * x22) + (-1 * x12 * x22) + (-1 * x13 * x21) + (-1 * x2 * x22) + (-1 * x21 * x8)))); // AtDD22: Liveness = 16; [AtDD22, kxx, kxy, kxz, kyy, kyz, kzz, x11, x12, x13, x14, x23, x24, x4, x6, x8]
        vreal x25 = (2 * x5); // x25: Dependency! Liveness = 8; [gxx, gxy, gxz, gyy, gyz, gzz, x4, x5]
        store(trK, stencil_idx_0_0_0_VVV, (x1 + x12 + x2 + (x0 * x25) + (x13 * x25) + (x25 * x8))); // trK: Liveness = 15; [kxx, kxy, kxz, kyy, kyz, kzz, trK, x11, x12, x13, x14, x25, x4, x6, x8]
        store(evo_lapse, stencil_idx_0_0_0_VVV, access(alp, stencil_idx_0_0_0_VVV)); // evo_lapse: Liveness = 11; [alp, evo_lapse, gxx, gxy, gxz, gyy, gyz, gzz, x3, x4, x5]
        store(evo_shiftU0, stencil_idx_0_0_0_VVV, access(betax, stencil_idx_0_0_0_VVV)); // evo_shiftU0: Liveness = 11; [betax, evo_shiftU0, gxx, gxy, gxz, gyy, gyz, gzz, x3, x4, x5]
        store(evo_shiftU1, stencil_idx_0_0_0_VVV, access(betay, stencil_idx_0_0_0_VVV)); // evo_shiftU1: Liveness = 11; [betay, evo_shiftU1, gxx, gxy, gxz, gyy, gyz, gzz, x3, x4, x5]
        store(evo_shiftU2, stencil_idx_0_0_0_VVV, access(betaz, stencil_idx_0_0_0_VVV)); // evo_shiftU2: Liveness = 11; [betaz, evo_shiftU2, gxx, gxy, gxz, gyy, gyz, gzz, x3, x4, x5]
        store(gtDD00, stencil_idx_0_0_0_VVV, (access(gxx, stencil_idx_0_0_0_VVV) * x4)); // gtDD00: Liveness = 9; [gtDD00, gxx, gxy, gxz, gyy, gyz, gzz, x4, x5]
        store(gtDD11, stencil_idx_0_0_0_VVV, (access(gyy, stencil_idx_0_0_0_VVV) * x4)); // gtDD11: Liveness = 9; [gtDD11, gxx, gxy, gxz, gyy, gyz, gzz, x4, x5]
        store(gtDD22, stencil_idx_0_0_0_VVV, (access(gzz, stencil_idx_0_0_0_VVV) * x4)); // gtDD22: Liveness = 9; [gtDD22, gxx, gxy, gxz, gyy, gyz, gzz, x4, x5]
        store(gtDD01, stencil_idx_0_0_0_VVV, (access(gxy, stencil_idx_0_0_0_VVV) * x4)); // gtDD01: Liveness = 9; [gtDD01, gxx, gxy, gxz, gyy, gyz, gzz, x4, x5]
        store(gtDD02, stencil_idx_0_0_0_VVV, (access(gxz, stencil_idx_0_0_0_VVV) * x4)); // gtDD02: Liveness = 9; [gtDD02, gxx, gxy, gxz, gyy, gyz, gzz, x4, x5]
        store(gtDD12, stencil_idx_0_0_0_VVV, (access(gyz, stencil_idx_0_0_0_VVV) * x4)); // gtDD12: Liveness = 9; [gtDD12, gxx, gxy, gxz, gyy, gyz, gzz, x4, x5]
        store(chi, stencil_idx_0_0_0_VVV, x4); // chi: Liveness = 2; [chi, x4]
        store(Theta, stencil_idx_0_0_0_VVV, 0); // Theta: Liveness = 1; [Theta]    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}