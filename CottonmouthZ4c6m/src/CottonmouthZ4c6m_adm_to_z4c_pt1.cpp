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
    const auto DX = CCTK_DELTA_SPACE(0);
    const auto DY = CCTK_DELTA_SPACE(1);
    const auto DZ = CCTK_DELTA_SPACE(2);
    const auto DT = cctk_delta_time;
    const auto DXI = (1.0 / DX);
    const auto DYI = (1.0 / DY);
    const auto DZI = (1.0 / DZ);
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // adm_to_z4c_pt1 loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        store(Theta, stencil_idx_0_0_0_VVV, 0); // Theta: 
        vreal x0 = access(gxx, stencil_idx_0_0_0_VVV); // x0: Dependency! Liveness = 16; [gxx, x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x1 = access(gyz, stencil_idx_0_0_0_VVV); // x1: Dependency! Liveness = 16; [gyz, x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x3 = access(gzz, stencil_idx_0_0_0_VVV); // x3: Dependency! Liveness = 16; [gzz, x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x4 = access(gxy, stencil_idx_0_0_0_VVV); // x4: Dependency! Liveness = 16; [gxy, x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x6 = access(gyy, stencil_idx_0_0_0_VVV); // x6: Dependency! Liveness = 16; [gyy, x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x7 = access(gxz, stencil_idx_0_0_0_VVV); // x7: Dependency! Liveness = 16; [gxz, x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x2 = pow2(x1); // x2: Dependency! Liveness = 4; [x1, x10, x4, x7]
        vreal x5 = pow2(x4); // x5: Dependency! Liveness = 3; [x10, x4, x7]
        vreal x8 = pow2(x7); // x8: Dependency! Liveness = 2; [x10, x7]
        vreal x9 = ((x0 * (x2 + (-1 * x3 * x6))) + (x3 * x5) + (x6 * x8) + (-2 * x1 * x4 * x7)); // x9: Dependency! Liveness = 30; [x0, x1, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x2, x20, x21, x22, x23, x24, x26, x28, x3, x30, x32, x34, x4, x5, x6, x7, x8, x9]
        vreal x10 = pow(static_cast<vreal>((-(x9))), (-1.0 / 3.0)); // x10: Dependency! Liveness = 15; [x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        store(chi, stencil_idx_0_0_0_VVV, x10); // chi: 
        vreal x12 = access(kzz, stencil_idx_0_0_0_VVV); // x12: Dependency! Liveness = 16; [kzz, x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x13 = (x12 * ((-(x5)) + (x0 * x6))); // x13: Dependency! Liveness = 30; [x0, x1, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x2, x20, x21, x22, x23, x24, x26, x28, x3, x30, x32, x34, x4, x5, x6, x7, x8, x9]
        vreal x14 = access(kyy, stencil_idx_0_0_0_VVV); // x14: Dependency! Liveness = 16; [kyy, x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x15 = (x14 * ((-(x8)) + (x0 * x3))); // x15: Dependency! Liveness = 28; [x0, x1, x10, x11, x13, x14, x15, x16, x17, x18, x19, x2, x20, x21, x22, x23, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x8, x9]
        vreal x16 = access(kxx, stencil_idx_0_0_0_VVV); // x16: Dependency! Liveness = 16; [kxx, x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x17 = (x16 * ((-(x2)) + (x3 * x6))); // x17: Dependency! Liveness = 26; [x0, x1, x10, x11, x13, x15, x16, x17, x18, x19, x2, x20, x21, x22, x23, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x20 = access(kxz, stencil_idx_0_0_0_VVV); // x20: Dependency! Liveness = 16; [kxz, x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x21 = (x20 * ((x1 * x4) + (-1 * x6 * x7))); // x21: Dependency! Liveness = 16; [x0, x1, x10, x11, x20, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x18 = access(kyz, stencil_idx_0_0_0_VVV); // x18: Dependency! Liveness = 16; [kyz, x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x22 = access(kxy, stencil_idx_0_0_0_VVV); // x22: Dependency! Liveness = 16; [kxy, x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x19 = ((x0 * x1) + (-1 * x4 * x7)); // x19: Dependency! Liveness = 15; [x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x23 = ((x3 * x4) + (-1 * x1 * x7)); // x23: Dependency! Liveness = 15; [x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        vreal x11 = pown<vreal>(x9, -1); // x11: Dependency! Liveness = 15; [x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7, x9]
        store(trK, stencil_idx_0_0_0_VVV, (x11 * ((-(x13)) + (-(x15)) + (-(x17)) + (-2 * x21) + (2 * x18 * x19) + (2 * x22 * x23)))); // trK: 
        store(gtDD00, stencil_idx_0_0_0_VVV, (x0 * x10)); // gtDD00: 
        store(gtDD01, stencil_idx_0_0_0_VVV, (x10 * x4)); // gtDD01: 
        store(gtDD11, stencil_idx_0_0_0_VVV, (x10 * x6)); // gtDD11: 
        store(gtDD02, stencil_idx_0_0_0_VVV, (x10 * x7)); // gtDD02: 
        store(gtDD12, stencil_idx_0_0_0_VVV, (x1 * x10)); // gtDD12: 
        store(gtDD22, stencil_idx_0_0_0_VVV, (x10 * x3)); // gtDD22: 
        vreal x24 = (x0 * x11); // x24: Dependency! Liveness = 8; [x0, x1, x10, x11, x3, x4, x6, x7]
        vreal x25 = ((1.0 / 3.0) * x24); // x25: Dependency! Liveness = 14; [x0, x1, x10, x11, x24, x26, x28, x3, x30, x32, x34, x4, x6, x7]
        store(AtDD00, stencil_idx_0_0_0_VVV, (-1 * x10 * ((-(x16)) + (-1 * x25 * (x13 + x15 + x17)) + ((-2.0 / 3.0) * x21 * x24) + ((2.0 / 3.0) * x0 * x11 * ((x18 * x19) + (x22 * x23)))))); // AtDD00: 
        vreal x26 = (x11 * x4); // x26: Dependency! Liveness = 5; [x1, x10, x11, x4, x7]
        vreal x27 = ((1.0 / 3.0) * x26); // x27: Dependency! Liveness = 13; [x0, x1, x10, x11, x26, x28, x3, x30, x32, x34, x4, x6, x7]
        store(AtDD01, stencil_idx_0_0_0_VVV, (-1 * x10 * ((-(x22)) + (-1 * x27 * (x13 + x15 + x17)) + ((-2.0 / 3.0) * x21 * x26) + ((2.0 / 3.0) * x11 * x4 * ((x18 * x19) + (x22 * x23)))))); // AtDD01: 
        vreal x28 = (x11 * x6); // x28: Dependency! Liveness = 7; [x1, x10, x11, x3, x4, x6, x7]
        vreal x29 = ((1.0 / 3.0) * x28); // x29: Dependency! Liveness = 12; [x0, x1, x10, x11, x28, x3, x30, x32, x34, x4, x6, x7]
        store(AtDD11, stencil_idx_0_0_0_VVV, (-1 * x10 * ((-(x14)) + (-1 * x29 * (x13 + x15 + x17)) + ((-2.0 / 3.0) * x21 * x28) + ((2.0 / 3.0) * x11 * x6 * ((x18 * x19) + (x22 * x23)))))); // AtDD11: 
        vreal x30 = (x11 * x7); // x30: Dependency! Liveness = 5; [x1, x10, x11, x4, x7]
        vreal x31 = ((1.0 / 3.0) * x30); // x31: Dependency! Liveness = 11; [x0, x1, x10, x11, x3, x30, x32, x34, x4, x6, x7]
        store(AtDD02, stencil_idx_0_0_0_VVV, (-1 * x10 * ((-(x20)) + (-1 * x31 * (x13 + x15 + x17)) + ((-2.0 / 3.0) * x21 * x30) + ((2.0 / 3.0) * x11 * x7 * ((x18 * x19) + (x22 * x23)))))); // AtDD02: 
        vreal x32 = (x1 * x11); // x32: Dependency! Liveness = 5; [x1, x10, x11, x4, x7]
        vreal x33 = ((1.0 / 3.0) * x32); // x33: Dependency! Liveness = 10; [x0, x1, x10, x11, x3, x32, x34, x4, x6, x7]
        store(AtDD12, stencil_idx_0_0_0_VVV, (-1 * x10 * ((-(x18)) + (-1 * x33 * (x13 + x15 + x17)) + ((-2.0 / 3.0) * x21 * x32) + ((2.0 / 3.0) * x1 * x11 * ((x18 * x19) + (x22 * x23)))))); // AtDD12: 
        vreal x34 = (x11 * x3); // x34: Dependency! Liveness = 6; [x1, x10, x11, x3, x4, x7]
        vreal x35 = ((1.0 / 3.0) * x34); // x35: Dependency! Liveness = 9; [x0, x1, x10, x11, x3, x34, x4, x6, x7]
        store(AtDD22, stencil_idx_0_0_0_VVV, (-1 * x10 * ((-(x12)) + (-1 * x35 * (x13 + x15 + x17)) + ((-2.0 / 3.0) * x21 * x34) + ((2.0 / 3.0) * x11 * x3 * ((x18 * x19) + (x22 * x23)))))); // AtDD22: 
        store(evo_lapse, stencil_idx_0_0_0_VVV, access(alp, stencil_idx_0_0_0_VVV)); // evo_lapse: 
        store(evo_shiftU0, stencil_idx_0_0_0_VVV, access(betax, stencil_idx_0_0_0_VVV)); // evo_shiftU0: 
        store(evo_shiftU1, stencil_idx_0_0_0_VVV, access(betay, stencil_idx_0_0_0_VVV)); // evo_shiftU1: 
        store(evo_shiftU2, stencil_idx_0_0_0_VVV, access(betaz, stencil_idx_0_0_0_VVV)); // evo_shiftU2:     
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}