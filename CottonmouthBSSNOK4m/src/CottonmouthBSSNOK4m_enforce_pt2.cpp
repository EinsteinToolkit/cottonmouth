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
void enforce_pt2(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_enforce_pt2;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("enforce_pt2");
    #endif
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define AtDD00_layout VVV_layout
    #define AtDD01_layout VVV_layout
    #define AtDD02_layout VVV_layout
    #define AtDD11_layout VVV_layout
    #define AtDD12_layout VVV_layout
    #define AtDD22_layout VVV_layout
    #define gtDD00_layout VVV_layout
    #define gtDD01_layout VVV_layout
    #define gtDD02_layout VVV_layout
    #define gtDD11_layout VVV_layout
    #define gtDD12_layout VVV_layout
    #define gtDD22_layout VVV_layout
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // enforce_pt2 loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x579 = access(AtDD12, stencil_idx_0_0_0_VVV); // x579: Dependency! Liveness = 7; [AtDD12, x139, x149, x47, x48, x50, x51]
        vreal x47 = access(gtDD00, stencil_idx_0_0_0_VVV); // x47: Dependency! Liveness = 4; [gtDD00, x139, x149, x47]
        vreal x48 = access(gtDD12, stencil_idx_0_0_0_VVV); // x48: Dependency! Liveness = 5; [gtDD12, x139, x149, x47, x48]
        vreal x49 = (x47 * x48); // x49: Dependency! Liveness = 6; [x47, x48, x50, x51, x62, x95]
        vreal x50 = access(gtDD01, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 6; [gtDD01, x139, x149, x47, x48, x50]
        vreal x51 = access(gtDD02, stencil_idx_0_0_0_VVV); // x51: Dependency! Liveness = 7; [gtDD02, x139, x149, x47, x48, x50, x51]
        vreal x148 = (x50 * x51); // x148: Dependency! Liveness = 6; [x47, x48, x50, x51, x62, x95]
        vreal x149 = (x49 + (-(x148))); // x149: Dependency! Liveness = 11; [x131, x139, x148, x149, x49, x573, x576, x577, x578, x579, x580]
        vreal x592 = (-(x149)); // x592: Dependency! Liveness = 8; [x139, x149, x47, x48, x50, x51, x62, x95]
        vreal x1014 = (x579 * x592); // x1014: Dependency! Liveness = 13; [x131, x139, x149, x161, x186, x222, x573, x576, x577, x578, x579, x580, x592]
        vreal x62 = access(gtDD22, stencil_idx_0_0_0_VVV); // x62: Dependency! Liveness = 8; [gtDD22, x139, x149, x47, x48, x50, x51, x62]
        vreal x63 = (x47 * x62); // x63: Dependency! Liveness = 6; [x47, x48, x50, x51, x62, x95]
        vreal x64 = pow2(x51); // x64: Dependency! Liveness = 3; [x48, x50, x51]
        vreal x131 = (x63 + (-(x64))); // x131: Dependency! Liveness = 9; [x131, x573, x576, x577, x578, x579, x580, x63, x64]
        vreal x578 = access(AtDD11, stencil_idx_0_0_0_VVV); // x578: Dependency! Liveness = 7; [AtDD11, x139, x149, x47, x48, x50, x51]
        vreal x1015 = (x131 * x578); // x1015: Dependency! Liveness = 11; [x131, x139, x149, x161, x186, x222, x573, x576, x577, x578, x580]
        vreal x576 = access(AtDD01, stencil_idx_0_0_0_VVV); // x576: Dependency! Liveness = 7; [AtDD01, x139, x149, x47, x48, x50, x51]
        vreal x73 = (x50 * x62); // x73: Dependency! Liveness = 6; [x47, x48, x50, x51, x62, x95]
        vreal x138 = (x48 * x51); // x138: Dependency! Liveness = 6; [x47, x48, x50, x51, x62, x95]
        vreal x139 = (x73 + (-(x138))); // x139: Dependency! Liveness = 10; [x131, x138, x139, x573, x576, x577, x578, x579, x580, x73]
        vreal x600 = (-(x139)); // x600: Dependency! Liveness = 7; [x139, x47, x48, x50, x51, x62, x95]
        vreal x1016 = (x576 * x600); // x1016: Dependency! Liveness = 10; [x139, x149, x161, x186, x222, x573, x576, x577, x580, x600]
        vreal x95 = access(gtDD11, stencil_idx_0_0_0_VVV); // x95: Dependency! Liveness = 9; [gtDD11, x139, x149, x47, x48, x50, x51, x62, x95]
        vreal x184 = (x47 * x95); // x184: Dependency! Liveness = 6; [x47, x48, x50, x51, x62, x95]
        vreal x185 = pow2(x50); // x185: Dependency! Liveness = 3; [x48, x50, x51]
        vreal x186 = (x184 + (-(x185))); // x186: Dependency! Liveness = 13; [x131, x139, x149, x161, x184, x185, x186, x573, x576, x577, x578, x579, x580]
        vreal x580 = access(AtDD22, stencil_idx_0_0_0_VVV); // x580: Dependency! Liveness = 7; [AtDD22, x139, x149, x47, x48, x50, x51]
        vreal x846 = (x186 * x580); // x846: Dependency! Liveness = 8; [x139, x149, x161, x186, x222, x573, x577, x580]
        vreal x96 = (x51 * x95); // x96: Dependency! Liveness = 6; [x47, x48, x50, x51, x62, x95]
        vreal x94 = (x48 * x50); // x94: Dependency! Liveness = 6; [x47, x48, x50, x51, x62, x95]
        vreal x222 = (x94 + (-(x96))); // x222: Dependency! Liveness = 14; [x131, x139, x149, x161, x186, x222, x573, x576, x577, x578, x579, x580, x94, x96]
        vreal x577 = access(AtDD02, stencil_idx_0_0_0_VVV); // x577: Dependency! Liveness = 7; [AtDD02, x139, x149, x47, x48, x50, x51]
        vreal x847 = (x222 * x577); // x847: Dependency! Liveness = 6; [x139, x149, x161, x222, x573, x577]
        vreal x159 = (x62 * x95); // x159: Dependency! Liveness = 6; [x47, x48, x50, x51, x62, x95]
        vreal x160 = pow2(x48); // x160: Dependency! Liveness = 3; [x48, x50, x51]
        vreal x161 = (x159 + (-(x160))); // x161: Dependency! Liveness = 12; [x131, x139, x149, x159, x160, x161, x573, x576, x577, x578, x579, x580]
        vreal x573 = access(AtDD00, stencil_idx_0_0_0_VVV); // x573: Dependency! Liveness = 7; [AtDD00, x139, x149, x47, x48, x50, x51]
        vreal x851 = (x161 * x573); // x851: Dependency! Liveness = 4; [x139, x149, x161, x573]
        vreal x1012 = ((1.0 / 3.0) * x47); // x1012: Dependency! Liveness = 6; [x47, x48, x50, x51, x62, x95]
        vreal x1013 = ((2.0 / 3.0) * x47); // x1013: Dependency! Liveness = 6; [x47, x48, x50, x51, x62, x95]
        store(AtDD00, stencil_idx_0_0_0_VVV, (x573 + (-1 * x1012 * x1015) + (-1 * x1012 * x846) + (-1 * x1012 * x851) + (-1 * x1013 * x1014) + (-1 * x1013 * x1016) + (-1 * x1013 * x847)));
        vreal x1017 = ((1.0 / 3.0) * x50); // x1017: Dependency! Liveness = 3; [x48, x50, x51]
        vreal x1018 = ((2.0 / 3.0) * x50); // x1018: Dependency! Liveness = 3; [x48, x50, x51]
        store(AtDD01, stencil_idx_0_0_0_VVV, (x576 + (-1 * x1014 * x1018) + (-1 * x1015 * x1017) + (-1 * x1016 * x1018) + (-1 * x1017 * x846) + (-1 * x1017 * x851) + (-1 * x1018 * x847)));
        vreal x1019 = ((1.0 / 3.0) * x51); // x1019: Dependency! Liveness = 2; [x48, x51]
        vreal x1020 = ((2.0 / 3.0) * x51); // x1020: Dependency! Liveness = 2; [x48, x51]
        store(AtDD02, stencil_idx_0_0_0_VVV, (x577 + (-1 * x1014 * x1020) + (-1 * x1015 * x1019) + (-1 * x1016 * x1020) + (-1 * x1019 * x846) + (-1 * x1019 * x851) + (-1 * x1020 * x847)));
        vreal x1021 = ((1.0 / 3.0) * x95); // x1021: Dependency! Liveness = 5; [x48, x50, x51, x62, x95]
        vreal x1022 = ((2.0 / 3.0) * x95); // x1022: Dependency! Liveness = 5; [x48, x50, x51, x62, x95]
        store(AtDD11, stencil_idx_0_0_0_VVV, (x578 + (-1 * x1014 * x1022) + (-1 * x1015 * x1021) + (-1 * x1016 * x1022) + (-1 * x1021 * x846) + (-1 * x1021 * x851) + (-1 * x1022 * x847)));
        vreal x1023 = ((1.0 / 3.0) * x48); // x1023: Dependency! Liveness = 1; [x48]
        vreal x1024 = ((2.0 / 3.0) * x48); // x1024: Dependency! Liveness = 1; [x48]
        store(AtDD12, stencil_idx_0_0_0_VVV, (x579 + (-1 * x1014 * x1024) + (-1 * x1015 * x1023) + (-1 * x1016 * x1024) + (-1 * x1023 * x846) + (-1 * x1023 * x851) + (-1 * x1024 * x847)));
        vreal x1025 = ((1.0 / 3.0) * x62); // x1025: Dependency! Liveness = 4; [x48, x50, x51, x62]
        vreal x1026 = ((2.0 / 3.0) * x62); // x1026: Dependency! Liveness = 4; [x48, x50, x51, x62]
        store(AtDD22, stencil_idx_0_0_0_VVV, (x580 + (-1 * x1014 * x1026) + (-1 * x1015 * x1025) + (-1 * x1016 * x1026) + (-1 * x1025 * x846) + (-1 * x1025 * x851) + (-1 * x1026 * x847)));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}