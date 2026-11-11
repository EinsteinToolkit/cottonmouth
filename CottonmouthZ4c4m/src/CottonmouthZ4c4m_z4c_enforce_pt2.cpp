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
void z4c_enforce_pt2(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_z4c_enforce_pt2;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("z4c_enforce_pt2");
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
    const auto DX = CCTK_DELTA_SPACE(0);
    const auto DY = CCTK_DELTA_SPACE(1);
    const auto DZ = CCTK_DELTA_SPACE(2);
    const auto DT = cctk_delta_time;
    const auto DXI = (1.0 / DX);
    const auto DYI = (1.0 / DY);
    const auto DZI = (1.0 / DZ);
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // z4c_enforce_pt2 loop 0
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x39 = access(gtDD11, stencil_idx_0_0_0_VVV); // x39: Dependency! Liveness = 7; [gtDD11, x36, x37, x38, x39, x48, x79]
        vreal x79 = access(gtDD00, stencil_idx_0_0_0_VVV); // x79: Dependency! Liveness = 7; [gtDD00, x36, x37, x38, x39, x48, x79]
        vreal x36 = access(gtDD01, stencil_idx_0_0_0_VVV); // x36: Dependency! Liveness = 7; [gtDD01, x36, x37, x38, x39, x48, x79]
        vreal x121 = pow2(x36); // x121: Dependency! Liveness = 3; [x36, x37, x38]
        vreal x122 = ((-(x121)) + (x39 * x79)); // x122: Dependency! Liveness = 7; [x121, x36, x37, x38, x39, x48, x79]
        vreal x834 = access(AtDD22, stencil_idx_0_0_0_VVV); // x834: Dependency! Liveness = 7; [AtDD22, x36, x37, x38, x39, x48, x79]
        vreal x835 = (x122 * x834); // x835: Dependency! Liveness = 11; [x122, x36, x37, x38, x39, x48, x79, x834, x836, x841, x844]
        vreal x836 = access(AtDD02, stencil_idx_0_0_0_VVV); // x836: Dependency! Liveness = 7; [AtDD02, x36, x37, x38, x39, x48, x79]
        vreal x37 = access(gtDD12, stencil_idx_0_0_0_VVV); // x37: Dependency! Liveness = 7; [gtDD12, x36, x37, x38, x39, x48, x79]
        vreal x38 = access(gtDD02, stencil_idx_0_0_0_VVV); // x38: Dependency! Liveness = 7; [gtDD02, x36, x37, x38, x39, x48, x79]
        vreal x40 = ((x36 * x37) + (-1 * x38 * x39)); // x40: Dependency! Liveness = 6; [x36, x37, x38, x39, x48, x79]
        vreal x837 = (x40 * x836); // x837: Dependency! Liveness = 10; [x36, x37, x38, x39, x40, x48, x79, x836, x841, x844]
        vreal x48 = access(gtDD22, stencil_idx_0_0_0_VVV); // x48: Dependency! Liveness = 7; [gtDD22, x36, x37, x38, x39, x48, x79]
        vreal x145 = pow2(x38); // x145: Dependency! Liveness = 3; [x36, x37, x38]
        vreal x146 = ((-(x145)) + (x48 * x79)); // x146: Dependency! Liveness = 7; [x145, x36, x37, x38, x39, x48, x79]
        vreal x844 = access(AtDD11, stencil_idx_0_0_0_VVV); // x844: Dependency! Liveness = 7; [AtDD11, x36, x37, x38, x39, x48, x79]
        vreal x851 = (x146 * x844); // x851: Dependency! Liveness = 9; [x146, x36, x37, x38, x39, x48, x79, x841, x844]
        vreal x57 = pow2(x37); // x57: Dependency! Liveness = 3; [x36, x37, x38]
        vreal x58 = ((-(x57)) + (x39 * x48)); // x58: Dependency! Liveness = 7; [x36, x37, x38, x39, x48, x57, x79]
        vreal x841 = access(AtDD00, stencil_idx_0_0_0_VVV); // x841: Dependency! Liveness = 7; [AtDD00, x36, x37, x38, x39, x48, x79]
        vreal x856 = (x58 * x841); // x856: Dependency! Liveness = 8; [x36, x37, x38, x39, x48, x58, x79, x841]
        vreal x838 = access(AtDD12, stencil_idx_0_0_0_VVV); // x838: Dependency! Liveness = 7; [AtDD12, x36, x37, x38, x39, x48, x79]
        vreal x842 = access(AtDD01, stencil_idx_0_0_0_VVV); // x842: Dependency! Liveness = 7; [AtDD01, x36, x37, x38, x39, x48, x79]
        vreal x49 = ((x36 * x48) + (-1 * x37 * x38)); // x49: Dependency! Liveness = 6; [x36, x37, x38, x39, x48, x79]
        vreal x80 = ((x37 * x79) + (-1 * x36 * x38)); // x80: Dependency! Liveness = 6; [x36, x37, x38, x39, x48, x79]
        vreal x963 = ((1.0 / 3.0) * x79); // x963: Dependency! Liveness = 6; [x36, x37, x38, x39, x48, x79]
        store(AtDD00, stencil_idx_0_0_0_VVV, (x841 + (-1 * x963 * (x835 + x851 + x856)) + ((2.0 / 3.0) * x79 * ((-(x837)) + (x49 * x842) + (x80 * x838)))));
        vreal x964 = ((1.0 / 3.0) * x36); // x964: Dependency! Liveness = 3; [x36, x37, x38]
        store(AtDD01, stencil_idx_0_0_0_VVV, (x842 + (-1 * x964 * (x835 + x851 + x856)) + ((2.0 / 3.0) * x36 * ((-(x837)) + (x49 * x842) + (x80 * x838)))));
        vreal x965 = ((1.0 / 3.0) * x39); // x965: Dependency! Liveness = 5; [x36, x37, x38, x39, x48]
        store(AtDD11, stencil_idx_0_0_0_VVV, (x844 + (-1 * x965 * (x835 + x851 + x856)) + ((2.0 / 3.0) * x39 * ((-(x837)) + (x49 * x842) + (x80 * x838)))));
        vreal x966 = ((1.0 / 3.0) * x38); // x966: Dependency! Liveness = 2; [x37, x38]
        store(AtDD02, stencil_idx_0_0_0_VVV, (x836 + (-1 * x966 * (x835 + x851 + x856)) + ((2.0 / 3.0) * x38 * ((-(x837)) + (x49 * x842) + (x80 * x838)))));
        vreal x967 = ((1.0 / 3.0) * x37); // x967: Dependency! Liveness = 1; [x37]
        store(AtDD12, stencil_idx_0_0_0_VVV, (x838 + (-1 * x967 * (x835 + x851 + x856)) + ((2.0 / 3.0) * x37 * ((-(x837)) + (x49 * x842) + (x80 * x838)))));
        vreal x968 = ((1.0 / 3.0) * x48); // x968: Dependency! Liveness = 4; [x36, x37, x38, x48]
        store(AtDD22, stencil_idx_0_0_0_VVV, (x834 + (-1 * x968 * (x835 + x851 + x856)) + ((2.0 / 3.0) * x48 * ((-(x837)) + (x49 * x842) + (x80 * x838)))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}