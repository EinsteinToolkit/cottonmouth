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
void z4c_to_adm(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_z4c_to_adm;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("z4c_to_adm");
    #endif
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define AtDD00_layout VVV_layout
    #define AtDD01_layout VVV_layout
    #define AtDD02_layout VVV_layout
    #define AtDD11_layout VVV_layout
    #define AtDD12_layout VVV_layout
    #define AtDD22_layout VVV_layout
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
    // z4c_to_adm loop 0
    grid.loop_all_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x79 = access(gtDD00, stencil_idx_0_0_0_VVV); // x79: Dependency! Liveness = 9; [gtDD00, x36, x37, x38, x39, x48, x522, x547, x79]
        vreal x522 = access(chi, stencil_idx_0_0_0_VVV); // x522: Dependency! Liveness = 9; [chi, x36, x37, x38, x39, x48, x522, x547, x79]
        vreal x547 = pown<vreal>(x522, -1); // x547: Dependency! Liveness = 13; [x36, x37, x38, x39, x48, x522, x547, x79, x812, x942, x958, x961, x964]
        vreal x564 = (x547 * x79); // x564: Dependency! Liveness = 7; [x36, x37, x38, x39, x48, x547, x79]
        store(gxx, stencil_idx_0_0_0_VVV, x564); // gxx: 
        vreal x36 = access(gtDD01, stencil_idx_0_0_0_VVV); // x36: Dependency! Liveness = 8; [gtDD01, x36, x37, x38, x39, x48, x547, x79]
        vreal x581 = (x36 * x547); // x581: Dependency! Liveness = 6; [x36, x37, x38, x39, x48, x547]
        store(gxy, stencil_idx_0_0_0_VVV, x581); // gxy: 
        vreal x39 = access(gtDD11, stencil_idx_0_0_0_VVV); // x39: Dependency! Liveness = 8; [gtDD11, x36, x37, x38, x39, x48, x547, x79]
        vreal x585 = (x39 * x547); // x585: Dependency! Liveness = 5; [x37, x38, x39, x48, x547]
        store(gyy, stencil_idx_0_0_0_VVV, x585); // gyy: 
        vreal x38 = access(gtDD02, stencil_idx_0_0_0_VVV); // x38: Dependency! Liveness = 8; [gtDD02, x36, x37, x38, x39, x48, x547, x79]
        vreal x589 = (x38 * x547); // x589: Dependency! Liveness = 4; [x37, x38, x48, x547]
        store(gxz, stencil_idx_0_0_0_VVV, x589); // gxz: 
        vreal x37 = access(gtDD12, stencil_idx_0_0_0_VVV); // x37: Dependency! Liveness = 8; [gtDD12, x36, x37, x38, x39, x48, x547, x79]
        vreal x594 = (x37 * x547); // x594: Dependency! Liveness = 3; [x37, x48, x547]
        store(gyz, stencil_idx_0_0_0_VVV, x594); // gyz: 
        vreal x48 = access(gtDD22, stencil_idx_0_0_0_VVV); // x48: Dependency! Liveness = 8; [gtDD22, x36, x37, x38, x39, x48, x547, x79]
        vreal x598 = (x48 * x547); // x598: Dependency! Liveness = 2; [x48, x547]
        store(gzz, stencil_idx_0_0_0_VVV, x598); // gzz: 
        vreal x831 = access(AtDD00, stencil_idx_0_0_0_VVV); // x831: Dependency! Liveness = 10; [AtDD00, x36, x37, x38, x39, x48, x522, x547, x79, x812]
        vreal x812 = access(trK, stencil_idx_0_0_0_VVV); // x812: Dependency! Liveness = 10; [trK, x36, x37, x38, x39, x48, x522, x547, x79, x812]
        vreal x1103 = ((1.0 / 3.0) * x812); // x1103: Dependency! Liveness = 12; [x36, x37, x38, x39, x48, x547, x79, x812, x942, x958, x961, x964]
        store(kxx, stencil_idx_0_0_0_VVV, (x547 * (x831 + (x1103 * x79)))); // kxx: 
        vreal x832 = access(AtDD01, stencil_idx_0_0_0_VVV); // x832: Dependency! Liveness = 10; [AtDD01, x36, x37, x38, x39, x48, x522, x547, x79, x812]
        store(kxy, stencil_idx_0_0_0_VVV, (x547 * (x832 + (x1103 * x36)))); // kxy: 
        vreal x834 = access(AtDD11, stencil_idx_0_0_0_VVV); // x834: Dependency! Liveness = 10; [AtDD11, x36, x37, x38, x39, x48, x522, x547, x79, x812]
        store(kyy, stencil_idx_0_0_0_VVV, (x547 * (x834 + (x1103 * x39)))); // kyy: 
        vreal x826 = access(AtDD02, stencil_idx_0_0_0_VVV); // x826: Dependency! Liveness = 10; [AtDD02, x36, x37, x38, x39, x48, x522, x547, x79, x812]
        store(kxz, stencil_idx_0_0_0_VVV, (x547 * (x826 + (x1103 * x38)))); // kxz: 
        vreal x828 = access(AtDD12, stencil_idx_0_0_0_VVV); // x828: Dependency! Liveness = 10; [AtDD12, x36, x37, x38, x39, x48, x522, x547, x79, x812]
        store(kyz, stencil_idx_0_0_0_VVV, (x547 * (x828 + (x1103 * x37)))); // kyz: 
        vreal x824 = access(AtDD22, stencil_idx_0_0_0_VVV); // x824: Dependency! Liveness = 10; [AtDD22, x36, x37, x38, x39, x48, x522, x547, x79, x812]
        store(kzz, stencil_idx_0_0_0_VVV, (x547 * (x824 + (x1103 * x48)))); // kzz: 
        vreal x942 = access(evo_lapse, stencil_idx_0_0_0_VVV); // x942: Dependency! Liveness = 11; [evo_lapse, x36, x37, x38, x39, x48, x522, x547, x79, x812, x942]
        store(alp, stencil_idx_0_0_0_VVV, x942); // alp: 
        vreal x958 = access(evo_shiftU0, stencil_idx_0_0_0_VVV); // x958: Dependency! Liveness = 12; [evo_shiftU0, x36, x37, x38, x39, x48, x522, x547, x79, x812, x942, x958]
        store(betax, stencil_idx_0_0_0_VVV, x958); // betax: 
        vreal x961 = access(evo_shiftU1, stencil_idx_0_0_0_VVV); // x961: Dependency! Liveness = 13; [evo_shiftU1, x36, x37, x38, x39, x48, x522, x547, x79, x812, x942, x958, x961]
        store(betay, stencil_idx_0_0_0_VVV, x961); // betay: 
        vreal x964 = access(evo_shiftU2, stencil_idx_0_0_0_VVV); // x964: Dependency! Liveness = 14; [evo_shiftU2, x36, x37, x38, x39, x48, x522, x547, x79, x812, x942, x958, x961, x964]
        store(betaz, stencil_idx_0_0_0_VVV, x964); // betaz:     
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}