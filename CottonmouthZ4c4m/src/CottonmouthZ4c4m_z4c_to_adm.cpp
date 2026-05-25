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
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // z4c_to_adm loop 0
    grid.loop_all_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x38 = access(gtDD00, stencil_idx_0_0_0_VVV); // x38: Dependency! Liveness = 8; [gtDD00, x38, x39, x41, x42, x46, x539, x55]
        vreal x845 = access(AtDD00, stencil_idx_0_0_0_VVV); // x845: Dependency! Liveness = 13; [AtDD00, x38, x39, x41, x42, x46, x507, x539, x55, x794, x797, x799, x801]
        vreal x507 = access(chi, stencil_idx_0_0_0_VVV); // x507: Dependency! Liveness = 9; [chi, x38, x39, x41, x42, x46, x507, x539, x55]
        vreal x539 = pown<vreal>(x507, -1); // x539: Dependency! Liveness = 13; [x38, x39, x41, x42, x46, x507, x539, x55, x794, x797, x799, x801, x954]
        x507 = access(trK, stencil_idx_0_0_0_VVV); // x954: Dependency! Liveness = 14; [trK, x38, x39, x41, x42, x46, x507, x539, x55, x794, x797, x799, x801, x954]
        vreal x1136 = ((1.0 / 3.0) * x507); // x1136: Dependency! Liveness = 12; [x38, x39, x41, x42, x46, x539, x55, x794, x797, x799, x801, x954]
        store(kxx, stencil_idx_0_0_0_VVV, (x539 * (x845 + (x1136 * x38)))); // kxx: Liveness = 5; [kxx, x1136, x38, x539, x845]
        x845 = access(gtDD01, stencil_idx_0_0_0_VVV); // x41: Dependency! Liveness = 8; [gtDD01, x38, x39, x41, x42, x46, x539, x55]
        vreal x831 = access(AtDD01, stencil_idx_0_0_0_VVV); // x831: Dependency! Liveness = 13; [AtDD01, x38, x39, x41, x42, x46, x507, x539, x55, x794, x797, x799, x801]
        store(kxy, stencil_idx_0_0_0_VVV, (x539 * (x831 + (x1136 * x845)))); // kxy: Liveness = 6; [kxy, x1136, x38, x41, x539, x831]
        x831 = access(gtDD02, stencil_idx_0_0_0_VVV); // x42: Dependency! Liveness = 8; [gtDD02, x38, x39, x41, x42, x46, x539, x55]
        vreal x841 = access(AtDD02, stencil_idx_0_0_0_VVV); // x841: Dependency! Liveness = 13; [AtDD02, x38, x39, x41, x42, x46, x507, x539, x55, x794, x797, x799, x801]
        store(kxz, stencil_idx_0_0_0_VVV, (x539 * (x841 + (x1136 * x831)))); // kxz: Liveness = 7; [kxz, x1136, x38, x41, x42, x539, x841]
        x841 = access(gtDD11, stencil_idx_0_0_0_VVV); // x46: Dependency! Liveness = 8; [gtDD11, x38, x39, x41, x42, x46, x539, x55]
        vreal x832 = access(AtDD11, stencil_idx_0_0_0_VVV); // x832: Dependency! Liveness = 13; [AtDD11, x38, x39, x41, x42, x46, x507, x539, x55, x794, x797, x799, x801]
        store(kyy, stencil_idx_0_0_0_VVV, (x539 * (x832 + (x1136 * x841)))); // kyy: Liveness = 8; [kyy, x1136, x38, x41, x42, x46, x539, x832]
        x832 = access(gtDD12, stencil_idx_0_0_0_VVV); // x39: Dependency! Liveness = 8; [gtDD12, x38, x39, x41, x42, x46, x539, x55]
        vreal x830 = access(AtDD12, stencil_idx_0_0_0_VVV); // x830: Dependency! Liveness = 13; [AtDD12, x38, x39, x41, x42, x46, x507, x539, x55, x794, x797, x799, x801]
        store(kyz, stencil_idx_0_0_0_VVV, (x539 * (x830 + (x1136 * x832)))); // kyz: Liveness = 9; [kyz, x1136, x38, x39, x41, x42, x46, x539, x830]
        x830 = access(gtDD22, stencil_idx_0_0_0_VVV); // x55: Dependency! Liveness = 9; [gtDD22, x38, x39, x41, x42, x46, x507, x539, x55]
        vreal x840 = access(AtDD22, stencil_idx_0_0_0_VVV); // x840: Dependency! Liveness = 13; [AtDD22, x38, x39, x41, x42, x46, x507, x539, x55, x794, x797, x799, x801]
        store(kzz, stencil_idx_0_0_0_VVV, (x539 * (x840 + (x1136 * x830)))); // kzz: Liveness = 10; [kzz, x1136, x38, x39, x41, x42, x46, x539, x55, x840]
        x1136 = access(evo_shiftU0, stencil_idx_0_0_0_VVV); // x794: Dependency! Liveness = 10; [evo_shiftU0, x38, x39, x41, x42, x46, x507, x539, x55, x794]
        x840 = access(evo_shiftU1, stencil_idx_0_0_0_VVV); // x797: Dependency! Liveness = 11; [evo_shiftU1, x38, x39, x41, x42, x46, x507, x539, x55, x794, x797]
        vreal x799 = access(evo_shiftU2, stencil_idx_0_0_0_VVV); // x799: Dependency! Liveness = 12; [evo_shiftU2, x38, x39, x41, x42, x46, x507, x539, x55, x794, x797, x799]
        vreal x801 = access(evo_lapse, stencil_idx_0_0_0_VVV); // x801: Dependency! Liveness = 13; [evo_lapse, x38, x39, x41, x42, x46, x507, x539, x55, x794, x797, x799, x801]
        store(alp, stencil_idx_0_0_0_VVV, x801); // alp: Liveness = 12; [alp, x38, x39, x41, x42, x46, x539, x55, x794, x797, x799, x801]
        store(betax, stencil_idx_0_0_0_VVV, x1136); // betax: Liveness = 11; [betax, x38, x39, x41, x42, x46, x539, x55, x794, x797, x799]
        store(betay, stencil_idx_0_0_0_VVV, x840); // betay: Liveness = 10; [betay, x38, x39, x41, x42, x46, x539, x55, x797, x799]
        store(betaz, stencil_idx_0_0_0_VVV, x799); // betaz: Liveness = 9; [betaz, x38, x39, x41, x42, x46, x539, x55, x799]
        store(gxx, stencil_idx_0_0_0_VVV, (x38 * x539)); // gxx: Liveness = 8; [gxx, x38, x39, x41, x42, x46, x539, x55]
        store(gxy, stencil_idx_0_0_0_VVV, (x539 * x845)); // gxy: Liveness = 7; [gxy, x39, x41, x42, x46, x539, x55]
        store(gxz, stencil_idx_0_0_0_VVV, (x539 * x831)); // gxz: Liveness = 6; [gxz, x39, x42, x46, x539, x55]
        store(gyy, stencil_idx_0_0_0_VVV, (x539 * x841)); // gyy: Liveness = 5; [gyy, x39, x46, x539, x55]
        store(gyz, stencil_idx_0_0_0_VVV, (x539 * x832)); // gyz: Liveness = 4; [gyz, x39, x539, x55]
        store(gzz, stencil_idx_0_0_0_VVV, (x539 * x830)); // gzz: Liveness = 3; [gzz, x539, x55]    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}