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
        vreal x89 = access(gtDD00, stencil_idx_0_0_0_VVV); // x89: Dependency! Liveness = 9; [gtDD00, x36, x37, x38, x39, x50, x687, x707, x89]
        vreal x687 = access(chi, stencil_idx_0_0_0_VVV); // x687: Dependency! Liveness = 9; [chi, x36, x37, x38, x39, x50, x687, x707, x89]
        vreal x707 = pown<vreal>(x687, -1); // x707: Dependency! Liveness = 13; [x36, x37, x38, x39, x50, x687, x707, x89, x966, x978, x980, x983, x985]
        vreal x717 = (x707 * x89); // x717: Dependency! Liveness = 7; [x36, x37, x38, x39, x50, x707, x89]
        store(gxx, stencil_idx_0_0_0_VVV, x717); // gxx: 
        vreal x36 = access(gtDD01, stencil_idx_0_0_0_VVV); // x36: Dependency! Liveness = 8; [gtDD01, x36, x37, x38, x39, x50, x707, x89]
        vreal x734 = (x36 * x707); // x734: Dependency! Liveness = 6; [x36, x37, x38, x39, x50, x707]
        store(gxy, stencil_idx_0_0_0_VVV, x734); // gxy: 
        vreal x39 = access(gtDD11, stencil_idx_0_0_0_VVV); // x39: Dependency! Liveness = 8; [gtDD11, x36, x37, x38, x39, x50, x707, x89]
        vreal x738 = (x39 * x707); // x738: Dependency! Liveness = 5; [x37, x38, x39, x50, x707]
        store(gyy, stencil_idx_0_0_0_VVV, x738); // gyy: 
        vreal x38 = access(gtDD02, stencil_idx_0_0_0_VVV); // x38: Dependency! Liveness = 8; [gtDD02, x36, x37, x38, x39, x50, x707, x89]
        vreal x742 = (x38 * x707); // x742: Dependency! Liveness = 4; [x37, x38, x50, x707]
        store(gxz, stencil_idx_0_0_0_VVV, x742); // gxz: 
        vreal x37 = access(gtDD12, stencil_idx_0_0_0_VVV); // x37: Dependency! Liveness = 8; [gtDD12, x36, x37, x38, x39, x50, x707, x89]
        vreal x747 = (x37 * x707); // x747: Dependency! Liveness = 3; [x37, x50, x707]
        store(gyz, stencil_idx_0_0_0_VVV, x747); // gyz: 
        vreal x50 = access(gtDD22, stencil_idx_0_0_0_VVV); // x50: Dependency! Liveness = 8; [gtDD22, x36, x37, x38, x39, x50, x707, x89]
        vreal x751 = (x50 * x707); // x751: Dependency! Liveness = 2; [x50, x707]
        store(gzz, stencil_idx_0_0_0_VVV, x751); // gzz: 
        vreal x1006 = access(AtDD00, stencil_idx_0_0_0_VVV); // x1006: Dependency! Liveness = 8; [AtDD00, x36, x37, x38, x39, x50, x707, x89]
        vreal x966 = access(trK, stencil_idx_0_0_0_VVV); // x966: Dependency! Liveness = 10; [trK, x36, x37, x38, x39, x50, x687, x707, x89, x966]
        vreal x1282 = ((1.0 / 3.0) * x966); // x1282: Dependency! Liveness = 12; [x36, x37, x38, x39, x50, x707, x89, x966, x978, x980, x983, x985]
        store(kxx, stencil_idx_0_0_0_VVV, (x707 * (x1006 + (x1282 * x89)))); // kxx: 
        vreal x1007 = access(AtDD01, stencil_idx_0_0_0_VVV); // x1007: Dependency! Liveness = 8; [AtDD01, x36, x37, x38, x39, x50, x707, x89]
        store(kxy, stencil_idx_0_0_0_VVV, (x707 * (x1007 + (x1282 * x36)))); // kxy: 
        vreal x1009 = access(AtDD11, stencil_idx_0_0_0_VVV); // x1009: Dependency! Liveness = 8; [AtDD11, x36, x37, x38, x39, x50, x707, x89]
        store(kyy, stencil_idx_0_0_0_VVV, (x707 * (x1009 + (x1282 * x39)))); // kyy: 
        vreal x1001 = access(AtDD02, stencil_idx_0_0_0_VVV); // x1001: Dependency! Liveness = 8; [AtDD02, x36, x37, x38, x39, x50, x707, x89]
        store(kxz, stencil_idx_0_0_0_VVV, (x707 * (x1001 + (x1282 * x38)))); // kxz: 
        vreal x1003 = access(AtDD12, stencil_idx_0_0_0_VVV); // x1003: Dependency! Liveness = 8; [AtDD12, x36, x37, x38, x39, x50, x707, x89]
        store(kyz, stencil_idx_0_0_0_VVV, (x707 * (x1003 + (x1282 * x37)))); // kyz: 
        vreal x999 = access(AtDD22, stencil_idx_0_0_0_VVV); // x999: Dependency! Liveness = 14; [AtDD22, x36, x37, x38, x39, x50, x687, x707, x89, x966, x978, x980, x983, x985]
        store(kzz, stencil_idx_0_0_0_VVV, (x707 * (x999 + (x1282 * x50)))); // kzz: 
        vreal x978 = access(evo_lapse, stencil_idx_0_0_0_VVV); // x978: Dependency! Liveness = 11; [evo_lapse, x36, x37, x38, x39, x50, x687, x707, x89, x966, x978]
        store(alp, stencil_idx_0_0_0_VVV, x978); // alp: 
        vreal x980 = access(evo_shiftU0, stencil_idx_0_0_0_VVV); // x980: Dependency! Liveness = 12; [evo_shiftU0, x36, x37, x38, x39, x50, x687, x707, x89, x966, x978, x980]
        store(betax, stencil_idx_0_0_0_VVV, x980); // betax: 
        vreal x983 = access(evo_shiftU1, stencil_idx_0_0_0_VVV); // x983: Dependency! Liveness = 13; [evo_shiftU1, x36, x37, x38, x39, x50, x687, x707, x89, x966, x978, x980, x983]
        store(betay, stencil_idx_0_0_0_VVV, x983); // betay: 
        vreal x985 = access(evo_shiftU2, stencil_idx_0_0_0_VVV); // x985: Dependency! Liveness = 14; [evo_shiftU2, x36, x37, x38, x39, x50, x687, x707, x89, x966, x978, x980, x983, x985]
        store(betaz, stencil_idx_0_0_0_VVV, x985); // betaz:     
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}