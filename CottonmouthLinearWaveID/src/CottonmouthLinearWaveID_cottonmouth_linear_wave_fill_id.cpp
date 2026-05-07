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
void cottonmouth_linear_wave_fill_id(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_cottonmouth_linear_wave_fill_id;
    DECLARE_CCTK_PARAMETERS;
    using vreal = CCTK_REAL;
    constexpr std::size_t vsize = 0;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("cottonmouth_linear_wave_fill_id");
    #endif
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define alp_layout VVV_layout
    #define betax_layout VVV_layout
    #define betay_layout VVV_layout
    #define betaz_layout VVV_layout
    #define dt2alp_layout VVV_layout
    #define dt2betax_layout VVV_layout
    #define dt2betay_layout VVV_layout
    #define dt2betaz_layout VVV_layout
    #define dtalp_layout VVV_layout
    #define dtbetax_layout VVV_layout
    #define dtbetay_layout VVV_layout
    #define dtbetaz_layout VVV_layout
    #define dtkxx_layout VVV_layout
    #define dtkxy_layout VVV_layout
    #define dtkxz_layout VVV_layout
    #define dtkyy_layout VVV_layout
    #define dtkyz_layout VVV_layout
    #define dtkzz_layout VVV_layout
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
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    // cottonmouth_linear_wave_fill_id loop 0
    grid.loop_all_device<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const vreal x = p.x;
        const vreal t = cctk_time;
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x0 = pown<vreal>(wavelength, -1);
        vreal x1 = (x0 * ((6.28318530717959 * x) + (-6.28318530717959 * t)));
        vreal x2 = (amplitude * sin(x1));
        vreal x3 = (19.7392088021787 * x2 * pown<vreal>(wavelength, -2));
        store(gzz, stencil_idx_0_0_0_VVV, (1 + (-(x2))));
        store(gyy, stencil_idx_0_0_0_VVV, (1 + x2));
        x2 = (3.14159265358979 * amplitude * x0 * cos(x1));
        store(dtkyy, stencil_idx_0_0_0_VVV, (-(x3)));
        store(dtkzz, stencil_idx_0_0_0_VVV, x3);
        store(kyy, stencil_idx_0_0_0_VVV, (-(x2)));
        store(kzz, stencil_idx_0_0_0_VVV, x2);
        store(alp, stencil_idx_0_0_0_VVV, 1);
        store(betax, stencil_idx_0_0_0_VVV, 0);
        store(betay, stencil_idx_0_0_0_VVV, 0);
        store(betaz, stencil_idx_0_0_0_VVV, 0);
        store(dt2alp, stencil_idx_0_0_0_VVV, 0);
        store(dt2betax, stencil_idx_0_0_0_VVV, 0);
        store(dt2betay, stencil_idx_0_0_0_VVV, 0);
        store(dt2betaz, stencil_idx_0_0_0_VVV, 0);
        store(dtalp, stencil_idx_0_0_0_VVV, 0);
        store(dtbetax, stencil_idx_0_0_0_VVV, 0);
        store(dtbetay, stencil_idx_0_0_0_VVV, 0);
        store(dtbetaz, stencil_idx_0_0_0_VVV, 0);
        store(dtkxx, stencil_idx_0_0_0_VVV, 0);
        store(dtkxy, stencil_idx_0_0_0_VVV, 0);
        store(dtkxz, stencil_idx_0_0_0_VVV, 0);
        store(dtkyz, stencil_idx_0_0_0_VVV, 0);
        store(gxx, stencil_idx_0_0_0_VVV, 1);
        store(gxy, stencil_idx_0_0_0_VVV, 0);
        store(gxz, stencil_idx_0_0_0_VVV, 0);
        store(gyz, stencil_idx_0_0_0_VVV, 0);
        store(kxx, stencil_idx_0_0_0_VVV, 0);
        store(kxy, stencil_idx_0_0_0_VVV, 0);
        store(kxz, stencil_idx_0_0_0_VVV, 0);
        store(kyz, stencil_idx_0_0_0_VVV, 0);    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}