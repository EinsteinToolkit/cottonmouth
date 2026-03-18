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
#define access(GF, IDX) (GF(p.mask, IDX))
#define store(GF, IDX, VAL) (GF.store(p.mask, IDX, VAL))
#define stencil(GF, IDX) (GF(p.mask, IDX))
#define CCTK_ASSERT(X) if(!(X)) { CCTK_Error(__LINE__, __FILE__, CCTK_THORNSTRING, "Assertion Failure: " #X); }
using namespace Arith;
using namespace Loop;
using std::cbrt,std::fmax,std::fmin,std::sqrt;
void cottonmouth_diagonal_linear_wave_fill_id(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_cottonmouth_diagonal_linear_wave_fill_id;
    DECLARE_CCTK_PARAMETERS;
    using vreal = Arith::simd<CCTK_REAL>;
    constexpr std::size_t vsize = std::tuple_size_v<vreal>;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("cottonmouth_diagonal_linear_wave_fill_id");
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
    grid.loop_all_device<VVV_centered[0], VVV_centered[1], VVV_centered[2], vsize>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const auto y = p.y;
        const vreal x = (p.x + (Arith::iota<vreal>() * p.dx));
        const vreal t = cctk_time;
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x1 = sqrt(2);
        vreal x0 = pown<vreal>(wavelength, -1);
        vreal x2 = ((1.0 / 2.0) * x0 * x1 * ((6.28318530717959 * x) + (-6.28318530717959 * y) + (-6.28318530717959 * t * x1)));
        x1 = (amplitude * sin(x2));
        vreal x5 = (19.7392088021787 * x1 * pown<vreal>(wavelength, -2));
        vreal x3 = (3.14159265358979 * amplitude * x0 * cos(x2));
        store(gzz, stencil_idx_0_0_0_VVV, (1 + (-(x1))));
        store(gyy, stencil_idx_0_0_0_VVV, (1 + x1));
        store(kzz, stencil_idx_0_0_0_VVV, (-(x3)));
        store(dtkzz, stencil_idx_0_0_0_VVV, (-(x5)));
        store(gxx, stencil_idx_0_0_0_VVV, 1);
        store(gxy, stencil_idx_0_0_0_VVV, 0);
        store(gxz, stencil_idx_0_0_0_VVV, 0);
        store(gyz, stencil_idx_0_0_0_VVV, 0);
        store(kxx, stencil_idx_0_0_0_VVV, 0);
        store(kxy, stencil_idx_0_0_0_VVV, 0);
        store(kyy, stencil_idx_0_0_0_VVV, x3);
        store(kxz, stencil_idx_0_0_0_VVV, 0);
        store(kyz, stencil_idx_0_0_0_VVV, 0);
        store(alp, stencil_idx_0_0_0_VVV, 1);
        store(betax, stencil_idx_0_0_0_VVV, 0);
        store(betay, stencil_idx_0_0_0_VVV, 0);
        store(betaz, stencil_idx_0_0_0_VVV, 0);
        store(dtalp, stencil_idx_0_0_0_VVV, 0);
        store(dtbetax, stencil_idx_0_0_0_VVV, 0);
        store(dtbetay, stencil_idx_0_0_0_VVV, 0);
        store(dtbetaz, stencil_idx_0_0_0_VVV, 0);
        store(dtkxx, stencil_idx_0_0_0_VVV, 0);
        store(dtkxy, stencil_idx_0_0_0_VVV, 0);
        store(dtkyy, stencil_idx_0_0_0_VVV, x5);
        store(dtkxz, stencil_idx_0_0_0_VVV, 0);
        store(dtkyz, stencil_idx_0_0_0_VVV, 0);
        store(dt2alp, stencil_idx_0_0_0_VVV, 0);
        store(dt2betax, stencil_idx_0_0_0_VVV, 0);
        store(dt2betay, stencil_idx_0_0_0_VVV, 0);
        store(dt2betaz, stencil_idx_0_0_0_VVV, 0);    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}