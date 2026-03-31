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
void bssn2adm(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_bssn2adm;
    DECLARE_CCTK_PARAMETERS;
    using vreal = Arith::simd<CCTK_REAL>;
    constexpr std::size_t vsize = std::tuple_size_v<vreal>;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("bssn2adm");
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
    #define w_layout VVV_layout
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    grid.loop_all_device<VVV_centered[0], VVV_centered[1], VVV_centered[2], vsize>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x550 = ((1.0 / 3.0) * access(trK, stencil_idx_0_0_0_VVV));
        vreal x548 = pow2(access(w, stencil_idx_0_0_0_VVV));
        vreal x549 = pown<vreal>(x548, -1);
        store(kxx, stencil_idx_0_0_0_VVV, (x549 * (access(AtDD00, stencil_idx_0_0_0_VVV) + (access(gtDD00, stencil_idx_0_0_0_VVV) * x550))));
        store(kxy, stencil_idx_0_0_0_VVV, (x549 * (access(AtDD01, stencil_idx_0_0_0_VVV) + (access(gtDD01, stencil_idx_0_0_0_VVV) * x550))));
        store(kxz, stencil_idx_0_0_0_VVV, (x549 * (access(AtDD02, stencil_idx_0_0_0_VVV) + (access(gtDD02, stencil_idx_0_0_0_VVV) * x550))));
        store(kyy, stencil_idx_0_0_0_VVV, (x549 * (access(AtDD11, stencil_idx_0_0_0_VVV) + (access(gtDD11, stencil_idx_0_0_0_VVV) * x550))));
        store(kyz, stencil_idx_0_0_0_VVV, (x549 * (access(AtDD12, stencil_idx_0_0_0_VVV) + (access(gtDD12, stencil_idx_0_0_0_VVV) * x550))));
        store(kzz, stencil_idx_0_0_0_VVV, (x549 * (access(AtDD22, stencil_idx_0_0_0_VVV) + (access(gtDD22, stencil_idx_0_0_0_VVV) * x550))));
        x550 = (access(gtDD00, stencil_idx_0_0_0_VVV) * x549);
        x548 = (access(gtDD01, stencil_idx_0_0_0_VVV) * x549);
        vreal x553 = (access(gtDD02, stencil_idx_0_0_0_VVV) * x549);
        vreal x554 = (access(gtDD11, stencil_idx_0_0_0_VVV) * x549);
        vreal x555 = (access(gtDD12, stencil_idx_0_0_0_VVV) * x549);
        vreal x556 = (access(gtDD22, stencil_idx_0_0_0_VVV) * x549);
        store(alp, stencil_idx_0_0_0_VVV, access(evo_lapse, stencil_idx_0_0_0_VVV));
        store(betax, stencil_idx_0_0_0_VVV, access(evo_shiftU0, stencil_idx_0_0_0_VVV));
        store(betay, stencil_idx_0_0_0_VVV, access(evo_shiftU1, stencil_idx_0_0_0_VVV));
        store(betaz, stencil_idx_0_0_0_VVV, access(evo_shiftU2, stencil_idx_0_0_0_VVV));
        store(gxx, stencil_idx_0_0_0_VVV, x550);
        store(gxy, stencil_idx_0_0_0_VVV, x548);
        store(gxz, stencil_idx_0_0_0_VVV, x553);
        store(gyy, stencil_idx_0_0_0_VVV, x554);
        store(gyz, stencil_idx_0_0_0_VVV, x555);
        store(gzz, stencil_idx_0_0_0_VVV, x556);    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}