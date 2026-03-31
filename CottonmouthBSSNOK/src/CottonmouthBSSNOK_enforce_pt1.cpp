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
void enforce_pt1(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_enforce_pt1;
    DECLARE_CCTK_PARAMETERS;
    using vreal = Arith::simd<CCTK_REAL>;
    constexpr std::size_t vsize = std::tuple_size_v<vreal>;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("enforce_pt1");
    #endif
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define evo_lapse_layout VVV_layout
    #define gtDD00_layout VVV_layout
    #define gtDD01_layout VVV_layout
    #define gtDD02_layout VVV_layout
    #define gtDD11_layout VVV_layout
    #define gtDD12_layout VVV_layout
    #define gtDD22_layout VVV_layout
    #define w_layout VVV_layout
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2], vsize>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x138 = pow2(access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x163 = pow2(access(gtDD01, stencil_idx_0_0_0_VVV));
        vreal x43 = pow2(access(gtDD02, stencil_idx_0_0_0_VVV));
        vreal x927 = pow(static_cast<vreal>(((-1 * access(gtDD00, stencil_idx_0_0_0_VVV) * x138) + (-1 * access(gtDD11, stencil_idx_0_0_0_VVV) * x43) + (-1 * access(gtDD22, stencil_idx_0_0_0_VVV) * x163) + (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD11, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV)) + (2 * access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV)))), (-1.0 / 3.0));
        store(gtDD00, stencil_idx_0_0_0_VVV, (access(gtDD00, stencil_idx_0_0_0_VVV) * x927));
        store(gtDD01, stencil_idx_0_0_0_VVV, (access(gtDD01, stencil_idx_0_0_0_VVV) * x927));
        store(gtDD02, stencil_idx_0_0_0_VVV, (access(gtDD02, stencil_idx_0_0_0_VVV) * x927));
        store(gtDD11, stencil_idx_0_0_0_VVV, (access(gtDD11, stencil_idx_0_0_0_VVV) * x927));
        store(gtDD12, stencil_idx_0_0_0_VVV, (access(gtDD12, stencil_idx_0_0_0_VVV) * x927));
        store(gtDD22, stencil_idx_0_0_0_VVV, (access(gtDD22, stencil_idx_0_0_0_VVV) * x927));
        store(evo_lapse, stencil_idx_0_0_0_VVV, max(access(evo_lapse, stencil_idx_0_0_0_VVV), evolved_lapse_floor));
        store(w, stencil_idx_0_0_0_VVV, max(access(w, stencil_idx_0_0_0_VVV), conformal_factor_floor));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}