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
void enforce_pt2(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_enforce_pt2;
    DECLARE_CCTK_PARAMETERS;
    using vreal = Arith::simd<CCTK_REAL>;
    constexpr std::size_t vsize = std::tuple_size_v<vreal>;
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
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2], vsize>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x931 = ((1.0 / 3.0) * access(gtDD00, stencil_idx_0_0_0_VVV));
        vreal x43 = pow2(access(gtDD02, stencil_idx_0_0_0_VVV));
        vreal x42 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV));
        vreal x109 = (x42 + (-(x43)));
        x42 = (access(AtDD11, stencil_idx_0_0_0_VVV) * x109);
        x109 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV));
        x43 = (access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x117 = (x109 + (-(x43)));
        vreal x567 = (-(x117));
        x117 = (access(AtDD01, stencil_idx_0_0_0_VVV) * x567);
        x567 = (access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD11, stencil_idx_0_0_0_VVV));
        vreal x73 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x200 = (x73 + (-(x567)));
        x73 = (access(AtDD02, stencil_idx_0_0_0_VVV) * x200);
        x200 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD02, stencil_idx_0_0_0_VVV));
        vreal x31 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x127 = (x31 + (-(x200)));
        x31 = (-(x127));
        x127 = (access(AtDD12, stencil_idx_0_0_0_VVV) * x31);
        vreal x163 = pow2(access(gtDD01, stencil_idx_0_0_0_VVV));
        vreal x162 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD11, stencil_idx_0_0_0_VVV));
        vreal x164 = (x162 + (-(x163)));
        x162 = (access(AtDD22, stencil_idx_0_0_0_VVV) * x164);
        x164 = ((2.0 / 3.0) * access(gtDD00, stencil_idx_0_0_0_VVV));
        x163 = pow2(access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x137 = (access(gtDD11, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV));
        vreal x139 = (x137 + (-(x163)));
        x137 = (access(AtDD00, stencil_idx_0_0_0_VVV) * x139);
        store(AtDD00, stencil_idx_0_0_0_VVV, (access(AtDD00, stencil_idx_0_0_0_VVV) + (-1 * x117 * x164) + (-1 * x127 * x164) + (-1 * x137 * x931) + (-1 * x162 * x931) + (-1 * x164 * x73) + (-1 * x42 * x931)));
        x931 = ((1.0 / 3.0) * access(gtDD01, stencil_idx_0_0_0_VVV));
        x139 = ((2.0 / 3.0) * access(gtDD01, stencil_idx_0_0_0_VVV));
        store(AtDD01, stencil_idx_0_0_0_VVV, (access(AtDD01, stencil_idx_0_0_0_VVV) + (-1 * x117 * x139) + (-1 * x127 * x139) + (-1 * x137 * x931) + (-1 * x139 * x73) + (-1 * x162 * x931) + (-1 * x42 * x931)));
        vreal x935 = ((1.0 / 3.0) * access(gtDD11, stencil_idx_0_0_0_VVV));
        vreal x934 = ((2.0 / 3.0) * access(gtDD11, stencil_idx_0_0_0_VVV));
        store(AtDD11, stencil_idx_0_0_0_VVV, (access(AtDD11, stencil_idx_0_0_0_VVV) + (-1 * x117 * x934) + (-1 * x127 * x934) + (-1 * x137 * x935) + (-1 * x162 * x935) + (-1 * x42 * x935) + (-1 * x73 * x934)));
        x934 = ((2.0 / 3.0) * access(gtDD02, stencil_idx_0_0_0_VVV));
        x935 = ((1.0 / 3.0) * access(gtDD02, stencil_idx_0_0_0_VVV));
        store(AtDD02, stencil_idx_0_0_0_VVV, (access(AtDD02, stencil_idx_0_0_0_VVV) + (-1 * x117 * x934) + (-1 * x127 * x934) + (-1 * x137 * x935) + (-1 * x162 * x935) + (-1 * x42 * x935) + (-1 * x73 * x934)));
        vreal x938 = ((2.0 / 3.0) * access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x939 = ((1.0 / 3.0) * access(gtDD12, stencil_idx_0_0_0_VVV));
        store(AtDD12, stencil_idx_0_0_0_VVV, (access(AtDD12, stencil_idx_0_0_0_VVV) + (-1 * x117 * x938) + (-1 * x127 * x938) + (-1 * x137 * x939) + (-1 * x162 * x939) + (-1 * x42 * x939) + (-1 * x73 * x938)));
        x938 = ((2.0 / 3.0) * access(gtDD22, stencil_idx_0_0_0_VVV));
        x939 = ((1.0 / 3.0) * access(gtDD22, stencil_idx_0_0_0_VVV));
        store(AtDD22, stencil_idx_0_0_0_VVV, (access(AtDD22, stencil_idx_0_0_0_VVV) + (-1 * x117 * x938) + (-1 * x127 * x938) + (-1 * x137 * x939) + (-1 * x162 * x939) + (-1 * x42 * x939) + (-1 * x73 * x938)));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}