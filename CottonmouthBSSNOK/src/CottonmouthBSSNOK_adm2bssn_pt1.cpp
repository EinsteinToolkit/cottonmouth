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
void adm2bssn_pt1(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_adm2bssn_pt1;
    DECLARE_CCTK_PARAMETERS;
    using vreal = Arith::simd<CCTK_REAL>;
    constexpr std::size_t vsize = std::tuple_size_v<vreal>;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("adm2bssn_pt1");
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
    #define dtbetax_layout VVV_layout
    #define dtbetay_layout VVV_layout
    #define dtbetaz_layout VVV_layout
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
    #define shift_BU0_layout VVV_layout
    #define shift_BU1_layout VVV_layout
    #define shift_BU2_layout VVV_layout
    #define trK_layout VVV_layout
    #define w_layout VVV_layout
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    CCTK_ASSERT((cctk_nghostzones[0] >= 2));
    CCTK_ASSERT((cctk_nghostzones[1] >= 2));
    CCTK_ASSERT((cctk_nghostzones[2] >= 2));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2], vsize>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x0 = pow2(access(gyz, stencil_idx_0_0_0_VVV));
        vreal x1 = pow2(access(gxy, stencil_idx_0_0_0_VVV));
        vreal x2 = pow2(access(gxz, stencil_idx_0_0_0_VVV));
        vreal x3 = ((access(gxx, stencil_idx_0_0_0_VVV) * x0) + (access(gyy, stencil_idx_0_0_0_VVV) * x2) + (access(gzz, stencil_idx_0_0_0_VVV) * x1) + (-1 * access(gxx, stencil_idx_0_0_0_VVV) * access(gyy, stencil_idx_0_0_0_VVV) * access(gzz, stencil_idx_0_0_0_VVV)) + (-2 * access(gxy, stencil_idx_0_0_0_VVV) * access(gxz, stencil_idx_0_0_0_VVV) * access(gyz, stencil_idx_0_0_0_VVV)));
        vreal x6 = pown<vreal>(x3, -1);
        vreal x7 = (access(kxx, stencil_idx_0_0_0_VVV) * x6 * (x0 + (-1 * access(gyy, stencil_idx_0_0_0_VVV) * access(gzz, stencil_idx_0_0_0_VVV))));
        x0 = (access(kyy, stencil_idx_0_0_0_VVV) * x6 * (x2 + (-1 * access(gxx, stencil_idx_0_0_0_VVV) * access(gzz, stencil_idx_0_0_0_VVV))));
        x2 = (access(kzz, stencil_idx_0_0_0_VVV) * x6 * (x1 + (-1 * access(gxx, stencil_idx_0_0_0_VVV) * access(gyy, stencil_idx_0_0_0_VVV))));
        x1 = (access(kxz, stencil_idx_0_0_0_VVV) * ((access(gxz, stencil_idx_0_0_0_VVV) * access(gyy, stencil_idx_0_0_0_VVV)) + (-1 * access(gxy, stencil_idx_0_0_0_VVV) * access(gyz, stencil_idx_0_0_0_VVV))));
        vreal x9 = (access(kxy, stencil_idx_0_0_0_VVV) * ((access(gxy, stencil_idx_0_0_0_VVV) * access(gzz, stencil_idx_0_0_0_VVV)) + (-1 * access(gxz, stencil_idx_0_0_0_VVV) * access(gyz, stencil_idx_0_0_0_VVV))));
        vreal x14 = (access(kyz, stencil_idx_0_0_0_VVV) * ((access(gxx, stencil_idx_0_0_0_VVV) * access(gyz, stencil_idx_0_0_0_VVV)) + (-1 * access(gxy, stencil_idx_0_0_0_VVV) * access(gxz, stencil_idx_0_0_0_VVV))));
        vreal x10 = ((2.0 / 3.0) * x6);
        vreal x11 = (access(gxx, stencil_idx_0_0_0_VVV) * x10);
        vreal x4 = (-(x3));
        x3 = pow(static_cast<vreal>(x4), (-1.0 / 3.0));
        vreal x8 = ((1.0 / 3.0) * access(gxx, stencil_idx_0_0_0_VVV));
        store(AtDD00, stencil_idx_0_0_0_VVV, (x3 * (access(kxx, stencil_idx_0_0_0_VVV) + (-1 * x0 * x8) + (-1 * x1 * x11) + (-1 * x11 * x14) + (-1 * x11 * x9) + (-1 * x2 * x8) + (-1 * x7 * x8))));
        x11 = (access(gxy, stencil_idx_0_0_0_VVV) * x10);
        x8 = ((1.0 / 3.0) * access(gxy, stencil_idx_0_0_0_VVV));
        store(AtDD01, stencil_idx_0_0_0_VVV, (x3 * (access(kxy, stencil_idx_0_0_0_VVV) + (-1 * x0 * x8) + (-1 * x1 * x11) + (-1 * x11 * x14) + (-1 * x11 * x9) + (-1 * x2 * x8) + (-1 * x7 * x8))));
        vreal x19 = (access(gxz, stencil_idx_0_0_0_VVV) * x10);
        vreal x18 = ((1.0 / 3.0) * access(gxz, stencil_idx_0_0_0_VVV));
        store(AtDD02, stencil_idx_0_0_0_VVV, (x3 * (access(kxz, stencil_idx_0_0_0_VVV) + (-1 * x0 * x18) + (-1 * x1 * x19) + (-1 * x14 * x19) + (-1 * x18 * x2) + (-1 * x18 * x7) + (-1 * x19 * x9))));
        x18 = (access(gyy, stencil_idx_0_0_0_VVV) * x10);
        x19 = ((1.0 / 3.0) * access(gyy, stencil_idx_0_0_0_VVV));
        store(AtDD11, stencil_idx_0_0_0_VVV, (x3 * (access(kyy, stencil_idx_0_0_0_VVV) + (-1 * x0 * x19) + (-1 * x1 * x18) + (-1 * x14 * x18) + (-1 * x18 * x9) + (-1 * x19 * x2) + (-1 * x19 * x7))));
        vreal x23 = (access(gyz, stencil_idx_0_0_0_VVV) * x10);
        vreal x22 = ((1.0 / 3.0) * access(gyz, stencil_idx_0_0_0_VVV));
        store(AtDD12, stencil_idx_0_0_0_VVV, (x3 * (access(kyz, stencil_idx_0_0_0_VVV) + (-1 * x0 * x22) + (-1 * x1 * x23) + (-1 * x14 * x23) + (-1 * x2 * x22) + (-1 * x22 * x7) + (-1 * x23 * x9))));
        x22 = (access(gzz, stencil_idx_0_0_0_VVV) * x10);
        x10 = ((1.0 / 3.0) * access(gzz, stencil_idx_0_0_0_VVV));
        store(AtDD22, stencil_idx_0_0_0_VVV, (x3 * (access(kzz, stencil_idx_0_0_0_VVV) + (-1 * x0 * x10) + (-1 * x1 * x22) + (-1 * x10 * x2) + (-1 * x10 * x7) + (-1 * x14 * x22) + (-1 * x22 * x9))));
        x23 = (DXI * access(betax, stencil_idx_0_0_0_VVV));
        vreal x28 = (DYI * access(betay, stencil_idx_0_0_0_VVV));
        vreal x29 = (DZI * access(betaz, stencil_idx_0_0_0_VVV));
        vreal x30 = ((4.0 / 3.0) * pown<vreal>(access(alp, stencil_idx_0_0_0_VVV), -1));
        store(shift_BU0, stencil_idx_0_0_0_VVV, (x30 * (access(dtbetax, stencil_idx_0_0_0_VVV) + (-1 * x23 * (((1.0 / 12.0) * (((-(stencil(betax, stencil_idx_2_0_0_VVV))) + stencil(betax, stencil_idx_m2_0_0_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betax, stencil_idx_m1_0_0_VVV))) + stencil(betax, stencil_idx_1_0_0_VVV)))))) + (-1 * x28 * (((1.0 / 12.0) * (((-(stencil(betax, stencil_idx_0_2_0_VVV))) + stencil(betax, stencil_idx_0_m2_0_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betax, stencil_idx_0_m1_0_VVV))) + stencil(betax, stencil_idx_0_1_0_VVV)))))) + (-1 * x29 * (((1.0 / 12.0) * (((-(stencil(betax, stencil_idx_0_0_2_VVV))) + stencil(betax, stencil_idx_0_0_m2_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betax, stencil_idx_0_0_m1_VVV))) + stencil(betax, stencil_idx_0_0_1_VVV)))))))));
        store(shift_BU1, stencil_idx_0_0_0_VVV, (x30 * (access(dtbetay, stencil_idx_0_0_0_VVV) + (-1 * x23 * (((1.0 / 12.0) * (((-(stencil(betay, stencil_idx_2_0_0_VVV))) + stencil(betay, stencil_idx_m2_0_0_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betay, stencil_idx_m1_0_0_VVV))) + stencil(betay, stencil_idx_1_0_0_VVV)))))) + (-1 * x28 * (((1.0 / 12.0) * (((-(stencil(betay, stencil_idx_0_2_0_VVV))) + stencil(betay, stencil_idx_0_m2_0_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betay, stencil_idx_0_m1_0_VVV))) + stencil(betay, stencil_idx_0_1_0_VVV)))))) + (-1 * x29 * (((1.0 / 12.0) * (((-(stencil(betay, stencil_idx_0_0_2_VVV))) + stencil(betay, stencil_idx_0_0_m2_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betay, stencil_idx_0_0_m1_VVV))) + stencil(betay, stencil_idx_0_0_1_VVV)))))))));
        store(shift_BU2, stencil_idx_0_0_0_VVV, (x30 * (access(dtbetaz, stencil_idx_0_0_0_VVV) + (-1 * x23 * (((1.0 / 12.0) * (((-(stencil(betaz, stencil_idx_2_0_0_VVV))) + stencil(betaz, stencil_idx_m2_0_0_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betaz, stencil_idx_m1_0_0_VVV))) + stencil(betaz, stencil_idx_1_0_0_VVV)))))) + (-1 * x28 * (((1.0 / 12.0) * (((-(stencil(betaz, stencil_idx_0_2_0_VVV))) + stencil(betaz, stencil_idx_0_m2_0_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betaz, stencil_idx_0_m1_0_VVV))) + stencil(betaz, stencil_idx_0_1_0_VVV)))))) + (-1 * x29 * (((1.0 / 12.0) * (((-(stencil(betaz, stencil_idx_0_0_2_VVV))) + stencil(betaz, stencil_idx_0_0_m2_VVV)))) + ((2.0 / 3.0) * (((-(stencil(betaz, stencil_idx_0_0_m1_VVV))) + stencil(betaz, stencil_idx_0_0_1_VVV)))))))));
        x28 = (2 * x6);
        store(trK, stencil_idx_0_0_0_VVV, (x0 + x2 + x7 + (x1 * x28) + (x14 * x28) + (x28 * x9)));
        store(w, stencil_idx_0_0_0_VVV, pow(static_cast<vreal>(x4), (-1.0 / 6.0)));
        store(evo_lapse, stencil_idx_0_0_0_VVV, access(alp, stencil_idx_0_0_0_VVV));
        store(evo_shiftU0, stencil_idx_0_0_0_VVV, access(betax, stencil_idx_0_0_0_VVV));
        store(evo_shiftU1, stencil_idx_0_0_0_VVV, access(betay, stencil_idx_0_0_0_VVV));
        store(evo_shiftU2, stencil_idx_0_0_0_VVV, access(betaz, stencil_idx_0_0_0_VVV));
        store(gtDD00, stencil_idx_0_0_0_VVV, (access(gxx, stencil_idx_0_0_0_VVV) * x3));
        store(gtDD11, stencil_idx_0_0_0_VVV, (access(gyy, stencil_idx_0_0_0_VVV) * x3));
        store(gtDD22, stencil_idx_0_0_0_VVV, (access(gzz, stencil_idx_0_0_0_VVV) * x3));
        store(gtDD01, stencil_idx_0_0_0_VVV, (access(gxy, stencil_idx_0_0_0_VVV) * x3));
        store(gtDD02, stencil_idx_0_0_0_VVV, (access(gxz, stencil_idx_0_0_0_VVV) * x3));
        store(gtDD12, stencil_idx_0_0_0_VVV, (access(gyz, stencil_idx_0_0_0_VVV) * x3));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}