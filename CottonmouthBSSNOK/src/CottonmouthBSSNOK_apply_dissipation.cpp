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
void apply_dissipation(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_apply_dissipation;
    DECLARE_CCTK_PARAMETERS;
    using vreal = Arith::simd<CCTK_REAL>;
    constexpr std::size_t vsize = std::tuple_size_v<vreal>;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("apply_dissipation");
    #endif
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define AtDD00_layout VVV_layout
    #define AtDD01_layout VVV_layout
    #define AtDD02_layout VVV_layout
    #define AtDD11_layout VVV_layout
    #define AtDD12_layout VVV_layout
    #define AtDD22_layout VVV_layout
    #define At_rhsDD00_layout VVV_layout
    #define At_rhsDD01_layout VVV_layout
    #define At_rhsDD02_layout VVV_layout
    #define At_rhsDD11_layout VVV_layout
    #define At_rhsDD12_layout VVV_layout
    #define At_rhsDD22_layout VVV_layout
    #define ConfConnectU0_layout VVV_layout
    #define ConfConnectU1_layout VVV_layout
    #define ConfConnectU2_layout VVV_layout
    #define ConfConnect_rhsU0_layout VVV_layout
    #define ConfConnect_rhsU1_layout VVV_layout
    #define ConfConnect_rhsU2_layout VVV_layout
    #define evo_lapse_layout VVV_layout
    #define evo_lapse_rhs_layout VVV_layout
    #define evo_shiftU0_layout VVV_layout
    #define evo_shiftU1_layout VVV_layout
    #define evo_shiftU2_layout VVV_layout
    #define evo_shift_rhsU0_layout VVV_layout
    #define evo_shift_rhsU1_layout VVV_layout
    #define evo_shift_rhsU2_layout VVV_layout
    #define gtDD00_layout VVV_layout
    #define gtDD01_layout VVV_layout
    #define gtDD02_layout VVV_layout
    #define gtDD11_layout VVV_layout
    #define gtDD12_layout VVV_layout
    #define gtDD22_layout VVV_layout
    #define gt_rhsDD00_layout VVV_layout
    #define gt_rhsDD01_layout VVV_layout
    #define gt_rhsDD02_layout VVV_layout
    #define gt_rhsDD11_layout VVV_layout
    #define gt_rhsDD12_layout VVV_layout
    #define gt_rhsDD22_layout VVV_layout
    #define shift_BU0_layout VVV_layout
    #define shift_BU1_layout VVV_layout
    #define shift_BU2_layout VVV_layout
    #define shift_B_rhsU0_layout VVV_layout
    #define shift_B_rhsU1_layout VVV_layout
    #define shift_B_rhsU2_layout VVV_layout
    #define trK_layout VVV_layout
    #define trK_rhs_layout VVV_layout
    #define w_layout VVV_layout
    #define w_rhs_layout VVV_layout
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    CCTK_ASSERT((cctk_nghostzones[0] >= 3));
    CCTK_ASSERT((cctk_nghostzones[1] >= 3));
    CCTK_ASSERT((cctk_nghostzones[2] >= 3));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2], vsize>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        vreal x289 = stencil(trK, stencil_idx_0_m2_0_VVV);
        vreal x290 = stencil(trK, stencil_idx_0_2_0_VVV);
        vreal x291 = stencil(trK, stencil_idx_0_m1_0_VVV);
        vreal x292 = stencil(trK, stencil_idx_0_1_0_VVV);
        vreal x293 = stencil(trK, stencil_idx_0_0_m2_VVV);
        vreal x294 = stencil(trK, stencil_idx_0_0_2_VVV);
        vreal x295 = stencil(trK, stencil_idx_0_0_m1_VVV);
        vreal x296 = stencil(trK, stencil_idx_0_0_1_VVV);
        vreal x285 = stencil(trK, stencil_idx_m2_0_0_VVV);
        vreal x286 = stencil(trK, stencil_idx_2_0_0_VVV);
        vreal x287 = stencil(trK, stencil_idx_m1_0_0_VVV);
        vreal x288 = stencil(trK, stencil_idx_1_0_0_VVV);
        vreal x284 = (-0.312500000000000 * stencil(trK, stencil_idx_0_0_0_VVV));
        store(trK_rhs, stencil_idx_0_0_0_VVV, (access(trK_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x284 + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_m3_0_0_VVV) + stencil(trK, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x287 + x288))) + (-0.0937500000000000 * ((x285 + x286))))) + (DYI * (x284 + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_0_m3_0_VVV) + stencil(trK, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x291 + x292))) + (-0.0937500000000000 * ((x289 + x290))))) + (DZI * (x284 + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_0_0_m3_VVV) + stencil(trK, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x295 + x296))) + (-0.0937500000000000 * ((x293 + x294)))))))));
        x284 = (-0.312500000000000 * stencil(gtDD00, stencil_idx_0_0_0_VVV));
        x285 = stencil(gtDD00, stencil_idx_m2_0_0_VVV);
        x286 = stencil(gtDD00, stencil_idx_2_0_0_VVV);
        x287 = ((x285 + x286));
        x288 = stencil(gtDD00, stencil_idx_m1_0_0_VVV);
        x289 = stencil(gtDD00, stencil_idx_1_0_0_VVV);
        x290 = ((x288 + x289));
        x291 = stencil(gtDD00, stencil_idx_0_m2_0_VVV);
        x292 = stencil(gtDD00, stencil_idx_0_2_0_VVV);
        x293 = ((x291 + x292));
        x294 = stencil(gtDD00, stencil_idx_0_m1_0_VVV);
        x295 = stencil(gtDD00, stencil_idx_0_1_0_VVV);
        x296 = ((x294 + x295));
        vreal x82 = stencil(gtDD00, stencil_idx_0_0_m2_VVV);
        vreal x83 = stencil(gtDD00, stencil_idx_0_0_2_VVV);
        vreal x247 = ((x82 + x83));
        x82 = stencil(gtDD00, stencil_idx_0_0_m1_VVV);
        x83 = stencil(gtDD00, stencil_idx_0_0_1_VVV);
        vreal x248 = ((x82 + x83));
        store(gt_rhsDD00, stencil_idx_0_0_0_VVV, (access(gt_rhsDD00, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x284 + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_m3_0_0_VVV) + stencil(gtDD00, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x290) + (-0.0937500000000000 * x287))) + (DYI * (x284 + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_0_m3_0_VVV) + stencil(gtDD00, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x296) + (-0.0937500000000000 * x293))) + (DZI * (x284 + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_0_0_m3_VVV) + stencil(gtDD00, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x248) + (-0.0937500000000000 * x247)))))));
        x247 = (-0.312500000000000 * stencil(gtDD01, stencil_idx_0_0_0_VVV));
        x248 = stencil(gtDD01, stencil_idx_m2_0_0_VVV);
        vreal x155 = stencil(gtDD01, stencil_idx_2_0_0_VVV);
        vreal x250 = ((x155 + x248));
        x155 = stencil(gtDD01, stencil_idx_m1_0_0_VVV);
        vreal x157 = stencil(gtDD01, stencil_idx_1_0_0_VVV);
        vreal x251 = ((x155 + x157));
        x157 = stencil(gtDD01, stencil_idx_0_m2_0_VVV);
        vreal x119 = stencil(gtDD01, stencil_idx_0_2_0_VVV);
        vreal x252 = ((x119 + x157));
        x119 = stencil(gtDD01, stencil_idx_0_m1_0_VVV);
        vreal x121 = stencil(gtDD01, stencil_idx_0_1_0_VVV);
        vreal x253 = ((x119 + x121));
        x121 = stencil(gtDD01, stencil_idx_0_0_m2_VVV);
        vreal x61 = stencil(gtDD01, stencil_idx_0_0_2_VVV);
        vreal x254 = ((x121 + x61));
        x61 = stencil(gtDD01, stencil_idx_0_0_m1_VVV);
        vreal x63 = stencil(gtDD01, stencil_idx_0_0_1_VVV);
        vreal x255 = ((x61 + x63));
        store(gt_rhsDD01, stencil_idx_0_0_0_VVV, (access(gt_rhsDD01, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x247 + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_m3_0_0_VVV) + stencil(gtDD01, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x251) + (-0.0937500000000000 * x250))) + (DYI * (x247 + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_0_m3_0_VVV) + stencil(gtDD01, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x253) + (-0.0937500000000000 * x252))) + (DZI * (x247 + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_0_0_m3_VVV) + stencil(gtDD01, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x255) + (-0.0937500000000000 * x254)))))));
        x250 = (-0.312500000000000 * stencil(gtDD02, stencil_idx_0_0_0_VVV));
        x251 = stencil(gtDD02, stencil_idx_m2_0_0_VVV);
        x252 = stencil(gtDD02, stencil_idx_2_0_0_VVV);
        x253 = ((x251 + x252));
        x254 = stencil(gtDD02, stencil_idx_m1_0_0_VVV);
        x255 = stencil(gtDD02, stencil_idx_1_0_0_VVV);
        x63 = ((x254 + x255));
        vreal x55 = stencil(gtDD02, stencil_idx_0_m2_0_VVV);
        vreal x56 = stencil(gtDD02, stencil_idx_0_2_0_VVV);
        vreal x259 = ((x55 + x56));
        x55 = stencil(gtDD02, stencil_idx_0_m1_0_VVV);
        x56 = stencil(gtDD02, stencil_idx_0_1_0_VVV);
        vreal x260 = ((x55 + x56));
        vreal x171 = stencil(gtDD02, stencil_idx_0_0_m2_VVV);
        vreal x172 = stencil(gtDD02, stencil_idx_0_0_2_VVV);
        vreal x261 = ((x171 + x172));
        x171 = stencil(gtDD02, stencil_idx_0_0_m1_VVV);
        x172 = stencil(gtDD02, stencil_idx_0_0_1_VVV);
        vreal x262 = ((x171 + x172));
        store(gt_rhsDD02, stencil_idx_0_0_0_VVV, (access(gt_rhsDD02, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x250 + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_m3_0_0_VVV) + stencil(gtDD02, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x63) + (-0.0937500000000000 * x253))) + (DYI * (x250 + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_0_m3_0_VVV) + stencil(gtDD02, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x260) + (-0.0937500000000000 * x259))) + (DZI * (x250 + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_0_0_m3_VVV) + stencil(gtDD02, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x262) + (-0.0937500000000000 * x261)))))));
        x259 = (-0.312500000000000 * stencil(gtDD11, stencil_idx_0_0_0_VVV));
        x260 = stencil(gtDD11, stencil_idx_m2_0_0_VVV);
        x261 = stencil(gtDD11, stencil_idx_2_0_0_VVV);
        x262 = ((x260 + x261));
        vreal x102 = stencil(gtDD11, stencil_idx_m1_0_0_VVV);
        vreal x103 = stencil(gtDD11, stencil_idx_1_0_0_VVV);
        vreal x265 = ((x102 + x103));
        x102 = stencil(gtDD11, stencil_idx_0_m2_0_VVV);
        x103 = stencil(gtDD11, stencil_idx_0_2_0_VVV);
        vreal x266 = ((x102 + x103));
        vreal x112 = stencil(gtDD11, stencil_idx_0_m1_0_VVV);
        vreal x113 = stencil(gtDD11, stencil_idx_0_1_0_VVV);
        vreal x267 = ((x112 + x113));
        x112 = stencil(gtDD11, stencil_idx_0_0_m2_VVV);
        x113 = stencil(gtDD11, stencil_idx_0_0_2_VVV);
        vreal x268 = ((x112 + x113));
        vreal x47 = stencil(gtDD11, stencil_idx_0_0_m1_VVV);
        vreal x48 = stencil(gtDD11, stencil_idx_0_0_1_VVV);
        vreal x269 = ((x47 + x48));
        store(gt_rhsDD11, stencil_idx_0_0_0_VVV, (access(gt_rhsDD11, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x259 + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_m3_0_0_VVV) + stencil(gtDD11, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x265) + (-0.0937500000000000 * x262))) + (DYI * (x259 + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_0_m3_0_VVV) + stencil(gtDD11, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x267) + (-0.0937500000000000 * x266))) + (DZI * (x259 + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_0_0_m3_VVV) + stencil(gtDD11, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x269) + (-0.0937500000000000 * x268)))))));
        x265 = (-0.312500000000000 * stencil(gtDD12, stencil_idx_0_0_0_VVV));
        x266 = stencil(gtDD12, stencil_idx_m2_0_0_VVV);
        x267 = stencil(gtDD12, stencil_idx_2_0_0_VVV);
        x268 = ((x266 + x267));
        x269 = stencil(gtDD12, stencil_idx_m1_0_0_VVV);
        x47 = stencil(gtDD12, stencil_idx_1_0_0_VVV);
        x48 = ((x269 + x47));
        vreal x128 = stencil(gtDD12, stencil_idx_0_m2_0_VVV);
        vreal x129 = stencil(gtDD12, stencil_idx_0_2_0_VVV);
        vreal x273 = ((x128 + x129));
        x128 = stencil(gtDD12, stencil_idx_0_m1_0_VVV);
        x129 = stencil(gtDD12, stencil_idx_0_1_0_VVV);
        vreal x274 = ((x128 + x129));
        vreal x178 = stencil(gtDD12, stencil_idx_0_0_m2_VVV);
        vreal x179 = stencil(gtDD12, stencil_idx_0_0_2_VVV);
        vreal x275 = ((x178 + x179));
        x178 = stencil(gtDD12, stencil_idx_0_0_m1_VVV);
        x179 = stencil(gtDD12, stencil_idx_0_0_1_VVV);
        vreal x276 = ((x178 + x179));
        store(gt_rhsDD12, stencil_idx_0_0_0_VVV, (access(gt_rhsDD12, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x265 + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_m3_0_0_VVV) + stencil(gtDD12, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x48) + (-0.0937500000000000 * x268))) + (DYI * (x265 + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_0_m3_0_VVV) + stencil(gtDD12, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x274) + (-0.0937500000000000 * x273))) + (DZI * (x265 + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_0_0_m3_VVV) + stencil(gtDD12, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x276) + (-0.0937500000000000 * x275)))))));
        x273 = (-0.312500000000000 * stencil(gtDD22, stencil_idx_0_0_0_VVV));
        x274 = stencil(gtDD22, stencil_idx_m2_0_0_VVV);
        x275 = stencil(gtDD22, stencil_idx_2_0_0_VVV);
        x276 = ((x274 + x275));
        vreal x78 = stencil(gtDD22, stencil_idx_m1_0_0_VVV);
        vreal x79 = stencil(gtDD22, stencil_idx_1_0_0_VVV);
        vreal x279 = ((x78 + x79));
        x78 = stencil(gtDD22, stencil_idx_0_m2_0_VVV);
        x79 = stencil(gtDD22, stencil_idx_0_2_0_VVV);
        vreal x280 = ((x78 + x79));
        vreal x38 = stencil(gtDD22, stencil_idx_0_m1_0_VVV);
        vreal x39 = stencil(gtDD22, stencil_idx_0_1_0_VVV);
        vreal x281 = ((x38 + x39));
        x38 = stencil(gtDD22, stencil_idx_0_0_m2_VVV);
        x39 = stencil(gtDD22, stencil_idx_0_0_2_VVV);
        vreal x282 = ((x38 + x39));
        vreal x167 = stencil(gtDD22, stencil_idx_0_0_m1_VVV);
        vreal x168 = stencil(gtDD22, stencil_idx_0_0_1_VVV);
        vreal x283 = ((x167 + x168));
        store(gt_rhsDD22, stencil_idx_0_0_0_VVV, (access(gt_rhsDD22, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x273 + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_m3_0_0_VVV) + stencil(gtDD22, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x279) + (-0.0937500000000000 * x276))) + (DYI * (x273 + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_0_m3_0_VVV) + stencil(gtDD22, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x281) + (-0.0937500000000000 * x280))) + (DZI * (x273 + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_0_0_m3_VVV) + stencil(gtDD22, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x283) + (-0.0937500000000000 * x282)))))));
        x279 = (-0.312500000000000 * stencil(w, stencil_idx_0_0_0_VVV));
        x280 = stencil(w, stencil_idx_m2_0_0_VVV);
        x281 = stencil(w, stencil_idx_2_0_0_VVV);
        x282 = ((x280 + x281));
        x283 = stencil(w, stencil_idx_m1_0_0_VVV);
        x167 = stencil(w, stencil_idx_1_0_0_VVV);
        x168 = ((x167 + x283));
        vreal x304 = stencil(w, stencil_idx_0_m2_0_VVV);
        vreal x305 = stencil(w, stencil_idx_0_2_0_VVV);
        vreal x306 = ((x304 + x305));
        x304 = stencil(w, stencil_idx_0_m1_0_VVV);
        x305 = stencil(w, stencil_idx_0_1_0_VVV);
        vreal x309 = ((x304 + x305));
        vreal x310 = stencil(w, stencil_idx_0_0_m2_VVV);
        vreal x311 = stencil(w, stencil_idx_0_0_2_VVV);
        vreal x312 = ((x310 + x311));
        x310 = stencil(w, stencil_idx_0_0_m1_VVV);
        x311 = stencil(w, stencil_idx_0_0_1_VVV);
        vreal x315 = ((x310 + x311));
        store(w_rhs, stencil_idx_0_0_0_VVV, (access(w_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x279 + ((1.0 / 64.0) * ((stencil(w, stencil_idx_m3_0_0_VVV) + stencil(w, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x168) + (-0.0937500000000000 * x282))) + (DYI * (x279 + ((1.0 / 64.0) * ((stencil(w, stencil_idx_0_m3_0_VVV) + stencil(w, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x309) + (-0.0937500000000000 * x306))) + (DZI * (x279 + ((1.0 / 64.0) * ((stencil(w, stencil_idx_0_0_m3_VVV) + stencil(w, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x315) + (-0.0937500000000000 * x312)))))));    
    });
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2], vsize>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        vreal x321 = stencil(AtDD00, stencil_idx_0_m2_0_VVV);
        vreal x322 = stencil(AtDD00, stencil_idx_0_2_0_VVV);
        vreal x323 = stencil(AtDD00, stencil_idx_0_m1_0_VVV);
        vreal x324 = stencil(AtDD00, stencil_idx_0_1_0_VVV);
        vreal x325 = stencil(AtDD00, stencil_idx_0_0_m2_VVV);
        vreal x326 = stencil(AtDD00, stencil_idx_0_0_2_VVV);
        vreal x327 = stencil(AtDD00, stencil_idx_0_0_m1_VVV);
        vreal x328 = stencil(AtDD00, stencil_idx_0_0_1_VVV);
        vreal x317 = stencil(AtDD00, stencil_idx_m2_0_0_VVV);
        vreal x318 = stencil(AtDD00, stencil_idx_2_0_0_VVV);
        vreal x319 = stencil(AtDD00, stencil_idx_m1_0_0_VVV);
        vreal x320 = stencil(AtDD00, stencil_idx_1_0_0_VVV);
        vreal x316 = (-0.312500000000000 * stencil(AtDD00, stencil_idx_0_0_0_VVV));
        store(At_rhsDD00, stencil_idx_0_0_0_VVV, (access(At_rhsDD00, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x316 + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_m3_0_0_VVV) + stencil(AtDD00, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x319 + x320))) + (-0.0937500000000000 * ((x317 + x318))))) + (DYI * (x316 + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_0_m3_0_VVV) + stencil(AtDD00, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x323 + x324))) + (-0.0937500000000000 * ((x321 + x322))))) + (DZI * (x316 + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_0_0_m3_VVV) + stencil(AtDD00, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x327 + x328))) + (-0.0937500000000000 * ((x325 + x326)))))))));
        x316 = stencil(AtDD01, stencil_idx_0_m2_0_VVV);
        x317 = stencil(AtDD01, stencil_idx_0_2_0_VVV);
        x318 = stencil(AtDD01, stencil_idx_0_m1_0_VVV);
        x319 = stencil(AtDD01, stencil_idx_0_1_0_VVV);
        x320 = stencil(AtDD01, stencil_idx_0_0_m2_VVV);
        x321 = stencil(AtDD01, stencil_idx_0_0_2_VVV);
        x322 = stencil(AtDD01, stencil_idx_0_0_m1_VVV);
        x323 = stencil(AtDD01, stencil_idx_0_0_1_VVV);
        x324 = stencil(AtDD01, stencil_idx_m2_0_0_VVV);
        x325 = stencil(AtDD01, stencil_idx_2_0_0_VVV);
        x326 = stencil(AtDD01, stencil_idx_m1_0_0_VVV);
        x327 = stencil(AtDD01, stencil_idx_1_0_0_VVV);
        x328 = (-0.312500000000000 * stencil(AtDD01, stencil_idx_0_0_0_VVV));
        store(At_rhsDD01, stencil_idx_0_0_0_VVV, (access(At_rhsDD01, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x328 + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_m3_0_0_VVV) + stencil(AtDD01, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x326 + x327))) + (-0.0937500000000000 * ((x324 + x325))))) + (DYI * (x328 + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_0_m3_0_VVV) + stencil(AtDD01, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x318 + x319))) + (-0.0937500000000000 * ((x316 + x317))))) + (DZI * (x328 + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_0_0_m3_VVV) + stencil(AtDD01, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x322 + x323))) + (-0.0937500000000000 * ((x320 + x321)))))))));
        vreal x347 = stencil(AtDD02, stencil_idx_0_m2_0_VVV);
        vreal x348 = stencil(AtDD02, stencil_idx_0_2_0_VVV);
        vreal x349 = stencil(AtDD02, stencil_idx_0_m1_0_VVV);
        vreal x350 = stencil(AtDD02, stencil_idx_0_1_0_VVV);
        vreal x351 = stencil(AtDD02, stencil_idx_0_0_m2_VVV);
        vreal x352 = stencil(AtDD02, stencil_idx_0_0_2_VVV);
        vreal x353 = stencil(AtDD02, stencil_idx_0_0_m1_VVV);
        vreal x354 = stencil(AtDD02, stencil_idx_0_0_1_VVV);
        vreal x343 = stencil(AtDD02, stencil_idx_m2_0_0_VVV);
        vreal x344 = stencil(AtDD02, stencil_idx_2_0_0_VVV);
        vreal x345 = stencil(AtDD02, stencil_idx_m1_0_0_VVV);
        vreal x346 = stencil(AtDD02, stencil_idx_1_0_0_VVV);
        vreal x342 = (-0.312500000000000 * stencil(AtDD02, stencil_idx_0_0_0_VVV));
        store(At_rhsDD02, stencil_idx_0_0_0_VVV, (access(At_rhsDD02, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x342 + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_m3_0_0_VVV) + stencil(AtDD02, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x345 + x346))) + (-0.0937500000000000 * ((x343 + x344))))) + (DYI * (x342 + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_0_m3_0_VVV) + stencil(AtDD02, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x349 + x350))) + (-0.0937500000000000 * ((x347 + x348))))) + (DZI * (x342 + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_0_0_m3_VVV) + stencil(AtDD02, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x353 + x354))) + (-0.0937500000000000 * ((x351 + x352)))))))));
        x342 = stencil(AtDD11, stencil_idx_0_m2_0_VVV);
        x343 = stencil(AtDD11, stencil_idx_0_2_0_VVV);
        x344 = stencil(AtDD11, stencil_idx_0_m1_0_VVV);
        x345 = stencil(AtDD11, stencil_idx_0_1_0_VVV);
        x346 = stencil(AtDD11, stencil_idx_0_0_m2_VVV);
        x347 = stencil(AtDD11, stencil_idx_0_0_2_VVV);
        x348 = stencil(AtDD11, stencil_idx_0_0_m1_VVV);
        x349 = stencil(AtDD11, stencil_idx_0_0_1_VVV);
        x350 = stencil(AtDD11, stencil_idx_m2_0_0_VVV);
        x351 = stencil(AtDD11, stencil_idx_2_0_0_VVV);
        x352 = stencil(AtDD11, stencil_idx_m1_0_0_VVV);
        x353 = stencil(AtDD11, stencil_idx_1_0_0_VVV);
        x354 = (-0.312500000000000 * stencil(AtDD11, stencil_idx_0_0_0_VVV));
        store(At_rhsDD11, stencil_idx_0_0_0_VVV, (access(At_rhsDD11, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x354 + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_m3_0_0_VVV) + stencil(AtDD11, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x352 + x353))) + (-0.0937500000000000 * ((x350 + x351))))) + (DYI * (x354 + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_0_m3_0_VVV) + stencil(AtDD11, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x344 + x345))) + (-0.0937500000000000 * ((x342 + x343))))) + (DZI * (x354 + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_0_0_m3_VVV) + stencil(AtDD11, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x348 + x349))) + (-0.0937500000000000 * ((x346 + x347)))))))));
        vreal x373 = stencil(AtDD12, stencil_idx_0_m2_0_VVV);
        vreal x374 = stencil(AtDD12, stencil_idx_0_2_0_VVV);
        vreal x375 = stencil(AtDD12, stencil_idx_0_m1_0_VVV);
        vreal x376 = stencil(AtDD12, stencil_idx_0_1_0_VVV);
        vreal x377 = stencil(AtDD12, stencil_idx_0_0_m2_VVV);
        vreal x378 = stencil(AtDD12, stencil_idx_0_0_2_VVV);
        vreal x379 = stencil(AtDD12, stencil_idx_0_0_m1_VVV);
        vreal x380 = stencil(AtDD12, stencil_idx_0_0_1_VVV);
        vreal x369 = stencil(AtDD12, stencil_idx_m2_0_0_VVV);
        vreal x370 = stencil(AtDD12, stencil_idx_2_0_0_VVV);
        vreal x371 = stencil(AtDD12, stencil_idx_m1_0_0_VVV);
        vreal x372 = stencil(AtDD12, stencil_idx_1_0_0_VVV);
        vreal x368 = (-0.312500000000000 * stencil(AtDD12, stencil_idx_0_0_0_VVV));
        store(At_rhsDD12, stencil_idx_0_0_0_VVV, (access(At_rhsDD12, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x368 + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_m3_0_0_VVV) + stencil(AtDD12, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x371 + x372))) + (-0.0937500000000000 * ((x369 + x370))))) + (DYI * (x368 + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_0_m3_0_VVV) + stencil(AtDD12, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x375 + x376))) + (-0.0937500000000000 * ((x373 + x374))))) + (DZI * (x368 + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_0_0_m3_VVV) + stencil(AtDD12, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x379 + x380))) + (-0.0937500000000000 * ((x377 + x378)))))))));
        x368 = stencil(AtDD22, stencil_idx_0_m2_0_VVV);
        x369 = stencil(AtDD22, stencil_idx_0_2_0_VVV);
        x370 = stencil(AtDD22, stencil_idx_0_m1_0_VVV);
        x371 = stencil(AtDD22, stencil_idx_0_1_0_VVV);
        x372 = stencil(AtDD22, stencil_idx_0_0_m2_VVV);
        x373 = stencil(AtDD22, stencil_idx_0_0_2_VVV);
        x374 = stencil(AtDD22, stencil_idx_0_0_m1_VVV);
        x375 = stencil(AtDD22, stencil_idx_0_0_1_VVV);
        x376 = stencil(AtDD22, stencil_idx_m2_0_0_VVV);
        x377 = stencil(AtDD22, stencil_idx_2_0_0_VVV);
        x378 = stencil(AtDD22, stencil_idx_m1_0_0_VVV);
        x379 = stencil(AtDD22, stencil_idx_1_0_0_VVV);
        x380 = (-0.312500000000000 * stencil(AtDD22, stencil_idx_0_0_0_VVV));
        store(At_rhsDD22, stencil_idx_0_0_0_VVV, (access(At_rhsDD22, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x380 + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_m3_0_0_VVV) + stencil(AtDD22, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x378 + x379))) + (-0.0937500000000000 * ((x376 + x377))))) + (DYI * (x380 + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_0_m3_0_VVV) + stencil(AtDD22, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x370 + x371))) + (-0.0937500000000000 * ((x368 + x369))))) + (DZI * (x380 + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_0_0_m3_VVV) + stencil(AtDD22, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x374 + x375))) + (-0.0937500000000000 * ((x372 + x373)))))))));
        vreal x399 = stencil(ConfConnectU0, stencil_idx_0_m2_0_VVV);
        vreal x400 = stencil(ConfConnectU0, stencil_idx_0_2_0_VVV);
        vreal x401 = stencil(ConfConnectU0, stencil_idx_0_m1_0_VVV);
        vreal x402 = stencil(ConfConnectU0, stencil_idx_0_1_0_VVV);
        vreal x403 = stencil(ConfConnectU0, stencil_idx_0_0_m2_VVV);
        vreal x404 = stencil(ConfConnectU0, stencil_idx_0_0_2_VVV);
        vreal x405 = stencil(ConfConnectU0, stencil_idx_0_0_m1_VVV);
        vreal x406 = stencil(ConfConnectU0, stencil_idx_0_0_1_VVV);
        vreal x395 = stencil(ConfConnectU0, stencil_idx_m2_0_0_VVV);
        vreal x396 = stencil(ConfConnectU0, stencil_idx_2_0_0_VVV);
        vreal x397 = stencil(ConfConnectU0, stencil_idx_m1_0_0_VVV);
        vreal x398 = stencil(ConfConnectU0, stencil_idx_1_0_0_VVV);
        vreal x394 = (-0.312500000000000 * stencil(ConfConnectU0, stencil_idx_0_0_0_VVV));
        store(ConfConnect_rhsU0, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU0, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x394 + ((1.0 / 64.0) * ((stencil(ConfConnectU0, stencil_idx_m3_0_0_VVV) + stencil(ConfConnectU0, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x397 + x398))) + (-0.0937500000000000 * ((x395 + x396))))) + (DYI * (x394 + ((1.0 / 64.0) * ((stencil(ConfConnectU0, stencil_idx_0_m3_0_VVV) + stencil(ConfConnectU0, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x401 + x402))) + (-0.0937500000000000 * ((x399 + x400))))) + (DZI * (x394 + ((1.0 / 64.0) * ((stencil(ConfConnectU0, stencil_idx_0_0_m3_VVV) + stencil(ConfConnectU0, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x405 + x406))) + (-0.0937500000000000 * ((x403 + x404)))))))));
        x394 = stencil(ConfConnectU1, stencil_idx_0_m2_0_VVV);
        x395 = stencil(ConfConnectU1, stencil_idx_0_2_0_VVV);
        x396 = stencil(ConfConnectU1, stencil_idx_0_m1_0_VVV);
        x397 = stencil(ConfConnectU1, stencil_idx_0_1_0_VVV);
        x398 = stencil(ConfConnectU1, stencil_idx_0_0_m2_VVV);
        x399 = stencil(ConfConnectU1, stencil_idx_0_0_2_VVV);
        x400 = stencil(ConfConnectU1, stencil_idx_0_0_m1_VVV);
        x401 = stencil(ConfConnectU1, stencil_idx_0_0_1_VVV);
        x402 = stencil(ConfConnectU1, stencil_idx_m2_0_0_VVV);
        x403 = stencil(ConfConnectU1, stencil_idx_2_0_0_VVV);
        x404 = stencil(ConfConnectU1, stencil_idx_m1_0_0_VVV);
        x405 = stencil(ConfConnectU1, stencil_idx_1_0_0_VVV);
        x406 = (-0.312500000000000 * stencil(ConfConnectU1, stencil_idx_0_0_0_VVV));
        store(ConfConnect_rhsU1, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU1, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x406 + ((1.0 / 64.0) * ((stencil(ConfConnectU1, stencil_idx_m3_0_0_VVV) + stencil(ConfConnectU1, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x404 + x405))) + (-0.0937500000000000 * ((x402 + x403))))) + (DYI * (x406 + ((1.0 / 64.0) * ((stencil(ConfConnectU1, stencil_idx_0_m3_0_VVV) + stencil(ConfConnectU1, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x396 + x397))) + (-0.0937500000000000 * ((x394 + x395))))) + (DZI * (x406 + ((1.0 / 64.0) * ((stencil(ConfConnectU1, stencil_idx_0_0_m3_VVV) + stencil(ConfConnectU1, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x400 + x401))) + (-0.0937500000000000 * ((x398 + x399)))))))));
        vreal x425 = stencil(ConfConnectU2, stencil_idx_0_m2_0_VVV);
        vreal x426 = stencil(ConfConnectU2, stencil_idx_0_2_0_VVV);
        vreal x427 = stencil(ConfConnectU2, stencil_idx_0_m1_0_VVV);
        vreal x428 = stencil(ConfConnectU2, stencil_idx_0_1_0_VVV);
        vreal x429 = stencil(ConfConnectU2, stencil_idx_0_0_m2_VVV);
        vreal x430 = stencil(ConfConnectU2, stencil_idx_0_0_2_VVV);
        vreal x431 = stencil(ConfConnectU2, stencil_idx_0_0_m1_VVV);
        vreal x432 = stencil(ConfConnectU2, stencil_idx_0_0_1_VVV);
        vreal x421 = stencil(ConfConnectU2, stencil_idx_m2_0_0_VVV);
        vreal x422 = stencil(ConfConnectU2, stencil_idx_2_0_0_VVV);
        vreal x423 = stencil(ConfConnectU2, stencil_idx_m1_0_0_VVV);
        vreal x424 = stencil(ConfConnectU2, stencil_idx_1_0_0_VVV);
        vreal x420 = (-0.312500000000000 * stencil(ConfConnectU2, stencil_idx_0_0_0_VVV));
        store(ConfConnect_rhsU2, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU2, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x420 + ((1.0 / 64.0) * ((stencil(ConfConnectU2, stencil_idx_m3_0_0_VVV) + stencil(ConfConnectU2, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x423 + x424))) + (-0.0937500000000000 * ((x421 + x422))))) + (DYI * (x420 + ((1.0 / 64.0) * ((stencil(ConfConnectU2, stencil_idx_0_m3_0_VVV) + stencil(ConfConnectU2, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x427 + x428))) + (-0.0937500000000000 * ((x425 + x426))))) + (DZI * (x420 + ((1.0 / 64.0) * ((stencil(ConfConnectU2, stencil_idx_0_0_m3_VVV) + stencil(ConfConnectU2, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x431 + x432))) + (-0.0937500000000000 * ((x429 + x430)))))))));    
    });
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2], vsize>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_m3_0_0_VVV(VVV_layout, p.I - 3*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_m3_0_VVV(VVV_layout, p.I - 3*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_m3_VVV(VVV_layout, p.I - 3*p.DI[2]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_3_VVV(VVV_layout, p.I + 3*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_0_3_0_VVV(VVV_layout, p.I + 3*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_3_0_0_VVV(VVV_layout, p.I + 3*p.DI[0]);
        vreal x514 = stencil(shift_BU0, stencil_idx_0_m2_0_VVV);
        vreal x515 = stencil(shift_BU0, stencil_idx_0_2_0_VVV);
        vreal x516 = stencil(shift_BU0, stencil_idx_0_m1_0_VVV);
        vreal x517 = stencil(shift_BU0, stencil_idx_0_1_0_VVV);
        vreal x518 = stencil(shift_BU0, stencil_idx_0_0_m2_VVV);
        vreal x519 = stencil(shift_BU0, stencil_idx_0_0_2_VVV);
        vreal x520 = stencil(shift_BU0, stencil_idx_0_0_m1_VVV);
        vreal x521 = stencil(shift_BU0, stencil_idx_0_0_1_VVV);
        vreal x510 = stencil(shift_BU0, stencil_idx_m2_0_0_VVV);
        vreal x511 = stencil(shift_BU0, stencil_idx_2_0_0_VVV);
        vreal x512 = stencil(shift_BU0, stencil_idx_m1_0_0_VVV);
        vreal x513 = stencil(shift_BU0, stencil_idx_1_0_0_VVV);
        vreal x509 = (-0.312500000000000 * stencil(shift_BU0, stencil_idx_0_0_0_VVV));
        store(shift_B_rhsU0, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU0, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x509 + ((1.0 / 64.0) * ((stencil(shift_BU0, stencil_idx_m3_0_0_VVV) + stencil(shift_BU0, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x512 + x513))) + (-0.0937500000000000 * ((x510 + x511))))) + (DYI * (x509 + ((1.0 / 64.0) * ((stencil(shift_BU0, stencil_idx_0_m3_0_VVV) + stencil(shift_BU0, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x516 + x517))) + (-0.0937500000000000 * ((x514 + x515))))) + (DZI * (x509 + ((1.0 / 64.0) * ((stencil(shift_BU0, stencil_idx_0_0_m3_VVV) + stencil(shift_BU0, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x520 + x521))) + (-0.0937500000000000 * ((x518 + x519)))))))));
        x509 = stencil(shift_BU1, stencil_idx_0_m2_0_VVV);
        x510 = stencil(shift_BU1, stencil_idx_0_2_0_VVV);
        x511 = stencil(shift_BU1, stencil_idx_0_m1_0_VVV);
        x512 = stencil(shift_BU1, stencil_idx_0_1_0_VVV);
        x513 = stencil(shift_BU1, stencil_idx_0_0_m2_VVV);
        x514 = stencil(shift_BU1, stencil_idx_0_0_2_VVV);
        x515 = stencil(shift_BU1, stencil_idx_0_0_m1_VVV);
        x516 = stencil(shift_BU1, stencil_idx_0_0_1_VVV);
        x517 = stencil(shift_BU1, stencil_idx_m2_0_0_VVV);
        x518 = stencil(shift_BU1, stencil_idx_2_0_0_VVV);
        x519 = stencil(shift_BU1, stencil_idx_m1_0_0_VVV);
        x520 = stencil(shift_BU1, stencil_idx_1_0_0_VVV);
        x521 = (-0.312500000000000 * stencil(shift_BU1, stencil_idx_0_0_0_VVV));
        store(shift_B_rhsU1, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU1, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x521 + ((1.0 / 64.0) * ((stencil(shift_BU1, stencil_idx_m3_0_0_VVV) + stencil(shift_BU1, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x519 + x520))) + (-0.0937500000000000 * ((x517 + x518))))) + (DYI * (x521 + ((1.0 / 64.0) * ((stencil(shift_BU1, stencil_idx_0_m3_0_VVV) + stencil(shift_BU1, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x511 + x512))) + (-0.0937500000000000 * ((x509 + x510))))) + (DZI * (x521 + ((1.0 / 64.0) * ((stencil(shift_BU1, stencil_idx_0_0_m3_VVV) + stencil(shift_BU1, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x515 + x516))) + (-0.0937500000000000 * ((x513 + x514)))))))));
        vreal x540 = stencil(shift_BU2, stencil_idx_0_m2_0_VVV);
        vreal x541 = stencil(shift_BU2, stencil_idx_0_2_0_VVV);
        vreal x542 = stencil(shift_BU2, stencil_idx_0_m1_0_VVV);
        vreal x543 = stencil(shift_BU2, stencil_idx_0_1_0_VVV);
        vreal x544 = stencil(shift_BU2, stencil_idx_0_0_m2_VVV);
        vreal x545 = stencil(shift_BU2, stencil_idx_0_0_2_VVV);
        vreal x546 = stencil(shift_BU2, stencil_idx_0_0_m1_VVV);
        vreal x547 = stencil(shift_BU2, stencil_idx_0_0_1_VVV);
        vreal x536 = stencil(shift_BU2, stencil_idx_m2_0_0_VVV);
        vreal x537 = stencil(shift_BU2, stencil_idx_2_0_0_VVV);
        vreal x538 = stencil(shift_BU2, stencil_idx_m1_0_0_VVV);
        vreal x539 = stencil(shift_BU2, stencil_idx_1_0_0_VVV);
        vreal x535 = (-0.312500000000000 * stencil(shift_BU2, stencil_idx_0_0_0_VVV));
        store(shift_B_rhsU2, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU2, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x535 + ((1.0 / 64.0) * ((stencil(shift_BU2, stencil_idx_m3_0_0_VVV) + stencil(shift_BU2, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x538 + x539))) + (-0.0937500000000000 * ((x536 + x537))))) + (DYI * (x535 + ((1.0 / 64.0) * ((stencil(shift_BU2, stencil_idx_0_m3_0_VVV) + stencil(shift_BU2, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x542 + x543))) + (-0.0937500000000000 * ((x540 + x541))))) + (DZI * (x535 + ((1.0 / 64.0) * ((stencil(shift_BU2, stencil_idx_0_0_m3_VVV) + stencil(shift_BU2, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x546 + x547))) + (-0.0937500000000000 * ((x544 + x545)))))))));
        x535 = (-0.312500000000000 * stencil(evo_lapse, stencil_idx_0_0_0_VVV));
        x536 = stencil(evo_lapse, stencil_idx_m2_0_0_VVV);
        x537 = stencil(evo_lapse, stencil_idx_2_0_0_VVV);
        x538 = ((x536 + x537));
        x539 = stencil(evo_lapse, stencil_idx_m1_0_0_VVV);
        x540 = stencil(evo_lapse, stencil_idx_1_0_0_VVV);
        x541 = ((x539 + x540));
        x542 = stencil(evo_lapse, stencil_idx_0_m2_0_VVV);
        x543 = stencil(evo_lapse, stencil_idx_0_2_0_VVV);
        x544 = ((x542 + x543));
        x545 = stencil(evo_lapse, stencil_idx_0_m1_0_VVV);
        x546 = stencil(evo_lapse, stencil_idx_0_1_0_VVV);
        x547 = ((x545 + x546));
        vreal x446 = stencil(evo_lapse, stencil_idx_0_0_m2_VVV);
        vreal x447 = stencil(evo_lapse, stencil_idx_0_0_2_VVV);
        vreal x448 = ((x446 + x447));
        x446 = stencil(evo_lapse, stencil_idx_0_0_m1_VVV);
        x447 = stencil(evo_lapse, stencil_idx_0_0_1_VVV);
        vreal x451 = ((x446 + x447));
        store(evo_lapse_rhs, stencil_idx_0_0_0_VVV, (access(evo_lapse_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x535 + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_m3_0_0_VVV) + stencil(evo_lapse, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x541) + (-0.0937500000000000 * x538))) + (DYI * (x535 + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_0_m3_0_VVV) + stencil(evo_lapse, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x547) + (-0.0937500000000000 * x544))) + (DZI * (x535 + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_0_0_m3_VVV) + stencil(evo_lapse, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x451) + (-0.0937500000000000 * x448)))))));
        x448 = (-0.312500000000000 * stencil(evo_shiftU0, stencil_idx_0_0_0_VVV));
        x451 = stencil(evo_shiftU0, stencil_idx_m2_0_0_VVV);
        vreal x454 = stencil(evo_shiftU0, stencil_idx_2_0_0_VVV);
        vreal x455 = ((x451 + x454));
        x454 = stencil(evo_shiftU0, stencil_idx_m1_0_0_VVV);
        vreal x457 = stencil(evo_shiftU0, stencil_idx_1_0_0_VVV);
        vreal x458 = ((x454 + x457));
        x457 = stencil(evo_shiftU0, stencil_idx_0_m2_0_VVV);
        vreal x460 = stencil(evo_shiftU0, stencil_idx_0_2_0_VVV);
        vreal x461 = ((x457 + x460));
        x460 = stencil(evo_shiftU0, stencil_idx_0_m1_0_VVV);
        vreal x463 = stencil(evo_shiftU0, stencil_idx_0_1_0_VVV);
        vreal x464 = ((x460 + x463));
        x463 = stencil(evo_shiftU0, stencil_idx_0_0_m2_VVV);
        vreal x466 = stencil(evo_shiftU0, stencil_idx_0_0_2_VVV);
        vreal x467 = ((x463 + x466));
        x466 = stencil(evo_shiftU0, stencil_idx_0_0_m1_VVV);
        vreal x469 = stencil(evo_shiftU0, stencil_idx_0_0_1_VVV);
        vreal x470 = ((x466 + x469));
        store(evo_shift_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU0, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x448 + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU0, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x458) + (-0.0937500000000000 * x455))) + (DYI * (x448 + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU0, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x464) + (-0.0937500000000000 * x461))) + (DZI * (x448 + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU0, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x470) + (-0.0937500000000000 * x467)))))));
        x455 = (-0.312500000000000 * stencil(evo_shiftU1, stencil_idx_0_0_0_VVV));
        x458 = stencil(evo_shiftU1, stencil_idx_m2_0_0_VVV);
        x461 = stencil(evo_shiftU1, stencil_idx_2_0_0_VVV);
        x464 = ((x458 + x461));
        x467 = stencil(evo_shiftU1, stencil_idx_m1_0_0_VVV);
        x470 = stencil(evo_shiftU1, stencil_idx_1_0_0_VVV);
        x469 = ((x467 + x470));
        vreal x478 = stencil(evo_shiftU1, stencil_idx_0_m2_0_VVV);
        vreal x479 = stencil(evo_shiftU1, stencil_idx_0_2_0_VVV);
        vreal x480 = ((x478 + x479));
        x478 = stencil(evo_shiftU1, stencil_idx_0_m1_0_VVV);
        x479 = stencil(evo_shiftU1, stencil_idx_0_1_0_VVV);
        vreal x483 = ((x478 + x479));
        vreal x484 = stencil(evo_shiftU1, stencil_idx_0_0_m2_VVV);
        vreal x485 = stencil(evo_shiftU1, stencil_idx_0_0_2_VVV);
        vreal x486 = ((x484 + x485));
        x484 = stencil(evo_shiftU1, stencil_idx_0_0_m1_VVV);
        x485 = stencil(evo_shiftU1, stencil_idx_0_0_1_VVV);
        vreal x489 = ((x484 + x485));
        store(evo_shift_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU1, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x455 + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU1, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x469) + (-0.0937500000000000 * x464))) + (DYI * (x455 + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU1, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x483) + (-0.0937500000000000 * x480))) + (DZI * (x455 + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU1, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x489) + (-0.0937500000000000 * x486)))))));
        x480 = (-0.312500000000000 * stencil(evo_shiftU2, stencil_idx_0_0_0_VVV));
        x483 = stencil(evo_shiftU2, stencil_idx_m2_0_0_VVV);
        x486 = stencil(evo_shiftU2, stencil_idx_2_0_0_VVV);
        x489 = ((x483 + x486));
        vreal x494 = stencil(evo_shiftU2, stencil_idx_m1_0_0_VVV);
        vreal x495 = stencil(evo_shiftU2, stencil_idx_1_0_0_VVV);
        vreal x496 = ((x494 + x495));
        x494 = stencil(evo_shiftU2, stencil_idx_0_m2_0_VVV);
        x495 = stencil(evo_shiftU2, stencil_idx_0_2_0_VVV);
        vreal x499 = ((x494 + x495));
        vreal x500 = stencil(evo_shiftU2, stencil_idx_0_m1_0_VVV);
        vreal x501 = stencil(evo_shiftU2, stencil_idx_0_1_0_VVV);
        vreal x502 = ((x500 + x501));
        x500 = stencil(evo_shiftU2, stencil_idx_0_0_m2_VVV);
        x501 = stencil(evo_shiftU2, stencil_idx_0_0_2_VVV);
        vreal x505 = ((x500 + x501));
        vreal x506 = stencil(evo_shiftU2, stencil_idx_0_0_m1_VVV);
        vreal x507 = stencil(evo_shiftU2, stencil_idx_0_0_1_VVV);
        vreal x508 = ((x506 + x507));
        store(evo_shift_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU2, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x480 + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU2, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x496) + (-0.0937500000000000 * x489))) + (DYI * (x480 + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU2, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x502) + (-0.0937500000000000 * x499))) + (DZI * (x480 + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU2, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x508) + (-0.0937500000000000 * x505)))))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}