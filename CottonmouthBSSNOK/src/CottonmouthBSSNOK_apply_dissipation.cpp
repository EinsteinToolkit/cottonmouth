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
        vreal x308 = stencil(trK, stencil_idx_0_m2_0_VVV);
        vreal x314 = stencil(trK, stencil_idx_0_0_m1_VVV);
        vreal x307 = stencil(trK, stencil_idx_1_0_0_VVV);
        vreal x306 = stencil(trK, stencil_idx_m1_0_0_VVV);
        vreal x303 = (-0.312500000000000 * stencil(trK, stencil_idx_0_0_0_VVV));
        vreal x309 = stencil(trK, stencil_idx_0_2_0_VVV);
        vreal x304 = stencil(trK, stencil_idx_m2_0_0_VVV);
        vreal x311 = stencil(trK, stencil_idx_0_1_0_VVV);
        vreal x312 = stencil(trK, stencil_idx_0_0_m2_VVV);
        vreal x315 = stencil(trK, stencil_idx_0_0_1_VVV);
        vreal x305 = stencil(trK, stencil_idx_2_0_0_VVV);
        vreal x313 = stencil(trK, stencil_idx_0_0_2_VVV);
        vreal x310 = stencil(trK, stencil_idx_0_m1_0_VVV);
        store(trK_rhs, stencil_idx_0_0_0_VVV, (access(trK_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x303 + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_m3_0_0_VVV) + stencil(trK, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x306 + x307))) + (-0.0937500000000000 * ((x304 + x305))))) + (DYI * (x303 + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_0_m3_0_VVV) + stencil(trK, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x310 + x311))) + (-0.0937500000000000 * ((x308 + x309))))) + (DZI * (x303 + ((1.0 / 64.0) * ((stencil(trK, stencil_idx_0_0_m3_VVV) + stencil(trK, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x314 + x315))) + (-0.0937500000000000 * ((x312 + x313)))))))));
        x303 = stencil(gtDD00, stencil_idx_m1_0_0_VVV);
        x304 = stencil(gtDD00, stencil_idx_1_0_0_VVV);
        x305 = ((x303 + x304));
        x306 = stencil(gtDD00, stencil_idx_0_0_1_VVV);
        x307 = stencil(gtDD00, stencil_idx_0_0_m1_VVV);
        x308 = ((x306 + x307));
        x309 = stencil(gtDD00, stencil_idx_0_0_2_VVV);
        x310 = stencil(gtDD00, stencil_idx_0_0_m2_VVV);
        x311 = ((x309 + x310));
        x312 = stencil(gtDD00, stencil_idx_m2_0_0_VVV);
        x313 = stencil(gtDD00, stencil_idx_2_0_0_VVV);
        x314 = ((x312 + x313));
        x315 = stencil(gtDD00, stencil_idx_0_m1_0_VVV);
        vreal x96 = stencil(gtDD00, stencil_idx_0_1_0_VVV);
        vreal x246 = ((x315 + x96));
        x96 = stencil(gtDD00, stencil_idx_0_m2_0_VVV);
        vreal x94 = stencil(gtDD00, stencil_idx_0_2_0_VVV);
        vreal x245 = ((x94 + x96));
        x94 = (-0.312500000000000 * stencil(gtDD00, stencil_idx_0_0_0_VVV));
        store(gt_rhsDD00, stencil_idx_0_0_0_VVV, (access(gt_rhsDD00, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x94 + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_m3_0_0_VVV) + stencil(gtDD00, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x305) + (-0.0937500000000000 * x314))) + (DYI * (x94 + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_0_m3_0_VVV) + stencil(gtDD00, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x246) + (-0.0937500000000000 * x245))) + (DZI * (x94 + ((1.0 / 64.0) * ((stencil(gtDD00, stencil_idx_0_0_m3_VVV) + stencil(gtDD00, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x308) + (-0.0937500000000000 * x311)))))));
        x245 = stencil(gtDD01, stencil_idx_0_2_0_VVV);
        x246 = stencil(gtDD01, stencil_idx_0_m2_0_VVV);
        vreal x252 = ((x245 + x246));
        vreal x156 = stencil(gtDD01, stencil_idx_m1_0_0_VVV);
        vreal x157 = stencil(gtDD01, stencil_idx_1_0_0_VVV);
        vreal x251 = ((x156 + x157));
        x156 = stencil(gtDD01, stencil_idx_0_0_m1_VVV);
        x157 = stencil(gtDD01, stencil_idx_0_0_1_VVV);
        vreal x255 = ((x156 + x157));
        vreal x249 = (-0.312500000000000 * stencil(gtDD01, stencil_idx_0_0_0_VVV));
        vreal x60 = stencil(gtDD01, stencil_idx_0_0_m2_VVV);
        vreal x61 = stencil(gtDD01, stencil_idx_0_0_2_VVV);
        vreal x254 = ((x60 + x61));
        x60 = stencil(gtDD01, stencil_idx_2_0_0_VVV);
        x61 = stencil(gtDD01, stencil_idx_m2_0_0_VVV);
        vreal x250 = ((x60 + x61));
        vreal x120 = stencil(gtDD01, stencil_idx_0_m1_0_VVV);
        vreal x121 = stencil(gtDD01, stencil_idx_0_1_0_VVV);
        vreal x253 = ((x120 + x121));
        store(gt_rhsDD01, stencil_idx_0_0_0_VVV, (access(gt_rhsDD01, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x249 + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_m3_0_0_VVV) + stencil(gtDD01, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x251) + (-0.0937500000000000 * x250))) + (DYI * (x249 + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_0_m3_0_VVV) + stencil(gtDD01, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x253) + (-0.0937500000000000 * x252))) + (DZI * (x249 + ((1.0 / 64.0) * ((stencil(gtDD01, stencil_idx_0_0_m3_VVV) + stencil(gtDD01, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x255) + (-0.0937500000000000 * x254)))))));
        x249 = stencil(gtDD11, stencil_idx_0_0_2_VVV);
        x250 = stencil(gtDD11, stencil_idx_0_0_m2_VVV);
        x251 = ((x249 + x250));
        x252 = stencil(gtDD11, stencil_idx_1_0_0_VVV);
        x253 = stencil(gtDD11, stencil_idx_m1_0_0_VVV);
        x254 = ((x252 + x253));
        x255 = (-0.312500000000000 * stencil(gtDD11, stencil_idx_0_0_0_VVV));
        x120 = stencil(gtDD11, stencil_idx_0_2_0_VVV);
        x121 = stencil(gtDD11, stencil_idx_0_m2_0_VVV);
        vreal x259 = ((x120 + x121));
        vreal x48 = stencil(gtDD11, stencil_idx_0_0_1_VVV);
        vreal x47 = stencil(gtDD11, stencil_idx_0_0_m1_VVV);
        vreal x262 = ((x47 + x48));
        x47 = stencil(gtDD11, stencil_idx_0_1_0_VVV);
        x48 = stencil(gtDD11, stencil_idx_0_m1_0_VVV);
        vreal x260 = ((x47 + x48));
        vreal x101 = stencil(gtDD11, stencil_idx_2_0_0_VVV);
        vreal x100 = stencil(gtDD11, stencil_idx_m2_0_0_VVV);
        vreal x257 = ((x100 + x101));
        store(gt_rhsDD11, stencil_idx_0_0_0_VVV, (access(gt_rhsDD11, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x255 + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_m3_0_0_VVV) + stencil(gtDD11, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x254) + (-0.0937500000000000 * x257))) + (DYI * (x255 + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_0_m3_0_VVV) + stencil(gtDD11, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x260) + (-0.0937500000000000 * x259))) + (DZI * (x255 + ((1.0 / 64.0) * ((stencil(gtDD11, stencil_idx_0_0_m3_VVV) + stencil(gtDD11, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x262) + (-0.0937500000000000 * x251)))))));
        x257 = stencil(gtDD02, stencil_idx_0_0_m1_VVV);
        x259 = stencil(gtDD02, stencil_idx_0_0_1_VVV);
        x260 = ((x257 + x259));
        x262 = stencil(gtDD02, stencil_idx_0_1_0_VVV);
        x100 = stencil(gtDD02, stencil_idx_0_m1_0_VVV);
        x101 = ((x100 + x262));
        vreal x149 = stencil(gtDD02, stencil_idx_1_0_0_VVV);
        vreal x148 = stencil(gtDD02, stencil_idx_m1_0_0_VVV);
        vreal x265 = ((x148 + x149));
        x148 = (-0.312500000000000 * stencil(gtDD02, stencil_idx_0_0_0_VVV));
        x149 = stencil(gtDD02, stencil_idx_0_2_0_VVV);
        vreal x55 = stencil(gtDD02, stencil_idx_0_m2_0_VVV);
        vreal x266 = ((x149 + x55));
        x55 = stencil(gtDD02, stencil_idx_0_0_m2_VVV);
        vreal x172 = stencil(gtDD02, stencil_idx_0_0_2_VVV);
        vreal x268 = ((x172 + x55));
        x172 = stencil(gtDD02, stencil_idx_2_0_0_VVV);
        vreal x146 = stencil(gtDD02, stencil_idx_m2_0_0_VVV);
        vreal x264 = ((x146 + x172));
        store(gt_rhsDD02, stencil_idx_0_0_0_VVV, (access(gt_rhsDD02, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x148 + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_m3_0_0_VVV) + stencil(gtDD02, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x265) + (-0.0937500000000000 * x264))) + (DYI * (x148 + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_0_m3_0_VVV) + stencil(gtDD02, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x101) + (-0.0937500000000000 * x266))) + (DZI * (x148 + ((1.0 / 64.0) * ((stencil(gtDD02, stencil_idx_0_0_m3_VVV) + stencil(gtDD02, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x260) + (-0.0937500000000000 * x268)))))));
        x264 = stencil(gtDD12, stencil_idx_0_m1_0_VVV);
        x265 = stencil(gtDD12, stencil_idx_0_1_0_VVV);
        x266 = ((x264 + x265));
        x268 = stencil(gtDD12, stencil_idx_0_m2_0_VVV);
        x146 = stencil(gtDD12, stencil_idx_0_2_0_VVV);
        vreal x273 = ((x146 + x268));
        vreal x270 = (-0.312500000000000 * stencil(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x68 = stencil(gtDD12, stencil_idx_1_0_0_VVV);
        vreal x67 = stencil(gtDD12, stencil_idx_m1_0_0_VVV);
        vreal x272 = ((x67 + x68));
        x67 = stencil(gtDD12, stencil_idx_0_0_2_VVV);
        x68 = stencil(gtDD12, stencil_idx_0_0_m2_VVV);
        vreal x275 = ((x67 + x68));
        vreal x66 = stencil(gtDD12, stencil_idx_2_0_0_VVV);
        vreal x65 = stencil(gtDD12, stencil_idx_m2_0_0_VVV);
        vreal x271 = ((x65 + x66));
        x65 = stencil(gtDD12, stencil_idx_0_0_m1_VVV);
        x66 = stencil(gtDD12, stencil_idx_0_0_1_VVV);
        vreal x276 = ((x65 + x66));
        store(gt_rhsDD12, stencil_idx_0_0_0_VVV, (access(gt_rhsDD12, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x270 + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_m3_0_0_VVV) + stencil(gtDD12, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x272) + (-0.0937500000000000 * x271))) + (DYI * (x270 + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_0_m3_0_VVV) + stencil(gtDD12, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x266) + (-0.0937500000000000 * x273))) + (DZI * (x270 + ((1.0 / 64.0) * ((stencil(gtDD12, stencil_idx_0_0_m3_VVV) + stencil(gtDD12, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x276) + (-0.0937500000000000 * x275)))))));
        x270 = stencil(gtDD22, stencil_idx_0_m2_0_VVV);
        x271 = stencil(gtDD22, stencil_idx_0_2_0_VVV);
        x272 = ((x270 + x271));
        x273 = stencil(gtDD22, stencil_idx_0_0_1_VVV);
        x275 = stencil(gtDD22, stencil_idx_0_0_m1_VVV);
        x276 = ((x273 + x275));
        vreal x165 = stencil(gtDD22, stencil_idx_0_0_m2_VVV);
        vreal x166 = stencil(gtDD22, stencil_idx_0_0_2_VVV);
        vreal x282 = ((x165 + x166));
        x165 = (-0.312500000000000 * stencil(gtDD22, stencil_idx_0_0_0_VVV));
        x166 = stencil(gtDD22, stencil_idx_m2_0_0_VVV);
        vreal x77 = stencil(gtDD22, stencil_idx_2_0_0_VVV);
        vreal x278 = ((x166 + x77));
        x77 = stencil(gtDD22, stencil_idx_1_0_0_VVV);
        vreal x78 = stencil(gtDD22, stencil_idx_m1_0_0_VVV);
        vreal x279 = ((x77 + x78));
        x78 = stencil(gtDD22, stencil_idx_0_m1_0_VVV);
        vreal x39 = stencil(gtDD22, stencil_idx_0_1_0_VVV);
        vreal x281 = ((x39 + x78));
        store(gt_rhsDD22, stencil_idx_0_0_0_VVV, (access(gt_rhsDD22, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x165 + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_m3_0_0_VVV) + stencil(gtDD22, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x279) + (-0.0937500000000000 * x278))) + (DYI * (x165 + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_0_m3_0_VVV) + stencil(gtDD22, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x281) + (-0.0937500000000000 * x272))) + (DZI * (x165 + ((1.0 / 64.0) * ((stencil(gtDD22, stencil_idx_0_0_m3_VVV) + stencil(gtDD22, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x276) + (-0.0937500000000000 * x282)))))));
        x278 = stencil(w, stencil_idx_0_2_0_VVV);
        x279 = stencil(w, stencil_idx_0_m2_0_VVV);
        x281 = ((x278 + x279));
        x282 = stencil(w, stencil_idx_m1_0_0_VVV);
        x39 = stencil(w, stencil_idx_1_0_0_VVV);
        vreal x290 = ((x282 + x39));
        vreal x297 = stencil(w, stencil_idx_0_0_m2_VVV);
        vreal x298 = stencil(w, stencil_idx_0_0_2_VVV);
        vreal x299 = ((x297 + x298));
        x297 = stencil(w, stencil_idx_0_0_1_VVV);
        x298 = stencil(w, stencil_idx_0_0_m1_VVV);
        vreal x302 = ((x297 + x298));
        vreal x286 = stencil(w, stencil_idx_2_0_0_VVV);
        vreal x285 = stencil(w, stencil_idx_m2_0_0_VVV);
        vreal x287 = ((x285 + x286));
        x285 = stencil(w, stencil_idx_0_m1_0_VVV);
        x286 = stencil(w, stencil_idx_0_1_0_VVV);
        vreal x296 = ((x285 + x286));
        vreal x284 = (-0.312500000000000 * stencil(w, stencil_idx_0_0_0_VVV));
        store(w_rhs, stencil_idx_0_0_0_VVV, (access(w_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x284 + ((1.0 / 64.0) * ((stencil(w, stencil_idx_m3_0_0_VVV) + stencil(w, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x290) + (-0.0937500000000000 * x287))) + (DYI * (x284 + ((1.0 / 64.0) * ((stencil(w, stencil_idx_0_m3_0_VVV) + stencil(w, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x296) + (-0.0937500000000000 * x281))) + (DZI * (x284 + ((1.0 / 64.0) * ((stencil(w, stencil_idx_0_0_m3_VVV) + stencil(w, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x302) + (-0.0937500000000000 * x299)))))));    
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
        vreal x322 = stencil(AtDD00, stencil_idx_0_2_0_VVV);
        vreal x328 = stencil(AtDD00, stencil_idx_0_0_1_VVV);
        vreal x326 = stencil(AtDD00, stencil_idx_0_0_2_VVV);
        vreal x321 = stencil(AtDD00, stencil_idx_0_m2_0_VVV);
        vreal x320 = stencil(AtDD00, stencil_idx_1_0_0_VVV);
        vreal x327 = stencil(AtDD00, stencil_idx_0_0_m1_VVV);
        vreal x318 = stencil(AtDD00, stencil_idx_2_0_0_VVV);
        vreal x325 = stencil(AtDD00, stencil_idx_0_0_m2_VVV);
        vreal x319 = stencil(AtDD00, stencil_idx_m1_0_0_VVV);
        vreal x316 = (-0.312500000000000 * stencil(AtDD00, stencil_idx_0_0_0_VVV));
        vreal x323 = stencil(AtDD00, stencil_idx_0_m1_0_VVV);
        vreal x317 = stencil(AtDD00, stencil_idx_m2_0_0_VVV);
        vreal x324 = stencil(AtDD00, stencil_idx_0_1_0_VVV);
        store(At_rhsDD00, stencil_idx_0_0_0_VVV, (access(At_rhsDD00, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x316 + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_m3_0_0_VVV) + stencil(AtDD00, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x319 + x320))) + (-0.0937500000000000 * ((x317 + x318))))) + (DYI * (x316 + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_0_m3_0_VVV) + stencil(AtDD00, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x323 + x324))) + (-0.0937500000000000 * ((x321 + x322))))) + (DZI * (x316 + ((1.0 / 64.0) * ((stencil(AtDD00, stencil_idx_0_0_m3_VVV) + stencil(AtDD00, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x327 + x328))) + (-0.0937500000000000 * ((x325 + x326)))))))));
        x316 = stencil(AtDD01, stencil_idx_m1_0_0_VVV);
        x317 = stencil(AtDD01, stencil_idx_0_0_m2_VVV);
        x318 = stencil(AtDD01, stencil_idx_0_1_0_VVV);
        x319 = stencil(AtDD01, stencil_idx_m2_0_0_VVV);
        x320 = stencil(AtDD01, stencil_idx_0_0_2_VVV);
        x321 = stencil(AtDD01, stencil_idx_2_0_0_VVV);
        x322 = stencil(AtDD01, stencil_idx_0_m2_0_VVV);
        x323 = stencil(AtDD01, stencil_idx_0_0_1_VVV);
        x324 = stencil(AtDD01, stencil_idx_0_0_m1_VVV);
        x325 = stencil(AtDD01, stencil_idx_1_0_0_VVV);
        x326 = (-0.312500000000000 * stencil(AtDD01, stencil_idx_0_0_0_VVV));
        x327 = stencil(AtDD01, stencil_idx_0_2_0_VVV);
        x328 = stencil(AtDD01, stencil_idx_0_m1_0_VVV);
        store(At_rhsDD01, stencil_idx_0_0_0_VVV, (access(At_rhsDD01, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x326 + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_m3_0_0_VVV) + stencil(AtDD01, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x316 + x325))) + (-0.0937500000000000 * ((x319 + x321))))) + (DYI * (x326 + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_0_m3_0_VVV) + stencil(AtDD01, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x318 + x328))) + (-0.0937500000000000 * ((x322 + x327))))) + (DZI * (x326 + ((1.0 / 64.0) * ((stencil(AtDD01, stencil_idx_0_0_m3_VVV) + stencil(AtDD01, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x323 + x324))) + (-0.0937500000000000 * ((x317 + x320)))))))));
        vreal x345 = stencil(AtDD11, stencil_idx_m1_0_0_VVV);
        vreal x349 = stencil(AtDD11, stencil_idx_0_m1_0_VVV);
        vreal x353 = stencil(AtDD11, stencil_idx_0_0_m1_VVV);
        vreal x351 = stencil(AtDD11, stencil_idx_0_0_m2_VVV);
        vreal x347 = stencil(AtDD11, stencil_idx_0_m2_0_VVV);
        vreal x344 = stencil(AtDD11, stencil_idx_2_0_0_VVV);
        vreal x350 = stencil(AtDD11, stencil_idx_0_1_0_VVV);
        vreal x348 = stencil(AtDD11, stencil_idx_0_2_0_VVV);
        vreal x354 = stencil(AtDD11, stencil_idx_0_0_1_VVV);
        vreal x346 = stencil(AtDD11, stencil_idx_1_0_0_VVV);
        vreal x342 = (-0.312500000000000 * stencil(AtDD11, stencil_idx_0_0_0_VVV));
        vreal x343 = stencil(AtDD11, stencil_idx_m2_0_0_VVV);
        vreal x352 = stencil(AtDD11, stencil_idx_0_0_2_VVV);
        store(At_rhsDD11, stencil_idx_0_0_0_VVV, (access(At_rhsDD11, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x342 + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_m3_0_0_VVV) + stencil(AtDD11, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x345 + x346))) + (-0.0937500000000000 * ((x343 + x344))))) + (DYI * (x342 + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_0_m3_0_VVV) + stencil(AtDD11, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x349 + x350))) + (-0.0937500000000000 * ((x347 + x348))))) + (DZI * (x342 + ((1.0 / 64.0) * ((stencil(AtDD11, stencil_idx_0_0_m3_VVV) + stencil(AtDD11, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x353 + x354))) + (-0.0937500000000000 * ((x351 + x352)))))))));
        x342 = stencil(AtDD02, stencil_idx_0_0_2_VVV);
        x343 = stencil(AtDD02, stencil_idx_2_0_0_VVV);
        x344 = stencil(AtDD02, stencil_idx_0_0_m2_VVV);
        x345 = stencil(AtDD02, stencil_idx_0_1_0_VVV);
        x346 = stencil(AtDD02, stencil_idx_m2_0_0_VVV);
        x347 = stencil(AtDD02, stencil_idx_0_m2_0_VVV);
        x348 = stencil(AtDD02, stencil_idx_0_2_0_VVV);
        x349 = stencil(AtDD02, stencil_idx_0_0_1_VVV);
        x350 = stencil(AtDD02, stencil_idx_0_0_m1_VVV);
        x351 = stencil(AtDD02, stencil_idx_0_m1_0_VVV);
        x352 = stencil(AtDD02, stencil_idx_m1_0_0_VVV);
        x353 = stencil(AtDD02, stencil_idx_1_0_0_VVV);
        x354 = (-0.312500000000000 * stencil(AtDD02, stencil_idx_0_0_0_VVV));
        store(At_rhsDD02, stencil_idx_0_0_0_VVV, (access(At_rhsDD02, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x354 + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_m3_0_0_VVV) + stencil(AtDD02, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x352 + x353))) + (-0.0937500000000000 * ((x343 + x346))))) + (DYI * (x354 + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_0_m3_0_VVV) + stencil(AtDD02, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x345 + x351))) + (-0.0937500000000000 * ((x347 + x348))))) + (DZI * (x354 + ((1.0 / 64.0) * ((stencil(AtDD02, stencil_idx_0_0_m3_VVV) + stencil(AtDD02, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x349 + x350))) + (-0.0937500000000000 * ((x342 + x344)))))))));
        vreal x370 = stencil(AtDD12, stencil_idx_2_0_0_VVV);
        vreal x376 = stencil(AtDD12, stencil_idx_0_1_0_VVV);
        vreal x375 = stencil(AtDD12, stencil_idx_0_m1_0_VVV);
        vreal x368 = (-0.312500000000000 * stencil(AtDD12, stencil_idx_0_0_0_VVV));
        vreal x373 = stencil(AtDD12, stencil_idx_0_m2_0_VVV);
        vreal x369 = stencil(AtDD12, stencil_idx_m2_0_0_VVV);
        vreal x372 = stencil(AtDD12, stencil_idx_1_0_0_VVV);
        vreal x378 = stencil(AtDD12, stencil_idx_0_0_2_VVV);
        vreal x377 = stencil(AtDD12, stencil_idx_0_0_m2_VVV);
        vreal x380 = stencil(AtDD12, stencil_idx_0_0_1_VVV);
        vreal x379 = stencil(AtDD12, stencil_idx_0_0_m1_VVV);
        vreal x371 = stencil(AtDD12, stencil_idx_m1_0_0_VVV);
        vreal x374 = stencil(AtDD12, stencil_idx_0_2_0_VVV);
        store(At_rhsDD12, stencil_idx_0_0_0_VVV, (access(At_rhsDD12, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x368 + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_m3_0_0_VVV) + stencil(AtDD12, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x371 + x372))) + (-0.0937500000000000 * ((x369 + x370))))) + (DYI * (x368 + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_0_m3_0_VVV) + stencil(AtDD12, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x375 + x376))) + (-0.0937500000000000 * ((x373 + x374))))) + (DZI * (x368 + ((1.0 / 64.0) * ((stencil(AtDD12, stencil_idx_0_0_m3_VVV) + stencil(AtDD12, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x379 + x380))) + (-0.0937500000000000 * ((x377 + x378)))))))));
        x368 = stencil(AtDD22, stencil_idx_m2_0_0_VVV);
        x369 = stencil(AtDD22, stencil_idx_0_0_m1_VVV);
        x370 = stencil(AtDD22, stencil_idx_0_0_m2_VVV);
        x371 = stencil(AtDD22, stencil_idx_0_0_1_VVV);
        x372 = stencil(AtDD22, stencil_idx_0_m1_0_VVV);
        x373 = stencil(AtDD22, stencil_idx_0_1_0_VVV);
        x374 = stencil(AtDD22, stencil_idx_1_0_0_VVV);
        x375 = stencil(AtDD22, stencil_idx_0_0_2_VVV);
        x376 = stencil(AtDD22, stencil_idx_0_2_0_VVV);
        x377 = stencil(AtDD22, stencil_idx_2_0_0_VVV);
        x378 = stencil(AtDD22, stencil_idx_m1_0_0_VVV);
        x379 = (-0.312500000000000 * stencil(AtDD22, stencil_idx_0_0_0_VVV));
        x380 = stencil(AtDD22, stencil_idx_0_m2_0_VVV);
        store(At_rhsDD22, stencil_idx_0_0_0_VVV, (access(At_rhsDD22, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x379 + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_m3_0_0_VVV) + stencil(AtDD22, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x374 + x378))) + (-0.0937500000000000 * ((x368 + x377))))) + (DYI * (x379 + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_0_m3_0_VVV) + stencil(AtDD22, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x372 + x373))) + (-0.0937500000000000 * ((x376 + x380))))) + (DZI * (x379 + ((1.0 / 64.0) * ((stencil(AtDD22, stencil_idx_0_0_m3_VVV) + stencil(AtDD22, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x369 + x371))) + (-0.0937500000000000 * ((x370 + x375)))))))));
        vreal x403 = stencil(ConfConnectU0, stencil_idx_0_0_m2_VVV);
        vreal x396 = stencil(ConfConnectU0, stencil_idx_2_0_0_VVV);
        vreal x397 = stencil(ConfConnectU0, stencil_idx_m1_0_0_VVV);
        vreal x405 = stencil(ConfConnectU0, stencil_idx_0_0_m1_VVV);
        vreal x404 = stencil(ConfConnectU0, stencil_idx_0_0_2_VVV);
        vreal x400 = stencil(ConfConnectU0, stencil_idx_0_2_0_VVV);
        vreal x394 = (-0.312500000000000 * stencil(ConfConnectU0, stencil_idx_0_0_0_VVV));
        vreal x399 = stencil(ConfConnectU0, stencil_idx_0_m2_0_VVV);
        vreal x398 = stencil(ConfConnectU0, stencil_idx_1_0_0_VVV);
        vreal x402 = stencil(ConfConnectU0, stencil_idx_0_1_0_VVV);
        vreal x401 = stencil(ConfConnectU0, stencil_idx_0_m1_0_VVV);
        vreal x406 = stencil(ConfConnectU0, stencil_idx_0_0_1_VVV);
        vreal x395 = stencil(ConfConnectU0, stencil_idx_m2_0_0_VVV);
        store(ConfConnect_rhsU0, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU0, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x394 + ((1.0 / 64.0) * ((stencil(ConfConnectU0, stencil_idx_m3_0_0_VVV) + stencil(ConfConnectU0, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x397 + x398))) + (-0.0937500000000000 * ((x395 + x396))))) + (DYI * (x394 + ((1.0 / 64.0) * ((stencil(ConfConnectU0, stencil_idx_0_m3_0_VVV) + stencil(ConfConnectU0, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x401 + x402))) + (-0.0937500000000000 * ((x399 + x400))))) + (DZI * (x394 + ((1.0 / 64.0) * ((stencil(ConfConnectU0, stencil_idx_0_0_m3_VVV) + stencil(ConfConnectU0, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x405 + x406))) + (-0.0937500000000000 * ((x403 + x404)))))))));
        x394 = stencil(ConfConnectU1, stencil_idx_m2_0_0_VVV);
        x395 = stencil(ConfConnectU1, stencil_idx_1_0_0_VVV);
        x396 = stencil(ConfConnectU1, stencil_idx_0_m2_0_VVV);
        x397 = (-0.312500000000000 * stencil(ConfConnectU1, stencil_idx_0_0_0_VVV));
        x398 = stencil(ConfConnectU1, stencil_idx_2_0_0_VVV);
        x399 = stencil(ConfConnectU1, stencil_idx_m1_0_0_VVV);
        x400 = stencil(ConfConnectU1, stencil_idx_0_1_0_VVV);
        x401 = stencil(ConfConnectU1, stencil_idx_0_0_2_VVV);
        x402 = stencil(ConfConnectU1, stencil_idx_0_0_m1_VVV);
        x403 = stencil(ConfConnectU1, stencil_idx_0_m1_0_VVV);
        x404 = stencil(ConfConnectU1, stencil_idx_0_0_m2_VVV);
        x405 = stencil(ConfConnectU1, stencil_idx_0_2_0_VVV);
        x406 = stencil(ConfConnectU1, stencil_idx_0_0_1_VVV);
        store(ConfConnect_rhsU1, stencil_idx_0_0_0_VVV, (access(ConfConnect_rhsU1, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x397 + ((1.0 / 64.0) * ((stencil(ConfConnectU1, stencil_idx_m3_0_0_VVV) + stencil(ConfConnectU1, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x395 + x399))) + (-0.0937500000000000 * ((x394 + x398))))) + (DYI * (x397 + ((1.0 / 64.0) * ((stencil(ConfConnectU1, stencil_idx_0_m3_0_VVV) + stencil(ConfConnectU1, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x400 + x403))) + (-0.0937500000000000 * ((x396 + x405))))) + (DZI * (x397 + ((1.0 / 64.0) * ((stencil(ConfConnectU1, stencil_idx_0_0_m3_VVV) + stencil(ConfConnectU1, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x402 + x406))) + (-0.0937500000000000 * ((x401 + x404)))))))));
        vreal x420 = (-0.312500000000000 * stencil(ConfConnectU2, stencil_idx_0_0_0_VVV));
        vreal x432 = stencil(ConfConnectU2, stencil_idx_0_0_1_VVV);
        vreal x421 = stencil(ConfConnectU2, stencil_idx_m2_0_0_VVV);
        vreal x426 = stencil(ConfConnectU2, stencil_idx_0_2_0_VVV);
        vreal x431 = stencil(ConfConnectU2, stencil_idx_0_0_m1_VVV);
        vreal x424 = stencil(ConfConnectU2, stencil_idx_1_0_0_VVV);
        vreal x429 = stencil(ConfConnectU2, stencil_idx_0_0_m2_VVV);
        vreal x428 = stencil(ConfConnectU2, stencil_idx_0_1_0_VVV);
        vreal x422 = stencil(ConfConnectU2, stencil_idx_2_0_0_VVV);
        vreal x430 = stencil(ConfConnectU2, stencil_idx_0_0_2_VVV);
        vreal x427 = stencil(ConfConnectU2, stencil_idx_0_m1_0_VVV);
        vreal x425 = stencil(ConfConnectU2, stencil_idx_0_m2_0_VVV);
        vreal x423 = stencil(ConfConnectU2, stencil_idx_m1_0_0_VVV);
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
        vreal x518 = stencil(shift_BU0, stencil_idx_0_0_m2_VVV);
        vreal x509 = (-0.312500000000000 * stencil(shift_BU0, stencil_idx_0_0_0_VVV));
        vreal x516 = stencil(shift_BU0, stencil_idx_0_m1_0_VVV);
        vreal x512 = stencil(shift_BU0, stencil_idx_m1_0_0_VVV);
        vreal x514 = stencil(shift_BU0, stencil_idx_0_m2_0_VVV);
        vreal x511 = stencil(shift_BU0, stencil_idx_2_0_0_VVV);
        vreal x517 = stencil(shift_BU0, stencil_idx_0_1_0_VVV);
        vreal x521 = stencil(shift_BU0, stencil_idx_0_0_1_VVV);
        vreal x510 = stencil(shift_BU0, stencil_idx_m2_0_0_VVV);
        vreal x513 = stencil(shift_BU0, stencil_idx_1_0_0_VVV);
        vreal x519 = stencil(shift_BU0, stencil_idx_0_0_2_VVV);
        vreal x520 = stencil(shift_BU0, stencil_idx_0_0_m1_VVV);
        vreal x515 = stencil(shift_BU0, stencil_idx_0_2_0_VVV);
        store(shift_B_rhsU0, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU0, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x509 + ((1.0 / 64.0) * ((stencil(shift_BU0, stencil_idx_m3_0_0_VVV) + stencil(shift_BU0, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x512 + x513))) + (-0.0937500000000000 * ((x510 + x511))))) + (DYI * (x509 + ((1.0 / 64.0) * ((stencil(shift_BU0, stencil_idx_0_m3_0_VVV) + stencil(shift_BU0, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x516 + x517))) + (-0.0937500000000000 * ((x514 + x515))))) + (DZI * (x509 + ((1.0 / 64.0) * ((stencil(shift_BU0, stencil_idx_0_0_m3_VVV) + stencil(shift_BU0, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x520 + x521))) + (-0.0937500000000000 * ((x518 + x519)))))))));
        x509 = stencil(shift_BU1, stencil_idx_0_0_m1_VVV);
        x510 = (-0.312500000000000 * stencil(shift_BU1, stencil_idx_0_0_0_VVV));
        x511 = stencil(shift_BU1, stencil_idx_0_0_1_VVV);
        x512 = stencil(shift_BU1, stencil_idx_0_m2_0_VVV);
        x513 = stencil(shift_BU1, stencil_idx_0_2_0_VVV);
        x514 = stencil(shift_BU1, stencil_idx_m2_0_0_VVV);
        x515 = stencil(shift_BU1, stencil_idx_0_0_m2_VVV);
        x516 = stencil(shift_BU1, stencil_idx_2_0_0_VVV);
        x517 = stencil(shift_BU1, stencil_idx_0_m1_0_VVV);
        x518 = stencil(shift_BU1, stencil_idx_0_1_0_VVV);
        x519 = stencil(shift_BU1, stencil_idx_m1_0_0_VVV);
        x520 = stencil(shift_BU1, stencil_idx_1_0_0_VVV);
        x521 = stencil(shift_BU1, stencil_idx_0_0_2_VVV);
        store(shift_B_rhsU1, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU1, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x510 + ((1.0 / 64.0) * ((stencil(shift_BU1, stencil_idx_m3_0_0_VVV) + stencil(shift_BU1, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x519 + x520))) + (-0.0937500000000000 * ((x514 + x516))))) + (DYI * (x510 + ((1.0 / 64.0) * ((stencil(shift_BU1, stencil_idx_0_m3_0_VVV) + stencil(shift_BU1, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x517 + x518))) + (-0.0937500000000000 * ((x512 + x513))))) + (DZI * (x510 + ((1.0 / 64.0) * ((stencil(shift_BU1, stencil_idx_0_0_m3_VVV) + stencil(shift_BU1, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x509 + x511))) + (-0.0937500000000000 * ((x515 + x521)))))))));
        vreal x546 = stencil(shift_BU2, stencil_idx_0_0_m1_VVV);
        vreal x539 = stencil(shift_BU2, stencil_idx_1_0_0_VVV);
        vreal x535 = (-0.312500000000000 * stencil(shift_BU2, stencil_idx_0_0_0_VVV));
        vreal x542 = stencil(shift_BU2, stencil_idx_0_m1_0_VVV);
        vreal x537 = stencil(shift_BU2, stencil_idx_2_0_0_VVV);
        vreal x540 = stencil(shift_BU2, stencil_idx_0_m2_0_VVV);
        vreal x545 = stencil(shift_BU2, stencil_idx_0_0_2_VVV);
        vreal x536 = stencil(shift_BU2, stencil_idx_m2_0_0_VVV);
        vreal x538 = stencil(shift_BU2, stencil_idx_m1_0_0_VVV);
        vreal x547 = stencil(shift_BU2, stencil_idx_0_0_1_VVV);
        vreal x543 = stencil(shift_BU2, stencil_idx_0_1_0_VVV);
        vreal x544 = stencil(shift_BU2, stencil_idx_0_0_m2_VVV);
        vreal x541 = stencil(shift_BU2, stencil_idx_0_2_0_VVV);
        store(shift_B_rhsU2, stencil_idx_0_0_0_VVV, (access(shift_B_rhsU2, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x535 + ((1.0 / 64.0) * ((stencil(shift_BU2, stencil_idx_m3_0_0_VVV) + stencil(shift_BU2, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * ((x538 + x539))) + (-0.0937500000000000 * ((x536 + x537))))) + (DYI * (x535 + ((1.0 / 64.0) * ((stencil(shift_BU2, stencil_idx_0_m3_0_VVV) + stencil(shift_BU2, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * ((x542 + x543))) + (-0.0937500000000000 * ((x540 + x541))))) + (DZI * (x535 + ((1.0 / 64.0) * ((stencil(shift_BU2, stencil_idx_0_0_m3_VVV) + stencil(shift_BU2, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * ((x546 + x547))) + (-0.0937500000000000 * ((x544 + x545)))))))));
        x535 = stencil(evo_lapse, stencil_idx_2_0_0_VVV);
        x536 = stencil(evo_lapse, stencil_idx_m2_0_0_VVV);
        x537 = ((x535 + x536));
        x538 = (-0.312500000000000 * stencil(evo_lapse, stencil_idx_0_0_0_VVV));
        x539 = stencil(evo_lapse, stencil_idx_0_1_0_VVV);
        x540 = stencil(evo_lapse, stencil_idx_0_m1_0_VVV);
        x541 = ((x539 + x540));
        x542 = stencil(evo_lapse, stencil_idx_0_m2_0_VVV);
        x543 = stencil(evo_lapse, stencil_idx_0_2_0_VVV);
        x544 = ((x542 + x543));
        x545 = stencil(evo_lapse, stencil_idx_0_0_m2_VVV);
        x546 = stencil(evo_lapse, stencil_idx_0_0_2_VVV);
        x547 = ((x545 + x546));
        vreal x438 = stencil(evo_lapse, stencil_idx_1_0_0_VVV);
        vreal x437 = stencil(evo_lapse, stencil_idx_m1_0_0_VVV);
        vreal x439 = ((x437 + x438));
        x437 = stencil(evo_lapse, stencil_idx_0_0_1_VVV);
        x438 = stencil(evo_lapse, stencil_idx_0_0_m1_VVV);
        vreal x451 = ((x437 + x438));
        store(evo_lapse_rhs, stencil_idx_0_0_0_VVV, (access(evo_lapse_rhs, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x538 + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_m3_0_0_VVV) + stencil(evo_lapse, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x439) + (-0.0937500000000000 * x537))) + (DYI * (x538 + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_0_m3_0_VVV) + stencil(evo_lapse, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x541) + (-0.0937500000000000 * x544))) + (DZI * (x538 + ((1.0 / 64.0) * ((stencil(evo_lapse, stencil_idx_0_0_m3_VVV) + stencil(evo_lapse, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x451) + (-0.0937500000000000 * x547)))))));
        x439 = stencil(evo_shiftU0, stencil_idx_m2_0_0_VVV);
        x451 = stencil(evo_shiftU0, stencil_idx_2_0_0_VVV);
        vreal x455 = ((x439 + x451));
        vreal x457 = stencil(evo_shiftU0, stencil_idx_1_0_0_VVV);
        vreal x456 = stencil(evo_shiftU0, stencil_idx_m1_0_0_VVV);
        vreal x458 = ((x456 + x457));
        x456 = stencil(evo_shiftU0, stencil_idx_0_m2_0_VVV);
        x457 = stencil(evo_shiftU0, stencil_idx_0_2_0_VVV);
        vreal x461 = ((x456 + x457));
        vreal x466 = stencil(evo_shiftU0, stencil_idx_0_0_2_VVV);
        vreal x465 = stencil(evo_shiftU0, stencil_idx_0_0_m2_VVV);
        vreal x467 = ((x465 + x466));
        x465 = stencil(evo_shiftU0, stencil_idx_0_0_m1_VVV);
        x466 = stencil(evo_shiftU0, stencil_idx_0_0_1_VVV);
        vreal x470 = ((x465 + x466));
        vreal x452 = (-0.312500000000000 * stencil(evo_shiftU0, stencil_idx_0_0_0_VVV));
        vreal x462 = stencil(evo_shiftU0, stencil_idx_0_m1_0_VVV);
        vreal x463 = stencil(evo_shiftU0, stencil_idx_0_1_0_VVV);
        vreal x464 = ((x462 + x463));
        store(evo_shift_rhsU0, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU0, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x452 + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU0, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x458) + (-0.0937500000000000 * x455))) + (DYI * (x452 + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU0, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x464) + (-0.0937500000000000 * x461))) + (DZI * (x452 + ((1.0 / 64.0) * ((stencil(evo_shiftU0, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU0, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x470) + (-0.0937500000000000 * x467)))))));
        x452 = stencil(evo_shiftU1, stencil_idx_0_1_0_VVV);
        x455 = stencil(evo_shiftU1, stencil_idx_0_m1_0_VVV);
        x458 = ((x452 + x455));
        x461 = stencil(evo_shiftU1, stencil_idx_0_2_0_VVV);
        x464 = stencil(evo_shiftU1, stencil_idx_0_m2_0_VVV);
        x467 = ((x461 + x464));
        x470 = (-0.312500000000000 * stencil(evo_shiftU1, stencil_idx_0_0_0_VVV));
        x462 = stencil(evo_shiftU1, stencil_idx_0_0_1_VVV);
        x463 = stencil(evo_shiftU1, stencil_idx_0_0_m1_VVV);
        vreal x489 = ((x462 + x463));
        vreal x485 = stencil(evo_shiftU1, stencil_idx_0_0_2_VVV);
        vreal x484 = stencil(evo_shiftU1, stencil_idx_0_0_m2_VVV);
        vreal x486 = ((x484 + x485));
        x484 = stencil(evo_shiftU1, stencil_idx_m1_0_0_VVV);
        x485 = stencil(evo_shiftU1, stencil_idx_1_0_0_VVV);
        vreal x477 = ((x484 + x485));
        vreal x473 = stencil(evo_shiftU1, stencil_idx_2_0_0_VVV);
        vreal x472 = stencil(evo_shiftU1, stencil_idx_m2_0_0_VVV);
        vreal x474 = ((x472 + x473));
        store(evo_shift_rhsU1, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU1, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x470 + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU1, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x477) + (-0.0937500000000000 * x474))) + (DYI * (x470 + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU1, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x458) + (-0.0937500000000000 * x467))) + (DZI * (x470 + ((1.0 / 64.0) * ((stencil(evo_shiftU1, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU1, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x489) + (-0.0937500000000000 * x486)))))));
        x474 = stencil(evo_shiftU2, stencil_idx_0_m2_0_VVV);
        x477 = stencil(evo_shiftU2, stencil_idx_0_2_0_VVV);
        x486 = ((x474 + x477));
        x489 = stencil(evo_shiftU2, stencil_idx_0_1_0_VVV);
        x472 = stencil(evo_shiftU2, stencil_idx_0_m1_0_VVV);
        x473 = ((x472 + x489));
        vreal x504 = stencil(evo_shiftU2, stencil_idx_0_0_2_VVV);
        vreal x503 = stencil(evo_shiftU2, stencil_idx_0_0_m2_VVV);
        vreal x505 = ((x503 + x504));
        x503 = stencil(evo_shiftU2, stencil_idx_2_0_0_VVV);
        x504 = stencil(evo_shiftU2, stencil_idx_m2_0_0_VVV);
        vreal x493 = ((x503 + x504));
        vreal x494 = stencil(evo_shiftU2, stencil_idx_m1_0_0_VVV);
        vreal x495 = stencil(evo_shiftU2, stencil_idx_1_0_0_VVV);
        vreal x496 = ((x494 + x495));
        x494 = stencil(evo_shiftU2, stencil_idx_0_0_1_VVV);
        x495 = stencil(evo_shiftU2, stencil_idx_0_0_m1_VVV);
        vreal x508 = ((x494 + x495));
        vreal x490 = (-0.312500000000000 * stencil(evo_shiftU2, stencil_idx_0_0_0_VVV));
        store(evo_shift_rhsU2, stencil_idx_0_0_0_VVV, (access(evo_shift_rhsU2, stencil_idx_0_0_0_VVV) + (dissipation_epsilon * ((DXI * (x490 + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_m3_0_0_VVV) + stencil(evo_shiftU2, stencil_idx_3_0_0_VVV)))) + (0.234375000000000 * x496) + (-0.0937500000000000 * x493))) + (DYI * (x490 + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_0_m3_0_VVV) + stencil(evo_shiftU2, stencil_idx_0_3_0_VVV)))) + (0.234375000000000 * x473) + (-0.0937500000000000 * x486))) + (DZI * (x490 + ((1.0 / 64.0) * ((stencil(evo_shiftU2, stencil_idx_0_0_m3_VVV) + stencil(evo_shiftU2, stencil_idx_0_0_3_VVV)))) + (0.234375000000000 * x508) + (-0.0937500000000000 * x505)))))));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}