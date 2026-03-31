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
void adm2bssn_pt2(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_adm2bssn_pt2;
    DECLARE_CCTK_PARAMETERS;
    using vreal = Arith::simd<CCTK_REAL>;
    constexpr std::size_t vsize = std::tuple_size_v<vreal>;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("adm2bssn_pt2");
    #endif
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define ConfConnectU0_layout VVV_layout
    #define ConfConnectU1_layout VVV_layout
    #define ConfConnectU2_layout VVV_layout
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
        vreal x110 = stencil(gtDD11, stencil_idx_0_m2_0_VVV);
        vreal x111 = stencil(gtDD11, stencil_idx_0_2_0_VVV);
        vreal x112 = stencil(gtDD11, stencil_idx_0_m1_0_VVV);
        vreal x113 = stencil(gtDD11, stencil_idx_0_1_0_VVV);
        vreal x45 = stencil(gtDD11, stencil_idx_0_0_m2_VVV);
        vreal x46 = stencil(gtDD11, stencil_idx_0_0_2_VVV);
        vreal x47 = stencil(gtDD11, stencil_idx_0_0_m1_VVV);
        vreal x48 = stencil(gtDD11, stencil_idx_0_0_1_VVV);
        vreal x100 = stencil(gtDD11, stencil_idx_m2_0_0_VVV);
        vreal x101 = stencil(gtDD11, stencil_idx_2_0_0_VVV);
        vreal x102 = stencil(gtDD11, stencil_idx_m1_0_0_VVV);
        vreal x103 = stencil(gtDD11, stencil_idx_1_0_0_VVV);
        vreal x137 = (access(gtDD11, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV));
        vreal x165 = stencil(gtDD22, stencil_idx_0_0_m2_VVV);
        vreal x166 = stencil(gtDD22, stencil_idx_0_0_2_VVV);
        vreal x167 = stencil(gtDD22, stencil_idx_0_0_m1_VVV);
        vreal x168 = stencil(gtDD22, stencil_idx_0_0_1_VVV);
        vreal x36 = stencil(gtDD22, stencil_idx_0_m2_0_VVV);
        vreal x37 = stencil(gtDD22, stencil_idx_0_2_0_VVV);
        vreal x38 = stencil(gtDD22, stencil_idx_0_m1_0_VVV);
        vreal x39 = stencil(gtDD22, stencil_idx_0_1_0_VVV);
        vreal x76 = stencil(gtDD22, stencil_idx_m2_0_0_VVV);
        vreal x77 = stencil(gtDD22, stencil_idx_2_0_0_VVV);
        vreal x78 = stencil(gtDD22, stencil_idx_m1_0_0_VVV);
        vreal x79 = stencil(gtDD22, stencil_idx_1_0_0_VVV);
        vreal x162 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD11, stencil_idx_0_0_0_VVV));
        vreal x42 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV));
        vreal x82 = stencil(gtDD00, stencil_idx_0_0_m2_VVV);
        vreal x83 = stencil(gtDD00, stencil_idx_0_0_2_VVV);
        vreal x84 = stencil(gtDD00, stencil_idx_0_0_m1_VVV);
        vreal x85 = stencil(gtDD00, stencil_idx_0_0_1_VVV);
        vreal x93 = stencil(gtDD00, stencil_idx_0_m2_0_VVV);
        vreal x94 = stencil(gtDD00, stencil_idx_0_2_0_VVV);
        vreal x95 = stencil(gtDD00, stencil_idx_0_m1_0_VVV);
        vreal x96 = stencil(gtDD00, stencil_idx_0_1_0_VVV);
        vreal x140 = stencil(gtDD00, stencil_idx_m2_0_0_VVV);
        vreal x141 = stencil(gtDD00, stencil_idx_2_0_0_VVV);
        vreal x142 = stencil(gtDD00, stencil_idx_m1_0_0_VVV);
        vreal x143 = stencil(gtDD00, stencil_idx_1_0_0_VVV);
        vreal x31 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x128 = stencil(gtDD12, stencil_idx_0_m2_0_VVV);
        vreal x129 = stencil(gtDD12, stencil_idx_0_2_0_VVV);
        vreal x130 = stencil(gtDD12, stencil_idx_0_m1_0_VVV);
        vreal x131 = stencil(gtDD12, stencil_idx_0_1_0_VVV);
        vreal x178 = stencil(gtDD12, stencil_idx_0_0_m2_VVV);
        vreal x179 = stencil(gtDD12, stencil_idx_0_0_2_VVV);
        vreal x180 = stencil(gtDD12, stencil_idx_0_0_m1_VVV);
        vreal x181 = stencil(gtDD12, stencil_idx_0_0_1_VVV);
        vreal x65 = stencil(gtDD12, stencil_idx_m2_0_0_VVV);
        vreal x66 = stencil(gtDD12, stencil_idx_2_0_0_VVV);
        vreal x67 = stencil(gtDD12, stencil_idx_m1_0_0_VVV);
        vreal x68 = stencil(gtDD12, stencil_idx_1_0_0_VVV);
        vreal x52 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV));
        vreal x53 = (((1.0 / 2.0) * x52) + ((-1.0 / 2.0) * access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV)));
        vreal x91 = ((2 * x52) + (-2 * access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV)));
        vreal x116 = (access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x74 = (access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD11, stencil_idx_0_0_0_VVV));
        vreal x171 = stencil(gtDD02, stencil_idx_0_0_m2_VVV);
        vreal x172 = stencil(gtDD02, stencil_idx_0_0_2_VVV);
        vreal x173 = stencil(gtDD02, stencil_idx_0_0_m1_VVV);
        vreal x174 = stencil(gtDD02, stencil_idx_0_0_1_VVV);
        vreal x55 = stencil(gtDD02, stencil_idx_0_m2_0_VVV);
        vreal x56 = stencil(gtDD02, stencil_idx_0_2_0_VVV);
        vreal x57 = stencil(gtDD02, stencil_idx_0_m1_0_VVV);
        vreal x58 = stencil(gtDD02, stencil_idx_0_1_0_VVV);
        vreal x146 = stencil(gtDD02, stencil_idx_m2_0_0_VVV);
        vreal x147 = stencil(gtDD02, stencil_idx_2_0_0_VVV);
        vreal x148 = stencil(gtDD02, stencil_idx_m1_0_0_VVV);
        vreal x149 = stencil(gtDD02, stencil_idx_1_0_0_VVV);
        vreal x32 = ((2 * x31) + (-2 * access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD02, stencil_idx_0_0_0_VVV)));
        vreal x34 = (((1.0 / 2.0) * x31) + ((-1.0 / 2.0) * access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD02, stencil_idx_0_0_0_VVV)));
        vreal x126 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD02, stencil_idx_0_0_0_VVV));
        vreal x73 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x118 = stencil(gtDD01, stencil_idx_0_m2_0_VVV);
        vreal x119 = stencil(gtDD01, stencil_idx_0_2_0_VVV);
        vreal x120 = stencil(gtDD01, stencil_idx_0_m1_0_VVV);
        vreal x121 = stencil(gtDD01, stencil_idx_0_1_0_VVV);
        vreal x60 = stencil(gtDD01, stencil_idx_0_0_m2_VVV);
        vreal x61 = stencil(gtDD01, stencil_idx_0_0_2_VVV);
        vreal x62 = stencil(gtDD01, stencil_idx_0_0_m1_VVV);
        vreal x63 = stencil(gtDD01, stencil_idx_0_0_1_VVV);
        vreal x154 = stencil(gtDD01, stencil_idx_m2_0_0_VVV);
        vreal x155 = stencil(gtDD01, stencil_idx_2_0_0_VVV);
        vreal x156 = stencil(gtDD01, stencil_idx_m1_0_0_VVV);
        vreal x157 = stencil(gtDD01, stencil_idx_1_0_0_VVV);
        vreal x138 = pow2(access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x163 = pow2(access(gtDD01, stencil_idx_0_0_0_VVV));
        vreal x43 = pow2(access(gtDD02, stencil_idx_0_0_0_VVV));
        vreal x117 = (x52 + (-(x116)));
        x116 = (x31 + (-(x126)));
        x126 = (DYI * (((1.0 / 12.0) * ((x128 + (-(x129))))) + ((2.0 / 3.0) * ((x131 + (-(x130)))))));
        x128 = (((1.0 / 12.0) * ((x45 + (-(x46))))) + ((2.0 / 3.0) * ((x48 + (-(x47))))));
        x45 = (x126 + ((-1.0 / 2.0) * DZI * x128));
        x46 = (((1.0 / 2.0) * x137) + ((-1.0 / 2.0) * x138));
        x47 = (x73 + (-(x74)));
        x48 = (DYI * (((1.0 / 12.0) * ((x110 + (-(x111))))) + ((2.0 / 3.0) * ((x113 + (-(x112)))))));
        x110 = (-(x53));
        x53 = (x110 * x48);
        x111 = (DXI * (((1.0 / 12.0) * ((x100 + (-(x101))))) + ((2.0 / 3.0) * ((x103 + (-(x102)))))));
        x100 = (((1.0 / 12.0) * ((x118 + (-(x119))))) + ((2.0 / 3.0) * ((x121 + (-(x120))))));
        x118 = (x111 + (-2 * DYI * x100));
        x119 = (-(x118));
        x120 = (x53 + (x119 * x46) + (x45 * x47));
        x121 = (DYI * (((1.0 / 12.0) * ((x36 + (-(x37))))) + ((2.0 / 3.0) * ((x39 + (-(x38)))))));
        x36 = (((1.0 / 12.0) * ((x178 + (-(x179))))) + ((2.0 / 3.0) * ((x181 + (-(x180))))));
        x178 = (DZI * x36);
        x179 = (((1.0 / 2.0) * x121) + (-(x178)));
        x180 = (DZI * (((1.0 / 12.0) * ((x165 + (-(x166))))) + ((2.0 / 3.0) * ((x168 + (-(x167)))))));
        x165 = (((1.0 / 2.0) * x73) + ((-1.0 / 2.0) * x74));
        x166 = (x165 * x180);
        x167 = (DXI * (((1.0 / 12.0) * ((x76 + (-(x77))))) + ((2.0 / 3.0) * ((x79 + (-(x78)))))));
        x76 = (((1.0 / 12.0) * ((x171 + (-(x172))))) + ((2.0 / 3.0) * ((x174 + (-(x173))))));
        x171 = (x167 + (-2 * DZI * x76));
        x172 = (-(x171));
        x173 = (x166 + (x117 * x179) + (x172 * x46));
        x174 = (DYI * (((1.0 / 12.0) * ((x55 + (-(x56))))) + ((2.0 / 3.0) * ((x58 + (-(x57)))))));
        x55 = (DZI * (((1.0 / 12.0) * ((x60 + (-(x61))))) + ((2.0 / 3.0) * ((x63 + (-(x62)))))));
        x60 = (DXI * (((1.0 / 12.0) * ((x65 + (-(x66))))) + ((2.0 / 3.0) * ((x68 + (-(x67)))))));
        x65 = (x174 + x55 + (-(x60)));
        x66 = (x121 * x165);
        x67 = (DZI * x128);
        x68 = (x110 * x67);
        x61 = (x66 + x68 + (x46 * x65));
        x62 = ((-2 * x74) + (2 * x73));
        x73 = (x42 + (-(x43)));
        x74 = (x137 + (-(x138)));
        x137 = (x162 + (-(x163)));
        x138 = (x165 * x167);
        x63 = (((1.0 / 12.0) * ((x82 + (-(x83))))) + ((2.0 / 3.0) * ((x85 + (-(x84))))));
        x82 = (DZI * x63);
        x83 = (x46 * x82);
        x84 = (x55 + x60 + (-(x174)));
        x85 = (x110 * x84);
        x56 = (x138 + x83 + x85);
        x57 = (x110 * x111);
        x58 = (((1.0 / 12.0) * ((x93 + (-(x94))))) + ((2.0 / 3.0) * ((x96 + (-(x95))))));
        x93 = (DYI * x58);
        x94 = (x46 * x93);
        x95 = (x174 + x60 + (-(x55)));
        x96 = (x165 * x95);
        x77 = (x57 + x94 + x96);
        x78 = (DXI * (((1.0 / 12.0) * ((x140 + (-(x141))))) + ((2.0 / 3.0) * ((x143 + (-(x142)))))));
        x140 = (x46 * x78);
        x141 = (DXI * (((1.0 / 12.0) * ((x146 + (-(x147))))) + ((2.0 / 3.0) * ((x149 + (-(x148)))))));
        x146 = (x141 + ((-1.0 / 2.0) * DZI * x63));
        x147 = (x146 * x47);
        x148 = (DXI * (((1.0 / 12.0) * ((x154 + (-(x155))))) + ((2.0 / 3.0) * ((x157 + (-(x156)))))));
        x154 = (x148 + ((-1.0 / 2.0) * DYI * x58));
        x155 = (-(x154));
        x156 = (x117 * x155);
        x157 = (x140 + x147 + x156);
        x149 = (-(x32));
        x32 = (-(x91));
        store(ConfConnectU0, stencil_idx_0_0_0_VVV, ((x120 * x73) + (x137 * x173) + (x149 * x61) + (x157 * x74) + (x32 * x77) + (x56 * x62)));
        x91 = (x110 * x78);
        x142 = (-(x146));
        x143 = (((1.0 / 2.0) * x42) + ((-1.0 / 2.0) * x43));
        x42 = (-(x143));
        x43 = ((2 * x148) + (-1 * DYI * x58));
        x79 = (-(x43));
        x168 = (x91 + (x116 * x142) + (x42 * x79));
        x181 = (DZI * x76);
        x37 = (((1.0 / 2.0) * x167) + (-(x181)));
        x38 = (-(x34));
        x34 = (x180 * x38);
        x39 = (x121 + (-2 * DZI * x36));
        x101 = (-(x39));
        x102 = (x34 + (x101 * x143) + (x117 * x37));
        x103 = (x167 * x38);
        x112 = (x110 * x82);
        x113 = (x103 + x112 + (x143 * x84));
        x129 = (x111 * x143);
        x130 = (x38 * x95);
        x131 = (x110 * x93);
        x31 = (x129 + x130 + x131);
        x52 = (x143 * x48);
        vreal x123 = (DYI * x100);
        vreal x124 = (((1.0 / 2.0) * x111) + (-(x123)));
        x123 = (x117 * x124);
        x117 = (-(x45));
        vreal x135 = (x116 * x117);
        vreal x136 = (x123 + x135 + x52);
        x135 = (x121 * x38);
        vreal x51 = (x143 * x67);
        vreal x71 = (x110 * x65);
        vreal x72 = (x135 + x51 + x71);
        store(ConfConnectU1, stencil_idx_0_0_0_VVV, ((x102 * x137) + (x113 * x62) + (x136 * x73) + (x149 * x72) + (x168 * x74) + (x31 * x32)));
        x136 = ((-2 * x126) + (DZI * x128));
        x72 = (x38 * x48);
        x51 = (-(x124));
        x124 = (((1.0 / 2.0) * x162) + ((-1.0 / 2.0) * x163));
        x162 = (-(x124));
        x163 = (x72 + (x136 * x162) + (x47 * x51));
        x71 = (x165 * x78);
        vreal x239 = ((2 * x141) + (-1 * DZI * x63));
        vreal x240 = (-(x239));
        x239 = (x71 + (x116 * x155) + (x162 * x240));
        x240 = (x111 * x38);
        vreal x226 = (x165 * x93);
        vreal x227 = (x226 + x240 + (x124 * x95));
        x226 = (x38 * x67);
        vreal x218 = (x121 * x124);
        vreal x219 = (x165 * x65);
        vreal x220 = (x218 + x219 + x226);
        x218 = (x165 * x82);
        x219 = (x124 * x167);
        vreal x223 = (x38 * x84);
        vreal x224 = (x218 + x219 + x223);
        x223 = (x124 * x180);
        vreal x229 = (-(x37));
        vreal x230 = (x229 * x47);
        x229 = (x116 * x179);
        vreal x232 = (x223 + x229 + x230);
        store(ConfConnectU2, stencil_idx_0_0_0_VVV, ((x137 * x232) + (x149 * x220) + (x163 * x73) + (x224 * x62) + (x227 * x32) + (x239 * x74)));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}